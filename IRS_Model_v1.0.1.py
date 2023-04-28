import cmath
import os

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.constants as constants
import scipy.optimize as optimize
from scipy.fft import fft2, ifft2
from tqdm import tqdm
import random


def reflection_coefficients(Z0, Z1_n):
    """Calculate reflection coefficients for given impedances."""
    return (Z1_n - Z0) / (Z1_n + Z0)


def freespace_impedance():
    epsilon_0 = constants.epsilon_0
    mu_0 = constants.mu_0
    return math.sqrt(mu_0 / epsilon_0)


def element_impedance(R, L, C, w):
    jwL = 1j * w * L
    jwC = 1j * w * C
    eq = R + 1 / jwC
    z = (jwL * eq) / (jwL + eq)
    return z


def calculate_dphi_dx_dy(transmitter, receiver, surface_size, element_size, element_spacing, wavelength, ni):
    k0 = 2 * np.pi / wavelength
    m_values, n_values = np.meshgrid(np.arange(surface_size[0]), np.arange(surface_size[1]), indexing='ij')

    x_mn = (element_size / 2) + (m_values * element_spacing) + (m_values * element_size)
    y_mn = (element_size / 2) + (n_values * element_spacing) + (n_values * element_size)
    z_mn = np.zeros_like(x_mn)

    I = np.stack((x_mn - transmitter[0], y_mn - transmitter[1], z_mn - transmitter[2]), axis=-1)
    R = np.stack((receiver[0] - x_mn, receiver[1] - y_mn, receiver[2] - z_mn), axis=-1)

    normal = np.array([0, 0, 1])

    I_norm = np.linalg.norm(I, axis=-1)
    R_norm = np.linalg.norm(R, axis=-1)

    theta_i = np.arccos(np.dot(-I, normal) / I_norm)
    # theta_r = np.arccos(np.dot(R, normal) / R_norm)
    #
    # # Calculate angle between plane of incidence and projection of reflected vector onto plane perpendicular to incident vector
    # I_unit = I / I_norm[..., np.newaxis]
    # R_proj = R - np.sum(R * I_unit, axis=-1)[..., np.newaxis] * I_unit
    # N_plane = np.cross(I, normal[np.newaxis, np.newaxis, :])
    # cos_phi_r = np.sum(R_proj * N_plane, axis=-1) / (R_norm * np.linalg.norm(N_plane, axis=-1))
    # sin_phi_r = np.linalg.norm(np.cross(R_proj, N_plane), axis=-1) / (R_norm * np.linalg.norm(N_plane, axis=-1))
    # phi_r = np.arctan2(sin_phi_r, cos_phi_r)

    R_proj = R.copy()
    R_proj[:, :, 0] = 0  # Projection of R onto the YZ plane
    R_proj_mag = np.linalg.norm(R_proj, axis=2)

    # Calculate theta_r the angle between the reflected vector and its projection onto the YZ plane
    dot_product = np.sum(R * R_proj, axis=2)
    theta_r = np.arccos(dot_product / (R_norm * R_proj_mag))

    # Calculate angle between the projection of reflected vector onto the YZ plane and the z-axis
    dot_product = np.sum(R_proj * normal, axis=2)
    phi_r = np.arccos(dot_product / R_proj_mag)

    dphi_dx = (np.sin(theta_r) - np.sin(theta_i)) * ni * k0
    dphi_dy = np.cos(theta_r) * np.sin(phi_r) * ni * k0

    return dphi_dx, dphi_dy


# Calculate the phase shift array from the phase gradient arrays (dphi_dx, dphi_dy) using Finite Difference Method
def calculate_phase_shifts_with_FDM(dphi_dx, dphi_dy, delta_x, delta_y, save_results=False):
    """
    f(x,y) = [f(x-1,y) + (df/dx * ∆x)] + [f(x,y-1) + (df/dy * ∆y)]
    """

    # Integrate along the x-axis
    phase_shifts_x_y0 = np.cumsum(dphi_dx * delta_x, axis=1)

    # Integrate along the y-axis
    phase_shifts_x0_y = np.cumsum(dphi_dy * delta_y, axis=0)

    phase_shifts = phase_shifts_x_y0 + phase_shifts_x0_y

    dphi_dx_recovered = np.gradient(phase_shifts, axis=1) / delta_x
    abs_diffdx = np.abs(dphi_dx - dphi_dx_recovered)
    maedx = np.mean(abs_diffdx)
    msedx = np.mean(abs_diffdx ** 2)
    dphi_dy_recovered = np.gradient(phase_shifts, axis=0) / delta_y
    abs_diffdy = np.abs(dphi_dy - dphi_dy_recovered)
    maedy = np.mean(abs_diffdy)
    msedy = np.mean(abs_diffdy ** 2)
    print("finite_difference_method")
    print("mae_dphi_dx", maedx)
    print("mse_dphi_dx", msedx)
    print("mae_dphi_dy", maedy)
    print("mse_dphi_dy", msedy)
    print()

    if save_results:
        results_file = open("./Results_model_v1-0-1/FunctionEstimationErrors.txt", "a")
        results_file.write("Finite Difference Method Errors:\n")
        results_file.write(f"\tmae_dphi_dx: {maedx} \n")
        results_file.write(f"\tmse_dphi_dx: {msedx} \n")
        results_file.write(f"\tmae_dphi_dy: {maedy} \n")
        results_file.write(f"\tmse_dphi_dy: {msedy} \n")
        results_file.write("\n")
        results_file.close()

    phase_shifts = np.mod(phase_shifts + np.pi, 2 * np.pi) - np.pi

    return phase_shifts


# Calculate the phase shift array from the phase gradient arrays (dphi_dx, dphi_dy)
def calculate_phase_shifts_with_second_derivative(dphi_dx, dphi_dy, delta_x, delta_y, save_results=False):
    dphi2_dxdy = np.zeros(dphi_dx.shape)

    for y in range(dphi_dx.shape[0]):
        for x in range(dphi_dx.shape[1]):
            count = 0
            term1, term2, term3, term4 = 0, 0, 0, 0
            if y != dphi_dx.shape[0] - 1:
                count += 1
                term1 = (dphi_dx[y + 1, x] - dphi_dx[y, x]) / delta_y
            if y != 0:
                count += 1
                term2 = (dphi_dx[y, x] - dphi_dx[y - 1, x]) / delta_y
            if x != dphi_dy.shape[1] - 1:
                count += 1
                term3 = (dphi_dy[y, x + 1] - dphi_dy[y, x]) / delta_x
            if x != 0:
                count += 1
                term4 = (dphi_dy[y, x] - dphi_dy[y, x - 1]) / delta_x

            dphi2_dxdy[y, x] = (term1 + term2 + term3 + term4) / count

    phase_shifts = np.cumsum(np.cumsum(dphi2_dxdy * delta_y, axis=0) * delta_x, axis=1)

    dphi_dx_recovered = np.gradient(phase_shifts, axis=1) / delta_x
    abs_diffdx = np.abs(dphi_dx - dphi_dx_recovered)
    maedx = np.mean(abs_diffdx)
    msedx = np.mean(abs_diffdx ** 2)
    dphi_dy_recovered = np.gradient(phase_shifts, axis=0) / delta_y
    abs_diffdy = np.abs(dphi_dy - dphi_dy_recovered)
    maedy = np.mean(abs_diffdy)
    msedy = np.mean(abs_diffdy ** 2)
    print("averaging_method")
    print("mae_dphi_dx", maedx)
    print("mse_dphi_dx", msedx)
    print("mae_dphi_dy", maedy)
    print("mse_dphi_dy", msedy)
    print()

    if save_results:
        results_file = open("./Results_model_v1-0-1/FunctionEstimationErrors.txt", "a")
        results_file.write("Averaging Method Errors:\n")
        results_file.write(f"\tmae_dphi_dx: {maedx} \n")
        results_file.write(f"\tmse_dphi_dx: {msedx} \n")
        results_file.write(f"\tmae_dphi_dy: {maedy} \n")
        results_file.write(f"\tmse_dphi_dy: {msedy} \n")
        results_file.write("\n")
        results_file.close()

    phase_shifts = np.mod(phase_shifts + np.pi, 2 * np.pi) - np.pi

    return phase_shifts


# Calculate the phase shift array from the phase gradient arrays (dphi_dx, dphi_dy) using Random Walk
def calculate_phase_shifts_with_random_walk(dphi_dx, dphi_dy, delta_x, delta_y, save_results=False):
    """
    Calculates the phase_shifts from the partial derivatives dphi_dx, dphi_dy using "Random Walk Method".
    Random Walk in a loop that mase sure that all elements are visited at least 100 times.
    """
    phase_shifts = np.zeros(dphi_dx.shape)

    curr_x, curr_y = 0, 0

    visited_elements = np.zeros(dphi_dx.shape, dtype=int)
    visited_elements[curr_y, curr_x] = 20

    i = 0
    while np.min(visited_elements) < 20:
        i += 1
        new_direction = random.randint(1, 4)
        # Directions:
        #     1 = Right (-->)
        #     2 = Left (<--)
        #     3 = Down
        #     4 = Up

        if not ((new_direction == 2 and curr_x == 1 and curr_y == 0) or (
                new_direction == 4 and curr_x == 0 and curr_y == 1)):

            if new_direction == 1 and curr_x < phase_shifts.shape[1] - 1:
                # phase_shifts[curr_y, curr_x + 1] = phase_shifts[curr_y, curr_x] + delta_x * dphi_dx[curr_y, curr_x]
                if phase_shifts[curr_y, curr_x + 1] != 0:
                    phase_shifts[curr_y, curr_x + 1] = (phase_shifts[curr_y, curr_x + 1] + phase_shifts[
                        curr_y, curr_x] + delta_x * dphi_dx[curr_y, curr_x]) / 2
                else:
                    phase_shifts[curr_y, curr_x + 1] = phase_shifts[curr_y, curr_x] + delta_x * dphi_dx[
                        curr_y, curr_x]
                curr_x += 1
            elif new_direction == 2 and curr_x > 0:
                # phase_shifts[curr_y, curr_x - 1] = phase_shifts[curr_y, curr_x] - delta_x * dphi_dx[curr_y, curr_x]
                if phase_shifts[curr_y, curr_x - 1] != 0:
                    phase_shifts[curr_y, curr_x - 1] = (phase_shifts[curr_y, curr_x - 1] + phase_shifts[
                        curr_y, curr_x] - delta_x * dphi_dx[curr_y, curr_x]) / 2
                else:
                    phase_shifts[curr_y, curr_x - 1] = phase_shifts[curr_y, curr_x] - delta_x * dphi_dx[
                        curr_y, curr_x]
                curr_x -= 1
            elif new_direction == 3 and curr_y < phase_shifts.shape[0] - 1:
                # phase_shifts[curr_y + 1, curr_x] = phase_shifts[curr_y, curr_x] + delta_y * dphi_dy[curr_y, curr_x]
                if phase_shifts[curr_y + 1, curr_x] != 0:
                    phase_shifts[curr_y + 1, curr_x] = (phase_shifts[curr_y + 1, curr_x] + phase_shifts[
                        curr_y, curr_x] + delta_y * dphi_dy[curr_y, curr_x]) / 2
                else:
                    phase_shifts[curr_y + 1, curr_x] = phase_shifts[curr_y, curr_x] + delta_y * dphi_dy[
                        curr_y, curr_x]
                curr_y += 1
            elif new_direction == 4 and curr_y > 0:
                # phase_shifts[curr_y - 1, curr_x] = phase_shifts[curr_y, curr_x] - delta_y * dphi_dy[curr_y, curr_x]
                if phase_shifts[curr_y - 1, curr_x] != 0:
                    phase_shifts[curr_y - 1, curr_x] = (phase_shifts[curr_y - 1, curr_x] + phase_shifts[
                        curr_y, curr_x] - delta_y * dphi_dy[curr_y, curr_x]) / 2
                else:
                    phase_shifts[curr_y - 1, curr_x] = phase_shifts[curr_y, curr_x] - delta_y * dphi_dy[
                        curr_y, curr_x]
                curr_y -= 1
            else:
                continue
            visited_elements[curr_y, curr_x] += 1

    dphi_dx_recovered = np.gradient(phase_shifts, axis=1) / delta_x
    abs_diffdx = np.abs(dphi_dx - dphi_dx_recovered)
    maedx = np.mean(abs_diffdx)
    msedx = np.mean(abs_diffdx ** 2)
    dphi_dy_recovered = np.gradient(phase_shifts, axis=0) / delta_y
    abs_diffdy = np.abs(dphi_dy - dphi_dy_recovered)
    maedy = np.mean(abs_diffdy)
    msedy = np.mean(abs_diffdy ** 2)
    print("random_walk_method Errors")
    print("mae_dphi_dx", maedx)
    print("mse_dphi_dx", msedx)
    print("mae_dphi_dy", maedy)
    print("mse_dphi_dy", msedy)

    print("Minimum Element visited:", np.min(visited_elements))
    print("Maximum Element visited:", np.max(visited_elements))
    print("Average number of times each element is visited", np.average(visited_elements))
    print("Standard Deviation between the number of visits for each element", np.round(np.std(visited_elements), 2))

    print()

    if save_results:
        results_file = open("./Results_model_v1-0-1/FunctionEstimationErrors.txt", "a")
        results_file.write("Random Walk Method:\n")
        results_file.write(f"\tmae_dphi_dx: {maedx} \n")
        results_file.write(f"\tmse_dphi_dx: {msedx} \n")
        results_file.write(f"\tmae_dphi_dy: {maedy} \n")
        results_file.write(f"\tmse_dphi_dy: {msedy} \n")
        results_file.write("\n")
        results_file.close()

    phase_shifts = np.mod(phase_shifts + np.pi, 2 * np.pi) - np.pi

    return phase_shifts


# Calculate the phase shift array from the phase gradient arrays (dphi_dx, dphi_dy) with random walk
def calculate_phase_shifts_with_random_walk2(dphi_dx, dphi_dy, delta_x, delta_y, save_results=False):
    """
    Calculates the phase_shifts from the partial derivatives dphi_dx, dphi_dy using "Random Walk Method".
    Random Walk in a loop that make sure that all elements are visited then repeat this process n times.
    """
    phase_shifts = np.zeros(dphi_dx.shape)

    for _ in tqdm(range(20)):
        curr_x, curr_y = 0, 0

        visited = np.zeros(dphi_dx.shape, dtype=bool)
        visited[curr_y, curr_x] = True

        while not np.all(visited):
            new_direction = random.randint(1, 4)
            # Directions:
            #     1 = Right (-->)
            #     2 = Left (<--)
            #     3 = Down
            #     4 = Up

            if not ((new_direction == 2 and curr_x == 1 and curr_y == 0) or (
                    new_direction == 4 and curr_x == 0 and curr_y == 1)):

                if new_direction == 1 and curr_x < phase_shifts.shape[1] - 1:
                    # phase_shifts[curr_y, curr_x + 1] = phase_shifts[curr_y, curr_x] + delta_x * dphi_dx[curr_y, curr_x]
                    if phase_shifts[curr_y, curr_x + 1] != 0:
                        phase_shifts[curr_y, curr_x + 1] = (phase_shifts[curr_y, curr_x + 1] + phase_shifts[
                            curr_y, curr_x] + delta_x * dphi_dx[curr_y, curr_x]) / 2
                    else:
                        phase_shifts[curr_y, curr_x + 1] = phase_shifts[curr_y, curr_x] + delta_x * dphi_dx[
                            curr_y, curr_x]
                    curr_x += 1
                elif new_direction == 2 and curr_x > 0:
                    # phase_shifts[curr_y, curr_x - 1] = phase_shifts[curr_y, curr_x] - delta_x * dphi_dx[curr_y, curr_x]
                    if phase_shifts[curr_y, curr_x - 1] != 0:
                        phase_shifts[curr_y, curr_x - 1] = (phase_shifts[curr_y, curr_x - 1] + phase_shifts[
                            curr_y, curr_x] - delta_x * dphi_dx[curr_y, curr_x]) / 2
                    else:
                        phase_shifts[curr_y, curr_x - 1] = phase_shifts[curr_y, curr_x] - delta_x * dphi_dx[
                            curr_y, curr_x]
                    curr_x -= 1
                elif new_direction == 3 and curr_y < phase_shifts.shape[0] - 1:
                    # phase_shifts[curr_y + 1, curr_x] = phase_shifts[curr_y, curr_x] + delta_y * dphi_dy[curr_y, curr_x]
                    if phase_shifts[curr_y + 1, curr_x] != 0:
                        phase_shifts[curr_y + 1, curr_x] = (phase_shifts[curr_y + 1, curr_x] + phase_shifts[
                            curr_y, curr_x] + delta_y * dphi_dy[curr_y, curr_x]) / 2
                    else:
                        phase_shifts[curr_y + 1, curr_x] = phase_shifts[curr_y, curr_x] + delta_y * dphi_dy[
                            curr_y, curr_x]
                    curr_y += 1
                elif new_direction == 4 and curr_y > 0:
                    # phase_shifts[curr_y - 1, curr_x] = phase_shifts[curr_y, curr_x] - delta_y * dphi_dy[curr_y, curr_x]
                    if phase_shifts[curr_y - 1, curr_x] != 0:
                        phase_shifts[curr_y - 1, curr_x] = (phase_shifts[curr_y - 1, curr_x] + phase_shifts[
                            curr_y, curr_x] - delta_y * dphi_dy[curr_y, curr_x]) / 2
                    else:
                        phase_shifts[curr_y - 1, curr_x] = phase_shifts[curr_y, curr_x] - delta_y * dphi_dy[
                            curr_y, curr_x]
                    curr_y -= 1
                else:
                    continue

                if not visited[curr_y, curr_x]:
                    visited[curr_y, curr_x] = True

    dphi_dx_recovered = np.gradient(phase_shifts, axis=1) / delta_x
    abs_diffdx = np.abs(dphi_dx - dphi_dx_recovered)
    maedx = np.mean(abs_diffdx)
    msedx = np.mean(abs_diffdx ** 2)
    dphi_dy_recovered = np.gradient(phase_shifts, axis=0) / delta_y
    abs_diffdy = np.abs(dphi_dy - dphi_dy_recovered)
    maedy = np.mean(abs_diffdy)
    msedy = np.mean(abs_diffdy ** 2)
    print("random_walk2_method Errors")
    print("mae_dphi_dx", maedx)
    print("mse_dphi_dx", msedx)
    print("mae_dphi_dy", maedy)
    print("mse_dphi_dy", msedy)
    print()

    if save_results:
        results_file = open("./Results_model_v1-0-1/FunctionEstimationErrors.txt", "a")
        results_file.write("Random Walk2 Method:\n")
        results_file.write(f"\tmae_dphi_dx: {maedx} \n")
        results_file.write(f"\tmse_dphi_dx: {msedx} \n")
        results_file.write(f"\tmae_dphi_dy: {maedy} \n")
        results_file.write(f"\tmse_dphi_dy: {msedy} \n")
        results_file.write("\n")
        results_file.close()

    phase_shifts = np.mod(phase_shifts + np.pi, 2 * np.pi) - np.pi

    return phase_shifts


def fft_poisson_solver(dphi_dx, dphi_dy, delta_x, delta_y):
    # Compute the Laplacian from the partial derivatives
    laplacian = (np.gradient(dphi_dx, delta_y, axis=0) + np.gradient(dphi_dy, delta_x, axis=1))

    # Perform 2D FFT on the Laplacian
    laplacian_fft = fft2(laplacian)

    # Compute the frequency components
    k_y, k_x = np.mgrid[0:dphi_dx.shape[0], 0:dphi_dx.shape[1]]
    k_y = k_y / dphi_dx.shape[0]
    k_x = k_x / dphi_dx.shape[1]
    k_y[k_y > 0.5] -= 1
    k_x[k_x > 0.5] -= 1
    k_y *= 2 * np.pi / delta_y
    k_x *= 2 * np.pi / delta_x

    # Divide by the frequency components' squares (excluding the zero-frequency component)
    k2 = k_x ** 2 + k_y ** 2
    k2[0, 0] = 1  # Avoid division by zero
    phase_shifts_fft = laplacian_fft / k2
    phase_shifts_fft[0, 0] = 0  # Set zero-frequency component to zero

    # Perform inverse 2D FFT to obtain the phase shift function
    phase_shifts = np.real(ifft2(phase_shifts_fft))

    dphi_dx_recovered = np.gradient(phase_shifts, axis=1) / delta_x
    abs_diffdx = np.abs(dphi_dx - dphi_dx_recovered)
    maedx = np.mean(abs_diffdx)
    msedx = np.mean(abs_diffdx ** 2)
    dphi_dy_recovered = np.gradient(phase_shifts, axis=0) / delta_y
    abs_diffdy = np.abs(dphi_dy - dphi_dy_recovered)
    maedy = np.mean(abs_diffdy)
    msedy = np.mean(abs_diffdy ** 2)
    print("fft_poisson_solver_method")
    print("mae_dphi_dx", maedx)
    print("mse_dphi_dx", msedx)
    print("mae_dphi_dy", maedy)
    print("mse_dphi_dy", msedy)
    print()

    phase_shifts = np.mod(phase_shifts + np.pi, 2 * np.pi) - np.pi

    return phase_shifts


def show_phase_shift_plots(phase_shifts, title, save_plot=False):
    plt.figure()
    plt.imshow(phase_shifts, cmap='viridis', origin='lower')
    plt.colorbar(label='Phase Shift (deg)')
    plt.title(title)
    plt.xlabel('Element Index (x)')
    plt.ylabel('Element Index (y)')
    if save_plot:
        plt.savefig(f"./Results_model_v1-0-1/{title}.png")


def draw_incident_reflected_wave(transmitter, receiver, surface_size, element_size, element_spacing, phi_matrix):
    phi_matrix_deg = np.rad2deg(phi_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw transmitter and receiver
    ax.scatter(*transmitter, color='red', label='Transmitter')
    ax.scatter(*receiver, color='blue', label='Receiver')
    # Add text labels
    ax.text(transmitter[0], transmitter[1], transmitter[2], 'Transmitter', fontsize=10, color='red')
    ax.text(receiver[0], receiver[1], receiver[2], 'Receiver', fontsize=10, color='blue')

    # Calculate the middle of the surface
    surface_middle = np.array([
        ((surface_size[1] * element_size) + ((surface_size[0] - 1) * element_spacing)) / 2,
        ((surface_size[0] * element_size) + ((surface_size[0] - 1) * element_spacing)) / 2,
        0
    ])

    # Create a colormap
    cmap = plt.get_cmap('viridis')

    # Draw IRS elements
    IRS_elements = []
    for i in range(surface_size[0]):
        for j in range(surface_size[1]):
            # element = np.array([j * element_size, i * element_size, 0])
            element = np.array(
                [(element_size / 2) + (j * element_spacing) + (j * element_size),
                 (element_size / 2) + (i * element_spacing) + (i * element_size), 0])
            IRS_elements.append(element)

            # Get the color from the phi_matrix
            color = cmap(phi_matrix_deg[i, j])
            ax.scatter(*element, color=color, marker='s')

    # Draw incident wave
    incident_wave = np.linspace(transmitter, surface_middle, 100)
    ax.plot(incident_wave[:, 0], incident_wave[:, 1], incident_wave[:, 2], 'r--', label='Incident Wave')

    # Draw reflected wave
    reflected_wave = np.linspace(surface_middle, receiver, 100)
    ax.plot(reflected_wave[:, 0], reflected_wave[:, 1], reflected_wave[:, 2], 'b--', label='Reflected Wave')

    # Draw normal vector to the surface
    normal_start = surface_middle
    normal_end = surface_middle + np.array([0, 0, max(transmitter[2], receiver[2])])
    ax.plot([normal_start[0], normal_end[0]], [normal_start[1], normal_end[1]], [normal_start[2], normal_end[2]], 'k-',
            label='Normal Vector')

    # Set axis labels and plot limits
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Set x-axis limits
    # ax.set_xlim(-0.05, 0.5)
    # ax.set_ylim(-0.1, 0.5)
    # ax.set_zlim(0, 1)

    # Set legend
    ax.legend()

    # plt.show()


def main():
    save_results = False
    # Parameters
    transmitter = np.array([1, 0.5, 10])  # Position of the transmitter
    receiver = np.array([1.5, 1.2, 40])  # Position of the receiver
    ni = 1  # Refractive index
    frequency = 2.4e9  # Frequency in Hz
    c = 3e8  # Speed of light in m/s
    wavelength = c / frequency  # Calculate wavelength
    surface_size = (50, 50)  # Metasurface dimensions (M, N)
    element_size = wavelength / 8
    element_spacing = wavelength / 8  # Element spacing in x and y

    surface_height = (surface_size[0] * element_size) + ((surface_size[0] - 1) * element_spacing)
    surface_width = (surface_size[1] * element_size) + ((surface_size[0] - 1) * element_spacing)
    surface_area = surface_height * surface_width
    print(f"Surface Height: {round(surface_height, 2)} m")
    print(f"Surface Width: {round(surface_width, 2)} m")
    print(f"Surface Area: {round(surface_area, 2)} m²")

    # calculate the phase shift needed
    dphi_dx, dphi_dy = calculate_dphi_dx_dy(transmitter, receiver, surface_size, element_size, element_spacing,
                                            wavelength, ni)

    if save_results:
        results_file = open("./Results_model_v1-0-1/FunctionEstimationErrors.txt", "w")
        results_file.close()

    delta = element_size + element_spacing
    phase_shifts = calculate_phase_shifts_with_FDM(dphi_dx, dphi_dy, delta, delta, save_results=save_results)
    phase_shifts1 = calculate_phase_shifts_with_second_derivative(dphi_dx, dphi_dy, delta, delta, save_results=save_results)
    phase_shifts2 = calculate_phase_shifts_with_random_walk(dphi_dx, dphi_dy, delta, delta, save_results=save_results)
    phase_shifts3 = calculate_phase_shifts_with_random_walk2(dphi_dx, dphi_dy, delta, delta, save_results=save_results)
    # phase_shifts3 = fft_poisson_solver(dphi_dx, dphi_dy, delta, delta)

    show_phase_shift_plots(np.degrees(phase_shifts), "phase_shifts - FDM", save_plot=save_results)
    show_phase_shift_plots(np.degrees(phase_shifts1), "phase_shifts - second derivative", save_plot=save_results)
    show_phase_shift_plots(np.degrees(phase_shifts2), "phase_shifts - random walk", save_plot=save_results)
    show_phase_shift_plots(np.degrees(phase_shifts3), "phase_shifts - random walk2", save_plot=save_results)
    # show_phase_shift_plots(np.degrees(phase_shifts4))

    if save_results:
        results_directory = "./Results_model_v1-0-1/"
        # os.makedirs(results_directory)

        results_file = open(os.path.join(results_directory, "results.txt"), "w")
        results_file.write(f"Incident Signal Wavelength: {round(wavelength, 3)} m\n")
        results_file.write(f"Surface Number of Elements: {surface_size}\n")
        results_file.write(f"Surface Elements Sizes: {round(element_size, 3)} m\n")
        results_file.write(f"Surface Elements spacings: {round(element_spacing, 3)} m\n")
        results_file.write(f"Surface Height: {round(surface_height, 2)} m\n")
        results_file.write(f"Surface Width: {round(surface_width, 2)} m\n")
        results_file.write(f"Surface Area: {round(surface_area, 2)} m²\n")
        results_file.close()

        np.savetxt(os.path.join(results_directory, "phase_shifts - FDM (in degrees).csv"), np.degrees(phase_shifts),
                   delimiter=",")
        np.savetxt(os.path.join(results_directory, "phase_shifts - second derivative (in degrees).csv"),
                   np.degrees(phase_shifts1), delimiter=",")
        np.savetxt(os.path.join(results_directory, "phase_shifts - random walk (in degrees).csv"),
                   np.degrees(phase_shifts2), delimiter=",")

    # draw_incident_reflected_wave(transmitter, receiver, surface_size, element_size, element_spacing, phase_shifts)
    plt.show()


if __name__ == "__main__":
    main()
