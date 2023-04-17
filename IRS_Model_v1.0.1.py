import cmath

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.constants as constants
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimize
from tqdm import tqdm


def cartesian_to_spherical(cartesian_coords):
    x, y, z = cartesian_coords.T
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan(y / x)
    spherical_coords = np.array([r, theta, phi]).T
    return spherical_coords


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


def phi(R, L, C, w):
    Z0 = freespace_impedance()
    Z1 = element_impedance(R, L, C, w)
    Z = reflection_coefficients(Z1, Z0)
    return cmath.phase(Z)


def find_required_capacitance(R, L, w, phi_z):
    Cn_guess = 1e-12  # Initial guess for Cn
    Cn_solution = optimize.fsolve(lambda C: phi(R, L, C, w) - phi_z, Cn_guess)
    return Cn_solution


def inverse_varactor(C, C_min, C_max, V_min, V_max):
    """Find the bias voltage needed to achieve the required capacitance value."""
    # Assuming a simple linear relationship between bias voltage and capacitance
    V = V_min + (V_max - V_min) * (C - C_min) / (C_max - C_min)
    return V


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

    theta_i = np.arccos(np.dot(I, normal) / I_norm)
    theta_r = np.arccos(np.dot(R, normal) / R_norm)

    # Calculate angle between plane of incidence and projection of reflected vector onto plane perpendicular to incident vector
    I_unit = I / I_norm[..., np.newaxis]
    R_proj = R - np.sum(R * I_unit, axis=-1)[..., np.newaxis] * I_unit
    N_plane = np.cross(I, normal[np.newaxis, np.newaxis, :])
    cos_phi_r = np.sum(R_proj * N_plane, axis=-1) / (R_norm * np.linalg.norm(N_plane, axis=-1))
    sin_phi_r = np.linalg.norm(np.cross(R_proj, N_plane), axis=-1) / (R_norm * np.linalg.norm(N_plane, axis=-1))
    phi_r = np.arctan2(sin_phi_r, cos_phi_r)

    # R_proj = R.copy()
    # R_proj[:, :, 0] = 0  # Projection of R onto the YZ plane
    # R_proj_mag = np.linalg.norm(R_proj, axis=2)
    #
    # # Calculate theta_r the angle between the reflected vector and its projection onto the YZ plane
    # dot_product = np.sum(R * R_proj, axis=2)
    # theta_r = np.arccos(dot_product / (R_norm * R_proj_mag))
    #
    # # Calculate angle between the projection of reflected vector onto the YZ plane and the z-axis
    # dot_product = np.sum(R_proj * normal, axis=2)
    # phi_r = np.arccos(dot_product / R_proj_mag)

    dphi_dx = (np.sin(theta_r) - np.sin(theta_i)) * ni * k0
    dphi_dy = np.cos(theta_r) * np.sin(phi_r) * ni * k0

    return dphi_dx, dphi_dy


# Calculate the phase shift array from the phase gradient arrays (dphi_dx, dphi_dy)
def calculate_phase_shifts_from_gradients(dphi_dx, dphi_dy, delta_x, delta_y, f_init=0):
    # Integrate along the x-axis
    phase_shifts_x_y0 = np.cumsum(dphi_dx * delta_x, axis=0) + f_init

    # Integrate along the y-axis
    phase_shifts = np.cumsum(dphi_dy * delta_y, axis=1) + phase_shifts_x_y0

    phase_shifts = np.mod(phase_shifts + np.pi, 2 * np.pi) - np.pi

    return phase_shifts


def power_received(phase_shifts, surface_size, element_size, element_spacing, transmitter, receiver, wavelength, ni):
    num_rows, num_columns = surface_size

    Z0 = freespace_impedance()
    R_value = 1
    L_value = 2.5e-9
    f = constants.speed_of_light / wavelength
    w = 2 * math.pi * f

    A_i = 1
    phi_i = math.radians(30)
    # incident_wave_n = A_i * np.cos(w * t + phi_i)

    capacitance_matrix = np.zeros(surface_size)

    elements_reflection_coefficients = []
    rays_distances = []

    transmitted_power = np.power(A_i, 2) / 2
    # minimum_distance = np.min(rays_distances)
    k = 2 * np.pi * ni / wavelength
    term1 = transmitted_power * np.power((wavelength / (4 * np.pi)), 2)
    term2 = 0
    received_powers = []

    pbar = tqdm(total=(num_rows * num_columns), desc='Progress')
    for y in range(num_rows):
        for x in range(num_columns):
            # Find the required capacitance value for the desired phase shift
            C_n = find_required_capacitance(R_value, L_value, w, phase_shifts[y, x])
            capacitance_matrix[y, x] = C_n

            # Calculate impedance for the current element
            Z1_n = element_impedance(R_value, L_value, C_n, w)

            # Calculate reflection coefficient for the current element
            reflection_coefficient = reflection_coefficients(Z0, Z1_n)
            # elements_reflection_coefficients.append(reflection_coefficient)

            # Position of the current element
            element_position = np.array(
                [(element_size / 2) + (x * element_spacing) + (x * element_size),
                 (element_size / 2) + (y * element_spacing) + (y * element_size), 0])
            # Calculate incident vector and reflected vector
            incident_vector = element_position - transmitter
            incidence_distance = np.linalg.norm(incident_vector)
            reflected_vector = receiver - element_position
            reflection_distance = np.linalg.norm(reflected_vector)

            # rays_distances.append(incidence_distance + reflection_distance)
            rays_distance = incidence_distance + reflection_distance
            rays_distances.append(rays_distance)

            term2 += reflection_coefficient * np.exp(1j * k * rays_distance) / rays_distance

            received_powers.append(term1 * np.power(np.abs(term2), 2))

            pbar.update(1)

    # for i in range(len(rays_distances)):
    #     # reflection_coefficient_amplitude = abs(elements_reflection_coefficients[i])
    #     # reflection_coefficient_phase = cmath.phase(elements_reflection_coefficients[i])
    #     # term2 += (reflection_coefficient_amplitude * np.exp(-1j * reflection_coefficient_phase)) / rays_distances[i]
    #
    #     term2 += elements_reflection_coefficients[i] * np.exp(1j * k * (rays_distances[i])) / \
    #              rays_distances[i]
    #
    #     if i % 100 == 0:
    #         print(i, end=", ")
    #         received_powers.append(term1 * np.power(np.abs(term2), 2))

    # term2_1 = np.sum(
    #     elements_reflection_coefficients * np.exp(1j * k * (rays_distances - minimum_distance)) / rays_distances)

    pbar.close()

    print("transmitted power (in watts):", transmitted_power)
    print("transmitted power (in dBm):", 10 * np.log10(transmitted_power / 1e-3))
    print("min distance:", np.min(rays_distances))
    print("max distance:", np.max(rays_distances))

    print("term 1:", term1)
    print("term 2 with for:", term2)
    print("abs term 2 with for:", np.abs(term2))
    # print("term 2 with np:", term2_1)

    # plot Power as function of number of elements
    received_powers_dB = 10 * np.log10(np.array(received_powers) / 1e-3)
    gain_dB = 10 * np.log10((np.array(received_powers) / 1e-3) / transmitted_power)
    transmitted_power_dB_array = np.full_like(received_powers_dB, 10 * np.log10(transmitted_power / 1e-3))
    plt.figure()

    plt.plot(transmitted_power_dB_array, label='Transmitted Power')
    plt.plot(received_powers_dB, label='Received Power')
    plt.plot(gain_dB, label='Gain')
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Number of Elements')
    plt.ylabel('Power (in dBm)')
    plt.legend()
    plt.title('Power vs Number of Elements')

    received_power = term1 * np.power(np.abs(term2), 2)

    return received_power, capacitance_matrix


def show_phase_shift_plots(phase_shifts):
    plt.figure()
    plt.imshow(phase_shifts, cmap='viridis', origin='lower')
    plt.colorbar(label='Phase Shift (deg)')
    plt.title("Phase Shifts")
    plt.xlabel('Element Index (x)')
    plt.ylabel('Element Index (y)')
    # plt.show()


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
    # Parameters
    transmitter = np.array([1, 0.5, 10.5])  # Position of the transmitter
    receiver = np.array([1.5, 2.2, 5.5])  # Position of the receiver
    ni = 1  # Refractive index
    frequency = 2.4e9  # Frequency in Hz
    c = 3e8  # Speed of light in m/s
    wavelength = c / frequency  # Calculate wavelength
    surface_size = (100, 100)  # Metasurface dimensions (M, N)
    element_size = wavelength / 8
    element_spacing = wavelength / 8  # Element spacing in x and y

    print("Surface Height:", ((surface_size[0] * element_size) + ((surface_size[0] - 1) * element_spacing)), "m")
    print("Surface Weight:", ((surface_size[1] * element_size) + ((surface_size[0] - 1) * element_spacing)), "m")

    # calculate the phase shift needed
    dphi_dx, dphi_dy = calculate_dphi_dx_dy(transmitter, receiver, surface_size, element_size, element_spacing,
                                            wavelength, ni)

    delta = element_size + element_spacing
    phase_shifts = calculate_phase_shifts_from_gradients(dphi_dx, dphi_dy, delta, delta)

    received_power, capacitance_matrix = power_received(phase_shifts, surface_size, element_size, element_spacing,
                                                        transmitter, receiver, wavelength, ni)

    print("Received Power (in Watts):", received_power)
    print("Received Power (in dBm):", 10 * math.log10(received_power / 1e-3))

    print("\nCapacitance Matrix: ")
    print(capacitance_matrix)

    show_phase_shift_plots(np.degrees(phase_shifts))
    # draw_incident_reflected_wave(transmitter, receiver, surface_size, element_size, element_spacing, phase_shifts)
    plt.show()


if __name__ == "__main__":
    main()
