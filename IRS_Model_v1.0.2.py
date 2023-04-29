import cmath

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.constants as constants
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


def estimate_capacitance_for_phase_shift(target_phase_shift, c_values, available_phase_shifts):
    return np.interp(target_phase_shift, available_phase_shifts, c_values, period=(2 * np.pi))


def required_varactor_bias_voltages(c):
    """Find the bias voltage needed to achieve the required capacitance value."""
    # Set the parameters
    # c0 = 867.3e-12
    c0 = 10e-12
    v0 = 2.9
    m = 1.66

    v = v0 * (np.power((c0 / c), (1 / m)) - 1)
    return np.round(v, 2)


def project_vector_onto_plane(vi, vr, uz):
    # Find the normal vector of plane Pi
    ni = np.cross(vi, uz)

    # Find the normal vector of plane Pr
    nr = np.cross(ni, uz)
    nr = nr / np.linalg.norm(nr, axis=-1, keepdims=True)

    # Calculate the projection of vr onto nr
    proj_vr_on_nr = (np.expand_dims(np.sum(vr * nr, axis=2), axis=2) / np.power(
        np.linalg.norm(nr, axis=-1, keepdims=True), 2)) * nr

    # Calculate the projection of vr onto plane Pr
    proj_vr_on_pr = vr - proj_vr_on_nr

    return proj_vr_on_pr


def calculate_dphi_dx_dy(transmitter, receiver, surface_size, element_size, element_spacing, wavelength, wave_number,
                         ni):
    m_values, n_values = np.meshgrid(np.arange(surface_size[0]), np.arange(surface_size[1]), indexing='ij')

    x_mn = (element_size / 2) + (m_values * element_spacing) + (m_values * element_size)
    y_mn = (element_size / 2) + (n_values * element_spacing) + (n_values * element_size)
    z_mn = np.zeros_like(x_mn)

    incident_vectors = np.stack((x_mn - transmitter[0], y_mn - transmitter[1], z_mn - transmitter[2]), axis=2)
    reflected_vectors = np.stack((receiver[0] - x_mn, receiver[1] - y_mn, receiver[2] - z_mn), axis=2)

    normal = np.array([0, 0, 1])

    incident_vectors_norms = np.linalg.norm(incident_vectors, axis=2)
    reflected_vectors_norms = np.linalg.norm(reflected_vectors, axis=2)

    theta_i = np.arccos(np.dot(-incident_vectors, normal) / incident_vectors_norms)
    # theta_r = np.arccos(np.dot(reflected_vectors, normal) / reflected_vectors_norms)

    # "projections" are the projection vectors of reflected vectors onto plane perpendicular to incident vectors
    # "theta_r" are the angles between reflected vectors and the "projections"
    # "phi_r" are the angles between projections and normal the metasurface (z axis)
    projections = project_vector_onto_plane(incident_vectors, reflected_vectors, normal)
    projections_mag = np.linalg.norm(projections, axis=2)

    theta_r = np.arccos(np.sum(projections * reflected_vectors, axis=2) / (projections_mag * reflected_vectors_norms))
    phi_r = np.arccos(np.sum(projections * normal, axis=2) / projections_mag)

    # If rounding to 2 digits: accurate to 0.57 degrees = 0.01 radiant
    # If rounding to 3 digits: accurate to 0.057 degrees = 0.001 radiant
    accuracy = 3
    pr0 = np.round(phi_r, accuracy) == 0
    titr = np.round(theta_i, accuracy) == np.round(theta_r, accuracy)
    s = np.logical_and(titr, pr0)
    ns = np.sum(s)
    ps = round((ns / s.size) * 100, 2)

    dphi_dx = (np.sin(theta_r) - np.sin(theta_i)) * ni * wave_number
    dphi_dy = np.cos(theta_r) * np.sin(phi_r) * ni * wave_number

    return dphi_dx, dphi_dy


# Calculate the phase shift array from the phase gradient arrays (dphi_dx, dphi_dy) using Finite Random Walk
def calculate_phase_shifts_from_gradients(dphi_dx, dphi_dy, delta_x, delta_y):
    """
    Calculates the phase_shifts from the partial derivatives dphi_dx, dphi_dy using "Random Walk Method".
    Random Walk in a loop that mase sure that all elements are visited at least 100 times.
    """
    phase_shifts = np.zeros(dphi_dx.shape)

    curr_x, curr_y = 0, 0

    visited_elements = np.zeros(dphi_dx.shape, dtype=int)
    visited_elements[curr_y, curr_x] = 100

    min_visits = 0
    pbar = tqdm(total=100)
    while np.min(visited_elements) < 100:
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

        if min_visits < np.min(visited_elements):
            min_visits = np.min(visited_elements)
            pbar.update(1)
    pbar.close()

    phase_shifts = np.mod(phase_shifts + np.pi, 2 * np.pi) - np.pi

    return phase_shifts


def power_received(transmitter, receiver, surface_size, element_size, element_spacing, phase_shifts, wavelength,
                   wave_number, angular_frequency, incident_amplitude, incident_phase, ni, plot_power=False,
                   save_plot=False):
    num_rows, num_columns = surface_size

    Z0 = freespace_impedance()
    R_value = 1
    L_value = 2.5e-9

    capacitance_matrix = np.zeros(surface_size)

    real_phase_shifts = np.zeros(surface_size)

    incidence_distances = []
    reflection_distances = []
    rays_distances = []

    transmitted_power = np.power(incident_amplitude, 2) / 2
    term1 = transmitted_power * np.power((wavelength / (4 * np.pi)), 2)
    term2 = 0
    received_powers = []

    capacitance_range = np.arange(0.25e-12, 6e-12, 0.01e-12)
    element_impedances = element_impedance(R_value, L_value, capacitance_range, angular_frequency)
    reflection_coeffs = reflection_coefficients(Z0, element_impedances)
    reflection_coefficients_amplitude = np.abs(reflection_coeffs)
    reflection_coefficients_phase_shifts = np.angle(reflection_coeffs)

    pbar = tqdm(total=(num_rows * num_columns), desc='Progress')
    for y in range(num_rows):
        for x in range(num_columns):
            # Find the required capacitance value for the desired phase shift
            C_n = estimate_capacitance_for_phase_shift(phase_shifts[y, x], capacitance_range,
                                                       reflection_coefficients_phase_shifts)
            capacitance_matrix[y, x] = C_n

            # Calculate impedance for the current element
            Z1_n = element_impedance(R_value, L_value, C_n, angular_frequency)

            # Calculate reflection coefficient for the current element
            reflection_coefficient = reflection_coefficients(Z0, Z1_n)
            # elements_reflection_coefficients.append(reflection_coefficient)

            real_phase_shifts[y, x] = cmath.phase(reflection_coefficient)

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
            incidence_distances.append(incidence_distance)
            reflection_distances.append(reflection_distance)
            rays_distance = incidence_distance + reflection_distance
            rays_distances.append(rays_distance)

            term2 += reflection_coefficient * np.exp(1j * wave_number * ni * rays_distance) / rays_distance

            received_powers.append(term1 * np.power(np.abs(term2), 2))

            pbar.update(1)
    pbar.close()

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

    received_power = term1 * np.power(np.abs(term2), 2)

    min_max_transmitter_distance = [np.round(np.min(incidence_distances), 2), np.round(np.max(incidence_distances), 2)]
    min_max_receiver_distance = [np.round(np.min(reflection_distances), 2), np.round(np.max(reflection_distances), 2)]
    min_max_distance = [np.round(np.min(rays_distances), 2), np.round(np.max(rays_distances), 2)]

    # plot Power as function of number of elements
    if plot_power:
        received_powers_dB = 10 * np.log10(np.array(received_powers) / 1e-3)
        gain_dB = 10 * np.log10((np.array(received_powers) / 1e-3) / transmitted_power)
        transmitted_power_dB_array = np.full_like(received_powers_dB, 10 * np.log10(transmitted_power / 1e-3))
        plt.figure()

        plt.plot(transmitted_power_dB_array, label='Transmitted Power')
        plt.plot(received_powers_dB, label='Received Power')
        plt.plot(gain_dB, label='Gain (Pr/Pt)')
        plt.xscale('log')
        plt.xlabel('Number of Elements')
        plt.ylabel('Power (in dBm)')
        plt.legend()
        plt.title('Received Power vs Number of Elements')

        if save_plot:
            plt.savefig("./Results_model_v1-0-2/Received Power vs Number of Elements.png")

    return real_phase_shifts, capacitance_matrix, received_power, min_max_transmitter_distance, min_max_receiver_distance, min_max_distance


def show_phase_shift_plots(phase_shifts, title, save_plot=False):
    plt.figure()
    plt.imshow(phase_shifts, cmap='viridis', origin='lower')
    plt.colorbar(label='Phase Shift (deg)')
    plt.title(title)
    plt.xlabel('Element Index (x)')
    plt.ylabel('Element Index (y)')
    if save_plot:
        plt.savefig(f"./Results_model_v1-0-2/{title}.png")


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
    transmitter = np.array([1, 0.5, 400])  # Position of the transmitter
    receiver = np.array([1.5, 1.2, 100])  # Position of the receiver
    frequency = 2.4e9  # Frequency in Hz
    c = constants.speed_of_light  # Speed of light in m/s
    wavelength = c / frequency  # Calculate wavelength
    angular_frequency = 2 * math.pi * frequency
    wave_number = 2 * np.pi / wavelength
    incident_amplitude = 1
    incident_phase = math.radians(30)
    # incident_wave_n = incident_amplitude * np.cos(w * t + incident_phase)

    ni = 1  # Refractive index
    surface_size = (50, 50)  # Metasurface dimensions (M, N)
    element_size = wavelength / 8
    element_spacing = wavelength / 8  # Element spacing in x and y
    delta = element_size + element_spacing

    surface_height = (surface_size[0] * element_size) + ((surface_size[0] - 1) * element_spacing)
    surface_width = (surface_size[1] * element_size) + ((surface_size[0] - 1) * element_spacing)
    surface_area = surface_height * surface_width
    print(f"Surface Height: {round(surface_height, 2)} m")
    print(f"Surface Width: {round(surface_width, 2)} m")
    print(f"Surface Area: {round(surface_area, 2)} m²")

    # calculate the phase shift needed
    dphi_dx, dphi_dy = calculate_dphi_dx_dy(transmitter, receiver, surface_size, element_size, element_spacing,
                                            wavelength, wave_number, ni)

    phase_shifts = calculate_phase_shifts_from_gradients(dphi_dx, dphi_dy, delta, delta)

    real_phase_shifts, capacitance_matrix, received_power, min_max_transmitter_distance, min_max_receiver_distance, min_max_distance = \
        power_received(transmitter, receiver, surface_size, element_size, element_spacing, phase_shifts, wavelength,
                       wave_number, angular_frequency, incident_amplitude, incident_phase, ni, plot_power=True,
                       save_plot=save_results)

    corresponding_varactor_voltages = required_varactor_bias_voltages(capacitance_matrix)

    transmitted_power = np.power(incident_amplitude, 2) / 2

    print(f"min LOS distance between emitter and surface through surface: {min_max_transmitter_distance[0]} m")
    print(f"max LOS distance between emitter and surface through surface: {min_max_transmitter_distance[1]} m")
    print(f"min LOS distance between surface and receiver through surface: {min_max_receiver_distance[0]} m")
    print(f"max LOS distance between surface and receiver through surface: {min_max_receiver_distance[1]} m")
    print(f"min NLOS distance between emitter and receiver through surface: {min_max_distance[0]} m")
    print(f"max NLOS distance between emitter and receiver through surface: {min_max_distance[1]} m")

    print(f"transmitted power (in Watts): {transmitted_power} W")
    print(f"transmitted power (in dBm): {round(10 * np.log10(transmitted_power / 1e-3), 2)} dBm")
    # print(f"Received Power (in milliWatts): {round(received_power * 1e3, 2)} mW")
    print(f"Received Power (in Watts): {received_power:.2e} W")
    print(f"Received Power (in dBm): {round(10 * math.log10(received_power / 1e-3), 2)} dBm")
    print(f"Percentage Received/Transmitted Power: {((received_power / transmitted_power) * 100):.2e}%")

    print("\nVaractors Capacitance Matrix (in picoFarad): ")
    print(np.round(np.multiply(capacitance_matrix, 1e12), 2))
    print("\nRequired Varactor Bias Voltages (in Volts):")
    print(corresponding_varactor_voltages)

    if save_results:
        results_file = open("./Results_model_v1-0-2/results.txt", "w")
        results_file.write(f"Incident Signal Wavelength: {round(wavelength, 3)} m\n")
        results_file.write(f"Surface Number of Elements: {surface_size}\n")
        results_file.write(f"Surface Elements Sizes: {round(element_size, 3)} m\n")
        results_file.write(f"Surface Elements spacings: {round(element_spacing, 3)} m\n")
        results_file.write(f"Surface Height: {round(surface_height, 2)} m\n")
        results_file.write(f"Surface Width: {round(surface_width, 2)} m\n")
        results_file.write(f"Surface Area: {round(surface_area, 2)} m²\n")
        results_file.write(
            f"min NLOS distance between emitter and receiver through surface: {min_max_transmitter_distance[0]} m\n")
        results_file.write(
            f"max NLOS distance between emitter and receiver through surface: {min_max_transmitter_distance[1]} m\n")
        results_file.write(f"transmitted power (in Watts): {transmitted_power} W\n")
        results_file.write(f"transmitted power (in dBm): {round(10 * np.log10(transmitted_power / 1e-3), 2)} dBm\n")
        results_file.write(f"Received Power (in Watts): {received_power:.2e} W\n")
        results_file.write(f"Received Power (in dBm): {round(10 * math.log10(received_power / 1e-3), 2)} dBm\n")
        results_file.close()

        np.savetxt("./Results_model_v1-0-2/required_phase_shifts(in degrees).csv", np.degrees(phase_shifts),
                   delimiter=",")
        np.savetxt("./Results_model_v1-0-2/real_phase_shifts(in degrees).csv", np.degrees(real_phase_shifts),
                   delimiter=",")
        np.savetxt("./Results_model_v1-0-2/varactors_capacitance_matrix(in picoFarad).csv",
                   np.round(np.multiply(capacitance_matrix, 1e12), 2),
                   delimiter=",")
        np.savetxt("./Results_model_v1-0-2/corresponding_varactor_voltages(in Volts).csv",
                   corresponding_varactor_voltages,
                   delimiter=",")

    show_phase_shift_plots(np.degrees(phase_shifts), "Required Phase Shifts", save_plot=save_results)
    show_phase_shift_plots(np.degrees(real_phase_shifts), "Real Phase Shifts", save_plot=save_results)
    # draw_incident_reflected_wave(transmitter, receiver, surface_size, element_size, element_spacing, phase_shifts)

    plt.show()


if __name__ == "__main__":
    main()
