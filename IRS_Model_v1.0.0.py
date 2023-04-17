import cmath

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.constants as constants
import scipy.optimize as optimize


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


def power_received(A_i, surface_size, surface_area, distance, transmitter_antenna_gain):
    # Transmitted Power
    transmitted_power = math.pow(A_i, 2) / 2

    # Calculate the received free-space channel gain
    beta_d = transmitter_antenna_gain * (surface_area / (4 * np.pi * math.pow(distance, 2)))

    N = surface_size[0] * surface_size[1]

    # Total Channel gain
    a1 = (N * beta_d) / (3 * (N * beta_d * np.pi + 1) * np.sqrt(2 * N * beta_d * np.pi + 1))
    a2 = (2 / 3 * np.pi) * np.arctan((N * beta_d) / np.sqrt(2 * N * beta_d * np.pi + 1))
    a_d_N = a1 + a2

    received_power = a_d_N * transmitted_power

    return received_power


def received_signal(phi_i, element_received_power, reflection_coefficient):
    # Calculate the amplitude of the received signal (A_r)
    A_r = np.sqrt((2 * element_received_power) * abs(reflection_coefficient))
    phi_r = phi_i + cmath.phase(reflection_coefficient)

    # Calculate the received signal (s_r(t))
    # received_signal = A_r * np.cos(w * t + phi_i)

    return A_r, phi_r


def reflected_signal(phase_shifts, surface_size, element_size, element_spacing, transmitter, wavelength):
    num_rows, num_columns = surface_size
    total_signal = 0 + 0j

    Z0 = freespace_impedance()
    R_value = 1
    L_value = 2.5e-9
    f = constants.speed_of_light / wavelength
    w = 2 * math.pi * f

    A_i = 1
    phi_i = math.radians(30)
    # incident_wave_n = A_i * np.cos(w * t + phi_i)

    surface_middle = np.array([
        ((surface_size[0] * element_size) + ((surface_size[0] - 1) * element_spacing)) / 2,
        ((surface_size[1] * element_size) + ((surface_size[1] - 1) * element_spacing)) / 2,
        0
    ])

    surface_area = math.pow(element_size, 2)
    distance = np.linalg.norm(surface_middle - transmitter)

    received_power = power_received(A_i, surface_size, surface_area, distance, 1)

    element_received_power = received_power / (num_rows * num_columns)

    capacitance_matrix = np.zeros(surface_size)

    total_signal = 0
    total_signal1 = 0
    pr = 0
    for y in range(num_rows):
        for x in range(num_columns):
            # Find the required capacitance value for the desired phase shift
            C_n = find_required_capacitance(R_value, L_value, w, phase_shifts[y, x])
            capacitance_matrix[y, x] = C_n

            # Calculate impedance for the current element
            Z1_n = element_impedance(R_value, L_value, C_n, w)

            # Calculate reflection coefficient for the current element
            reflection_coefficient = reflection_coefficients(Z0, Z1_n)

            # # Position of the current element
            # element_position = np.array(
            #     [(element_size / 2) + (x * element_spacing) + (x * element_size),
            #      (element_size / 2) + (y * element_spacing) + (y * element_size), 0])
            # # Calculate incident vector and reflected vector
            # incident_vector = element_position - transmitter
            # distance = np.linalg.norm(incident_vector)

            A_r_xy, phi_r_xy = received_signal(phi_i, element_received_power, reflection_coefficient)
            # Calculate the complex received signal (s_r) for each element
            s_r_xy = A_r_xy * np.exp(1j * phi_r_xy)
            # Sum up the complex received signals
            total_signal += s_r_xy
            total_signal1 += A_r_xy * np.cos(phi_r_xy) + 1j * A_r_xy * np.sin(phi_r_xy)

    reflected_amplitude = np.abs(total_signal)
    reflected_phase = np.angle(total_signal)

    return reflected_amplitude, reflected_phase, capacitance_matrix


def show_phase_shift_plots(phase_shifts):
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
    surface_size = (50, 50)  # Metasurface dimensions (M, N)
    element_size = wavelength / 8
    element_spacing = wavelength / 8  # Element spacing in x and y

    print("Surface Height:", ((surface_size[0] * element_size) + ((surface_size[0] - 1) * element_spacing)), "m")
    print("Surface Weight:", ((surface_size[1] * element_size) + ((surface_size[0] - 1) * element_spacing)), "m")

    # calculate the phase shift needed
    dphi_dx, dphi_dy = calculate_dphi_dx_dy(transmitter, receiver, surface_size, element_size, element_spacing,
                                            wavelength, ni)

    delta = element_size + element_spacing
    phase_shifts = calculate_phase_shifts_from_gradients(dphi_dx, dphi_dy, delta, delta)

    reflected_amplitude, reflected_phase, capacitance_matrix = reflected_signal(phase_shifts, surface_size,
                                                                                element_size, element_spacing,
                                                                                transmitter, wavelength)

    A_i = 1
    print("Incident Amplitude:", A_i)
    print("Reflected Amplitude:", reflected_amplitude)
    print("Incident power (in Watts):", math.pow(A_i, 2) / 2)
    print("Reflected power (in Watts):", math.pow(reflected_amplitude, 2) / 2)
    print("Incident Power (in dB):", 10 * math.log10(math.pow(1, 2) / 2))
    print("Reflected Power (in dB):", 10 * math.log10(math.pow(reflected_amplitude, 2) / 2))
    print("Reflected Phase (in radiant):", reflected_phase)
    print("Reflected Phase (in degrees):", math.degrees(reflected_phase))

    print("\nCapacitance Matrix: ")
    print(capacitance_matrix)

    show_phase_shift_plots(np.degrees(phase_shifts))
    # draw_incident_reflected_wave(transmitter, receiver, surface_size, element_size, element_spacing, phase_shifts)
    plt.show()


if __name__ == "__main__":
    main()
