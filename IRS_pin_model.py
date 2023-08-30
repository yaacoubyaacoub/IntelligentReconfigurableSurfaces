import os

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.constants as constants
from tqdm import tqdm
import random


def reflection_coefficients(Z0, Z1_n):
    """
    Calculate reflection coefficients for given impedances.
    :param Z0: impedance of freespace
    :param Z1_n: impedance of a given surface
    :return: reflection_coefficients of the surfaces
    """
    return (Z1_n - Z0) / (Z1_n + Z0)


def freespace_impedance():
    """
    calculates the impedance of freespace
    Z0 = μ0 / ε0
    ε0: Permittivity of freespace
    μ0: Permeability of freespace
    :return: impedance of freespace
    """
    epsilon_0 = constants.epsilon_0
    mu_0 = constants.mu_0
    return math.sqrt(mu_0 / epsilon_0)


def elements_coordinates(surface_size, element_size, element_spacing):
    """
    Calculates the surface elements coordinates based on their numbers, their sizes and their spacings.
    :param surface_size: number of elements in both x and y directions of the surface (y_n, x_n)
    :param element_size: (size_x, size_y) the width and height of a single element
    :param element_spacing: spacing between 2 elements in both x and y directions
           (spacing between elements is the same in both directions)
    :return: elements_coordinates_array: array containing the coordinates of each element of the surface based on
                                         their numbers, their sizes and their spacings
    """
    y_indices, x_indices = np.meshgrid(np.arange(surface_size[0]), np.arange(surface_size[1]), indexing='ij')

    x_values = (element_size[1] / 2) + (x_indices * element_spacing) + (x_indices * element_size[1])
    y_values = (element_size[0] / 2) + (y_indices * element_spacing) + (y_indices * element_size[0])
    z_values = np.zeros_like(x_values)

    elements_coordinates_array = np.stack((x_values, y_values, z_values), axis=2)

    return elements_coordinates_array


def calculates_incident_reflected_vectors(transmitter, receiver, elements_coordinates_array):
    """
    Computes the incidents and the reflections vectors.
    Computes the distances between the transmitter and each element of the surface (incidents vectors norms)
    Computes the distances between the receiver and each element of the surface (reflected vectors norms)
    :param transmitter: the coordinates of the transmitter
    :param receiver: the coordinates of the receiver
    :param elements_coordinates_array: array containing the coordinates of each element of the surface based on
                                       their numbers, their sizes and their spacings
    :return: incident_vectors: array of incident vectors
             incident_vectors_norms: distances between the transmitter and each element of the surface
             reflected_vectors: array of reflected vectors
             reflected_vectors_norms: distances between the receiver and each element of the surface
    """
    incident_vectors = elements_coordinates_array - transmitter
    incident_vectors_norms = np.linalg.norm(incident_vectors, axis=2)
    reflected_vectors = receiver - elements_coordinates_array
    reflected_vectors_norms = np.linalg.norm(reflected_vectors, axis=2)

    return incident_vectors, incident_vectors_norms, reflected_vectors, reflected_vectors_norms


def calculate_wave_travelled_distances(incidence_distances, reflection_distances):
    """
    Calculates the distances travelled by the waves from the transmitter to receiver through the surface.
    :param incidence_distances: distances between the transmitter and each element of the surface
    :param reflection_distances: distances between the receiver and each element of the surface
    :return: rays_distances: distances between the transmitter and the receiver through each element of the surface
             min_total_distance: min distance between the transmitter and the receiver through the surface
             max_total_distance: max distance between the transmitter and the receiver through the surface
             average_total_distance: average distance between the transmitter and the receiver through the surface
             min_transmitter_surface_distance: min distance between the transmitter and the surface
             max_transmitter_surface_distance: max distance between the transmitter and the surface
             average_transmitter_surface_distance: average distance between the transmitter and the surface
             min_surface_receiver_distance: min distance between the surface and the receiver
             max_surface_receiver_distance: max distance between the surface and the receiver
             average_surface_receiver_distance: average distance between the surface and the receiver
    """
    rays_distances = incidence_distances + reflection_distances

    min_transmitter_surface_distance = np.round(np.min(incidence_distances), 2)
    max_transmitter_surface_distance = np.round(np.max(incidence_distances), 2)
    average_transmitter_surface_distance = np.round(np.average(incidence_distances), 2)

    min_surface_receiver_distance = np.round(np.min(reflection_distances), 2)
    max_surface_receiver_distance = np.round(np.max(reflection_distances), 2)
    average_surface_receiver_distance = np.round(np.average(reflection_distances), 2)

    min_total_distance = np.round(np.min(rays_distances), 2)
    max_total_distance = np.round(np.max(rays_distances), 2)
    average_total_distance = np.round(np.average(rays_distances), 2)

    return rays_distances, min_total_distance, max_total_distance, average_total_distance, \
           min_transmitter_surface_distance, max_transmitter_surface_distance, average_transmitter_surface_distance, \
           min_surface_receiver_distance, max_surface_receiver_distance, average_surface_receiver_distance


def calculate_angles(transmitter, receiver, elements_coordinates_array):
    """
    Calculates the angles of the reflection phenomenon based on snell's generalized law of reflection
    this calculation is done geometrically at this stage.
    :param transmitter: the coordinates of the transmitter
    :param receiver: the coordinates of the receiver
    :param elements_coordinates_array: array containing the coordinates of each element of the surface based on
                                       their numbers, their sizes and their spacings
    :return: theta_i: array of incidence angles.
                     (angle between the incidence vector and the normal to the reflection surface)
             theta_r: array of reflection angles.
                     (angle between the reflected vector and its projection onto the plane perpendicular to the plane
                     of incidence)
             phi_tx: array of angles between the projection the transmitted vector onto the xy plane and the x-axis
             phi_rx: array of angles between the projection the reflected vector onto the xy plane and the x-axis
    """
    incident_vectors, incident_vectors_norms, reflected_vectors, reflected_vectors_norms = calculates_incident_reflected_vectors(
        transmitter, receiver, elements_coordinates_array)

    normal = np.array([0, 0, 1])

    # theta_i are the angles between the incident vectors and the normal to the reflection surface
    theta_i = np.arccos(np.dot(-incident_vectors, normal) / incident_vectors_norms)

    # theta_r are the angles between the reflected vectors and the normal to the reflection surface
    theta_r = np.arccos(np.dot(reflected_vectors, normal) / reflected_vectors_norms)

    proj_t_on_z = np.expand_dims(np.sum(-incident_vectors * normal, axis=2), axis=2) * normal
    proj_t_on_xy = -incident_vectors - proj_t_on_z

    proj_r_on_z = np.expand_dims(np.sum(reflected_vectors * normal, axis=2), axis=2) * normal
    proj_r_on_xy = reflected_vectors - proj_r_on_z

    x_axis = np.array([0, 1, 0])

    phi_tx = np.arccos(np.dot(proj_t_on_xy, x_axis) / np.linalg.norm(proj_t_on_xy, axis=2))
    phi_rx = np.arccos(np.dot(proj_r_on_xy, x_axis) / np.linalg.norm(proj_r_on_xy, axis=2))

    return theta_i, theta_r, phi_tx, phi_rx


def calculate_phase_shifts(elements_coordinates_array, wave_number, theta_i, theta_r, phi_tx, phi_rx,
                           incidence_distances):
    rx = wave_number * np.sin(theta_r) * ((elements_coordinates_array[:, :, 0] * np.cos(phi_rx)) + (
            elements_coordinates_array[:, :, 1] * np.sin(phi_rx)))
    # tx = wave_number * np.sin(theta_i) * ((elements_coordinates_array[:, :, 0] * np.cos(phi_tx)) + (
    #         elements_coordinates_array[:, :, 1] * np.sin(phi_tx)))
    # phase_shifts = np.mod((tx - rx) + np.pi, 2 * np.pi) - np.pi

    xtx = incidence_distances * np.sin(theta_i) * np.cos(phi_tx)
    ytx = incidence_distances * np.sin(theta_i) * np.sin(phi_tx)
    ztx = incidence_distances * np.cos(theta_i)

    phi_nm = wave_number * np.sqrt(np.power(np.sqrt(elements_coordinates_array[:, :, 0] * xtx), 2) + np.power(
        np.sqrt(elements_coordinates_array[:, :, 1] * ytx), 2) + np.power(ztx, 2))
    phase_shifts = np.mod(phi_nm + np.pi, 2 * np.pi) - np.pi
    return phase_shifts


def quantizing_phase_shifts(phase_shifts):
    quantized_phase_shifts = np.where((-np.pi / 2 < phase_shifts) & (phase_shifts < np.pi / 2), 0, math.radians(170))
    return quantized_phase_shifts


def show_phase_shift_plots(phase_shifts, title, save_plot=False, results_directory_path=None):
    """
    Plot the phase shift on a heatmap
    :param phase_shifts: 2D phase shift matrix resembling the metasurface where every entry of this matrix represents
                         the phase shift realized by the corresponding element of the surface.
    :param title: title of the plot
    :param save_plot: flag indicating if the plot is saved as a png or not
    :param results_directory_path: path for the directory to save the plot as png
    """
    plt.figure()
    plt.imshow(phase_shifts, cmap='viridis', origin='lower')
    plt.colorbar(label='Phase Shift (deg)')
    plt.title(title)
    plt.xlabel('Element Index (x)')
    plt.ylabel('Element Index (y)')
    if save_plot and results_directory_path is not None:
        plt.savefig(os.path.join(results_directory_path, f"{title}.png"))


def gradient_2d_periodic(f, delta_x=1.0, delta_y=1.0):
    """
    Calculates the gradient of a function f.
    taking into account the [-π π] periodicity of the function; when calculating the difference between 2 values of
    the function, we will perform modulo [-π π] to the result.
    ""
    x=0: δf/δx = (f(x+1,y) - f(x,y))/Δx
    x=-1: δf/δx = (f(x,y) - f(x-1,y))/Δx                    (x=-1 means the last x value of the array)
    x=[1,...,-2]: δf/δx = (f(x+1,y) - f(x-1,y))/2*Δx        (x=-2 means the value before the last x of the array)

    y=0: δf/δy = (f(x,y+1) - f(x,y))/Δy
    y=-1: δf/δy = (f(x,y) - f(x,y-1))/Δy                    (y=-1 means the last y value of the array)
    y=[1,...,-2]: δf/δy = (f(x,y+1) - f(x,y-1))/2*Δy        (y=-2 means the value before the last y of the array)
    ""
    :param f: the function to derive in x and y directions
    :param delta_x: the difference between an element and the next one in the x direction
    :param delta_y: the difference between an element and the next one in the y direction
    :return: dphi_dx: the gradient of the function f in the x direction
             dphi_dy: the gradient of the function f in the y direction
    """
    # Initialize gradient arrays
    df_dy = np.zeros_like(f)
    df_dx = np.zeros_like(f)

    # Compute gradients along rows (axis 0)
    df_dy[0] = (np.mod(f[1] - f[0] + np.pi, 2 * np.pi) - np.pi) / delta_y  # Forward difference for first row
    df_dy[-1] = (np.mod(f[-1] - f[-2] + np.pi, 2 * np.pi) - np.pi) / delta_y  # Backward difference for last row
    df_dy[1:-1] = (np.mod(f[2:] - f[:-2] + np.pi, 2 * np.pi) - np.pi) / (
            2 * delta_y)  # Central difference for interior rows

    # Compute gradients along columns (axis 1)
    df_dx[:, 0] = (np.mod(f[:, 1] - f[:, 0] + np.pi,
                          2 * np.pi) - np.pi) / delta_x  # Forward difference for first column
    df_dx[:, -1] = (np.mod(f[:, -1] - f[:, -2] + np.pi,
                           2 * np.pi) - np.pi) / delta_x  # Backward difference for last column
    df_dx[:, 1:-1] = (np.mod(f[:, 2:] - f[:, :-2] + np.pi, 2 * np.pi) - np.pi) / (
            2 * delta_x)  # Central difference for interior columns

    return df_dx, df_dy


def calculate_real_reflected_angles(theta_i, phase_shifts, delta_x, delta_y, wave_number, ni):
    """
    Calculates the phase shifts gradient realized by the surface using the phase shifts array.
    calculate the 3D reflection angles 'θr' and 'φr' this calculation is based on snell's generalized law of reflection:
    ""
    sin(θr) - sin(θi) = 1/ni*k0 * dΦ/dx
    cos(θt)sin(φr) = 1/ni*k0 * dΦ/dy
    ""
    :param theta_i: array of incidence angles.
                    (angle between the incidence vector and the normal to the reflection surface)
    :param phase_shifts: 2D array resembling the metasurface where every entry of this matrix represents the
                         phase shift realized by the corresponding element of the surface.
    :param delta_x: the difference between an element and the next one in the x direction, taken between the middle of
                    two adjacent elements. element sizes and spacing are the same between every two adjacent elements
                    in the x direction, so delta_x is uniform.
                    delta_x = element_size_x + element_spacing
    :param delta_y: the difference between an element and the next one in the y direction, taken between the middle of
                    two adjacent elements. element sizes and spacing are the same between every two adjacent elements
                    in the y direction, so delta_y is uniform.
                    delta_y = element_size_y + element_spacing
    :param wave_number: the number of complete wave cycles of an electromagnetic field that exist in one meter.
                        k0=2π/λ
    :param ni: index of refraction of the medium in which the reflection is taking place
    :return: theta_r: array of reflection angles.
                      (angle between the reflected vector and its projection onto the plane perpendicular to the plane
                      of incidence)
             phi_r: array of angles of diversion from the plane of incidence.
                    (angle between the projection the reflected vector onto the plane perpendicular to the plane
                    of incidence and the normal to the reflection surface)
    """
    # dphi_dx = np.gradient(phase_shifts, delta_x, axis=1)
    # dphi_dy = np.gradient(phase_shifts, delta_y, axis=0)
    dphi_dx, dphi_dy = gradient_2d_periodic(phase_shifts, delta_x, delta_y)

    theta_r = np.arcsin((dphi_dx / (ni * wave_number)) + np.sin(theta_i))
    phi_r = np.arcsin(dphi_dy / (wave_number * np.cos(theta_r)))

    return theta_r, phi_r


def quantized_reflection_coefficients(w, quantized_phase_shifts):
    reflection_coefficient_170deg = -0.9816435307138043 + 0.1815248655766346j
    reflection_coefficient_0deg = 0.12324074472713799 + 0.0003495138418978822j

    Z0 = freespace_impedance()
    impedance_off = 50 + (-1j * (1 / (28e-15 * w))) + (1j * 30e-12 * w)
    impedance_on = 50 + 7.8 + (1j * 30e-12 * w)
    reflection_coefficient_off = reflection_coefficients(Z0, impedance_off)
    reflection_coefficient_on = reflection_coefficients(Z0, impedance_on)
    reflection_coefficients_array = np.where((quantized_phase_shifts == 0), reflection_coefficient_off,
                                             reflection_coefficient_on)
    return reflection_coefficients_array


def radiation_pattern(theta, phi, elements_coordinates_array, reflection_coefficients, incident_amplitude, theta_i,
                      wave_number):
    E = np.cos(theta) * incident_amplitude * np.sum(reflection_coefficients * np.cos(theta_i) * np.exp(
        -1j * wave_number * np.sin(theta) * ((elements_coordinates_array[:, :, 0] * np.cos(phi)) + (
                elements_coordinates_array[:, :, 1] * np.sin(phi)))))
    return E


def main():
    # Parameters
    transmitter = np.array([-1.73, 0.15, 30])  # Position of the transmitter
    receiver = np.array([2.27, 5, 30])  # Position of the receiver
    # transmitter = np.array([1, 0.5, 3])  # Position of the transmitter
    # receiver = np.array([1.5, 1.2, 2])  # Position of the receiver
    # transmitter = np.array([0.2, 1.2, 0.5])  # Position of the transmitter
    # receiver = np.array([0.3, 1.6, 0.5])  # Position of the receiver
    frequency = 10e9  # Frequency in Hz
    c = constants.speed_of_light  # Speed of light in m/s
    wavelength = c / frequency  # Calculate wavelength
    angular_frequency = 2 * math.pi * frequency
    wave_number = 2 * np.pi / wavelength
    incident_amplitude = 0.1
    incident_phase = math.radians(30)
    # incident_wave_n = incident_amplitude * np.cos(angular_frequency * t + incident_phase)

    ni = 1  # Refractive index

    # Varactor Parameters
    R_value = 1
    # for f = 2.4GHz varactor components values
    # L1_value = 2.5e-9
    # L2_value = 0.7e-9
    # capacitance_range = np.arange(0.25e-12, 6e-12, 0.01e-12)
    # for f = 5GHz varactor components values
    # L1_value = 0.65e-9
    # L2_value = 0.5e-9
    # capacitance_range = np.arange(0.01e-12, 2e-12, 0.001e-12)
    # for f = 10GHz varactor components values
    L1_value = 0.35e-9
    L2_value = 0.25e-9
    capacitance_range = np.arange(0.2e-12, 0.8e-12, 0.001e-12)

    # Metasurface Parameters
    # surface_size = (5, 5)  # Metasurface dimensions (M, N)
    surface_size = (20, 55)  # Metasurface dimensions (M, N)
    # surface_size = (50, 50)  # Metasurface dimensions (M, N)
    element_spacing = wavelength / 4  # Element spacing in x and y
    element_size_x = wavelength / 4
    element_size_y = wavelength / 4
    delta_x = element_size_x + element_spacing
    delta_y = element_size_y + element_spacing

    surface_height = (surface_size[0] * element_size_y) + ((surface_size[0] - 1) * element_spacing)
    surface_width = (surface_size[1] * element_size_x) + ((surface_size[0] - 1) * element_spacing)
    surface_area = surface_height * surface_width

    # Calculates surface elements coordinates
    elements_coordinates_array = elements_coordinates(surface_size, (element_size_x, element_size_y), element_spacing)

    # Calculate Incident and Reflected vectors
    incident_vectors, incidence_distances, reflected_vectors, reflection_distances = calculates_incident_reflected_vectors(
        transmitter, receiver, elements_coordinates_array)
    # Calculates ray travelled distances
    rays_distances, min_total_distance, max_total_distance, average_total_distance, \
    min_transmitter_surface_distance, max_transmitter_surface_distance, average_transmitter_surface_distance, \
    min_surface_receiver_distance, max_surface_receiver_distance, average_surface_receiver_distance \
        = calculate_wave_travelled_distances(incidence_distances, reflection_distances)

    # calculate the phase shifts needed
    theta_i, theta_r, phi_tx, phi_rx = calculate_angles(transmitter, receiver, elements_coordinates_array)

    avg_theta_r = math.degrees(np.average(theta_r))
    avg_phi_r = math.degrees(np.average(phi_rx))

    print(f"Reflected angle: {avg_theta_r}")
    print(f"Reflected angle phi: {avg_phi_r}")

    phase_shifts = calculate_phase_shifts(elements_coordinates_array, wave_number, theta_i, theta_r, phi_tx, phi_rx,
                                          incidence_distances)

    quantized_phase_shifts = quantizing_phase_shifts(phase_shifts)

    reflection_coefficients_array = quantized_reflection_coefficients(angular_frequency, quantized_phase_shifts)

    # Generate the data for the radiation pattern
    theta = np.linspace(-np.pi / 2, np.pi / 2, 180)  # Range of theta from 0 to π
    phi = np.linspace(-np.pi, np.pi, 360)  # Range of phi from 0 to 2π
    # theta = np.linspace(0, np.pi, 180)  # Range of theta from 0 to π
    # phi = np.linspace(0, 2 * np.pi, 360)  # Range of phi from 0 to 2π
    E = np.zeros((len(phi), len(theta)), dtype=complex)
    for i in range(len(phi)):
        for j in range(len(theta)):
            E[i, j] = radiation_pattern(theta[j], phi[i], elements_coordinates_array, reflection_coefficients_array,
                                        incident_amplitude, theta_i, wave_number)
    # E = np.zeros(len(theta))
    # for i in range(len(theta)):
    #     E[i] = radiation_pattern(theta[i], avg_phi_r / 2, elements_coordinates_array, reflection_coefficients_array,
    #                              incident_amplitude, theta_i, wave_number)
    E_amplitude = np.abs(10 * np.log10(np.abs(E)))

    Theta, Phi = np.meshgrid(theta, phi)  # Create a meshgrid

    # Convert spherical coordinates to Cartesian coordinates
    x = E_amplitude * np.sin(Theta) * np.cos(Phi)
    y = E_amplitude * np.sin(Theta) * np.sin(Phi)
    z = E_amplitude * np.cos(Theta)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='jet')
    # ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(E), rstride=1, cstride=1, linewidth=0, antialiased=False)

    # Add color legend
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Antenna Radiation Pattern')

    # Create a polar plot
    plt.figure()
    plt.polar(theta[1:179], E_amplitude[21, 1:179])

    phi = np.pi  # Fixed phi value
    # Set plot title and labels
    plt.title('Radiation Pattern (phi = π/2)')
    plt.xlabel('Angle (theta)')
    plt.ylabel('Amplitude (E)')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
