import cmath
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


def element_impedance(R, L1, L2, C, w):
    """
    Calculates the impedance of an element of the surface.
    Equivalent circuit of one element of the metasurface:
                ------------L1-----------
            ____|                       |____
                |                       |
                -----L2-----R-----C------
    :param R: effective resistance of element
    :param L1: bottom layer inductance of the element
    :param L2: top layer inductance of the element
    :param C: effective capacitance of an element
    :param w: angular frequency (w = 2 * π * frequency)
    :return: the element impedance
    """
    jwL1 = 1j * w * L1
    jwL2 = 1j * w * L2
    jwC = 1j * w * C

    node1 = jwL1
    node2 = jwL2 + (1 / jwC) + R
    z_eq = (node1 * node2) / (node1 + node2)
    return z_eq


def estimate_capacitance_for_phase_shift(target_phase_shift, c_values, available_phase_shifts):
    """
    Estimate the capacitance needed for a required phase shift.
    Done by interpolation between the available phase shifts that could be realized by a given varactor.
    these values are calculates by setting a capacitance value to a given range realizable by the varactor, calculating
    the phase shift realized using each capacitance value and saving the capacitance value and their corresponding
    the phase shifts.
    :param target_phase_shift: the phase shift that we need to achieve for a given element
    :param c_values: 1D array of the capacitance value range achievable by tha varactor
    :param available_phase_shifts: 1D array of the corresponding phase shifts realized by the varactor given c_values
    :return: the estimated capacitance value for the target phase shift
    """
    return np.interp(target_phase_shift, available_phase_shifts, c_values, period=(2 * np.pi))


def required_varactor_bias_voltages(c):
    """
    Find the bias voltage for the varactor in order to achieve the required capacitance value.
    varactor capacitance-voltage relationship model:
        c = c0 / ((1 + v / v0) ** m)
            c0: capacitance at v=0
            v0: characteristic voltage of the varactor
            m: non-linearity factor
            v: bias voltage applied to the varactor
            c: varactor capacitance that corresponds to a given bias voltage v.
    :param c: required capacitance value
    :return: corresponding varactor voltage
    """
    # Set the parameters
    # c0 = 867.3e-12
    c0 = 10e-12
    v0 = 2.9
    m = 1.66

    v = v0 * (np.power((c0 / c), (1 / m)) - 1)
    return np.round(v, 2)


def calculate_normal_plane_vector(incident_vector, uz=np.array([0, 0, 1])):
    """
    Calculates the normal vectors of the planes perpendicular to the incident planes
    where the incident plane is the plane that includes the incident vector
    :param incident_vector: array of incident vectors
    :param uz: unit vector normal to the plane of the metasurface
    :return: array of normal vector to each plane of each incident vector
    """
    # Find the normal vector of plane of incidence Pi
    ni = np.cross(incident_vector, uz)

    # Find the normal vector of plane Pr_p perpendicular to Pi
    nr_p = np.cross(ni, uz)
    nr_p = nr_p / np.linalg.norm(nr_p, axis=2, keepdims=True)

    return nr_p


def project_vector_onto_plane(reflected_vector, nr_p):
    """
    Calculates the projections of the reflected vector onto the plane perpendicular to the incident plane. This plane is
    denoted by its normal vector nr_p
    :param reflected_vector: array of reflected vectors
    :param nr_p: array of normal vectors ot the planes
    :return: the projections of the reflected vector onto the plane perpendicular to the incident plane
    """
    # Calculate the projection of vr onto nr_p
    proj_vr_on_nr_p = (np.expand_dims(np.sum(reflected_vector * nr_p, axis=2), axis=2) / np.power(
        np.linalg.norm(nr_p, axis=2, keepdims=True), 2)) * nr_p

    # Calculate the projection of vr onto plane Pr_p
    proj_vr_on_pr_p = reflected_vector - proj_vr_on_nr_p

    return proj_vr_on_pr_p


def find_projected_reflection_vector(origin_points, receiver, nr_p, phi_r):
    """
    Finds the projection of the reflected vector onto the plane P perpendicular to the incident plane using the angle
    between the projected vector and the z-axis phi_r
    :param origin_points: origin of the projection of the reflected vector on plane P
    :param receiver: the coordinates of the receiver
    :param nr_p: normal vector to the plane P
    :param phi_r: angle between the z-axis and the projection of the reflected vector onto P
    :return: the reflected projection vector
    """
    # Projecting the receiver onto the planes perpendicular to the planes of incidence
    reflected_vectors = receiver - origin_points
    proj_vr_on_nr_p = (np.expand_dims(np.sum(reflected_vectors * nr_p, axis=2), axis=2) / np.power(
        np.linalg.norm(nr_p, axis=2, keepdims=True), 2)) * nr_p
    proj_r_on_pr_p = np.subtract(receiver, proj_vr_on_nr_p)

    d = -1 * np.expand_dims(np.sum(origin_points * nr_p, axis=2), axis=2)
    planes = np.concatenate((nr_p, d), axis=2)  # [a, b, c, d]

    vz = np.cos(phi_r)
    z2 = vz + origin_points[:, :, 2]

    X = (planes[:, :, 2] * z2) + planes[:, :, 3]

    A = ((planes[:, :, 0] ** 2) / (planes[:, :, 1] ** 2)) + 1
    B = ((2 * X * planes[:, :, 0]) / (planes[:, :, 1] ** 2)) + (
            (2 * planes[:, :, 0] * origin_points[:, :, 1]) / planes[:, :, 1]) - (2 * origin_points[:, :, 0])
    C = ((X ** 2) / (planes[:, :, 1] ** 2)) + ((2 * X * origin_points[:, :, 1]) / planes[:, :, 1]) + (
            origin_points[:, :, 0] ** 2) + (
                origin_points[:, :, 1] ** 2) + (vz ** 2) - 1

    sol1 = (-B + np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
    sol2 = (-B - np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)

    x2 = np.where(np.isnan(sol1), sol2, np.where(np.isnan(sol2), sol1,
                                                 np.where(origin_points[:, :, 0] > proj_r_on_pr_p[:, :, 0],
                                                          np.minimum(sol1, sol2), np.maximum(sol1, sol2))))

    y2 = (-1 / planes[:, :, 1]) * ((planes[:, :, 0] * x2) + (planes[:, :, 2] * z2) + planes[:, :, 3])

    unit_reflected_projection_vector = np.stack(
        (x2 - origin_points[:, :, 0], y2 - origin_points[:, :, 1], z2 - origin_points[:, :, 2]), axis=2)

    t = (receiver[2] - origin_points[:, :, 2]) / unit_reflected_projection_vector[:, :, 2]

    x = origin_points[:, :, 0] + unit_reflected_projection_vector[:, :, 0] * t
    y = origin_points[:, :, 1] + unit_reflected_projection_vector[:, :, 1] * t

    reflected_projection_vector = np.stack(
        (x - origin_points[:, :, 0], y - origin_points[:, :, 1], receiver[2] - origin_points[:, :, 2]), axis=2)

    return reflected_projection_vector


def find_reflection_vector(origin_points, receiver, projection, theta_r):
    """
    Finds the reflected vectors based on its projections onto the plane P perpendicular to the incident plane and the
    angle of reflection theta_r
    :param origin_points: origin of the reflected vector
    :param receiver: the coordinates of the receiver
    :param projection: projection of the reflected vector onto plane P array [a, b, c]
    :param theta_r: angle between the reflection vector and its projection onto the plane P
    :return: the reflection vector
    """
    projection_magnitude = np.linalg.norm(projection, axis=2)
    reflected_vector_magnitude = projection_magnitude / np.cos(theta_r)

    zr = projection[:, :, 2]

    X = reflected_vector_magnitude * projection_magnitude * np.cos(theta_r) - zr ** 2

    A = ((projection[:, :, 0] ** 2) / (projection[:, :, 1] ** 2)) + 1
    B = (-2 * X * projection[:, :, 0]) / (projection[:, :, 1] ** 2)
    C = ((X ** 2) / (projection[:, :, 1] ** 2)) + (zr ** 2) - (reflected_vector_magnitude ** 2)

    sol1 = (-B + np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
    estimate_sol1 = sol1 + origin_points[:, :, 0]
    sol2 = (-B - np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
    estimate_sol2 = sol2 + origin_points[:, :, 0]

    xr = np.where(np.isnan(sol1), sol2, np.where(np.isnan(sol2), sol1, np.where(
        np.abs(estimate_sol1 - receiver[0]) < np.abs(estimate_sol2 - receiver[0]), sol1, sol2)))

    yr = (X - (projection[:, :, 0] * xr)) / projection[:, :, 1]

    reflected_vector = np.stack((xr, yr, zr), axis=2)

    return reflected_vector


def elements_coordinates(surface_size, element_size, element_spacing):
    """
    Calculates the surface elements coordinates based on their numbers, their sizes and their spacings.
    :param surface_size: number of elements in both x and y directions of the surface (y_n, x_n)
    :param element_size: size of each edge of a square element
    :param element_spacing: spacing between 2 elements in both x and y directions
           (spacing between elements is the same in both directions)
    :return: elements_coordinates_array: array containing the coordinates of each element of the surface based on
                                         their numbers, their sizes and their spacings
    """
    y_indices, x_indices = np.meshgrid(np.arange(surface_size[0]), np.arange(surface_size[1]), indexing='ij')

    x_values = (element_size / 2) + (x_indices * element_spacing) + (x_indices * element_size)
    y_values = (element_size / 2) + (y_indices * element_spacing) + (y_indices * element_size)
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
             min_transmitter_surface_distance: min distance between the transmitter and the surface
             max_transmitter_surface_distance: max distance between the transmitter and the surface
             min_surface_receiver_distance: min distance between the surface and the receiver
             max_surface_receiver_distance: max distance between the surface and the receiver
    """
    rays_distances = incidence_distances + reflection_distances

    min_transmitter_surface_distance = np.round(np.min(incidence_distances), 2)
    max_transmitter_surface_distance = np.round(np.max(incidence_distances), 2)

    min_surface_receiver_distance = np.round(np.min(reflection_distances), 2)
    max_surface_receiver_distance = np.round(np.max(reflection_distances), 2)

    min_total_distance = np.round(np.min(rays_distances), 2)
    max_total_distance = np.round(np.max(rays_distances), 2)
    average_total_distance = np.round(np.average(rays_distances), 2)

    return rays_distances, min_total_distance, max_total_distance, average_total_distance, \
           min_transmitter_surface_distance, max_transmitter_surface_distance, \
           min_surface_receiver_distance, max_surface_receiver_distance


def calculate_angles(transmitter, receiver, surface_size, element_size, element_spacing):
    """
    Calculates the angles of the reflection phenomenon based on snell's generalized law of reflection
    this calculation is done geometrically at this stage.
    :param transmitter: the coordinates of the transmitter
    :param receiver: the coordinates of the receiver
    :param surface_size: number of elements in both x and y directions of the surface (y_n, x_n)
    :param element_size: size of each edge of a square element
    :param element_spacing: spacing between 2 elements in both x and y directions
           (spacing between elements is the same in both directions)
    :return: theta_i: array of incidence angles.
                     (angle between the incidence vector and the normal to the reflection surface)
             theta_r: array of reflection angles.
                     (angle between the reflected vector and its projection onto the plane perpendicular to the plane
                     of incidence)
             phi_r: array of angles of diversion from the plane of incidence.
                     (angle between the projection the reflected vector onto the plane perpendicular to the plane
                     of incidence and the normal to the reflection surface)
    """
    elements_coordinates_array = elements_coordinates(surface_size, element_size, element_spacing)
    incident_vectors, incident_vectors_norms, reflected_vectors, reflected_vectors_norms = calculates_incident_reflected_vectors(
        transmitter, receiver, elements_coordinates_array)

    normal = np.array([0, 0, 1])

    # theta_i are the angles between the incident vectors and the normal to the reflection surface
    theta_i = np.arccos(np.dot(-incident_vectors, normal) / incident_vectors_norms)

    # theta_r are the angles between the reflected vectors and the normal to the reflection surface
    # theta_r = np.arccos(np.dot(reflected_vectors, normal) / reflected_vectors_norms)

    # "projections" are the projection vectors of reflected vectors onto plane perpendicular to incident vectors plane
    nr_p = calculate_normal_plane_vector(incident_vectors, normal)
    projections = project_vector_onto_plane(reflected_vectors, nr_p)
    projections_mag = np.linalg.norm(projections, axis=2)

    # "theta_r" are the angles between reflected vectors and the "projections"
    theta_r = np.arccos(np.sum(projections * reflected_vectors, axis=2) / (projections_mag * reflected_vectors_norms))

    # "phi_r" are the angles between projections and normal the metasurface (z axis)
    phi_r = np.arccos(np.dot(projections, normal) / projections_mag)

    # ############################################# Used only for testing ##############################################
    # testing getting projected_vector and reflected_vector using phi_r and theta_r
    proj_vect = find_projected_reflection_vector(elements_coordinates_array, receiver, nr_p, phi_r)
    ref_vec = find_reflection_vector(elements_coordinates_array, receiver, proj_vect, theta_r)
    diffp = proj_vect - projections
    diffr = ref_vec - reflected_vectors

    # Number of elements of the surface that are following the original snell's law 'θi = θr', 'φr = 0'
    # If rounding to 2 digits: accurate to 0.57 degrees = 0.01 radiant
    # If rounding to 3 digits: accurate to 0.057 degrees = 0.001 radiant
    accuracy = 3
    phi_r__0 = np.round(phi_r, accuracy) == 0
    theta_i__theta_r = np.round(theta_i, accuracy) == np.round(theta_r, accuracy)
    original_snell_law = np.logical_and(theta_i__theta_r, phi_r__0)
    number_original_snell_law = np.sum(original_snell_law)
    percentage_original_snell_law = round((number_original_snell_law / original_snell_law.size) * 100, 2)
    # ##################################################################################################################

    return theta_i, theta_r, phi_r


def calculate_dphi_dx_dy(theta_i, theta_r, phi_r, wave_number, ni):
    """
    Calculates the phase gradients in both x and y directions based on snell's generalized law of reflection
    ""
    sin(θr) - sin(θi) = 1/ni*k0 * dΦ/dx
    cos(θr)sin(φr) = 1/ni*k0 * dΦ/dy
    ""
    :param theta_i: array of incidence angles.
                    (angle between the incidence vector and the normal to the reflection surface)
    :param theta_r: array of reflection angles.
                    (angle between the reflected vector and its projection onto the plane perpendicular to the plane
                    of incidence)
    :param phi_r: array of angles of diversion from the plane of incidence.
                  (angle between the projection the reflected vector onto the plane perpendicular to the plane
                  of incidence and the normal to the reflection surface)
    :param wave_number: the number of complete wave cycles of an electromagnetic field that exist in one meter.
                        k0=2π/λ
    :param ni: index of refraction of the medium in which the reflection is taking place
    :return: dphi_dx: the gradient of the phase in the x direction (based on snell's generalized law of reflection)
             dphi_dy: the gradient of the phase in the y direction (based on snell's generalized law of reflection)
    """
    dphi_dx = (np.sin(theta_r) - np.sin(theta_i)) * ni * wave_number
    dphi_dy = np.cos(theta_r) * np.sin(phi_r) * ni * wave_number

    return dphi_dx, dphi_dy


# Calculate the phase shift array from the phase gradient arrays (dphi_dx, dphi_dy) using Random Walk
def calculate_phase_shifts_from_gradients(dphi_dx, dphi_dy, delta_x, delta_y):
    """
    Calculates the phase_shifts from the partial derivatives dphi_dx, dphi_dy using "Random Walk Method".
    Random Walk in a loop that make sure that all elements are visited at least 10 times.
    Updates are made by taking the average between the new and the old value of a location.
    :param dphi_dx: the gradient of the phase in the x direction (based on snell's generalized law of reflection)
    :param dphi_dy: the gradient of the phase in the y direction (based on snell's generalized law of reflection)
    :param delta_x: the difference between an element and the next one in the x direction, taken between the middle of
                    two adjacent elements. element sizes and spacing are the same between every two adjacent elements
                    in the x direction, so delta_x is uniform.
                    delta_x = element_size + element_spacing
    :param delta_y: the difference between an element and the next one in the y direction, taken between the middle of
                    two adjacent elements. element sizes and spacing are the same between every two adjacent elements
                    in the y direction, so delta_y is uniform.
                    delta_y = element_size + element_spacing
    :return: phase_shifts: 2D array resembling the metasurface where every entry of this matrix represents the
             phase shift required by the corresponding element of the surface.
    """
    phase_shifts = np.zeros(dphi_dx.shape)

    curr_x, curr_y = 0, 0

    visited_elements = np.zeros(dphi_dx.shape, dtype=int)

    current_min_visits = 0
    target_min_visits = 10
    pbar = tqdm(total=target_min_visits)
    while np.min(visited_elements) < target_min_visits:
        new_direction = random.randint(1, 4)
        # Directions:
        #     1 = Right (-->)
        #     2 = Left (<--)
        #     3 = Down
        #     4 = Up

        if (new_direction == 2 and curr_x == 1 and curr_y == 0) or (new_direction == 4 and curr_x == 0 and curr_y == 1):
            curr_x, curr_y = 0, 0
        else:
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

        if current_min_visits < np.min(visited_elements):
            current_min_visits = np.min(visited_elements)
            pbar.update(1)
    pbar.close()

    phase_shifts = np.mod(phase_shifts + np.pi, 2 * np.pi) - np.pi

    return phase_shifts


# Calculate the phase shift array from the phase gradient arrays (dphi_dx, dphi_dy)
def calculate_phase_shifts_from_gradients1(dphi_dx, dphi_dy, delta_x, delta_y):
    """
    Calculates the phase_shifts from the partial derivatives dphi_dx, dphi_dy.
    Create 2 phase_shifts arrays (phase_shifts_x, phase_shifts_y) one calculated based on dphi_dx and the other based
    on dphi_dy.
    In both phase_shifts arrays, the fist column (x=0) is always calculated based on dphi_dx and delta_x.
    In both phase_shifts arrays, the fist row (y=0) is always calculated based on dphi_dy and delta_y.
    then completing both array by using the following equations:
    ""
    x=0: f(x+1,y) = (δf/δx * Δx) + f(x,y)
    x=-1: f(x,y) = (δf/δx * Δx) + f(x-1,y)                  (x=-1 means the last x value of the array)
    x=[1,...,-2]: f(x+1,y) = (δf/δx * 2*Δx) + f(x-1,y)      (x=-2 means the value before the last x of the array)

    y=0: f(x,y+1) = (δf/δy * Δy) + f(x,y)
    y=-1: f(x,y) = (δf/δy * Δy) + f(x,y-1)                  (y=-1 means the last y value of the array)
    y=[1,...,-2]: f(x,y+1) = (δf/δx * 2*Δx) + f(x,y-1)      (y=-2 means the value before the last y of the array)
    ""
    Then based on the first column (x=0) and the first row (y=0) both phase_shifts arrays
    (phase_shifts_x, phase_shifts_y) should theoretically be the same, but due to estimation error we will have some
    differences. So finally to calculate the final phase shift array we take the average of both phase_shifts arrays
    (phase_shifts_x, phase_shifts_y) by adding them together (on an element basis) and then dividing by 2.
    "" phase_shifts = (phase_shifts_x + phase_shifts_y) / 2 ""
    :param dphi_dx: the gradient of the phase in the x direction (based on snell's generalized law of reflection)
    :param dphi_dy: the gradient of the phase in the y direction (based on snell's generalized law of reflection)
    :param delta_x: the difference between an element and the next one in the x direction, taken between the middle of
                    two adjacent elements. element sizes and spacing are the same between every two adjacent elements
                    in the x direction, so delta_x is uniform.
                    delta_x = element_size + element_spacing
    :param delta_y: the difference between an element and the next one in the y direction, taken between the middle of
                    two adjacent elements. element sizes and spacing are the same between every two adjacent elements
                    in the y direction, so delta_y is uniform.
                    delta_y = element_size + element_spacing
    :return: phase_shifts: 2D array resembling the metasurface where every entry of this matrix represents the
             phase shift required by the corresponding element of the surface.
    """
    phase_shifts_x = np.zeros(dphi_dx.shape)
    phase_shifts_y = np.zeros(dphi_dx.shape)

    for curr_y in range(phase_shifts_x.shape[0]):
        for curr_x in range(phase_shifts_x.shape[1]):
            # Fill the phase_shifts_x array
            if curr_x == 0:
                phase_shifts_x[curr_y, curr_x + 1] = (delta_x * dphi_dx[curr_y, curr_x]) + phase_shifts_x[
                    curr_y, curr_x]
                # Fill the first column of the phase_shifts_x (x=0) using dphi_dy and delta_y
                if curr_y == 0:
                    phase_shifts_x[curr_y + 1, curr_x] = (delta_y * dphi_dy[curr_y, curr_x]) + phase_shifts_x[
                        curr_y, curr_x]
                elif curr_y == phase_shifts_y.shape[0] - 1:
                    phase_shifts_x[curr_y, curr_x] = (delta_y * dphi_dy[curr_y, curr_x]) + phase_shifts_x[
                        curr_y - 1, curr_x]
                else:
                    phase_shifts_x[curr_y + 1, curr_x] = (2 * delta_y * dphi_dy[curr_y, curr_x]) + phase_shifts_x[
                        curr_y - 1, curr_x]
                #  ###############################################################################
            elif curr_x == phase_shifts_x.shape[1] - 1:
                phase_shifts_x[curr_y, curr_x] = (delta_x * dphi_dx[curr_y, curr_x]) + phase_shifts_x[
                    curr_y, curr_x - 1]
            else:  # 0 < curr_x < (phase_shifts.shape[1] - 1)
                phase_shifts_x[curr_y, curr_x + 1] = (2 * delta_x * dphi_dx[curr_y, curr_x]) + phase_shifts_x[
                    curr_y, curr_x - 1]
            #  ###############################################################################

            # Fill the phase_shifts_y array
            if curr_y == 0:
                phase_shifts_y[curr_y + 1, curr_x] = (delta_y * dphi_dy[curr_y, curr_x]) + phase_shifts_y[
                    curr_y, curr_x]
                # Fill the first row of the phase_shifts_y (y=0) using dphi_dx and delta_x
                if curr_x == 0:
                    phase_shifts_y[curr_y, curr_x + 1] = (delta_x * dphi_dx[curr_y, curr_x]) + phase_shifts_y[
                        curr_y, curr_x]
                elif curr_x == phase_shifts_y.shape[1] - 1:
                    phase_shifts_y[curr_y, curr_x] = (delta_x * dphi_dx[curr_y, curr_x]) + phase_shifts_y[
                        curr_y, curr_x - 1]
                else:
                    phase_shifts_y[curr_y, curr_x + 1] = (2 * delta_x * dphi_dx[curr_y, curr_x]) + phase_shifts_y[
                        curr_y, curr_x - 1]
                #  ###############################################################################
            elif curr_y == phase_shifts_y.shape[0] - 1:
                phase_shifts_y[curr_y, curr_x] = (delta_y * dphi_dy[curr_y, curr_x]) + phase_shifts_y[
                    curr_y - 1, curr_x]
            else:  # 0 < curr_y < (phase_shifts.shape[0] - 1)
                phase_shifts_y[curr_y + 1, curr_x] = (2 * delta_y * dphi_dy[curr_y, curr_x]) + phase_shifts_y[
                    curr_y - 1, curr_x]
            #  ###############################################################################

    phase_shifts = (phase_shifts_x + phase_shifts_y) / 2

    phase_shifts = np.mod(phase_shifts + np.pi, 2 * np.pi) - np.pi

    return phase_shifts


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
                    delta_x = element_size + element_spacing
    :param delta_y: the difference between an element and the next one in the y direction, taken between the middle of
                    two adjacent elements. element sizes and spacing are the same between every two adjacent elements
                    in the y direction, so delta_y is uniform.
                    delta_y = element_size + element_spacing
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


def calculate_capacitance_matrix(R_value, L1_value, L2_value, capacitance_range, phase_shifts, angular_frequency):
    """
    Calculates the required capacitance of each element of the surface based on the frequency of the incoming signal
    and the phase shift that must be introduced by each element.
    :param R_value: resistance of every element on the surface
    :param L1_value: bottom layer inductance of every element on the surface
    :param L2_value: top layer inductance of every element on the surface
    :param capacitance_range: capacitance range that the varactor is able to produce
    :param phase_shifts: 2D matrix of the required phase shift of each element of the surface
    :param angular_frequency: w = 2 * π * frequency
    :return: capacitance_matrix: estimated capacitance of each element of the surface based on the frequency of the
                                 incoming signal and the required phase shift.
    """
    Z0 = freespace_impedance()
    elements_achievable_impedances = element_impedance(R_value, L1_value, L2_value, capacitance_range,
                                                       angular_frequency)
    elements_achievable_reflection_coefficients = reflection_coefficients(Z0, elements_achievable_impedances)
    reflection_coefficients_amplitude = np.abs(elements_achievable_reflection_coefficients)
    reflection_coefficients_phase_shifts = np.angle(elements_achievable_reflection_coefficients)
    capacitance_matrix = estimate_capacitance_for_phase_shift(phase_shifts, capacitance_range,
                                                              reflection_coefficients_phase_shifts)
    return capacitance_matrix


def calculate_real_phase_shifts(R_value, L1_value, L2_value, capacitance_matrix, angular_frequency):
    """
    Calculates the real reflection coefficients and the real phase shifts introduced by each element of the surface
    based on the frequency of the incoming signal, and the capacitance of each element.
    :param R_value: resistance of every element on the surface
    :param L1_value: bottom layer inductance of every element on the surface
    :param L2_value: top layer inductance of every element on the surface
    :param capacitance_matrix: estimated capacitance of each element of the surface based on the frequency of the
                               incoming signal and the required phase shift.
    :param angular_frequency: w = 2 * π * frequency
    :return: real_reflection_coefficients_array: 2D array of complex numbers representing the real reflection
                                                 coefficients of each element of the surface.
             real_phase_shifts: 2D matrix of the real phase shift introduced by each element of the surface
    """
    Z0 = freespace_impedance()
    real_elements_impedance = element_impedance(R_value, L1_value, L2_value, capacitance_matrix, angular_frequency)
    real_reflection_coefficients_array = reflection_coefficients(Z0, real_elements_impedance)
    real_phase_shifts = np.angle(real_reflection_coefficients_array)
    return real_reflection_coefficients_array, real_phase_shifts


def compute_successful_reflections(receiver, elements_coordinates_array, incident_vectors, real_theta_r, real_phi_r):
    """
    Compute the real trajectory of the reflected rays based on the real reflection angles (θt, φr) calculated form the
    real phase shift introduced by each element of the metasurface.
    Estimate if the reflected vector will be hitting the receiver antenna based on the antenna shape and dimensions
    :param receiver: the coordinates of the receiver in space
    :param elements_coordinates_array: array containing the coordinates of eah element of the metasurface
    :param incident_vectors: array containing the incident vectors from the transmitter to each element of the surface
    :param real_theta_r: theta_r: array of reflection angles.
                                  (angle between the reflected vector and its projection onto the plane perpendicular
                                  to the plane of incidence)
    :param real_phi_r: array of angles of diversion from the plane of incidence.
                       (angle between the projection the reflected vector onto the plane perpendicular to the plane of
                       incidence and the normal to the reflection surface)
    :return: successful_reflections: 2D boolean array where each entry represent an element of the metasurface.
                                      - 'True': if the reflected vector was successful in hitting the receiver antenna
                                      - 'False': if the reflected vector was misses the receiver antenna
    """
    nr_p = calculate_normal_plane_vector(incident_vectors)
    real_projected_vectors = find_projected_reflection_vector(elements_coordinates_array, receiver, nr_p, real_phi_r)
    real_reflected_vectors = find_reflection_vector(elements_coordinates_array, receiver, real_projected_vectors,
                                                    real_theta_r)

    real_destination_reached = real_reflected_vectors + elements_coordinates_array

    # Ignoring the rays that will not hit the receiver. (±1 degrees = ±π/180 radiant)
    # successful_reflections = np.logical_and((np.abs(real_theta_r - theoretical_theta_r) < (np.pi / 180)),
    #                              (np.abs(real_phi_r - theoretical_phi_r) < (np.pi / 180)))

    # Ignoring the rays that will not hit the receiver. If the hit location is outside the antenna radius.
    # receiver_antenna_radius = 0.05  # Receiver antenna radius in meters
    # successful_reflections = np.linalg.norm(receiver - real_destination_reached, axis=2) < receiver_antenna_radius

    # Ignoring the rays that will not hit the receiver antenna.
    # If the hit location is outside the rectangular antenna dimensions the ray will be ignored.
    # the antenna rectangular dimensions are modeled by the ranges [x_min, x_max] and [y_min, y_max]
    antenna_width, antenna_height = 0.05, 0.1  # Receiver antenna dimensions (width, height) in meters
    antenna = [receiver[0] - antenna_width, receiver[0] + antenna_width, receiver[1] - antenna_height,
               receiver[1] + antenna_height]  # [x_min, x_max, y_min, y_max]
    x_mask = (real_destination_reached[:, :, 0] > antenna[0]) & (real_destination_reached[:, :, 0] < antenna[1])
    y_mask = (real_destination_reached[:, :, 1] > antenna[2]) & (real_destination_reached[:, :, 1] < antenna[3])
    successful_reflections = x_mask & y_mask
    accurate_elements_percentage = successful_reflections.mean()

    return successful_reflections, accurate_elements_percentage


def power_received(wavelength, wave_number, incident_amplitude, incident_phase, ni, real_reflection_coefficients_array,
                   rays_distances, successful_reflections):
    """
    Calculates the power received by the receiver antenna.
    the calculation is based on the two ray model but ignoring the line of sight component.
    ""
    Pr = Pt * (λ/4π)² * (Σ(Γ(θ) * exp(j*k₀*d*nᵢ)/d))²
    ""
    :param wavelength: the wavelength of the transmitted signal
    :param wave_number: the number of complete wave cycles of an electromagnetic field that exist in one meter.
                        k0=2π/λ
    :param incident_amplitude: the amplitude of the incident wave
    :param incident_phase: the phase of the incident wave
    :param ni: index of refraction of the medium in which the reflection is taking place
    :param real_reflection_coefficients_array: 2D array of complex numbers representing the real reflection
                                               coefficients of each element of the surface
    :param rays_distances:  distances between the transmitter and the receiver through each element of the surface
    :param successful_reflections: 2D boolean array where each entry represent an element of the metasurface.
    :return: received_power: the power received by the receiver antenna
    """
    transmitted_power = np.power(incident_amplitude, 2) / 2

    term1 = transmitted_power * np.power((wavelength / (4 * np.pi)), 2)

    term2 = real_reflection_coefficients_array * np.exp(1j * wave_number * ni * rays_distances) / rays_distances
    term2 = term2 * successful_reflections

    term2 = np.cumsum(term2.flatten())
    received_powers = term1 * np.power(np.abs(term2), 2)

    received_power = received_powers[-1]

    return received_powers, received_power


def plot_power_graph(transmitted_power, received_powers, save_plot=False, results_directory_path=None):
    """
    Plot the transmitted power.
    Plot the received power vs number of elements
    :param transmitted_power: Power transmitted by the transmitter of the signal
    :param received_powers: 1D array containing the power received by the receiver antenna based on the number of
                            elements of the reflecting metasurface
    :param save_plot: flag indicating if the plot is saved as a png or not
    :param results_directory_path: path for the directory to save the plot as png
    """
    received_powers_dB = 10 * np.log10(np.array(received_powers) / 1e-3)
    gain_dB = 10 * np.log10((np.array(received_powers)) / transmitted_power)
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

    if save_plot and results_directory_path is not None:
        plt.savefig(os.path.join(results_directory_path, "Received Power vs Number of Elements.png"))


def find_snells_angle(transmitter, receiver, normal):
    """
    Find the reflection angle based on the normal reflection law (snell's law 'θi = θr')
    (the case where no metasurface is implemented)
    Assuming the surface of reflection lies in the xy-plane (z=0), then the normal to the surface is the unit vector uz
    parallel to z-axis.
    The transmitter and the receiver are points located in 3D space.
    ""
    theta_i = theta_r
    ((vi.normal) / |vi|) = ((vr.normal) / |vr|)
    (zi / |vi|) = (zr / |vr|)
    ((x - xi)^2 + (y - yi)^2 + (zi)^2) / ((x - xr)^2 + (y - yr)^2 + (zr)^2) = (zi / zr)^2
    ""
    :param transmitter: the coordinates of the transmitter
    :param receiver: the coordinates of the receiver
    :param normal: unit vector normal to the plane of the metasurface
    :return: theta_i: the angle of reflection. in this case we have θi = θr
    """
    xi, yi, zi = transmitter
    xr, yr, zr = receiver

    def f(x, y):
        return (((x - xi) ** 2 + (y - yi) ** 2 + zi ** 2) / ((x - xr) ** 2 + (y - yr) ** 2 + zr ** 2)) - (
                (zi / zr) ** 2)

    # Define a grid of points to evaluate the function
    X, Y = np.meshgrid(np.linspace(min(xi, xr), max(xi, xr), 1000), np.linspace(min(yi, yr), max(yi, yr), 1000))

    # Evaluate the function on the grid
    Z = f(X, Y)

    # Find the (x, y) coordinates where the function is closest to zero
    idx = np.argmin(np.abs(Z))
    x_value, y_value = X.flat[idx], Y.flat[idx]
    p0 = np.array([x_value, y_value, 0])

    vi = transmitter - p0
    theta_i = np.arccos(np.dot(vi, normal) / np.linalg.norm(vi))
    # vr = receiver - p0
    # theta_r = np.arccos(np.dot(vr, normal) / np.linalg.norm(vr))

    return theta_i


def power_without_intelligent_surface(transmitted_power, wavelength, wave_number, ni, distance, theta_i, epilon_r,
                                      parallel_perpendicular=0):
    """
    Find the power received by a receiver when a signal is sent from a transmitter and no line of sight between the
    transmitter and the receiver, the signal that reaches the receiver comes only from reflection of the transmitted
    signal on a normal smooth surface.
    In this case the reflection will happen according to the original snell's law 'θi = θr'
    :param transmitted_power: the power of the transmitted signal
    :param wavelength: the wavelength of the transmitted signal
    :param wave_number: the number of complete wave cycles of an electromagnetic field that exist in one meter.
                        k0=2π/λ
    :param ni: index of refraction of the medium in which the reflection is taking place
    :param distance: the total non-line of sight distance between the transmitter and the receiver following the
                     signal's path. distance = distance_transmitter_surface + distance_surface_receiver
    :param theta_i: angle of incidence of the signal into the surface.
    :param epilon_r: permittivity εr of the material composing the surface on which the signal will reflect back
    :param parallel_perpendicular: what kind of polarization this signal has.
                                    parallel polarization: 'parallel_perpendicular = 0'
                                    perpendicular polarization: 'parallel_perpendicular = 1'
    :return: received_power: the received power of the reflected signal on the receiver side
    """
    sqrt_term = np.sqrt(epilon_r - np.power(np.sin(theta_i), 2))
    reflection_coefficient_parallel = (np.cos(theta_i) - sqrt_term) / (np.cos(theta_i) + sqrt_term)
    reflection_coefficient_perpendicular = ((epilon_r * np.cos(theta_i)) - sqrt_term) / (
            (epilon_r * np.cos(theta_i)) + sqrt_term)

    if parallel_perpendicular == 0:
        reflection_coefficient_amplitude = abs(reflection_coefficient_parallel)
        reflection_coefficient_phase = np.arccos(reflection_coefficient_parallel / abs(reflection_coefficient_parallel))
        reflection_coefficient = cmath.rect(reflection_coefficient_amplitude, reflection_coefficient_phase)
    else:
        reflection_coefficient_amplitude = abs(reflection_coefficient_perpendicular)
        reflection_coefficient_phase = np.arccos(
            reflection_coefficient_perpendicular / abs(reflection_coefficient_perpendicular))
        reflection_coefficient = cmath.rect(reflection_coefficient_amplitude, reflection_coefficient_phase)

    term1 = transmitted_power * np.power((wavelength / (4 * np.pi)), 2)
    term2 = reflection_coefficient * np.exp(1j * wave_number * ni * distance) / distance
    received_power = term1 * np.power(np.abs(term2), 2)
    return received_power


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


def draw_incident_reflected_wave(transmitter, receiver, surface_size, element_size, element_spacing, phi_matrix):
    """
    Drawing the surface the transmitter the receiver as a dot. and show the reflection path
    :param transmitter: the coordinates of the transmitter
    :param receiver: the coordinates of the receiver
    :param surface_size: number of elements in both x and y directions of the surface (y_n, x_n)
    :param element_size: size of each edge of a square element
    :param element_spacing: spacing between 2 elements in both x and y directions
           (spacing between elements is the same in both directions)
    :param phi_matrix: 2D phase shift matrix resembling the metasurface where every entry of this matrix represents
                       the phase shift realized by the corresponding element of the surface.
    """
    phi_matrix_deg = np.rad2deg(phi_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw transmitter and receiver
    ax.scatter(transmitter[0], transmitter[1], transmitter[2], color='red', label='Transmitter')
    ax.scatter(receiver[0], receiver[1], receiver[2], color='blue', label='Receiver')
    # Add text labels
    ax.text(transmitter[0], transmitter[1], transmitter[2], 'Transmitter', fontsize=10, color='red')
    ax.text(receiver[0], receiver[1], receiver[2], 'Receiver', fontsize=10, color='blue')

    elements_coordinates_array = elements_coordinates(surface_size, element_size, element_spacing)
    # Convert the coordinates into flat arrays
    x_coordinates = elements_coordinates_array[:, :, 0].flatten()
    y_coordinates = elements_coordinates_array[:, :, 1].flatten()
    phi_values = phi_matrix_deg.flatten()

    # Create a color array from colormap
    cmap = plt.get_cmap('viridis')
    colors = cmap(phi_values)

    # Draw IRS elements
    ax.scatter(x_coordinates, y_coordinates, c=colors, marker='s')

    # Calculate the middle of the surface
    surface_middle = np.array([
        ((surface_size[1] * element_size) + ((surface_size[1] - 1) * element_spacing)) / 2,
        ((surface_size[0] * element_size) + ((surface_size[0] - 1) * element_spacing)) / 2,
        0
    ])

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

    # Set legend
    ax.legend()

    # Set axis labels and plot limits
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Set x-axis limits
    # ax.set_xlim(-0.05, 0.5)
    # ax.set_ylim(-0.1, 0.5)
    # ax.set_zlim(0, 1)

    # plt.show()


def main():
    print_results = True
    save_results = True
    # Parameters
    transmitter = np.array([1, 0.5, 3])  # Position of the transmitter
    receiver = np.array([1.5, 1.2, 2])  # Position of the receiver
    # transmitter = np.array([0.2, 1.2, 0.5])  # Position of the transmitter
    # receiver = np.array([0.3, 1.6, 0.5])  # Position of the receiver
    frequency = 10e9  # Frequency in Hz
    c = constants.speed_of_light  # Speed of light in m/s
    wavelength = c / frequency  # Calculate wavelength
    angular_frequency = 2 * math.pi * frequency
    wave_number = 2 * np.pi / wavelength
    incident_amplitude = 0.1
    incident_phase = math.radians(30)
    # incident_wave_n = incident_amplitude * np.cos(w * t + incident_phase)

    ni = 1  # Refractive index

    # Varactor Parameters
    R_value = 1
    # for f = 2.4GHz varactor components values
    # L1_value = 2.5e-9
    # L2_value = 0.7e-9
    # capacitance_range = np.arange(0.25e-12, 6e-12, 0.01e-12)
    # for f = 10GHz varactor components values
    L1_value = 0.35e-9
    L2_value = 0.25e-9
    capacitance_range = np.arange(0.2e-12, 0.8e-12, 0.001e-12)

    # Metasurface Parameters
    surface_size = (20, 55)  # Metasurface dimensions (M, N)
    # surface_size = (50, 50)  # Metasurface dimensions (M, N)
    element_size = wavelength / 4
    element_spacing = wavelength / 4  # Element spacing in x and y
    delta = element_size + element_spacing

    surface_height = (surface_size[0] * element_size) + ((surface_size[0] - 1) * element_spacing)
    surface_width = (surface_size[1] * element_size) + ((surface_size[0] - 1) * element_spacing)
    surface_area = surface_height * surface_width

    # Calculates surface elements coordinates
    elements_coordinates_array = elements_coordinates(surface_size, element_size, element_spacing)

    # Calculate Incident and Reflected vectors
    incident_vectors, incidence_distances, reflected_vectors, reflection_distances = calculates_incident_reflected_vectors(
        transmitter, receiver, elements_coordinates_array)
    # Calculates ray travelled distances
    rays_distances, min_total_distance, max_total_distance, average_total_distance, \
    min_transmitter_surface_distance, max_transmitter_surface_distance, \
    min_surface_receiver_distance, max_surface_receiver_distance = calculate_wave_travelled_distances(
        incidence_distances, reflection_distances)

    # calculate the phase shifts needed
    theta_i, theta_r, phi_r = calculate_angles(transmitter, receiver, surface_size, element_size, element_spacing)
    dphi_dx, dphi_dy = calculate_dphi_dx_dy(theta_i, theta_r, phi_r, wave_number, ni)
    # phase_shifts = calculate_phase_shifts_from_gradients(dphi_dx, dphi_dy, delta, delta)
    phase_shifts = calculate_phase_shifts_from_gradients1(dphi_dx, dphi_dy, delta, delta)

    # Estimate the capacitance of each element of the surface to achieve the required phase shift
    capacitance_matrix = calculate_capacitance_matrix(R_value, L1_value, L2_value, capacitance_range, phase_shifts,
                                                      angular_frequency)
    # calculate the real phase shifts
    real_reflection_coefficients_array, real_phase_shifts = calculate_real_phase_shifts(R_value, L1_value, L2_value,
                                                                                        capacitance_matrix,
                                                                                        angular_frequency)
    # Calculate the real reflection angles
    real_theta_r, real_phi_r = calculate_real_reflected_angles(theta_i, real_phase_shifts, delta, delta,
                                                               wave_number, ni)

    # compute the successful reflections matrix
    successful_reflections, accurate_elements_percentage = compute_successful_reflections(receiver,
                                                                                          elements_coordinates_array,
                                                                                          incident_vectors,
                                                                                          real_theta_r, real_phi_r)

    # Calculate the received power
    received_powers, received_power = power_received(wavelength, wave_number, incident_amplitude, incident_phase, ni,
                                                     real_reflection_coefficients_array, rays_distances,
                                                     successful_reflections)

    # Calculate the required varactor bias voltages to achieve the required capacitance
    corresponding_varactor_voltages = required_varactor_bias_voltages(capacitance_matrix)

    transmitted_power = np.power(incident_amplitude, 2) / 2

    # Calculate the incident and the reflected angles based on the original snell's law
    original_snells_law_theta_i = find_snells_angle(transmitter, receiver, np.array([0, 0, 1]))

    # calculate the power that could have been received by the receiver antenna without the metasurface
    received_power_no_intelligent_surface = power_without_intelligent_surface(transmitted_power, wavelength,
                                                                              wave_number, ni, average_total_distance,
                                                                              original_snells_law_theta_i, 5)

    results_directory_path = None
    if save_results:
        current_file = os.path.splitext(os.path.basename(__file__))[0]
        results_directory_path = f"./Results_{current_file}/"
        os.makedirs(results_directory_path, exist_ok=True)

        results_file = open(os.path.join(results_directory_path, "results.txt"), "w")
        results_file.write(f"Incident Signal frequency: {frequency * 1e-9} GHz\n")
        results_file.write(f"Incident Signal Wavelength: {round(wavelength * 1e3, 3)} mm\n")
        results_file.write(f"Surface Number of Elements: {surface_size}\n")
        results_file.write(f"Surface Elements Sizes: {round(element_size * 1e3, 3)} mm\n")
        results_file.write(f"Surface Elements spacings: {round(element_spacing * 1e3, 3)} mm\n")
        results_file.write(f"Surface Height: {round(surface_height * 1e2, 2)} cm\n")
        results_file.write(f"Surface Width: {round(surface_width * 1e2, 2)} cm\n")
        results_file.write(f"Surface Area: {round(surface_area, 2)} m²\n")
        results_file.write(
            f"min LOS distance between emitter and surface through surface: {min_transmitter_surface_distance} m\n")
        results_file.write(
            f"max LOS distance between emitter and surface through surface: {max_transmitter_surface_distance} m\n")
        results_file.write(
            f"min LOS distance between surface and receiver through surface: {min_surface_receiver_distance} m\n")
        results_file.write(
            f"max LOS distance between surface and receiver through surface: {max_surface_receiver_distance} m\n")
        results_file.write(
            f"min NLOS distance between emitter and receiver through surface: {min_total_distance} m\n")
        results_file.write(
            f"max NLOS distance between emitter and receiver through surface: {max_total_distance} m\n")
        results_file.write(
            f"average NLOS distance between emitter and receiver through surface: {average_total_distance} m\n")
        results_file.write(f"transmitted power (in Watts): {transmitted_power:.2e} W\n")
        results_file.write(f"transmitted power (in dBm): {round(10 * np.log10(transmitted_power / 1e-3), 2)} dBm\n")
        results_file.write(f"Received Power (in Watts): {received_power:.2e} W\n")
        results_file.write(f"Received Power (in dBm): {round(10 * math.log10(received_power / 1e-3), 2)} dBm\n")

        results_file.write("Number of elements with correct reflection: "
                           f"{round(accurate_elements_percentage * successful_reflections.size)}/{successful_reflections.size}\n")
        results_file.write(
            f"Elements with correct reflection percentage: {round(accurate_elements_percentage * 100, 2)}%\n")

        results_file.write(f"Original Snell's law angle: {np.round(np.degrees(original_snells_law_theta_i), 2)}\n")

        if received_power_no_intelligent_surface != 0:
            results_file.write(
                f"Received Power without IRS (in Watts): {received_power_no_intelligent_surface:.2e} W\n")
            results_file.write(
                f"Received Power without IRS (in dBm): "
                f"{round(10 * math.log10(received_power_no_intelligent_surface / 1e-3), 2)} dBm\n")
            results_file.write(
                f"Additional received power with IRS: "
                f"{round((10 * math.log10(received_power / 1e-3)) - (10 * math.log10(received_power_no_intelligent_surface / 1e-3)), 2)} dBm\n")
        else:
            results_file.write("No received power without the intelligent metasurface.\n")
        results_file.close()

        np.savetxt(os.path.join(results_directory_path, "required_phase_shifts(in degrees).csv"),
                   np.rad2deg(phase_shifts), delimiter=",")
        np.savetxt(os.path.join(results_directory_path, "real_phase_shifts(in degrees).csv"),
                   np.rad2deg(real_phase_shifts), delimiter=",")
        np.savetxt(os.path.join(results_directory_path, "varactors_capacitance_matrix(in picoFarad).csv"),
                   np.round(np.multiply(capacitance_matrix, 1e12), 2), delimiter=",")
        np.savetxt(os.path.join(results_directory_path, "corresponding_varactor_voltages(in Volts).csv"),
                   corresponding_varactor_voltages, delimiter=",")

    if print_results:
        print(f"Surface Height: {round(surface_height * 1e2, 2)} cm")
        print(f"Surface Width: {round(surface_width * 1e2, 2)} cm")
        print(f"Surface Area: {round(surface_area, 2)} m²")

        print(f"min LOS distance between emitter and surface through surface: {min_transmitter_surface_distance} m")
        print(f"max LOS distance between emitter and surface through surface: {max_transmitter_surface_distance} m")
        print(f"min LOS distance between surface and receiver through surface: {min_surface_receiver_distance} m")
        print(f"max LOS distance between surface and receiver through surface: {max_surface_receiver_distance} m")
        print(f"min NLOS distance between emitter and receiver through surface: {min_total_distance} m")
        print(f"max NLOS distance between emitter and receiver through surface: {max_total_distance} m")
        print(f"average NLOS distance between emitter and receiver through surface: {average_total_distance} m")

        print(f"transmitted power (in Watts): {transmitted_power:.2e} W")
        print(f"transmitted power (in dBm): {round(10 * np.log10(transmitted_power / 1e-3), 2)} dBm")
        # print(f"Received Power (in milliWatts): {round(received_power * 1e3, 2)} mW")
        print(f"Received Power (in Watts): {received_power:.2e} W")
        print(f"Received Power (in dBm): {round(10 * math.log10(received_power / 1e-3), 2)} dBm")
        print(f"Percentage Received/Transmitted Power: {((received_power / transmitted_power) * 100):.2e}%")

        print("Number of elements with correct reflection: "
              f"{round(accurate_elements_percentage * successful_reflections.size)}/{successful_reflections.size}")
        print(f"Elements with correct reflection percentage: {round(accurate_elements_percentage * 100, 2)}%")

        print(f"Original Snell's law angle: {np.round(np.degrees(original_snells_law_theta_i), 2)}")
        print(f"Received Power without IRS (in Watts): {received_power_no_intelligent_surface:.2e} W")
        if received_power_no_intelligent_surface != 0:
            print(
                f"Received Power without IRS (in dBm): "
                f"{round(10 * math.log10(received_power_no_intelligent_surface / 1e-3), 2)} dBm")
            print(
                f"Percentage Received/Transmitted Power without IRS: "
                f"{((received_power_no_intelligent_surface / transmitted_power) * 100):.2e}%")
            print(
                f"Additional received power with IRS: "
                f"{round((10 * math.log10(received_power / 1e-3)) - (10 * math.log10(received_power_no_intelligent_surface / 1e-3)), 2)} dBm")
        else:
            print("No received power without the intelligent metasurface.")

        print("\nVaractors Capacitance Matrix (in picoFarad): ")
        print(np.round(np.multiply(capacitance_matrix, 1e12), 2))
        print("\nRequired Varactor Bias Voltages (in Volts):")
        print(corresponding_varactor_voltages)

        show_phase_shift_plots(np.rad2deg(phase_shifts), "Required Phase Shifts", save_plot=save_results,
                               results_directory_path=results_directory_path)
        show_phase_shift_plots(np.rad2deg(real_phase_shifts), "Real Phase Shifts", save_plot=save_results,
                               results_directory_path=results_directory_path)
        # show_phase_shift_plots(np.rad2deg(np.mod(phase_shifts - real_phase_shifts + np.pi, 2 * np.pi) - np.pi), "Difference")
        draw_incident_reflected_wave(transmitter, receiver, surface_size, element_size, element_spacing, phase_shifts)
        plot_power_graph(transmitted_power, received_powers, save_plot=save_results,
                         results_directory_path=results_directory_path)

        plt.show()


if __name__ == "__main__":
    main()
