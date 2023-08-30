import numpy as np


def find_projected_reflection_vector(point, receiver, nr_p, phi_r):
    """
    :param point: origin of the projection of the reflected vector on plane P
    :param receiver: the coordinates of the receiver
    :param nr_p: normal vector to the plane P
    :param phi_r: angle between the z-axis and the projection of the reflected vector onto P
    :return: the reflected projection vector
    """
    # Projecting the receiver onto the planes perpendicular to the planes of incidence
    reflected_vector = receiver - point
    proj_vr_on_nr_p = (np.dot(reflected_vector, nr_p) / np.power(np.linalg.norm(nr_p), 2)) * nr_p
    proj_r_on_pr_p = np.subtract(receiver, proj_vr_on_nr_p)

    d = -1 * ((nr_p[0] * point[0]) + (nr_p[1] * point[1]) + (nr_p[2] * point[2]))
    plane = [nr_p[0], nr_p[1], nr_p[2], d]  # [a, b, c, d]

    vz = np.cos(phi_r)
    z2 = vz + point[2]

    X = (plane[2] * z2) + plane[3]

    A = ((plane[0] ** 2) / (plane[1] ** 2)) + 1
    B = ((2 * X * plane[0]) / (plane[1] ** 2)) + ((2 * plane[0] * point[1]) / plane[1]) - (2 * point[0])
    C = ((X ** 2) / (plane[1] ** 2)) + ((2 * X * point[1]) / plane[1]) + (point[0] ** 2) + (point[1] ** 2) + (
            vz ** 2) - 1

    s1, s2 = np.roots((A, B, C))
    sol1 = (-B + np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
    sol2 = (-B - np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)

    x2 = sol2 if np.isnan(sol1) else sol1 if np.isnan(sol2) else np.minimum(sol1, sol2) if (
            point[0] > proj_r_on_pr_p[0]) else np.maximum(sol1, sol2)

    y2 = (-1 / plane[1]) * ((plane[0] * x2) + (plane[2] * z2) + plane[3])

    unit_reflected_projection_vector = np.array([x2 - point[0], y2 - point[1], z2 - point[2]])

    t = (receiver[2] - point[2]) / unit_reflected_projection_vector[2]

    x = point[0] + unit_reflected_projection_vector[0] * t
    y = point[1] + unit_reflected_projection_vector[1] * t

    reflected_projection_vector = np.array([x - point[0], y - point[1], receiver[2] - point[2]])

    return reflected_projection_vector


def find_reflection_vector(point, receiver, projection, theta_r):
    """
     :param point: origin of the projection of the reflected vector on plane P
     :param receiver: the coordinates of the receiver
     :param projection: projection of the reflected vector onto plane P array [a, b, c]
     :param theta_r: angle between the reflection vector and its projection onto the plane P
     :return: the reflection vector
     """
    projection_magnitude = np.linalg.norm(projection)
    reflected_vector_magnitude = projection_magnitude / np.cos(theta_r)

    zr = projection[2]

    X = reflected_vector_magnitude * projection_magnitude * np.cos(theta_r) - zr ** 2

    A = ((projection[0] ** 2) / (projection[1] ** 2)) + 1
    B = (-2 * X * projection[0]) / (projection[1] ** 2)
    C = ((X ** 2) / (projection[1] ** 2)) + (zr ** 2) - (reflected_vector_magnitude ** 2)

    s1, s2 = np.roots((A, B, C))
    sol1 = (-B + np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
    estimate_sol1 = sol1 + point[0]
    sol2 = (-B - np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
    estimate_sol2 = sol2 + point[0]

    xr = sol2 if np.isnan(sol1) else sol1 if np.isnan(sol2) else sol1 if (
            np.abs(estimate_sol1 - receiver[0]) < np.abs(estimate_sol2 - receiver[0])) else sol2

    yr = (X - (projection[0] * xr)) / projection[1]

    reflected_vector = np.array([xr, yr, zr])

    return reflected_vector
