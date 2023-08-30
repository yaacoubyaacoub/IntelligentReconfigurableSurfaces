import numpy as np


def project_vector_onto_plane(vi, vr, uz):
    # Find the normal vector of plane Pi
    ni = np.cross(vi, uz)

    # Find the normal vector of plane Pr
    nr = np.cross(ni, uz)
    nr = nr / np.linalg.norm(nr)

    # Calculate the projection of vr onto nr
    proj_vr_on_nr = (np.dot(vr, nr) / np.power(np.linalg.norm(nr), 2)) * nr
    # proj_vr_on_nr_1 = (np.dot(vr, nr) / np.dot(nr, nr)) * nr

    # Calculate the projection of vr onto plane Pr
    proj_vr_on_pr = vr - proj_vr_on_nr

    return proj_vr_on_pr


def project_vector_onto_plane_3d(vi, vr, uz):
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


def main():
    uz = np.array([0, 0, 1])

    vi = np.array([-2, 0, -3])
    vr = np.array([3, 2, 3])
    proj_vr_on_pr = project_vector_onto_plane(vi, vr, uz)
    print(proj_vr_on_pr)

    vii = np.array([[[-2, 0, -3], [2, 0, 3]], [[-2, 0, 3], [2, 0, -3]]])
    vrr = np.array([[[3, 2, 3], [-3, 2, 3]], [[3, -2, 3], [3, 2, -3]]])
    proj_vrr_on_pr = project_vector_onto_plane_3d(vii, vrr, uz)
    print(proj_vrr_on_pr)


if __name__ == "__main__":
    main()
