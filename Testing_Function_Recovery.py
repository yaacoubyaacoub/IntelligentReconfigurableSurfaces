import numpy as np
import matplotlib.pyplot as plt
import random


def gradient_2d(f, delta_x=1.0, delta_y=1.0):
    rows, cols = f.shape

    # Initialize gradient arrays
    df_dy = np.zeros_like(f)
    df_dx = np.zeros_like(f)

    # Compute gradients along rows (axis 0)
    df_dy[0] = (f[1] - f[0]) / delta_y  # Forward difference for first row
    df_dy[-1] = (f[-1] - f[-2]) / delta_y  # Backward difference for last row
    df_dy[1:-1] = (f[2:] - f[:-2]) / (2 * delta_y)  # Central difference for interior rows

    # Compute gradients along columns (axis 1)
    df_dx[:, 0] = (f[:, 1] - f[:, 0]) / delta_x  # Forward difference for first column
    df_dx[:, -1] = (f[:, -1] - f[:, -2]) / delta_x  # Backward difference for last column
    df_dx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * delta_x)  # Central difference for interior columns

    return df_dy, df_dx


def gradient_2d_periodic(f, delta_x=1.0, delta_y=1.0):
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


def recover_function(df_dy, df_dx, delta_x=1.0, delta_y=1.0, f00=0):
    rows, cols = df_dy.shape

    # Integrate along columns (axis 1) to recover partial function g
    g = np.zeros_like(df_dx)
    g[:, 0] = f00
    g[:, 1:] = np.cumsum(df_dx[:, :-1] * delta_x, axis=1)

    # Integrate along rows (axis 0) to recover the original function f
    f = np.zeros_like(df_dy)
    f[0] = g[0]
    f[1:] = np.cumsum(df_dy[:-1] * delta_y, axis=0) + g[1:]

    return f


def trapezoidal_integration_2d(df_dy, df_dx, delta_x=1.0, delta_y=1.0, f00=0):
    rows, cols = df_dy.shape

    # Integrate along columns (axis 1) to recover partial function g
    g = np.zeros_like(df_dx)
    g[:, 0] = f00
    g[:, 1:] = np.cumsum(0.5 * (df_dx[:, :-1] + df_dx[:, 1:]) * delta_x, axis=1)

    # Integrate along rows (axis 0) to recover the original function f
    f = np.zeros_like(df_dy)
    f[0] = g[0]
    f[1:] = np.cumsum(0.5 * (df_dy[:-1] + df_dy[1:]) * delta_y, axis=0) + g[1:]

    return f


def simpsons_integration_2d(df_dy, df_dx, delta_x=1.0, delta_y=1.0, f00=0):
    if df_dy.shape[0] % 2 != 0 or df_dy.shape[1] % 2 != 0:
        raise ValueError("Simpson's rule requires an even number of intervals in each dimension.")

    rows, cols = df_dy.shape

    # Integrate along columns (axis 1) to recover partial function g
    g = np.zeros_like(df_dx)
    g[:, 0] = f00
    weights_x = np.ones(cols - 1)
    weights_x[1::2] = 4
    weights_x[2:-1:2] = 2
    g[:, 1:] = np.cumsum(df_dx[:, :-1] * weights_x * (delta_x / 3), axis=1)

    # Integrate along rows (axis 0) to recover the original function f
    f = np.zeros_like(df_dy)
    f[0] = g[0]
    weights_y = np.ones(rows - 1)
    weights_y[1::2] = 4
    weights_y[2:-1:2] = 2
    f[1:] = np.cumsum(df_dy[:-1] * weights_y[:, np.newaxis] * (delta_y / 3), axis=0) + g[1:]

    return f


def random_walk_integration_2d(dphi_dx, dphi_dy, delta_x, delta_y):
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

    return phase_shifts


def integration_2d(dphi_dx, dphi_dy, delta_x, delta_y):
    # Integrate along the x-axis
    phase_shifts_x_y0 = np.cumsum(dphi_dx * delta_x, axis=1)

    # Integrate along the y-axis
    phase_shifts_x0_y = np.cumsum(dphi_dy * delta_y, axis=0)

    phase_shifts = phase_shifts_x_y0 + phase_shifts_x0_y

    return phase_shifts


def averaging_integration_2d(dphi_dx, dphi_dy, delta_x, delta_y):
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

    return phase_shifts


# Calculate the phase shift array from the phase gradient arrays (dphi_dx, dphi_dy)
def calculate_phase_shifts_from_gradients(dphi_dx, dphi_dy, delta_x, delta_y, f00=0):
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
    phase_shifts_x[0, 0] = f00
    phase_shifts_y[0, 0] = f00

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

    # Compute the average of phase_shifts_x and phase_shifts_y
    phase_shifts = (phase_shifts_x + phase_shifts_y) / 2

    return phase_shifts_x, phase_shifts_y, phase_shifts


def plot_3D_function(f, delta_x, delta_y, title):
    # Create x and y arrays for plotting
    x_vals, y_vals = np.meshgrid(np.arange(0, f.shape[1]) * delta_x,
                                 np.arange(0, f.shape[0]) * delta_y)

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the recovered function
    ax.plot_surface(x_vals, y_vals, f, cmap='viridis', alpha=0.8)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot title
    ax.set_title(title)

    # plt.show()


def main():
    # Define the 2D function z = x^2 + y^2
    x, y = np.meshgrid(np.arange(0, 6), np.arange(0, 6))
    # x, y = np.meshgrid(np.arange(-3, 3), np.arange(-3, 3))
    z = (0.1 * ((x ** 2) + (3 * (y ** 2)) + (2 * x) - y)) + 0.2
    # z = 0.1 * (x ** 2 + y ** 2)
    delta_x = 0.1
    delta_y = 0.1

    # Compute the gradients using gradient_2d
    # dz_dy, dz_dx = gradient_2d(z, delta_x, delta_y)
    dz_dx = np.gradient(z, delta_x, axis=1)
    dz_dy = np.gradient(z, delta_y, axis=0)

    # Recover the function using recover_function
    # recovered_z = recover_function(dz_dy, dz_dx, delta_x, delta_y, f00=z[0, 0])
    # recovered_z = trapezoidal_integration_2d(dz_dy, dz_dx, delta_x, delta_y, f00=z[0, 0])
    # recovered_z = simpsons_integration_2d(dz_dy, dz_dx, delta_x, delta_y, f00=z[0, 0])
    # recovered_z = random_walk_integration_2d(dz_dx, dz_dy, delta_x, delta_y)
    # recovered_z = integration_2d(dz_dx, dz_dy, delta_x, delta_y)
    # recovered_z = averaging_integration_2d(dz_dx, dz_dy, delta_x, delta_y)

    recovered_z_x, recovered_z_y, recovered_z = calculate_phase_shifts_from_gradients(dz_dx, dz_dy, delta_x, delta_y,
                                                                                      f00=z[0, 0])

    difference_function = np.abs(recovered_z - z)

    print("Original function:\n", z)
    print("Recovered function:\n", recovered_z)
    print("Difference function:\n", difference_function)
    print("dz_dx:\n", dz_dx)
    print("dz_dy:\n", dz_dy)

    plot_3D_function(z, delta_x, delta_y, "function")
    plot_3D_function(recovered_z, delta_x, delta_y, "recovered function")
    plot_3D_function(difference_function, delta_x, delta_y, "difference function")
    plt.show()


if __name__ == "__main__":
    main()
