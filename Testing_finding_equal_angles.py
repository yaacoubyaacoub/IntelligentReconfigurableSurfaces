import numpy as np


def find_xy(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    def f(x, y):
        return ((x - x1) ** 2 + (y - y1) ** 2) / ((x - x2) ** 2 + (y - y2) ** 2) - (z1 / z2) ** 2

    # Define a grid of points to evaluate the function
    X, Y = np.meshgrid(np.linspace(min(x1, x2), max(x1, x2), 1000), np.linspace(min(y1, y2), max(y1, y2), 1000))

    # Evaluate the function on the grid
    Z = f(X, Y)

    # Find the (x, y) coordinates where the function is closest to zero
    idx = np.argmin(np.abs(Z))
    x, y = X.flat[idx], Y.flat[idx]

    return x, y


def angles(p0, p1, p2, normal):
    v1 = p1 - p0
    v2 = p2 - p0
    theta1 = np.arccos(np.dot(v1, normal) / np.linalg.norm(v1))
    theta2 = np.arccos(np.dot(v2, normal) / np.linalg.norm(v2))

    return theta1, theta2


# p1 = np.array([1, 5, 3])
# p2 = np.array([4, 2, 6])
p1 = np.array([10, 0.5, 500])
p2 = np.array([15, 1.2, 15])
x, y = find_xy(p1, p2)
print("x =", x)
print("y =", y)
p0 = np.array([x, y, 0])
normal = np.array([0, 0, 1])
theta1, theta2 = angles(p0, p1, p2, normal)

print("theta1 =", np.rad2deg(theta1))
print("theta2 =", np.rad2deg(theta2))
