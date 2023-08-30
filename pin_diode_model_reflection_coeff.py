import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import scipy.constants as constants
from scipy.optimize import fsolve
from tqdm import tqdm


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


def plot_text(text):
    # Add the text to the plot
    plt.text(0, 0, text, fontsize=12, fontfamily="Arial", fontweight="bold")

    # Remove the axis ticks and labels
    plt.axis("off")


Ra = 50
Rs = 5.2
Cg = 0.025e-12

Rj = 1000
Ls = 1e-9
# Ls_range = np.arange(1e-9, 10e-9, 0.01e-9)
Cj_range = np.arange(0.01e-12, 10e-12, 0.01e-12)
Cb_range = np.arange(0.01e-12, 10e-12, 0.01e-12)
# print(f"Ls_range length: {len(Ls_range)}")
print(f"Cj_range length: {len(Cj_range)}")
print(f"Cb_range length: {len(Cb_range)}")

frequency = 10e9  # Frequency in Hz
c = constants.speed_of_light  # Speed of light in m/s
wavelength = c / frequency  # Calculate wavelength
w = 2 * math.pi * frequency

Z0 = freespace_impedance()

plt.figure()
break_flag = False
theta_rad_min, Ls_n, Cj_n, Cb_n = 180, 0, 0, 0
theta_rad_max, Ls_m, Cj_m, Cb_m = 0, 0, 0, 0

for Cj in tqdm(Cj_range):
    for Cb in Cb_range:
        A = (1j * w * Ls) + Rs + (((-1j * Rj) / (w * Cj)) / (Rj - (1j / (w * Cj)))) - (1j / (w * Cb))
        B = -1j / (w * Cg)

        Z = Ra + (A * B) / (A + B)

        ref_cof = reflection_coefficients(Z0, Z)

        r, theta_rad = cmath.polar(ref_cof)

        # if Cb == Cb_range[0] or Cb == Cb_range[-1]:
        #     theta_deg = math.degrees(theta_rad)
        #     plt.clf()
        #     output = '\n'.join([
        #         "Intermediate Polar coordinates:",
        #         f"   Magnitude: {r}",
        #         f"   Angle: {theta_deg}°"
        #     ])
        #     plot_text(output)
        #     plt.tight_layout()
        #     plt.draw()
        #     plt.pause(0.1)

        if abs(theta_rad) <= math.radians(1):
            theta_deg = math.degrees(theta_rad)
            print(f"\nPolar coordinates: {r}<{theta_deg}")
            print(f"Ls = {Ls}")
            print(f"Cj = {Cj}")
            print(f"Cb = {Cb}")
            break_flag = True
            break

        if theta_rad > theta_rad_max:
            theta_rad_max = theta_rad
            Ls_m, Cj_m, Cb_m = Ls, Cj, Cb
        if theta_rad > 0 and theta_rad < theta_rad_min:
            theta_rad_min = theta_rad
            Ls_n, Cj_n, Cb_n = Ls, Cj, Cb

    if break_flag:
        break

theta_deg_max = math.degrees(theta_rad_max)
print(f"\nMax Angle: {theta_deg_max}")
print(f"Ls = {Ls_m}")
print(f"Cj = {Cj_m}")
print(f"Cb = {Cb_m}")
theta_deg_min = math.degrees(theta_rad_min)
print(f"\nMin Angle: {theta_deg_min}")
print(f"Ls = {Ls_n}")
print(f"Cj = {Cj_n}")
print(f"Cb = {Cb_n}")
