import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.constants as constants


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


def estimate_capacitance_for_phase_shift(target_phase_shift, c_values, phase_shifts):
    return np.interp(target_phase_shift, phase_shifts, c_values, period=360)


def main():
    # Parameters
    Z0 = freespace_impedance()
    R_value = 1

    # 2.4GHz frequency
    # L1_value = 2.5e-9
    # L2_value = 0.7e-9
    # frequency = 2.4e9  # Frequency in Hz
    # c_values = np.arange(0.01e-12, 4e-12, 0.01e-12)

    # 5GHz frequency
    L1_value = 0.65e-9
    L2_value = 0.5e-9
    frequency = 5e9  # Frequency in Hz
    c_values = np.arange(0.01e-12, 2e-12, 0.01e-12)

    # 10GHz frequency
    # L1_value = 0.35e-9
    # L2_value = 0.25e-9
    # frequency = 10e9  # Frequency in Hz
    # c_values = np.arange(0.2e-12, 0.8e-12, 0.001e-12)

    # 20GHz frequency
    # L1_value = 0.2e-9
    # L2_value = 80e-12
    # frequency = 20e9  # Frequency in Hz
    # c_values = np.arange(0.1e-12, 0.5e-12, 0.001e-12)

    # 300GHz frequency
    # L1_value = 11e-12
    # L2_value = 1e-12
    # frequency = 300e9  # Frequency in Hz
    # c_values = np.arange(15e-15, 40e-15, 0.001e-15)

    wavelength = constants.speed_of_light / frequency
    print(f"Wavelength = {round(wavelength * 1e3, 2)} mm")

    w = 2 * math.pi * frequency

    element_impedances = element_impedance(R_value, L1_value, L2_value, c_values, w)

    reflection_coeffs = reflection_coefficients(Z0, element_impedances)

    # reflection_coefficients_amplitude = 10 * np.log10(np.abs(reflection_coeffs)) # in dB
    reflection_coefficients_amplitude = np.abs(reflection_coeffs)
    phase_shifts = np.rad2deg(np.angle(reflection_coeffs))

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(8, 8))

    # Plotting the reflection coefficients amplitude
    axs[0].plot(c_values, reflection_coefficients_amplitude)
    axs[0].set_xlabel('C Values')
    axs[0].set_ylabel('Reflection Coefficient Amplitude')
    axs[0].grid(True)

    # Plotting the phase shifts
    axs[1].plot(c_values, phase_shifts)
    axs[1].set_xlabel('C Values')
    axs[1].set_ylabel('Phase Shift (degrees)')
    axs[1].grid(True)

    target_phase_shift = 38  # Desired phase shift in degrees
    estimated_C = estimate_capacitance_for_phase_shift(target_phase_shift, c_values, phase_shifts)
    print(f"To achieve a phase shift of {target_phase_shift} degrees,", end=" ")
    # print(f"the estimated capacitance value is: {np.round(estimated_C * 1e12, 2)}pF")
    print(f"the estimated capacitance value is: {np.round(estimated_C * 1e15, 2)}fF")

    axs[1].scatter(estimated_C, target_phase_shift, color='r', marker='o', s=20,
                   label=f'Desired Phase Shift ({target_phase_shift}°)')
    axs[1].text((estimated_C + 0.1e-12), (target_phase_shift + 2),
                f'({np.round(estimated_C * 1e12, 2)}pF, {target_phase_shift}°)', fontsize=10, color='blue')
    axs[1].axhline(y=target_phase_shift, color='r', linestyle='--', clip_on=True)
    axs[1].axvline(x=estimated_C, color='r', linestyle='--')
    axs[1].legend()

    plt.show()


if __name__ == "__main__":
    main()
