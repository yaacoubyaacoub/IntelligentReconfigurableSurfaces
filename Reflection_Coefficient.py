import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.constants as constants


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


def estimate_capacitance_for_phase_shift(target_phase_shift, c_values, phase_shifts):
    return np.interp(target_phase_shift, phase_shifts, c_values, period=360)


def main():
    # Parameters
    Z0 = freespace_impedance()
    R_value = 1
    L_value = 2.5e-9
    wavelength = 0.125
    # frequency = constants.speed_of_light / wavelength
    frequency = 2.4e9  # Frequency in Hz
    w = 2 * math.pi * frequency
    c_values = np.arange(0.01e-12, 6e-12, 0.01e-12)

    element_impedances = element_impedance(R_value, L_value, c_values, w)

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
    print(f"Estimated capacitance value: {np.round(estimated_C * 1e12, 2)}pF")

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
