function [real_reflection_coefficients_array, real_phase_shifts] = calculate_real_phase_shifts(R_value, L1_value, L2_value, capacitance_matrix, angular_frequency)
    % Calculates the real reflection coefficients and the real phase shifts introduced by each element of the surface
    % based on the frequency of the incoming signal, and the capacitance of each element.
    %
    % Inputs:
    %   - R_value: resistance of every element on the surface
    %   - L1_value: bottom layer inductance of every element on the surface
    %   - L2_value: top layer inductance of every element on the surface
    %   - capacitance_matrix: estimated capacitance of each element of the surface based on the frequency of the
    %                         incoming signal and the required phase shift.
    %   - angular_frequency: w = 2 * pi * frequency
    %
    % Outputs:
    %   - real_reflection_coefficients_array: 2D array of complex numbers representing the real reflection
    %                                         coefficients of each element of the surface.
    %   - real_phase_shifts: 2D matrix of the real phase shift introduced by each element of the surface

    Z0 = freespace_impedance();
    real_elements_impedance = element_impedance(R_value, L1_value, L2_value, capacitance_matrix, angular_frequency);
    real_reflection_coefficients_array = reflection_coefficients(Z0, real_elements_impedance);
    real_phase_shifts = angle(real_reflection_coefficients_array);
end
