function capacitance_matrix = calculate_capacitance_matrix(R_value, L1_value, L2_value, capacitance_range, phase_shifts, angular_frequency)
    % Calculates the required capacitance of each element of the surface based on the frequency of the incoming signal
    % and the phase shift that must be introduced by each element.
    %
    % Inputs:
    %   - R_value: resistance of every element on the surface
    %   - L1_value: bottom layer inductance of every element on the surface
    %   - L2_value: top layer inductance of every element on the surface
    %   - capacitance_range: capacitance range that the varactor is able to produce
    %   - phase_shifts: 2D matrix of the required phase shift of each element of the surface
    %   - angular_frequency: w = 2 * pi * frequency
    %
    % Output:
    %   - capacitance_matrix: estimated capacitance of each element of the surface based on the frequency of the
    %                         incoming signal and the required phase shift.

    Z0 = freespace_impedance();
    elements_achievable_impedances = element_impedance(R_value, L1_value, L2_value, capacitance_range, angular_frequency);
    elements_achievable_reflection_coefficients = reflection_coefficients(Z0, elements_achievable_impedances);
    reflection_coefficients_amplitude = abs(elements_achievable_reflection_coefficients);
    reflection_coefficients_phase_shifts = angle(elements_achievable_reflection_coefficients);
    capacitance_matrix = estimate_capacitance_for_phase_shift(phase_shifts, capacitance_range, reflection_coefficients_phase_shifts);
end
