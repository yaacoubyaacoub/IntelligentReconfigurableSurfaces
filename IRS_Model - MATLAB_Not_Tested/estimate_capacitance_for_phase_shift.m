function estimated_capacitance = estimate_capacitance_for_phase_shift(target_phase_shift, c_values, available_phase_shifts)
    % Estimate the capacitance needed for a required phase shift.
    % Done by interpolation between the available phase shifts that could be realized by a given varactor.
    % These values are calculated by setting a capacitance value to a given range realizable by the varactor,
    % calculating the phase shift realized using each capacitance value and saving the capacitance value and their
    % corresponding phase shifts.
    % target_phase_shift: the phase shift that we need to achieve for a given element
    % c_values: 1D array of the capacitance value range achievable by the varactor
    % available_phase_shifts: 1D array of the corresponding phase shifts realized by the varactor given c_values
    % estimated_capacitance: the estimated capacitance value for the target phase shift
    
    estimated_capacitance = interp1(available_phase_shifts, c_values, target_phase_shift, 'linear', 'extrap');
end
