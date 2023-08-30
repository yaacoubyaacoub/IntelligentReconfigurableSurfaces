function reflection_coefficients = reflection_coefficients(Z0, Z1_n)
    % Calculate reflection coefficients for given impedances.
    % Z0: impedance of freespace
    % Z1_n: impedance of a given surface
    % reflection_coefficients: reflection coefficients of the surfaces
    
    reflection_coefficients = (Z1_n - Z0) / (Z1_n + Z0);
end