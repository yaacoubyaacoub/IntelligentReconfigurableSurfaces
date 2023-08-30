function [received_powers, received_power] = power_received(wavelength, wave_number, incident_amplitude, incident_phase, ni, real_reflection_coefficients_array, rays_distances, successful_reflections)
    % Calculates the power received by the receiver antenna.
    % The calculation is based on the two-ray model but ignoring the line of sight component.
    %
    % Inputs:
    %   - wavelength: the wavelength of the transmitted signal
    %   - wave_number: the number of complete wave cycles of an electromagnetic field that exist in one meter.
    %                  k0 = 2π / λ
    %   - incident_amplitude: the amplitude of the incident wave
    %   - incident_phase: the phase of the incident wave
    %   - ni: index of refraction of the medium in which the reflection is taking place
    %   - real_reflection_coefficients_array: 2D array of complex numbers representing the real reflection
    %                                         coefficients of each element of the surface
    %   - rays_distances: distances between the transmitter and the receiver through each element of the surface
    %   - successful_reflections: 2D boolean array where each entry represents an element of the metasurface.
    %
    % Outputs:
    %   - received_powers: an array of received powers at each step along the rays_distances
    %   - received_power: the total received power by the receiver antenna

    transmitted_power = abs(incident_amplitude).^2 / 2;

    term1 = transmitted_power * (wavelength / (4 * pi))^2;

    term2 = real_reflection_coefficients_array .* exp(1j * wave_number * ni * rays_distances) ./ rays_distances;
    term2 = term2 .* successful_reflections;

    term2 = cumsum(term2(:));
    received_powers = term1 * abs(term2).^2;

    received_power = received_powers(end);
end
