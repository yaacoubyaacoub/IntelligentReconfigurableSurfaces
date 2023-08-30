function received_power = power_without_intelligent_surface(transmitted_power, wavelength, wave_number, ni, distance, theta_i, epilon_r, parallel_perpendicular)
    % Find the power received by a receiver when a signal is sent from a transmitter and no line of sight between the
    % transmitter and the receiver, the signal that reaches the receiver comes only from reflection of the transmitted
    % signal on a normal smooth surface.
    % In this case the reflection will happen according to the original snell's law 'θi = θr'
    %
    % Inputs:
    %   - transmitted_power: the power of the transmitted signal
    %   - wavelength: the wavelength of the transmitted signal
    %   - wave_number: the number of complete wave cycles of an electromagnetic field that exist in one meter.
    %                  k0=2π/λ
    %   - ni: index of refraction of the medium in which the reflection is taking place
    %   - distance: the total non-line of sight distance between the transmitter and the receiver following the
    %               signal's path. distance = distance_transmitter_surface + distance_surface_receiver
    %   - theta_i: angle of incidence of the signal into the surface.
    %   - epilon_r: permittivity εr of the material composing the surface on which the signal will reflect back
    %   - parallel_perpendicular: what kind of polarization this signal has.
    %                             parallel polarization: 'parallel_perpendicular = 0'
    %                             perpendicular polarization: 'parallel_perpendicular = 1'
    %
    % Output:
    %   - received_power: the received power of the reflected signal on the receiver side

    sqrt_term = sqrt(epilon_r - sin(theta_i)^2);

    reflection_coefficient_parallel = (cos(theta_i) - sqrt_term) / (cos(theta_i) + sqrt_term);
    reflection_coefficient_perpendicular = ((epilon_r * cos(theta_i)) - sqrt_term) / ((epilon_r * cos(theta_i)) + sqrt_term);

    if parallel_perpendicular == 0
        reflection_coefficient_amplitude = abs(reflection_coefficient_parallel);
        reflection_coefficient_phase = acos(reflection_coefficient_parallel / abs(reflection_coefficient_parallel));
        reflection_coefficient = reflection_coefficient_amplitude * exp(1i * reflection_coefficient_phase);
    else
        reflection_coefficient_amplitude = abs(reflection_coefficient_perpendicular);
        reflection_coefficient_phase = acos(reflection_coefficient_perpendicular / abs(reflection_coefficient_perpendicular));
        reflection_coefficient = reflection_coefficient_amplitude * exp(1i * reflection_coefficient_phase);
    end

    term1 = transmitted_power * (wavelength / (4 * pi))^2;
    term2 = reflection_coefficient * exp(1i * wave_number * ni * distance) / distance;
    received_power = term1 * abs(term2)^2;
end
