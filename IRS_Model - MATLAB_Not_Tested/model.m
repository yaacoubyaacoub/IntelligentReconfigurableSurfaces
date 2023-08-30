function model(transmitter, receiver, room_sizes)
    print_results = false;
    save_results = false;

    % Parameters
    frequency = 10e9;  % Frequency in Hz
    c = 3e8;  % Speed of light in m/s
    wavelength = c / frequency;  % Calculate wavelength
    angular_frequency = 2 * pi * frequency;
    wave_number = 2 * pi / wavelength;
    incident_amplitude = 0.1;
    incident_phase = deg2rad(30);

    ni = 1;  % Refractive index

    % Varactor Parameters
    R_value = 1;
    % for f = 2.4GHz varactor components values
    % L1_value = 2.5e-9;
    % L2_value = 0.7e-9;
    % capacitance_range = linspace(0.25e-12, 6e-12, 0.01e-12);
    % for f = 10GHz varactor components values
    L1_value = 0.35e-9;
    L2_value = 0.25e-9;
    capacitance_range = linspace(0.2e-12, 0.8e-12, 0.001e-12);

    % Metasurface Parameters
    surface_size = [20, 55];  % Metasurface dimensions (M, N)
    % surface_size = [50, 50];  % Metasurface dimensions (M, N)
    element_size = wavelength / 4;
    element_spacing = wavelength / 4;  % Element spacing in x and y
    delta = element_size + element_spacing;

    surface_height = (surface_size(1) * element_size) + ((surface_size(1) - 1) * element_spacing);
    surface_width = (surface_size(2) * element_size) + ((surface_size(1) - 1) * element_spacing);
    surface_area = surface_height * surface_width;

    % Calculates surface elements coordinates
    elements_coordinates_array = elements_coordinates(surface_size, element_size, element_spacing);

    % Calculate Incident and Reflected vectors
    [incident_vectors, incidence_distances, reflected_vectors, reflection_distances] = calculates_incident_reflected_vectors(transmitter, receiver, elements_coordinates_array);

    % Calculates ray travelled distances
    [rays_distances, min_total_distance, max_total_distance, average_total_distance, min_transmitter_surface_distance, max_transmitter_surface_distance, average_transmitter_surface_distance, min_surface_receiver_distance, max_surface_receiver_distance, average_surface_receiver_distance] = calculate_wave_travelled_distances(incidence_distances, reflection_distances);

    % calculate the phase shifts needed
    [theta_i, theta_r, phi_r] = calculate_angles(transmitter, receiver, surface_size, element_size, element_spacing);
    [dphi_dx, dphi_dy] = calculate_dphi_dx_dy(theta_i, theta_r, phi_r, wave_number, ni);
    % phase_shifts = calculate_phase_shifts_from_gradients(dphi_dx, dphi_dy, delta, delta);
    phase_shifts = calculate_phase_shifts_from_gradients1(dphi_dx, dphi_dy, delta, delta);

    % Estimate the capacitance of each element of the surface to achieve the required phase shift
    capacitance_matrix = calculate_capacitance_matrix(R_value, L1_value, L2_value, capacitance_range, phase_shifts, angular_frequency);

    % calculate the real phase shifts
    [real_theta_r, real_phi_r] = calculate_real_reflected_angles(theta_i, real_phase_shifts, delta, delta, ...
                                                            wave_number, ni);

    % Compute the successful reflections matrix
    [successful_reflections, accurate_elements_percentage] = compute_successful_reflections(receiver, ...
                                                                                          elements_coordinates_array, ...
                                                                                          incident_vectors, ...
                                                                                          real_theta_r, real_phi_r);
    
    % Calculate the received power
    [received_powers, received_power] = power_received(wavelength, wave_number, incident_amplitude, incident_phase, ni, ...
                                                       real_reflection_coefficients_array, rays_distances, ...
                                                       successful_reflections);
    
    % Calculate the required varactor bias voltages to achieve the required capacitance
    corresponding_varactor_voltages = required_varactor_bias_voltages(capacitance_matrix);
    
    transmitted_power = (incident_amplitude.^2) / 2;
    
    % Calculate the incident and the reflected angles based on the original Snell's law
    original_snells_law_theta_i = find_snells_angle(transmitter, receiver, [0, 0, 1]);
    
    % Calculate the power that could have been received by the receiver antenna without the metasurface
    received_power_no_intelligent_surface = power_without_intelligent_surface(transmitted_power, wavelength, ...
                                                                              wave_number, ni, average_total_distance, ...
                                                                              original_snells_law_theta_i, 5);
    
    results_directory_path = '';
    if save_results
        disp('Saving Results...');
        current_file = erase(extractBefore(mfilename('fullpath'), '.m'), '.');
        results_directory_path = ['./Results_', current_file, '/'];
        mkdir(results_directory_path);
    
        results_file = fopen([results_directory_path, 'results.txt'], 'w');
        fprintf(results_file, 'Incident Signal frequency: %.2f GHz\n', frequency * 1e-9);
        fprintf(results_file, 'Incident Signal Wavelength: %.3f mm\n', wavelength * 1e3);
        fprintf(results_file, 'Surface Number of Elements: %d\n', surface_size);
        fprintf(results_file, 'Surface Elements Sizes: %.3f mm\n', element_size * 1e3);
        fprintf(results_file, 'Surface Elements spacings: %.3f mm\n', element_spacing * 1e3);
        fprintf(results_file, 'Surface Height: %.2f cm\n', surface_height * 1e2);
        fprintf(results_file, 'Surface Width: %.2f cm\n', surface_width * 1e2);
        fprintf(results_file, 'Surface Area: %.2f m²\n', surface_area);
        fprintf(results_file, 'Min LOS distance between emitter and surface: %.2f m\n', min_transmitter_surface_distance);
        fprintf(results_file, 'Max LOS distance between emitter and surface: %.2f m\n', max_transmitter_surface_distance);
        fprintf(results_file, 'Average LOS distance between emitter and surface: %.2f m\n', average_transmitter_surface_distance);
        fprintf(results_file, 'Min LOS distance between surface and receiver: %.2f m\n', min_surface_receiver_distance);
        fprintf(results_file, 'Max LOS distance between surface and receiver: %.2f m\n', max_surface_receiver_distance);
        fprintf(results_file, 'Average LOS distance between surface and receiver: %.2f m\n', average_surface_receiver_distance);
        fprintf(results_file, 'Min NLOS distance between emitter and receiver through surface: %.2f m\n', min_total_distance);
        fprintf(results_file, 'Max NLOS distance between emitter and receiver through surface: %.2f m\n', max_total_distance);
        fprintf(results_file, 'Average NLOS distance between emitter and receiver through surface: %.2f m\n', average_total_distance);
        fprintf(results_file, 'Transmitted power (in Watts): %.2e W\n', transmitted_power);
        fprintf(results_file, 'Transmitted power (in dBm): %.2f dBm\n', 10*log10(transmitted_power / 1e-3));
        fprintf(results_file, 'Received Power (in Watts): %.2e W\n', received_power);
        fprintf(results_file, 'Received Power (in dBm): %.2f dBm\n', 10*log10(received_power / 1e-3));
    
        fprintf(results_file, 'Number of elements with correct reflection: %d/%d\n', round(accurate_elements_percentage * numel(successful_reflections)), numel(successful_reflections));
        fprintf(results_file, 'Elements with correct reflection percentage: %.2f%%\n', accurate_elements_percentage * 100);
    
        fprintf(results_file, 'Original Snell''s law angle (in degrees): %.2f°\n', rad2deg(original_snells_law_theta_i));
    
        if received_power_no_intelligent_surface ~= 0
            fprintf(results_file, 'Received Power without IRS (in Watts): %.2e W\n', received_power_no_intelligent_surface);
            fprintf(results_file, 'Received Power without IRS (in dBm): %.2f dBm\n', 10*log10(received_power_no_intelligent_surface / 1e-3));
            fprintf(results_file, 'Additional received power with IRS: %.2f dBm\n', 10*log10(received_power / 1e-3) - 10*log10(received_power_no_intelligent_surface / 1e-3));
        else
            fprintf(results_file, 'No received power without the intelligent metasurface.\n');
        end
        
        fclose(results_file);
        
        writematrix([results_directory_path, 'required_phase_shifts(in degrees).csv'], rad2deg(phase_shifts), 'delimiter', ',');
        writematrix([results_directory_path, 'real_phase_shifts(in degrees).csv'], rad2deg(real_phase_shifts), 'delimiter', ',');
        writematrix([results_directory_path, 'varactors_capacitance_matrix(in picoFarad).csv'], round(capacitance_matrix * 1e12, 2), 'delimiter', ',');
        writematrix([results_directory_path, 'corresponding_varactor_voltages(in Volts).csv'], corresponding_varactor_voltages, 'delimiter', ',');
        
        disp('Results Saved.');
    end
    
    if print_results
        disp('Simulation Results:');
        
        disp(['Transmitter Location: ', mat2str(transmitter)]);
        disp(['Receiver Location: ', mat2str(receiver)]);
        
        disp(['Surface Height: ', num2str(round(surface_height * 1e2, 2)), ' cm']);
        disp(['Surface Width: ', num2str(round(surface_width * 1e2, 2)), ' cm']);
        disp(['Surface Area: ', num2str(round(surface_area, 2)), ' m²']);
        
        disp(['Min LOS distance between emitter and surface: ', num2str(min_transmitter_surface_distance), ' m']);
        disp(['Max LOS distance between emitter and surface: ', num2str(max_transmitter_surface_distance), ' m']);
        disp(['Average LOS distance between emitter and surface: ', num2str(average_transmitter_surface_distance), ' m']);
        disp(['Min LOS distance between surface and receiver: ', num2str(min_surface_receiver_distance), ' m']);
        disp(['Max LOS distance between surface and receiver: ', num2str(max_surface_receiver_distance), ' m']);
        disp(['Average LOS distance between surface and receiver: ', num2str(average_surface_receiver_distance), ' m']);
        disp(['Min NLOS distance between emitter and receiver through surface: ', num2str(min_total_distance), ' m']);
        disp(['Max NLOS distance between emitter and receiver through surface: ', num2str(max_total_distance), ' m']);
        disp(['Average NLOS distance between emitter and receiver through surface: ', num2str(average_total_distance), ' m']);
        
        disp(['transmitted power (in Watts): ', num2str(transmitted_power, '%.2e'), ' W']);
        disp(['transmitted power (in dBm): ', num2str(10*log10(transmitted_power / 1e-3), '%.2f'), ' dBm']);
        disp(['Received Power (in Watts): ', num2str(received_power, '%.2e'), ' W']);
        disp(['Received Power (in dBm): ', num2str(10*log10(received_power / 1e-3), '%.2f'), ' dBm']);
        disp(['Percentage Received/Transmitted Power: ', num2str((received_power / transmitted_power) * 100, '%.2e'), '%']);
        
        disp(['Number of elements with correct reflection: ', num2str(round(accurate_elements_percentage * numel(successful_reflections))), '/', num2str(numel(successful_reflections))]);
        disp(['Elements with correct reflection percentage: ', num2str(round(accurate_elements_percentage * 100, 2)), '%']);
        
        disp(['Original Snell''s law angle (in degrees): ', num2str(round(rad2deg(original_snells_law_theta_i), 2)), '°']);
        disp(['Received Power without IRS (in Watts): ', num2str(received_power_no_intelligent_surface, '%.2e'), ' W']);
        if received_power_no_intelligent_surface ~= 0
            disp(['Received Power without IRS (in dBm): ', num2str(10*log10(received_power_no_intelligent_surface / 1e-3), '%.2f'), ' dBm']);
            disp(['Percentage Received/Transmitted Power without IRS: ', num2str((received_power_no_intelligent_surface / transmitted_power) * 100, '%.2e'), '%']);
            disp(['Additional received power with IRS: ', num2str(round(10*log10(received_power / 1e-3) - 10*log10(received_power_no_intelligent_surface / 1e-3), 2)), ' dBm']);
        else
            disp('No received power without the intelligent metasurface.');
        end
        
        disp('Varactors Capacitance Matrix (in picoFarad): ');
        disp(round(capacitance_matrix * 1e12, 2));
        
        disp('Required Varactor Bias Voltages (in Volts):');
        disp(corresponding_varactor_voltages);
    end

    clf;

    show_phase_shift_plots(rad2deg(phase_shifts), 'Required Phase Shifts',save_results,results_directory_path, [3, 2, 1]);
    show_phase_shift_plots(rad2deg(real_phase_shifts), 'Real Phase Shifts',save_results, results_directory_path, [3, 2, 3]);
    draw_incident_reflected_wave(transmitter, receiver, surface_size, element_size, element_spacing, phase_shifts, room_sizes, [3, 2, [2, 4]]);
    plot_power_graph(transmitted_power, received_powers, save_results, results_directory_path, [3, 2, 5]);
    
    output = sprintf('\n%s\n', ...
        ['Transmitter Location: ', mat2str(transmitter)], ...
        ['Receiver Location: ', mat2str(receiver)], ...
        ['Surface Height: ', num2str(round(surface_height * 1e2, 2)), ' cm'], ...
        ['Surface Width: ', num2str(round(surface_width * 1e2, 2)), ' cm'], ...
        ['Surface Area: ', num2str(round(surface_area, 2)), ' m²'], ...
        ['Average LOS distance between emitter and surface: ', num2str(average_transmitter_surface_distance), ' m'], ...
        ['Average LOS distance between surface and receiver: ', num2str(average_surface_receiver_distance), ' m'], ...
        ['Average NLOS distance between emitter and receiver through surface: ', num2str(average_total_distance), ' m'], ...
        ['Incident angle (in degrees): ', num2str(round(rad2deg(theta_i(floor(size(theta_i, 1) / 2), floor(size(theta_i, 2) / 2))), 2)), '°'], ...
        ['Theoretical reflected angle (in degrees): ', num2str(round(rad2deg(theta_r(floor(size(theta_r, 1) / 2), floor(size(theta_r, 2) / 2))), 2)), '°'], ...
        ['Real reflected angle (in degrees): ', num2str(round(rad2deg(real_theta_r(floor(size(real_theta_r, 1) / 2), floor(size(real_theta_r, 2) / 2))), 2)), '°'], ...
        ['Theoretical Phi angle (in degrees): ', num2str(round(rad2deg(phi_r(floor(size(phi_r, 1) / 2), floor(size(phi_r, 2) / 2))), 2)), '°'], ...
        ['Real Phi angle (in degrees): ', num2str(round(rad2deg(real_phi_r(floor(size(real_phi_r, 1) / 2), floor(size(real_phi_r, 2) / 2))), 2)), '°'], ...
        ['Transmitted power (in Watts): ', num2str(transmitted_power, '%.2e'), ' W'], ...
        ['Transmitted power (in dBm): ', num2str(round(10 * log10(transmitted_power / 1e-3), 2)), ' dBm'], ...
        ['Received Power (in Watts): ', num2str(received_power, '%.2e'), ' W'], ...
        ['Received Power (in dBm): ', num2str(round(10 * log10(received_power / 1e-3), 2)), ' dBm'], ...
        ['Percentage Received/Transmitted Power: ', num2str((received_power / transmitted_power) * 100, '%.2e'), '%'], ...
        ['Number of elements with correct reflection: ', num2str(round(accurate_elements_percentage * numel(successful_reflections)), '%d'), '/', num2str(numel(successful_reflections), '%d')], ...
        ['Elements with correct reflection percentage: ', num2str(round(accurate_elements_percentage * 100, 2)), '%']);
    
    plot_text(output,[3, 2, 6]);
    
    tight_layout;

    
