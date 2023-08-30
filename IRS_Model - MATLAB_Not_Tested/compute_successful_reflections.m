function [successful_reflections, accurate_elements_percentage] = compute_successful_reflections(receiver, elements_coordinates_array, incident_vectors, real_theta_r, real_phi_r)
    % Compute the real trajectory of the reflected rays based on the real reflection angles (θt, φr) calculated from the
    % real phase shift introduced by each element of the metasurface.
    % Estimate if the reflected vector will be hitting the receiver antenna based on the antenna shape and dimensions
    %
    % Inputs:
    %   - receiver: the coordinates of the receiver in space
    %   - elements_coordinates_array: array containing the coordinates of each element of the metasurface
    %   - incident_vectors: array containing the incident vectors from the transmitter to each element of the surface
    %   - real_theta_r: theta_r: array of reflection angles.
    %                                (angle between the reflected vector and its projection onto the plane perpendicular
    %                                to the plane of incidence)
    %   - real_phi_r: array of angles of diversion from the plane of incidence.
    %                  (angle between the projection the reflected vector onto the plane perpendicular to the plane of
    %                  incidence and the normal to the reflection surface)
    %
    % Outputs:
    %   - successful_reflections: 2D boolean array where each entry represents an element of the metasurface.
    %                             - 'true': if the reflected vector was successful in hitting the receiver antenna
    %                             - 'false': if the reflected vector misses the receiver antenna
    %   - accurate_elements_percentage: percentage of accurate elements that successfully hit the receiver antenna

    nr_p = calculate_normal_plane_vector(incident_vectors);
    real_projected_vectors = find_projected_reflection_vector(elements_coordinates_array, receiver, nr_p, real_phi_r);
    real_reflected_vectors = find_reflection_vector(elements_coordinates_array, receiver, real_projected_vectors, real_theta_r);

    real_destination_reached = real_reflected_vectors + elements_coordinates_array;

    % Ignoring the rays that will not hit the receiver. If the hit location is outside the rectangular antenna dimensions,
    % the ray will be ignored. The antenna rectangular dimensions are modeled by the ranges [x_min, x_max] and [y_min, y_max].
    antenna_width = 0.05;  % Receiver antenna width in meters
    antenna_height = 0.1;  % Receiver antenna height in meters
    antenna = [receiver(1) - antenna_width, receiver(1) + antenna_width, receiver(2) - antenna_height, receiver(2) + antenna_height];  % [x_min, x_max, y_min, y_max]
    x_mask = (real_destination_reached(:, :, 1) > antenna(1)) & (real_destination_reached(:, :, 1) < antenna(2));
    y_mask = (real_destination_reached(:, :, 2) > antenna(3)) & (real_destination_reached(:, :, 2) < antenna(4));
    successful_reflections = x_mask & y_mask;
    accurate_elements_percentage = mean(successful_reflections(:));

end
