function [theta_i, theta_r, phi_r] = calculate_angles(transmitter, receiver, surface_size, element_size, element_spacing)
    % Calculates the angles of the reflection phenomenon based on Snell's generalized law of reflection.
    % This calculation is done geometrically at this stage.
    % transmitter: the coordinates of the transmitter
    % receiver: the coordinates of the receiver
    % surface_size: number of elements in both x and y directions of the surface (y_n, x_n)
    % element_size: size of each edge of a square element
    % element_spacing: spacing between 2 elements in both x and y directions
    % theta_i: array of incidence angles (angle between the incidence vector and the normal to the reflection surface)
    % theta_r: array of reflection angles (angle between the reflected vector and its projection onto the plane perpendicular to the plane of incidence)
    % phi_r: array of angles of diversion from the plane of incidence (angle between the projection of the reflected vector onto the plane perpendicular to the plane of incidence and the normal to the reflection surface)

    elements_coordinates_array = elements_coordinates(surface_size, element_size, element_spacing);
    [incident_vectors, incident_vectors_norms, reflected_vectors, reflected_vectors_norms] = ...
        calculates_incident_reflected_vectors(transmitter, receiver, elements_coordinates_array);

    normal = [0, 0, 1];

    % theta_i: angles between the incident vectors and the normal to the reflection surface
    theta_i = acos(dot(-incident_vectors, normal, 2) ./ incident_vectors_norms);

    % theta_r: angles between the reflected vectors and the normal to the reflection surface
    % theta_r = acos(dot(reflected_vectors, normal, 2) ./ reflected_vectors_norms);

    % "projections" are the projection vectors of reflected vectors onto the plane perpendicular to incident vectors plane
    nr_p = calculate_normal_plane_vector(incident_vectors, normal);
    projections = project_vector_onto_plane(reflected_vectors, nr_p);
    projections_mag = vecnorm(projections, 2, 2);

    % "theta_r" are the angles between reflected vectors and the "projections"
    theta_r = acos(dot(projections, reflected_vectors, 2) ./ (projections_mag .* reflected_vectors_norms));

    % "phi_r" are the angles between projections and normal to the metasurface (z-axis)
    phi_r = acos(dot(projections, normal, 2) ./ projections_mag);

    % ############################################# Used only for testing ##############################################
    % testing getting projected_vector and reflected_vector using phi_r and theta_r
    proj_vect = find_projected_reflection_vector(elements_coordinates_array, receiver, nr_p, phi_r);
    ref_vec = find_reflection_vector(elements_coordinates_array, receiver, proj_vect, theta_r);
    diffp = proj_vect - projections;
    diffr = ref_vec - reflected_vectors;

    % Number of elements of the surface that are following the original Snell's law 'θi = θr', 'φr = 0'
    % If rounding to 2 digits: accurate to 0.57 degrees = 0.01 radiant
    % If rounding to 3 digits: accurate to 0.057 degrees = 0.001 radiant
    accuracy = 3;
    phi_r__0 = round(phi_r, accuracy) == 0;
    theta_i__theta_r = round(theta_i, accuracy) == round(theta_r, accuracy);
    original_snell_law = theta_i__theta_r & phi_r__0;
    number_original_snell_law = sum(original_snell_law(:));
    percentage_original_snell_law = round((number_original_snell_law / numel(original_snell_law)) * 100, 2);
    % ##################################################################################################################

end
