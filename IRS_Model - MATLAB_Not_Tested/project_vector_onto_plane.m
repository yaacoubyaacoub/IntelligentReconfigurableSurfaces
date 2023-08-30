function proj_vr_on_pr_p = project_vector_onto_plane(reflected_vector, nr_p)
    % Calculates the projections of the reflected vector onto the plane perpendicular to the incident plane.
    % This plane is denoted by its normal vector nr_p
    % reflected_vector: array of reflected vectors
    % nr_p: array of normal vectors of the planes
    % proj_vr_on_pr_p: the projections of the reflected vector onto the plane perpendicular to the incident plane
    
    % Calculate the projection of vr onto nr_p
    proj_vr_on_nr_p = sum(reflected_vector .* nr_p, 3) ./ vecnorm(nr_p, 2, 3).^2 .* nr_p;

    % Calculate the projection of vr onto plane Pr_p
    proj_vr_on_pr_p = reflected_vector - proj_vr_on_nr_p;

end
