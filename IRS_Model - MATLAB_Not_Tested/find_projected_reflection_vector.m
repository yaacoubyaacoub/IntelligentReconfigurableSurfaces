function reflected_projection_vector = find_projected_reflection_vector(origin_points, receiver, nr_p, phi_r)
    % Finds the projection of the reflected vector onto the plane P perpendicular to the incident plane
    % using the angle between the projected vector and the z-axis phi_r
    % origin_points: origin of the projection of the reflected vector on plane P
    % receiver: the coordinates of the receiver
    % nr_p: normal vector to the plane P
    % phi_r: angle between the z-axis and the projection of the reflected vector onto P
    % reflected_projection_vector: the reflected projection vector
    
    % Projecting the receiver onto the planes perpendicular to the planes of incidence
    reflected_vectors = receiver - origin_points;
    proj_vr_on_nr_p = (sum(reflected_vectors .* nr_p, 3) ./ vecnorm(nr_p, 2, 3).^2) .* nr_p;
    proj_r_on_pr_p = receiver - proj_vr_on_nr_p;

    d = -1 * sum(origin_points .* nr_p, 3);
    planes = cat(3, nr_p, d); % [a, b, c, d]

    vz = cos(phi_r);
    z2 = vz + origin_points(:, :, 3);

    X = (planes(:, :, 3) .* z2) + planes(:, :, 4);

    A = ((planes(:, :, 1) .^ 2) ./ (planes(:, :, 2) .^ 2)) + 1;
    B = ((2 * X .* planes(:, :, 1)) ./ (planes(:, :, 2) .^ 2)) + ...
        ((2 * planes(:, :, 1) .* origin_points(:, :, 2)) ./ planes(:, :, 2)) - (2 * origin_points(:, :, 1));
    C = ((X .^ 2) ./ (planes(:, :, 2) .^ 2)) + ...
        ((2 * X .* origin_points(:, :, 2)) ./ planes(:, :, 2)) + ...
        (origin_points(:, :, 1) .^ 2) + (origin_points(:, :, 2) .^ 2) + (vz .^ 2) - 1;

    sol1 = (-B + sqrt((B .^ 2) - (4 * A .* C))) ./ (2 * A);
    sol2 = (-B - sqrt((B .^ 2) - (4 * A .* C))) ./ (2 * A);

    x2 = nan(size(sol1));
    y2 = nan(size(sol1));
    x2(origin_points(:, :, 1) > proj_r_on_pr_p(:, :, 1)) = min(sol1(origin_points(:, :, 1) > proj_r_on_pr_p(:, :, 1)), ...
        sol2(origin_points(:, :, 1) > proj_r_on_pr_p(:, :, 1)));
    x2(origin_points(:, :, 1) <= proj_r_on_pr_p(:, :, 1)) = max(sol1(origin_points(:, :, 1) <= proj_r_on_pr_p(:, :, 1)), ...
        sol2(origin_points(:, :, 1) <= proj_r_on_pr_p(:, :, 1)));

    y2(origin_points(:, :, 1) > proj_r_on_pr_p(:, :, 1)) = ...
        (-1 ./ planes(:, :, 2)) .* ((planes(:, :, 1) .* x2(origin_points(:, :, 1) > proj_r_on_pr_p(:, :, 1))) + ...
        (planes(:, :, 3) .* z2(origin_points(:, :, 1) > proj_r_on_pr_p(:, :, 1))) + planes(:, :, 4));
    y2(origin_points(:, :, 1) <= proj_r_on_pr_p(:, :, 1)) = ...
        (-1 ./ planes(:, :, 2)) .* ((planes(:, :, 1) .* x2(origin_points(:, :, 1) <= proj_r_on_pr_p(:, :, 1))) + ...
        (planes(:, :, 3) .* z2(origin_points(:, :, 1) <= proj_r_on_pr_p(:, :, 1))) + planes(:, :, 4));

    unit_reflected_projection_vector = cat(3, ...
        x2 - origin_points(:, :, 1), y2 - origin_points(:, :, 2), z2 - origin_points(:, :, 3));

    t = (receiver(3) - origin_points(:, :, 3)) ./ unit_reflected_projection_vector(:, :, 3);

    x = origin_points(:, :, 1) + unit_reflected_projection_vector(:, :, 1) .* t;
    y = origin_points(:, :, 2) + unit_reflected_projection_vector(:, :, 2) .* t;

    reflected_projection_vector = cat(3, ...
        x - origin_points(:, :, 1), y - origin_points(:, :, 2), receiver(3) - origin_points(:, :, 3));

end
