function reflected_vector = find_reflection_vector(origin_points, receiver, projection, theta_r)
    % Finds the reflected vectors based on its projections onto the plane P perpendicular to the incident plane
    % and the angle of reflection theta_r
    % origin_points: origin of the reflected vector
    % receiver: the coordinates of the receiver
    % projection: projection of the reflected vector onto plane P array [a, b, c]
    % theta_r: angle between the reflection vector and its projection onto the plane P
    % reflected_vector: the reflection vector
    
    projection_magnitude = vecnorm(projection, 2, 3);
    reflected_vector_magnitude = projection_magnitude ./ cos(theta_r);

    zr = projection(:, :, 3);

    X = reflected_vector_magnitude .* projection_magnitude .* cos(theta_r) - zr .^ 2;

    A = ((projection(:, :, 1) .^ 2) ./ (projection(:, :, 2) .^ 2)) + 1;
    B = (-2 * X .* projection(:, :, 1)) ./ (projection(:, :, 2) .^ 2);
    C = ((X .^ 2) ./ (projection(:, :, 2) .^ 2)) + (zr .^ 2) - (reflected_vector_magnitude .^ 2);

    sol1 = (-B + sqrt((B .^ 2) - (4 * A .* C))) ./ (2 * A);
    estimate_sol1 = sol1 + origin_points(:, :, 1);
    sol2 = (-B - sqrt((B .^ 2) - (4 * A .* C))) ./ (2 * A);
    estimate_sol2 = sol2 + origin_points(:, :, 1);

    xr = nan(size(sol1));
    yr = nan(size(sol1));
    xr(abs(estimate_sol1 - receiver(1)) < abs(estimate_sol2 - receiver(1))) = ...
        sol1(abs(estimate_sol1 - receiver(1)) < abs(estimate_sol2 - receiver(1)));
    xr(abs(estimate_sol1 - receiver(1)) >= abs(estimate_sol2 - receiver(1))) = ...
        sol2(abs(estimate_sol1 - receiver(1)) >= abs(estimate_sol2 - receiver(1)));

    yr = (X - (projection(:, :, 1) .* xr)) ./ projection(:, :, 2);

    reflected_vector = cat(3, xr, yr, zr);

end
