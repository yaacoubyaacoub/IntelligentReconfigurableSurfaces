function theta_i = find_snells_angle(transmitter, receiver, normal)
    % Find the reflection angle based on the normal reflection law (snell's law 'θi = θr')
    % (the case where no metasurface is implemented)
    % Assuming the surface of reflection lies in the xy-plane (z=0), then the normal to the surface is the unit vector uz
    % parallel to z-axis.
    % The transmitter and the receiver are points located in 3D space.
    %
    % Inputs:
    %   - transmitter: the coordinates of the transmitter
    %   - receiver: the coordinates of the receiver
    %   - normal: unit vector normal to the plane of the metasurface
    %
    % Output:
    %   - theta_i: the angle of reflection. in this case we have θi = θr

    xi = transmitter(1);
    yi = transmitter(2);
    zi = transmitter(3);
    xr = receiver(1);
    yr = receiver(2);
    zr = receiver(3);

    f = @(x, y) (((x - xi).^2 + (y - yi).^2 + zi.^2) ./ ((x - xr).^2 + (y - yr).^2 + zr.^2)) - ((zi ./ zr).^2);

    % Define a grid of points to evaluate the function
    [X, Y] = meshgrid(linspace(min(xi, xr), max(xi, xr), 1000), linspace(min(yi, yr), max(yi, yr), 1000));

    % Evaluate the function on the grid
    Z = f(X, Y);

    % Find the (x, y) coordinates where the function is closest to zero
    [~, idx] = min(abs(Z(:)));
    [x_value, y_value] = ind2sub(size(Z), idx);
    p0 = [X(x_value, y_value), Y(x_value, y_value), 0];

    vi = transmitter - p0;
    theta_i = acos(dot(vi, normal) / norm(vi));
end
