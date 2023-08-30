function [dphi_dx, dphi_dy] = calculate_dphi_dx_dy(theta_i, theta_r, phi_r, wave_number, ni)
    % Calculates the phase gradients in both x and y directions based on Snell's generalized law of reflection.
    % theta_i: array of incidence angles (angle between the incidence vector and the normal to the reflection surface)
    % theta_r: array of reflection angles (angle between the reflected vector and its projection onto the plane perpendicular to the plane of incidence)
    % phi_r: array of angles of diversion from the plane of incidence (angle between the projection of the reflected vector onto the plane perpendicular to the plane of incidence and the normal to the reflection surface)
    % wave_number: the number of complete wave cycles of an electromagnetic field that exist in one meter (k0 = 2π/λ)
    % ni: index of refraction of the medium in which the reflection is taking place
    % dphi_dx: the gradient of the phase in the x direction (based on Snell's generalized law of reflection)
    % dphi_dy: the gradient of the phase in the y direction (based on Snell's generalized law of reflection)

    dphi_dx = (sin(theta_r) - sin(theta_i)) .* ni .* wave_number;
    dphi_dy = cos(theta_r) .* sin(phi_r) .* ni .* wave_number;

end
