function [theta_r, phi_r] = calculate_real_reflected_angles(theta_i, phase_shifts, delta_x, delta_y, wave_number, ni)
    % Calculates the phase shifts gradient realized by the surface using the phase shifts array.
    % Calculates the 3D reflection angles 'theta_r' and 'phi_r' based on Snell's generalized law of reflection:
    %
    % sin(theta_r) - sin(theta_i) = (1/(ni * k0)) * dPhi_dx
    % cos(theta_t) * sin(phi_r) = (1/(ni * k0)) * dPhi_dy
    %
    % Inputs:
    %   - theta_i: array of incidence angles (angle between the incidence vector and the normal to the reflection surface)
    %   - phase_shifts: 2D array resembling the metasurface where every entry represents the phase shift
    %     realized by the corresponding element of the surface.
    %   - delta_x: difference between an element and the next one in the x direction (taken between the middle of two adjacent elements)
    %   - delta_y: difference between an element and the next one in the y direction (taken between the middle of two adjacent elements)
    %   - wave_number: the number of complete wave cycles of an electromagnetic field that exist in one meter (k0 = 2*pi/lambda)
    %   - ni: index of refraction of the medium in which the reflection is taking place
    %
    % Outputs:
    %   - theta_r: array of reflection angles (angle between the reflected vector and its projection onto the plane perpendicular to the plane of incidence)
    %   - phi_r: array of angles of diversion from the plane of incidence (angle between the projection of the reflected vector onto the plane perpendicular to the plane of incidence and the normal to the reflection surface)

    % Compute phase shift gradients
    [dphi_dx, dphi_dy] = gradient_2d_periodic(phase_shifts, delta_x, delta_y);

    % Compute reflection angles
    theta_r = asin((dphi_dx ./ (ni * wave_number)) + sin(theta_i));
    phi_r = asin(dphi_dy ./ (wave_number * cos(theta_r)));
end
