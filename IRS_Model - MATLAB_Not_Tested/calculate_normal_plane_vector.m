function nr_p = calculate_normal_plane_vector(incident_vector, uz)
    % Calculates the normal vectors of the planes perpendicular to the incident planes
    % where the incident plane is the plane that includes the incident vector
    % incident_vector: array of incident vectors
    % uz: unit vector normal to the plane of the metasurface
    % nr_p: array of normal vector to each plane of each incident vector
    
    % Default value for uz if not provided
    if nargin < 2
        uz = [0, 0, 1];
    end

    % Find the normal vector of plane of incidence Pi
    ni = cross(incident_vector, uz);

    % Find the normal vector of plane Pr_p perpendicular to Pi
    nr_p = cross(ni, uz);
    nr_p = nr_p ./ vecnorm(nr_p, 2, 3);

end
