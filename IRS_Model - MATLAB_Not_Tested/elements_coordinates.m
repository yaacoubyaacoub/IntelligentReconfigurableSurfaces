function elements_coordinates_array = elements_coordinates(surface_size, element_size, element_spacing)
    % Calculates the surface elements coordinates based on their numbers, sizes, and spacings.
    % surface_size: number of elements in both x and y directions of the surface (y_n, x_n)
    % element_size: size of each edge of a square element
    % element_spacing: spacing between 2 elements in both x and y directions
    %                  (spacing between elements is the same in both directions)
    % elements_coordinates_array: array containing the coordinates of each element of the surface based on
    %                             their numbers, sizes, and spacings
    
    [y_indices, x_indices] = meshgrid(0:surface_size(1)-1, 0:surface_size(2)-1);

    x_values = (element_size / 2) + (x_indices * element_spacing) + (x_indices * element_size);
    y_values = (element_size / 2) + (y_indices * element_spacing) + (y_indices * element_size);
    z_values = zeros(size(x_values));

    elements_coordinates_array = cat(3, x_values, y_values, z_values);

end
