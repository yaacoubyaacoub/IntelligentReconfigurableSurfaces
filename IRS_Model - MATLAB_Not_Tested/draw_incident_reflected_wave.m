function draw_incident_reflected_wave(transmitter, receiver, surface_size, element_size, element_spacing, phi_matrix, room_sizes, subplot_position)
    % Drawing the surface the transmitter the receiver as a dot. and show the reflection path
    %
    % Inputs:
    %   - transmitter: the coordinates of the transmitter
    %   - receiver: the coordinates of the receiver
    %   - surface_size: number of elements in both x and y directions of the surface (y_n, x_n)
    %   - element_size: size of each edge of a square element
    %   - element_spacing: spacing between 2 elements in both x and y directions
    %                      (spacing between elements is the same in both directions)
    %   - phi_matrix: 2D phase shift matrix resembling the metasurface where every entry of this matrix represents
    %                 the phase shift realized by the corresponding element of the surface
    %   - room_sizes: List of the form [x_coord_min, x_coord_max, y_coord_min, y_coord_max, z_coord_min, z_coord_max]
    %                  defining the 3D sizes of the room using 3D coordinates relative to the position of the
    %                  metasurface which start at position (0, 0, 0)
    %   - subplot_position: Plot location on the figure in case we want all plots on the same figure.
    %                        None, if each plot is an independent figure.
    
    phi_matrix_deg = rad2deg(phi_matrix);

    if ~isempty(subplot_position)
        ax = subplot(subplot_position(1), subplot_position(2), subplot_position(3), 'projection', '3d');
    else
        fig = figure;
        ax = axes('Parent', fig, 'projection', '3d');
    end

    % Draw transmitter and receiver
    scatter3(ax, transmitter(1), transmitter(2), transmitter(3), 'red', 'filled', 'DisplayName', 'Transmitter');
    hold(ax, 'on');
    scatter3(ax, receiver(1), receiver(2), receiver(3), 'blue', 'filled', 'DisplayName', 'Receiver');
    % Add text labels
    text(ax, transmitter(1), transmitter(2), transmitter(3), 'Transmitter', 'FontSize', 10, 'Color', 'red');
    text(ax, receiver(1), receiver(2), receiver(3), 'Receiver', 'FontSize', 10, 'Color', 'blue');

    elements_coordinates_array = elements_coordinates(surface_size, element_size, element_spacing);
    % Convert the coordinates into flat arrays
    x_coordinates = reshape(elements_coordinates_array(:, :, 1), [], 1);
    y_coordinates = reshape(elements_coordinates_array(:, :, 2), [], 1);
    phi_values = phi_matrix_deg(:);

    % Create a color array from colormap
    cmap = colormap('viridis');
    colors = interp1(linspace(min(phi_values), max(phi_values), size(cmap, 1)), cmap, phi_values);

    % Draw IRS elements
    scatter3(ax, x_coordinates, y_coordinates, zeros(size(x_coordinates)), 36, colors, 's', 'filled');

    % Calculate the middle of the surface
    surface_middle = [
        ((surface_size(2) * element_size) + ((surface_size(2) - 1) * element_spacing)) / 2,
        ((surface_size(1) * element_size) + ((surface_size(1) - 1) * element_spacing)) / 2,
        0
    ];

    % Draw incident wave
    incident_wave = linspace(transmitter, surface_middle, 100);
    plot3(ax, incident_wave(:, 1), incident_wave(:, 2), incident_wave(:, 3), 'r--', 'DisplayName', 'Incident Wave');

    % Draw reflected wave
    reflected_wave = linspace(surface_middle, receiver, 100);
    plot3(ax, reflected_wave(:, 1), reflected_wave(:, 2), reflected_wave(:, 3), 'b--', 'DisplayName', 'Reflected Wave');

    % Draw normal vector to the surface
    normal_start = surface_middle;
    normal_end = surface_middle + [0, 0, max(transmitter(3), receiver(3))];
    plot3(ax, [normal_start(1), normal_end(1)], [normal_start(2), normal_end(2)], [normal_start(3), normal_end(3)], 'k-', 'DisplayName', 'Normal Vector');

    % Set legend
    legend(ax, 'Location', 'northwest');

    % Set title
    title(ax, 'IRS Reflection Model');

    % Set axis labels
    xlabel(ax, 'X-axis');
    ylabel(ax, 'Y-axis');
    zlabel(ax, 'Z-axis');

    % Set axis limits (Room Size)
    if ~isempty(room_sizes)
        % Room size 3D coordinates
        x_coord_min = room_sizes(1);
        x_coord_max = room_sizes(2);
        y_coord_min = room_sizes(3);
        y_coord_max = room_sizes(4);
        z_coord_min = room_sizes(5);
        z_coord_max = room_sizes(6);
        xlim(ax, [x_coord_min, x_coord_max]);
        ylim(ax, [y_coord_min, y_coord_max]);
        zlim(ax, [z_coord_min, z_coord_max]);
    end
end
