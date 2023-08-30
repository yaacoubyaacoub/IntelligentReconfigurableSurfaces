function coords = coordinates_update(coords, step_size_x, step_size_y, step_size_z, room_sizes)
    % Update the coordinates by randomly moving in x, y, and z directions
    %
    % Inputs:
    %   - coords: the current coordinates [x, y, z]
    %   - step_size_x: step size in the x direction
    %   - step_size_y: step size in the y direction
    %   - step_size_z: step size in the z direction
    %   - room_sizes: List of the form [x_coord_min, x_coord_max, y_coord_min, y_coord_max, z_coord_min, z_coord_max]
    %                  defining the 3D sizes of the room using 3D coordinates relative to the position of the
    %                  metasurface which start at position (0, 0, 0)
    %
    % Outputs:
    %   - coords: the updated coordinates after random movement

    if ~isempty(room_sizes)
        % Room size 3D coordinates (Limits where the transmitter and the receiver could be in the room)
        x_coord_min = room_sizes(1);
        x_coord_max = room_sizes(2);
        y_coord_min = room_sizes(3);
        y_coord_max = room_sizes(4);
        z_coord_min = room_sizes(5);
        z_coord_max = room_sizes(6);
    end

    % Taking direction randomly
    directions_taken = datasample(["x", "y", "z"], randi([0, 3]), 'Replace', false);

    for i = 1:numel(directions_taken)
        d = directions_taken(i);
        if d == "x"
            move_x = step_size_x * randsample([1, -1], 1);
            x_coord = coords(1) + move_x;
            if isempty(room_sizes) || (x_coord_min <= x_coord && x_coord <= x_coord_max)
                coords(1) = x_coord;
            end
        elseif d == "y"
            move_y = step_size_y * randsample([1, -1], 1);
            y_coord = coords(2) + move_y;
            if isempty(room_sizes) || (y_coord_min <= y_coord && y_coord <= y_coord_max)
                coords(2) = y_coord;
            end
        elseif d == "z"
            move_z = step_size_z * randsample([1, -1], 1);
            z_coord = coords(3) + move_z;
            if isempty(room_sizes) || (z_coord_min <= z_coord && z_coord <= z_coord_max)
                coords(3) = z_coord;
            end
        end
    end
end
