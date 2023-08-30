function phase_shifts = calculate_phase_shifts_from_gradients1(dphi_dx, dphi_dy, delta_x, delta_y)
    % Calculates the phase_shifts from the partial derivatives dphi_dx, dphi_dy.
    % Create 2 phase_shifts arrays (phase_shifts_x, phase_shifts_y) one calculated based on dphi_dx and the other based
    % on dphi_dy.
    % In both phase_shifts arrays, the first column (x=0) is always calculated based on dphi_dx and delta_x.
    % In both phase_shifts arrays, the first row (y=0) is always calculated based on dphi_dy and delta_y.
    % Then complete both arrays using the following equations:
    %
    % x=0: f(x+1,y) = (δf/δx * Δx) + f(x,y)
    % x=-1: f(x,y) = (δf/δx * Δx) + f(x-1,y)                  (x=-1 means the last x value of the array)
    % x=[1,...,-2]: f(x+1,y) = (δf/δx * 2*Δx) + f(x-1,y)      (x=-2 means the value before the last x of the array)
    %
    % y=0: f(x,y+1) = (δf/δy * Δy) + f(x,y)
    % y=-1: f(x,y) = (δf/δy * Δy) + f(x,y-1)                  (y=-1 means the last y value of the array)
    % y=[1,...,-2]: f(x,y+1) = (δf/δx * 2*Δx) + f(x,y-1)      (y=-2 means the value before the last y of the array)
    %
    % Then based on the first column (x=0) and the first row (y=0) of both phase_shifts arrays
    % (phase_shifts_x, phase_shifts_y), they should theoretically be the same, but due to estimation error,
    % we will have some differences. So finally, to calculate the final phase shift array, we take the average of
    % both phase_shifts arrays (phase_shifts_x, phase_shifts_y) by adding them together (on an element basis)
    % and then dividing by 2.
    % phase_shifts = (phase_shifts_x + phase_shifts_y) / 2

    % Initialize phase_shifts_x and phase_shifts_y arrays with zeros
    phase_shifts_x = zeros(size(dphi_dx));
    phase_shifts_y = zeros(size(dphi_dx));

    for curr_y = 1:size(phase_shifts_x, 1)
        for curr_x = 1:size(phase_shifts_x, 2)
            % Fill the phase_shifts_x array
            if curr_x == 1
                % Calculate phase shift for x=0 using dphi_dx and delta_x
                phase_shifts_x(curr_y, curr_x + 1) = (delta_x * dphi_dx(curr_y, curr_x)) + phase_shifts_x(curr_y, curr_x);

                % Fill the first column of the phase_shifts_x (x=0) using dphi_dy and delta_y
                if curr_y == 1
                    % Calculate phase shift for y=0 using dphi_dy and delta_y
                    phase_shifts_x(curr_y + 1, curr_x) = (delta_y * dphi_dy(curr_y, curr_x)) + phase_shifts_x(curr_y, curr_x);
                elseif curr_y == size(phase_shifts_y, 1)
                    % Calculate phase shift for y=-1 (last y value) using dphi_dy and delta_y
                    phase_shifts_x(curr_y, curr_x) = (delta_y * dphi_dy(curr_y, curr_x)) + phase_shifts_x(curr_y - 1, curr_x);
                else
                    % Calculate phase shift for y=[1,...,-2] using dphi_dy and delta_y
                    phase_shifts_x(curr_y + 1, curr_x) = (2 * delta_y * dphi_dy(curr_y, curr_x)) + phase_shifts_x(curr_y - 1, curr_x);
                end
            elseif curr_x == size(phase_shifts_x, 2)
                % Calculate phase shift for x=-1 (last x value) using dphi_dx and delta_x
                phase_shifts_x(curr_y, curr_x) = (delta_x * dphi_dx(curr_y, curr_x)) + phase_shifts_x(curr_y, curr_x - 1);
            else
                % Calculate phase shift for x=[1,...,-2] using dphi_dx and delta_x
                phase_shifts_x(curr_y, curr_x + 1) = (2 * delta_x * dphi_dx(curr_y, curr_x)) + phase_shifts_x(curr_y, curr_x - 1);
            end

            % Fill the phase_shifts_y array
            if curr_y == 1
                % Calculate phase shift for y=0 using dphi_dy and delta_y
                phase_shifts_y(curr_y + 1, curr_x) = (delta_y * dphi_dy(curr_y, curr_x)) + phase_shifts_y(curr_y, curr_x);

                % Fill the first row of the phase_shifts_y (y=0) using dphi_dx and delta_x
                if curr_x == 1
                    % Calculate phase shift for x=0 using dphi_dx and delta_x
                    phase_shifts_y(curr_y, curr_x + 1) = (delta_x * dphi_dx(curr_y, curr_x)) + phase_shifts_y(curr_y, curr_x);
                elseif curr_x == size(phase_shifts_y, 2)
                    % Calculate phase shift for x=-1 (last x value) using dphi_dx and delta_x
                    phase_shifts_y(curr_y, curr_x) = (delta_x * dphi_dx(curr_y, curr_x)) + phase_shifts_y(curr_y, curr_x - 1);
                else
                    % Calculate phase shift for x=[1,...,-2] using dphi_dx and delta_x
                    phase_shifts_y(curr_y, curr_x + 1) = (2 * delta_x * dphi_dx(curr_y, curr_x)) + phase_shifts_y(curr_y, curr_x - 1);
                end
            elseif curr_y == size(phase_shifts_y, 1)
                % Calculate phase shift for y=-1 (last y value) using dphi_dy and delta_y
                phase_shifts_y(curr_y, curr_x) = (delta_y * dphi_dy(curr_y, curr_x)) + phase_shifts_y(curr_y - 1, curr_x);
            else
                % Calculate phase shift for y=[1,...,-2] using dphi_dy and delta_y
                phase_shifts_y(curr_y + 1, curr_x) = (2 * delta_y * dphi_dy(curr_y, curr_x)) + phase_shifts_y(curr_y - 1, curr_x);
            end
        end
    end

    % Calculate the final phase_shifts array by taking the average of phase_shifts_x and phase_shifts_y
    phase_shifts = (phase_shifts_x + phase_shifts_y) / 2;

    % Wrap phase_shifts values to the range [-pi, pi]
    phase_shifts = mod(phase_shifts + pi, 2 * pi) - pi;
end
