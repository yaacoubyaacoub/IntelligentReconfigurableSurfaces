function phase_shifts = calculate_phase_shifts_from_gradients(dphi_dx, dphi_dy, delta_x, delta_y)
    % Calculates the phase_shifts from the partial derivatives dphi_dx, dphi_dy using the "Random Walk Method".
    % dphi_dx: the gradient of the phase in the x direction (based on Snell's generalized law of reflection)
    % dphi_dy: the gradient of the phase in the y direction (based on Snell's generalized law of reflection)
    % delta_x: the difference between an element and the next one in the x direction, taken between the middle of two adjacent elements
    % delta_y: the difference between an element and the next one in the y direction, taken between the middle of two adjacent elements
    % phase_shifts: 2D array resembling the metasurface where every entry of this matrix represents the phase shift required by the corresponding element of the surface

    phase_shifts = zeros(size(dphi_dx));

    curr_x = 1;
    curr_y = 1;

    visited_elements = zeros(size(dphi_dx));

    current_min_visits = 0;
    target_min_visits = 10;
    while min(visited_elements(:)) < target_min_visits
        new_direction = randi([1, 4]);
        % Directions:
        %     1 = Right (-->)
        %     2 = Left (<--)
        %     3 = Down
        %     4 = Up

        if (new_direction == 2 && curr_x == 2 && curr_y == 1) || (new_direction == 4 && curr_x == 1 && curr_y == 2)
            curr_x = 1;
            curr_y = 1;
        else
            if new_direction == 1 && curr_x < size(phase_shifts, 2)
                if phase_shifts(curr_y, curr_x + 1) ~= 0
                    phase_shifts(curr_y, curr_x + 1) = (phase_shifts(curr_y, curr_x + 1) + phase_shifts(curr_y, curr_x) + delta_x * dphi_dx(curr_y, curr_x)) / 2;
                else
                    phase_shifts(curr_y, curr_x + 1) = phase_shifts(curr_y, curr_x) + delta_x * dphi_dx(curr_y, curr_x);
                end
                curr_x = curr_x + 1;
            elseif new_direction == 2 && curr_x > 1
                if phase_shifts(curr_y, curr_x - 1) ~= 0
                    phase_shifts(curr_y, curr_x - 1) = (phase_shifts(curr_y, curr_x - 1) + phase_shifts(curr_y, curr_x) - delta_x * dphi_dx(curr_y, curr_x)) / 2;
                else
                    phase_shifts(curr_y, curr_x - 1) = phase_shifts(curr_y, curr_x) - delta_x * dphi_dx(curr_y, curr_x);
                end
                curr_x = curr_x - 1;
            elseif new_direction == 3 && curr_y < size(phase_shifts, 1)
                if phase_shifts(curr_y + 1, curr_x) ~= 0
                    phase_shifts(curr_y + 1, curr_x) = (phase_shifts(curr_y + 1, curr_x) + phase_shifts(curr_y, curr_x) + delta_y * dphi_dy(curr_y, curr_x)) / 2;
                else
                    phase_shifts(curr_y + 1, curr_x) = phase_shifts(curr_y, curr_x) + delta_y * dphi_dy(curr_y, curr_x);
                end
                curr_y = curr_y + 1;
            elseif new_direction == 4 && curr_y > 1
                if phase_shifts(curr_y - 1, curr_x) ~= 0
                    phase_shifts(curr_y - 1, curr_x) = (phase_shifts(curr_y - 1, curr_x) + phase_shifts(curr_y, curr_x) - delta_y * dphi_dy(curr_y, curr_x)) / 2;
                else
                    phase_shifts(curr_y - 1, curr_x) = phase_shifts(curr_y, curr_x) - delta_y * dphi_dy(curr_y, curr_x);
                end
                curr_y = curr_y - 1;
            else
                continue;
            end
        end
        visited_elements(curr_y, curr_x) = visited_elements(curr_y, curr_x) + 1;

        if current_min_visits < min(visited_elements(:))
            current_min_visits = min(visited_elements(:));
        end
    end

    phase_shifts = mod(phase_shifts + pi, 2 * pi) - pi;

end
