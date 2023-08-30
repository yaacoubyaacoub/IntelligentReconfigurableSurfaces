function [df_dx, df_dy] = gradient_2d_periodic(f, delta_x, delta_y)
    % Calculates the gradient of a function f, taking into account the [-pi pi] periodicity of the function.
    % When calculating the difference between 2 values of the function, modulo [-pi pi] is performed on the result.
    %
    % x=0: df_dx = (f(x+1,y) - f(x,y))/delta_x
    % x=-1: df_dx = (f(x,y) - f(x-1,y))/delta_x          (x=-1 means the last x value of the array)
    % x=[1,...,-2]: df_dx = (f(x+1,y) - f(x-1,y))/(2*delta_x)    (x=-2 means the value before the last x of the array)
    %
    % y=0: df_dy = (f(x,y+1) - f(x,y))/delta_y
    % y=-1: df_dy = (f(x,y) - f(x,y-1))/delta_y          (y=-1 means the last y value of the array)
    % y=[1,...,-2]: df_dy = (f(x,y+1) - f(x,y-1))/(2*delta_y)    (y=-2 means the value before the last y of the array)
    %
    % Inputs:
    %   - f: the function to derive in x and y directions
    %   - delta_x: the difference between an element and the next one in the x direction
    %   - delta_y: the difference between an element and the next one in the y direction
    %
    % Outputs:
    %   - df_dx: the gradient of the function f in the x direction
    %   - df_dy: the gradient of the function f in the y direction
    
    % Initialize gradient arrays
    df_dy = zeros(size(f));
    df_dx = zeros(size(f));

    % Compute gradients along rows (axis 1)
    df_dy(1,:) = (mod(f(2,:) - f(1,:) + pi, 2*pi) - pi) / delta_y;  % Forward difference for first row
    df_dy(end,:) = (mod(f(end,:) - f(end-1,:) + pi, 2*pi) - pi) / delta_y;  % Backward difference for last row
    df_dy(2:end-1,:) = (mod(f(3:end,:) - f(1:end-2,:) + pi, 2*pi) - pi) / (2*delta_y);  % Central difference for interior rows

    % Compute gradients along columns (axis 2)
    df_dx(:,1) = (mod(f(:,2) - f(:,1) + pi, 2*pi) - pi) / delta_x;  % Forward difference for first column
    df_dx(:,end) = (mod(f(:,end) - f(:,end-1) + pi, 2*pi) - pi) / delta_x;  % Backward difference for last column
    df_dx(:,2:end-1) = (mod(f(:,3:end) - f(:,1:end-2) + pi, 2*pi) - pi) / (2*delta_x);  % Central difference for interior columns
end
