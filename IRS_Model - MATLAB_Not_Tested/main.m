transmitter = [-1.73, 0.15, 3];  % Position of the transmitter
receiver = [2.27, 0.15, 3];  % Position of the receiver
% transmitter = [-4, 0.15, 3];  % Position of the transmitter
% receiver = [6, 0.15, 8];  % Position of the receiver

% Room size 3D coordinates
x_coord_min = -5;
x_coord_max = 7;
y_coord_min = -1.5;
y_coord_max = 1.5;
z_coord_min = 0;
z_coord_max = 12;
room_sizes = [x_coord_min, x_coord_max, y_coord_min, y_coord_max, z_coord_min, z_coord_max];

figure;
set(gcf, 'WindowState', 'maximized');

while true
    model(transmitter, receiver, room_sizes);
    drawnow;
    pause(1);

    transmitter = coordinates_update(transmitter, 'room_sizes', room_sizes);
    receiver = coordinates_update(receiver, 'room_sizes', room_sizes);
end