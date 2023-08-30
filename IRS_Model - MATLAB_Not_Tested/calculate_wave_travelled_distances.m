function [rays_distances, min_total_distance, max_total_distance, average_total_distance, ...
    min_transmitter_surface_distance, max_transmitter_surface_distance, average_transmitter_surface_distance, ...
    min_surface_receiver_distance, max_surface_receiver_distance, average_surface_receiver_distance] = ...
    calculate_wave_travelled_distances(incidence_distances, reflection_distances)
    % Calculates the distances traveled by the waves from the transmitter to the receiver through the surface.
    % incidence_distances: distances between the transmitter and each element of the surface
    % reflection_distances: distances between the receiver and each element of the surface
    % rays_distances: distances between the transmitter and the receiver through each element of the surface
    % min_total_distance: min distance between the transmitter and the receiver through the surface
    % max_total_distance: max distance between the transmitter and the receiver through the surface
    % average_total_distance: average distance between the transmitter and the receiver through the surface
    % min_transmitter_surface_distance: min distance between the transmitter and the surface
    % max_transmitter_surface_distance: max distance between the transmitter and the surface
    % average_transmitter_surface_distance: average distance between the transmitter and the surface
    % min_surface_receiver_distance: min distance between the surface and the receiver
    % max_surface_receiver_distance: max distance between the surface and the receiver
    % average_surface_receiver_distance: average distance between the surface and the receiver
    
    rays_distances = incidence_distances + reflection_distances;

    min_transmitter_surface_distance = round(min(incidence_distances(:)), 2);
    max_transmitter_surface_distance = round(max(incidence_distances(:)), 2);
    average_transmitter_surface_distance = round(mean(incidence_distances(:)), 2);

    min_surface_receiver_distance = round(min(reflection_distances(:)), 2);
    max_surface_receiver_distance = round(max(reflection_distances(:)), 2);
    average_surface_receiver_distance = round(mean(reflection_distances(:)), 2);

    min_total_distance = round(min(rays_distances(:)), 2);
    max_total_distance = round(max(rays_distances(:)), 2);
    average_total_distance = round(mean(rays_distances(:)), 2);

end
