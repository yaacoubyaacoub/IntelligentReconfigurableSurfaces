function [incident_vectors, incident_vectors_norms, reflected_vectors, reflected_vectors_norms] = ...
    calculates_incident_reflected_vectors(transmitter, receiver, elements_coordinates_array)
    % Computes the incident and reflected vectors.
    % Computes the distances between the transmitter and each element of the surface (incident vectors norms).
    % Computes the distances between the receiver and each element of the surface (reflected vectors norms).
    % transmitter: the coordinates of the transmitter
    % receiver: the coordinates of the receiver
    % elements_coordinates_array: array containing the coordinates of each element of the surface based on
    %                             their numbers, sizes, and spacings
    % incident_vectors: array of incident vectors
    % incident_vectors_norms: distances between the transmitter and each element of the surface
    % reflected_vectors: array of reflected vectors
    % reflected_vectors_norms: distances between the receiver and each element of the surface
    
    incident_vectors = elements_coordinates_array - transmitter;
    incident_vectors_norms = vecnorm(incident_vectors, 2, 3);
    reflected_vectors = receiver - elements_coordinates_array;
    reflected_vectors_norms = vecnorm(reflected_vectors, 2, 3);

end
