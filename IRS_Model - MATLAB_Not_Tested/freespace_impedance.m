function Z0 = freespace_impedance()
    % Calculate the impedance of freespace.
    % Z0 = μ0 / ε0
    % ε0: Permittivity of freespace
    % μ0: Permeability of freespace
    % Z0: Impedance of freespace
    
    epsilon_0 = 8.8541878128e-12;
    mu_0 =  4*pi*1e-7;
    Z0 = sqrt(mu_0 / epsilon_0);
end
