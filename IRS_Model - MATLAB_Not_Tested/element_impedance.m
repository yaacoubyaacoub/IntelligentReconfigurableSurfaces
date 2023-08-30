function z_eq = element_impedance(R, L1, L2, C, w)
    % Calculate the impedance of an element of the surface.
    % Equivalent circuit of one element of the metasurface:
    %             ------------L1-----------
    %         ____|                       |____
    %             |                       |
    %             -----L2-----R-----C------
    % R: effective resistance of element
    % L1: bottom layer inductance of the element
    % L2: top layer inductance of the element
    % C: effective capacitance of an element
    % w: angular frequency (w = 2 * π * frequency)
    % z_eq: the element impedance
    
    jwL1 = 1j * w * L1;
    jwL2 = 1j * w * L2;
    jwC = 1j * w * C;

    node1 = jwL1;
    node2 = jwL2 + (1 / jwC) + R;
    z_eq = (node1 * node2) / (node1 + node2);
end
