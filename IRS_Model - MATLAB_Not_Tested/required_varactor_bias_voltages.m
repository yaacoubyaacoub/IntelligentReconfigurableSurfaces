function v = required_varactor_bias_voltages(c)
    % Find the bias voltage for the varactor in order to achieve the required capacitance value.
    % varactor capacitance-voltage relationship model:
    %    c = c0 / ((1 + v / v0) ** m)
    %        c0: capacitance at v=0
    %        v0: characteristic voltage of the varactor
    %        m: non-linearity factor
    %        v: bias voltage applied to the varactor
    %        c: varactor capacitance that corresponds to a given bias voltage v.
    % c: required capacitance value
    % v: corresponding varactor voltage
    
    % Set the parameters
    c0 = 10e-12;
    v0 = 2.9;
    m = 1.66;

    v = v0 * ((c0 / c) ^ (1 / m) - 1);
    v = round(v, 2);
end
