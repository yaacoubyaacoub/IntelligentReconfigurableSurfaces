function plot_power_graph(transmitted_power, received_powers, save_plot, results_directory_path, subplot_position)
    % Plot the transmitted power.
    % Plot the received power vs number of elements.
    %
    % Inputs:
    %   - transmitted_power: Power transmitted by the transmitter of the signal
    %   - received_powers: 1D array containing the power received by the receiver antenna based on the number of
    %                      elements of the reflecting metasurface
    %   - save_plot: flag indicating if the plot is saved as a png or not
    %   - results_directory_path: path for the directory to save the plot as png
    %   - subplot_position: Plot location on the figure in case we want all plots on the same figure.
    %                       None, if each plot is an independent figure.

    received_powers_dB = 10 * log10(received_powers / 1e-3);
    gain_dB = 10 * log10(received_powers / transmitted_power);
    transmitted_power_dB_array = repmat(10 * log10(transmitted_power / 1e-3), size(received_powers_dB));

    if ~isempty(subplot_position)
        subplot(subplot_position{:});
    else
        figure;
    end

    plot(transmitted_power_dB_array, 'DisplayName', 'Transmitted Power');
    hold on;
    plot(received_powers_dB, 'DisplayName', 'Received Power');
    plot(gain_dB, 'DisplayName', 'Gain (Pr/Pt)');
    hold off;
    set(gca, 'XScale', 'log');
    xlabel('Number of Elements');
    ylabel('Power (in dBm)');
    legend;
    title('Received Power vs Number of Elements');

    if save_plot && ~isempty(results_directory_path)
        saveas(gcf, fullfile(results_directory_path, 'Received Power vs Number of Elements.png'));
    end
end
