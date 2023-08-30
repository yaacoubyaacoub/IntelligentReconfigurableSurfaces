function show_phase_shift_plots(phase_shifts, title, save_plot, results_directory_path, subplot_position)
    % Plot the phase shift on a heatmap
    %
    % Inputs:
    %   - phase_shifts: 2D phase shift matrix resembling the metasurface where every entry of this matrix represents
    %                    the phase shift realized by the corresponding element of the surface.
    %   - title: title of the plot
    %   - save_plot: flag indicating if the plot is saved as a png or not
    %   - results_directory_path: path for the directory to save the plot as png
    %   - subplot_position: Plot location on the figure in case we want all plots on the same figure.
    %                       None, if each plot is an independent figure.
    
    if nargin < 5
        subplot_position = [];
    end

    if ~isempty(subplot_position)
        subplot(subplot_position(1), subplot_position(2), subplot_position(3));
    else
        figure;
    end

    imagesc(phase_shifts);
    colormap('viridis');
    colorbar('label', 'Phase Shift (deg)');
    title(title);
    xlabel('Element Index (x)');
    ylabel('Element Index (y)');

    if save_plot && ~isempty(results_directory_path)
        saveas(gcf, fullfile(results_directory_path, [title '.png']));
    end
end
