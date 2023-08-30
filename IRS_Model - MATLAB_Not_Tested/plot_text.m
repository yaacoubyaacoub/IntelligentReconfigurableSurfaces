function plot_text(text, subplot_position)
    % Plot the text
    %
    % Inputs:
    %   - text: the text to be plotted
    %   - subplot_position: Plot location on the figure in case we want all plots on the same figure.
    %                        None, if each plot is an independent figure.

    if ~isempty(subplot_position)
        ax = subplot(subplot_position(1), subplot_position(2), subplot_position(3));
    else
        figure;
        ax = axes;
    end

    % Add the text to the plot
    text(ax, 0, 0, text, 'FontSize', 12, 'FontName', 'Arial', 'FontWeight', 'bold');

    % Remove the axis ticks and labels
    axis(ax, 'off');
end
