function mm_saveRegistrationHeatmap(I_registered, I_visible, save_path, fig_title)
%MM_SAVEREGISTRATIONHEATMAP  Overlay pixel-error heatmap on visible image and save.
%
%   mm_saveRegistrationHeatmap(I_registered, I_visible, save_path, fig_title)
%
%   Computes absolute pixel difference, Gaussian-smooths it, overlays as
%   jet colormap at 60% opacity on the visible image.

if nargin < 4, fig_title = 'Registration Error Heatmap'; end

try
    A = im2double(I_registered);
    B = im2double(I_visible);
    if ~isequal(size(A), size(B))
        r = min(size(A,1),size(B,1));  c = min(size(A,2),size(B,2));
        A = A(1:r,1:c);  B = B(1:r,1:c);
    end
    err_map = abs(A - B);

    % Gaussian smoothing for visual clarity
    h_gauss  = fspecial('gaussian', 15, 5);
    err_sm   = imfilter(err_map, h_gauss, 'replicate');

    % Normalise to [0 1]
    err_norm = (err_sm - min(err_sm(:))) / max(eps, max(err_sm(:)) - min(err_sm(:)));

    % Build RGB heatmap via jet colormap
    cmap      = jet(256);
    idx_map   = round(err_norm * 255) + 1;
    idx_map   = max(1, min(256, idx_map));
    hmap_rgb  = reshape(cmap(idx_map(:),:), [size(err_norm), 3]);

    % Blend with visible image at 60% heatmap opacity
    if size(B,3) == 1
        B_rgb = repmat(B, [1 1 3]);
    else
        B_rgb = B;
    end
    alpha   = 0.60;
    overlay = (1-alpha)*B_rgb + alpha*hmap_rgb;
    overlay = max(0, min(1, overlay));

    fig = figure('Visible','off');
    imshow(overlay);
    colorbar_ax = colorbar();
    colormap(jet);
    clim([0 1]);
    ylabel(colorbar_ax, 'Pixel Intensity Error');
    title(fig_title, 'FontSize', 11);

    % Ensure output directory exists
    [dir_out, ~, ~] = fileparts(save_path);
    if ~isempty(dir_out) && ~isfolder(dir_out)
        mkdir(dir_out);
    end
    print(fig, save_path, '-dpng', '-r150');
    close(fig);
    fprintf('  Heatmap saved: %s\n', save_path);
catch ME
    fprintf('  mm_saveRegistrationHeatmap failed: %s\n', ME.message);
end
end
