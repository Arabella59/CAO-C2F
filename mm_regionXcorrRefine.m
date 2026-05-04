function [offsets, confidences, xcorr_maps] = mm_regionXcorrRefine(I_therm_coarse, I_vis, masks)
%MM_REGIONXCORRREFINE  Per-region normalised cross-correlation refinement.
%
%   [offsets, confidences, xcorr_maps] = mm_regionXcorrRefine(I_therm_coarse, I_vis, masks)
%
%   I_therm_coarse : coarsely registered thermal image (double, same size as I_vis)
%   I_vis          : visible image (double)
%   masks          : struct of binary masks from mm_buildFaceMasks
%
%   offsets     : [7 x 2] [row_offset, col_offset] per region
%   confidences : [7 x 1] peak NCC value per region
%   xcorr_maps  : cell array of NCC maps for visualisation

region_names = fieldnames(masks);
n_regions    = length(region_names);
offsets      = zeros(n_regions, 2);
confidences  = zeros(n_regions, 1);
xcorr_maps   = cell(n_regions, 1);

I_t = double(I_therm_coarse);
I_v = double(I_vis);

% Normalise each image to [0,1] for NCC stability
I_t = (I_t - min(I_t(:))) / max(1, max(I_t(:)) - min(I_t(:)));
I_v = (I_v - min(I_v(:))) / max(1, max(I_v(:)) - min(I_v(:)));

for k = 1:n_regions
    mask = masks.(region_names{k});
    try
        % Extract bounding box of the mask for efficiency
        [rows_m, cols_m] = find(mask);
        if isempty(rows_m)
            offsets(k,:) = [0, 0];
            confidences(k) = 0;
            xcorr_maps{k} = [];
            continue;
        end
        r1 = min(rows_m); r2 = max(rows_m);
        c1 = min(cols_m); c2 = max(cols_m);

        patch_t = I_t(r1:r2, c1:c2) .* double(mask(r1:r2, c1:c2));
        patch_v = I_v(r1:r2, c1:c2) .* double(mask(r1:r2, c1:c2));

        % Skip degenerate patches
        if std(patch_t(:)) < 1e-6 || std(patch_v(:)) < 1e-6 || ...
                sum(mask(:)) < 9
            offsets(k,:) = [0, 0];
            confidences(k) = 0;
            xcorr_maps{k} = [];
            fprintf('  Region %s: degenerate patch, keeping coarse.\n', region_names{k});
            continue;
        end

        % Template = thermal patch, search in visible (same-size region)
        C = normxcorr2(patch_t, patch_v);
        xcorr_maps{k} = C;

        [peak_val, peak_idx] = max(C(:));
        [r_peak, c_peak]     = ind2sub(size(C), peak_idx);

        % Offset = shift of thermal relative to visible at peak
        row_off = r_peak - size(patch_t,1);
        col_off = c_peak - size(patch_t,2);

        confidences(k)  = peak_val;
        if peak_val < 0.1
            offsets(k,:) = [0, 0];
            fprintf('  Region %s: low confidence (%.3f), keeping coarse.\n', ...
                region_names{k}, peak_val);
        else
            offsets(k,:) = [row_off, col_off];
        end
    catch ME
        offsets(k,:) = [0, 0];
        confidences(k) = 0;
        xcorr_maps{k} = [];
        fprintf('  Region %s xcorr failed: %s\n', region_names{k}, ME.message);
    end
end
end
