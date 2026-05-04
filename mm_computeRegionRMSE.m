function region_rmse = mm_computeRegionRMSE(I_registered, I_visible, masks)
%MM_COMPUTEREGIONRMSE  Per-region pixel RMSE between registered and visible image.
%
%   region_rmse = mm_computeRegionRMSE(I_registered, I_visible, masks)
%
%   masks: struct of binary masks (from mm_buildFaceMasks) or [].
%   If masks is empty, returns scalar global RMSE.
%   Returns struct with same field names as masks, or scalar if no masks.

A = double(I_registered);
B = double(I_visible);
if ~isequal(size(A), size(B))
    r = min(size(A,1),size(B,1));  c = min(size(A,2),size(B,2));
    A = A(1:r,1:c);  B = B(1:r,1:c);
end

if isempty(masks)
    d = A-B;
    region_rmse = sqrt(mean(d(:).^2));
    return;
end

fnames = fieldnames(masks);
region_rmse = struct();
for k = 1:length(fnames)
    try
        mask = masks.(fnames{k});
        if size(mask,1) ~= size(A,1) || size(mask,2) ~= size(A,2)
            mask = imresize(mask, [size(A,1), size(A,2)], 'nearest');
        end
        pix_t = A(mask);
        pix_v = B(mask);
        if isempty(pix_t)
            region_rmse.(fnames{k}) = NaN;
        else
            region_rmse.(fnames{k}) = sqrt(mean((pix_t - pix_v).^2));
        end
    catch
        region_rmse.(fnames{k}) = NaN;
    end
end
end
