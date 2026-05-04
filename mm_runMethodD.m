function result = mm_runMethodD(I_thermal, I_visible, Lc)
%MM_RUNMETHODD  Full proposed method: CLAHE + Hybrid + Landmark NCC + PWL blend.
%
%   result = mm_runMethodD(I_thermal, I_visible, Lc)
%
%   Pipeline:
%     1. CLAHE on thermal
%     2. Coarse registration via cp_registration_v2 with hybrid orientation
%     3. Facial landmark detection on visible
%     4. 7-region binary mask construction
%     5. Per-region normxcorr2 refinement
%     6. Piecewise-linear blending via fitgeotrans 'pwl'
%
%   Returns standardised result struct with all intermediate data.

if nargin < 3, Lc = 6; end

result   = mm_empty_result();
maxRMSE  = 4*ceil(size(I_visible,1)/300);
rt       = struct('clahe',0,'coarse',0,'landmark',0,'segment',0,'xcorr',0,'blend',0);

try
    %% --- CLAHE ---
    t_clahe = tic;
    I_t_raw = im2gray_safe(I_thermal);
    I_v     = double(im2gray_safe(I_visible));
    I_clahe = adapthisteq(uint8(I_t_raw), 'NumTiles',[8 8], 'ClipLimit',0.02);
    I_t_in  = double(I_clahe);
    rt.clahe = toc(t_clahe);

    %% --- Coarse registration ---
    t_coarse = tic;
    [P1, P2, ~, ~, extra] = cp_registration_v2(I_t_in, I_v, 20, maxRMSE, 0, 1, 0, Lc, 0, I_v, I_clahe);
    result.pts1 = P1;  result.pts2 = P2;
    result.n_kp1      = extra.n_kp1;
    result.n_kp2      = extra.n_kp2;
    result.n_putative = extra.n_putative;
    result.n_final    = extra.n_final;
    result.kp1        = extra.cor1;  result.kp2 = extra.cor2;
    result.orientation1   = extra.orientation1;
    result.orient_method1 = extra.orient_method1;
    result.putative_pts1  = extra.putative_pts1;
    result.putative_pts2  = extra.putative_pts2;

    if size(P1,1) < 3
        error('Insufficient matches for coarse homography (%d).', size(P1,1));
    end
    [result.H, tform_coarse] = mm_computeHomographyFromPoints(P1, P2);
    refObj   = imref2d(size(I_v));
    I_coarse = imwarp(I_t_in, tform_coarse, 'OutputView', refObj);
    result.I_coarse = I_coarse;
    rt.coarse = toc(t_coarse);

    %% --- Landmark detection ---
    t_lm = tic;
    I_vis_u8 = uint8(255 * mat2gray(I_visible));
    if size(I_vis_u8,3) == 1, I_vis_u8 = repmat(I_vis_u8,[1 1 3]); end
    [pts_vis, bbox, detection_mode] = mm_detectFaceRegions(I_vis_u8);
    result.landmarks_vis  = pts_vis;
    result.bbox           = bbox;
    result.detection_mode = detection_mode;

    % Estimate landmark positions in thermal using coarse H
    if ~isempty(pts_vis) && ~any(isnan(result.H(:)))
        pts_h = [pts_vis, ones(size(pts_vis,1),1)];
        H_inv = inv(result.H);
        pts_t = (H_inv * pts_h')';
        result.landmarks_therm = pts_t(:,1:2) ./ pts_t(:,3);
    end
    rt.landmark = toc(t_lm);

    %% --- 7-region segmentation ---
    t_seg = tic;
    masks = [];
    if ~isempty(pts_vis) && ~isempty(bbox)
        try
            masks = mm_buildFaceMasks(pts_vis, size(I_v), bbox);
            result.masks = masks;
        catch ME_mask
            fprintf('  Mask building failed: %s\n', ME_mask.message);
        end
    end
    rt.segment = toc(t_seg);

    %% --- Per-region NCC refinement ---
    t_xcorr = tic;
    region_offsets = zeros(7,2);
    region_conf    = zeros(7,1);
    if ~isempty(masks)
        try
            [region_offsets, region_conf, ~] = ...
                mm_regionXcorrRefine(I_coarse, I_v, masks);
            result.region_offsets = region_offsets;
            result.region_conf    = region_conf;
        catch ME_xcorr
            fprintf('  Region xcorr failed: %s\n', ME_xcorr.message);
        end
    end
    rt.xcorr = toc(t_xcorr);

    %% --- Piecewise-linear blending ---
    t_blend = tic;
    I_fine = I_coarse;  % fallback
    if ~isempty(pts_vis) && size(pts_vis,1) >= 4
        try
            fnames = fieldnames(masks);
            src_pts = pts_vis;   % landmark positions in coarse-registered thermal
            dst_pts = pts_vis;   % will be shifted by regional offsets

            % Assign each landmark to a region and apply its offset
            for k = 1:length(fnames)
                mask_k = masks.(fnames{k});
                for lm = 1:size(pts_vis,1)
                    px = round(pts_vis(lm,1));  py = round(pts_vis(lm,2));
                    px = max(1,min(size(mask_k,2),px));
                    py = max(1,min(size(mask_k,1),py));
                    if mask_k(py,px)
                        dst_pts(lm,:) = pts_vis(lm,:) + region_offsets(k,[2,1]);
                    end
                end
            end

            tform_fine = fitgeotrans(src_pts, dst_pts, 'pwl');
            I_fine = imwarp(I_coarse, tform_fine, 'OutputView', refObj);
        catch ME_blend
            fprintf('  PWL blending failed (%s), using coarse result.\n', ME_blend.message);
        end
    else
        fprintf('  PWL: fewer than 4 landmarks, using coarse result.\n');
    end
    result.I_registered = I_fine;
    rt.blend = toc(t_blend);

    %% --- Final homography (approximated from landmark pairs) ---
    if ~isempty(pts_vis) && size(pts_vis,1) >= 4 && ~isempty(result.landmarks_therm)
        try
            [H_fine, ~] = mm_computeHomographyFromPoints(result.landmarks_therm, pts_vis);
            result.H_fine = H_fine;
        catch
            result.H_fine = result.H;
        end
    else
        result.H_fine = result.H;
    end

    %% --- Metrics ---
    [result.rmse, result.ncc, result.ssim] = ...
        mm_computeImageMetrics(result.I_registered, I_v);
    [result.rmse_coarse, result.ncc_coarse, result.ssim_coarse] = ...
        mm_computeImageMetrics(I_coarse, I_v);

    % Region RMSE
    if ~isempty(masks)
        result.region_rmse_fine   = mm_computeRegionRMSE(result.I_registered, I_v, masks);
        result.region_rmse_coarse = mm_computeRegionRMSE(I_coarse, I_v, masks);
    end

    result.runtime = rt.clahe + rt.coarse + rt.landmark + rt.segment + rt.xcorr + rt.blend;
    result.runtime_stages = rt;
    result.success = true;

catch ME
    fprintf('  Method D failed: %s\n', ME.message);
    result.error   = ME.message;
    result.runtime = 0;
end
end

function I = im2gray_safe(I)
if size(I,3) == 3, I = rgb2gray(I); end
I = double(I);
end
