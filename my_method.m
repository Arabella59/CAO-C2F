% MY_METHOD.M
% Standalone demonstration of the full proposed thermal-visible face registration
% pipeline for driver state monitoring.
%
% Dissertation: "Developing a Thermal-Visible Dual Camera System with Image
% Registration for Driver State Monitoring"
% Base method: CAO-C2F (Jiang et al., 2021)
% Area-based inspiration: TWMM (Meng et al., 2022)
%
% Novel contribution: CLAHE + Hybrid CAO orientation + Landmark-guided
% per-region NCC refinement + Piecewise-linear blending.

clear; close all; clc;
thisFolder = fileparts(mfilename('fullpath'));
addpath(thisFolder);
addpath(genpath(thisFolder));

%% System information
fprintf('=== MY_METHOD: Proposed Pipeline ===\n');
fprintf('MATLAB: %s\n', version);
fprintf('CPU cores: %d\n', feature('numcores'));
try; [~,s]=memory; fprintf('RAM: %.1fGB\n', s.PhysicalMemory.Total/1e9); catch; end
fprintf('Date: %s\n', datestr(now));

%% Output folder setup
outRoot   = fullfile(thisFolder, 'Performance Metriczz');
outFig    = fullfile(outRoot, 'figures');
outTable  = fullfile(outRoot, 'tables');
outMosaic = fullfile(outRoot, 'mosaics');
outHeat   = fullfile(outRoot, 'heatmaps');
outHist   = fullfile(outRoot, 'histograms');
for d = {outRoot, outFig, outTable, outMosaic, outHeat, outHist}
    if ~isfolder(d{1}), mkdir(d{1}); end
end

Lc = 6;  % CAO curvature region-of-support parameter

%% Runtime table
RT = struct('clahe',0,'coarse',0,'landmark',0,'segment',0,'xcorr',0,'blend',0);

% =========================================================================
%  STEP 1 — Image loading
% =========================================================================
fprintf('\n--- STEP 1: Image loading ---\n');
[f1,p1] = uigetfile({'*.jpg;*.bmp;*.png;*.tif','Images'},'Select THERMAL image');
if isequal(f1,0), error('No thermal image selected.'); end
[f2,p2] = uigetfile({'*.jpg;*.bmp;*.png;*.tif','Images'},'Select VISIBLE image');
if isequal(f2,0), error('No visible image selected.'); end

I_thermal_raw = imread(fullfile(p1,f1));
I_visible_raw = imread(fullfile(p2,f2));

pair_id = f1(1:max(1,end-4));  % thermal filename without extension

% Auto-resize if height > 480
function I = resize_480(I)
    if size(I,1) > 480
        sf = 480/size(I,1);
        I  = imresize(I, sf);
    end
end
I_thermal_raw = resize_480(I_thermal_raw);
I_visible_raw = resize_480(I_visible_raw);

fprintf('Thermal image: %dx%d  (%s)\n', size(I_thermal_raw,2), size(I_thermal_raw,1), f1);
fprintf('Visible image: %dx%d  (%s)\n', size(I_visible_raw,2), size(I_visible_raw,1), f2);

% Convert to grayscale for processing
if size(I_thermal_raw,3)==3, I_thermal_g = rgb2gray(I_thermal_raw);
else,                         I_thermal_g = I_thermal_raw; end
if size(I_visible_raw,3)==3, I_visible_g = rgb2gray(I_visible_raw);
else,                         I_visible_g = I_visible_raw; end

% =========================================================================
%  STEP 2 — CLAHE preprocessing
% =========================================================================
fprintf('\n--- STEP 2: CLAHE preprocessing ---\n');
t_clahe = tic;
I_clahe_u8 = adapthisteq(uint8(I_thermal_g), 'NumTiles',[8 8], 'ClipLimit',0.02);
I_clahe    = double(I_clahe_u8);
I_thermal_d = double(I_thermal_g);
I_visible_d = double(I_visible_g);
RT.clahe = toc(t_clahe);
fprintf('  CLAHE applied in %.3f s\n', RT.clahe);

% Quick keypoint-count preview for printed output
try
    [c_pre,~,~]  = cp_cornerDetection(I_thermal_d,[],[],[],[],0,1,1,Lc,1);
    [c_post,~,~] = cp_cornerDetection(I_clahe,[],[],[],[],0,1,1,Lc,1);
    fprintf('  Keypoints before CLAHE: %d  |  after CLAHE: %d\n', size(c_pre,1), size(c_post,1));
catch
    fprintf('  (keypoint preview skipped)\n');
end

% --- STEP 2a: CLAHE visualisation ---
fprintf('  Saving CLAHE visualisation figures...\n');
try
    fig2a = figure('Visible','off','Name','CLAHE Preprocessing');
    subplot(1,3,1); imshow(uint8(I_thermal_d)); title('Original Thermal');
    subplot(1,3,2); imshow(I_clahe_u8);          title('CLAHE Enhanced Thermal');
    subplot(1,3,3); imshow(I_visible_g);          title('Visible Image');
    sgtitle('CLAHE Preprocessing');
    print(fig2a, fullfile(outHist,[pair_id '_clahe_images.png']),'-dpng','-r150');
    close(fig2a);
catch ME2a
    fprintf('  CLAHE image figure warning: %s\n', ME2a.message);
end

try
    px_t  = I_thermal_d(:);
    px_c  = I_clahe(:);
    px_v  = I_visible_d(:);
    thresh = 200;
    fig2b = figure('Visible','off','Name','Intensity Histograms');
    sgtitle('Effect of CLAHE Preprocessing on Thermal Image Intensity Distribution');

    subplot(1,3,1);
    histogram(px_t, 64, 'FaceColor',[0.2 0.4 0.8]);
    title('Raw Thermal'); xlabel('Intensity'); ylabel('Count');
    text(0.05,0.9,sprintf('Mean=%.1f\nStd=%.1f\nAbove%d=%d', ...
        mean(px_t),std(px_t),thresh,sum(px_t>thresh)),'Units','normalized','FontSize',8);

    subplot(1,3,2);
    histogram(px_c, 64, 'FaceColor',[0.8 0.4 0.2]);
    title('CLAHE Thermal'); xlabel('Intensity');
    text(0.05,0.9,sprintf('Mean=%.1f\nStd=%.1f\nAbove%d=%d', ...
        mean(px_c),std(px_c),thresh,sum(px_c>thresh)),'Units','normalized','FontSize',8);

    subplot(1,3,3);
    histogram(px_v, 64, 'FaceColor',[0.2 0.7 0.3]);
    title('Visible'); xlabel('Intensity');
    text(0.05,0.9,sprintf('Mean=%.1f\nStd=%.1f\nAbove%d=%d', ...
        mean(px_v),std(px_v),thresh,sum(px_v>thresh)),'Units','normalized','FontSize',8);

    print(fig2b, fullfile(outHist,[pair_id '_clahe_histograms.png']),'-dpng','-r150');
    close(fig2b);
catch ME2b
    fprintf('  Histogram figure warning: %s\n', ME2b.message);
end

% =========================================================================
%  STEP 3 — Coarse registration using hybrid CAO-C2F
% =========================================================================
fprintf('\n--- STEP 3: Coarse registration (CLAHE + Hybrid orientation) ---\n');
t_coarse = tic;
maxRMSE  = 4*ceil(size(I_visible_d,1)/300);
try
    [P1, P2, ~, ~, extra] = cp_registration_v2(I_clahe, I_visible_d, 20, maxRMSE, ...
        0, 1, 0, Lc, 0, I_visible_d, I_clahe);
catch ME3
    error('Coarse registration failed: %s', ME3.message);
end
RT.coarse = toc(t_coarse);

n_kp1 = extra.n_kp1;  n_kp2 = extra.n_kp2;
n_put = extra.n_putative;  n_fin = extra.n_final;
ori1  = extra.orientation1;
om1   = extra.orient_method1;
cor1  = extra.cor1;  % [row col]
cor2  = extra.cor2;

fprintf('  Thermal keypoints: %d  |  Visible keypoints: %d\n', n_kp1, n_kp2);
fprintf('  Putative matches: %d  |  Final matches (post-RANSAC): %d\n', n_put, n_fin);
fprintf('  Runtime coarse: %.3f s\n', RT.coarse);

% Compute coarse homography
[H_coarse, tform_coarse] = mm_computeHomographyFromPoints(P1, P2);
refObj   = imref2d(size(I_visible_d));
I_coarse = imwarp(I_clahe, tform_coarse, 'OutputView', refObj);

fprintf('\n  H_coarse (3x3 projective homography):\n');
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_coarse(1,:));
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_coarse(2,:));
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_coarse(3,:));
fprintf('\n  Interpretation:\n');
fprintf('    H(1,3)=%.2f, H(2,3)=%.2f : translation (pixel shift between cameras)\n', H_coarse(1,3), H_coarse(2,3));
fprintf('    H(1:2,1:2) = [%.3f %.3f; %.3f %.3f] : rotation + scale\n', ...
    H_coarse(1,1),H_coarse(1,2),H_coarse(2,1),H_coarse(2,2));
fprintf('    H(3,1)=%.4f, H(3,2)=%.4f : perspective distortion\n', H_coarse(3,1), H_coarse(3,2));

% --- STEP 3a: Coarse registration visualisation ---
fprintf('  Saving coarse registration figures...\n');
try
    if ~isempty(extra.putative_pts1)
        fig3a = figure('Visible','off','Name','Putative Matches');
        cp_showMatch(I_clahe, I_visible_d, extra.putative_pts1, extra.putative_pts2, [], 'Putative Matches');
        print(fig3a, fullfile(outFig,[pair_id '_putative_matches.png']),'-dpng','-r150');
        close(fig3a);
    end
catch; end

try
    fig3b = figure('Visible','off','Name','Final Matches');
    cp_showMatch(I_clahe, I_visible_d, P1, P2, [], 'Final Matches (post-RANSAC)');
    print(fig3b, fullfile(outFig,[pair_id '_final_matches.png']),'-dpng','-r150');
    close(fig3b);
catch; end

% Hybrid orientation quiver plot (colour-coded)
try
    fig3c = figure('Visible','off','Name','Hybrid Orientation Quiver');
    imshow(uint8(I_clahe)); hold on;
    if ~isempty(cor1) && ~isempty(ori1) && ~isempty(om1) && length(ori1)==size(cor1,1)
        xu = 8*cos(ori1);  yv = 8*sin(ori1);
        % cor1 is [row col]; quiver wants (x=col, y=row)
        cao_idx  = om1 == 1;
        grad_idx = om1 == 0;
        if any(cao_idx)
            quiver(cor1(cao_idx,2), cor1(cao_idx,1), xu(cao_idx), yv(cao_idx), ...
                0, 'b', 'LineWidth',0.8);  % CAO = blue
        end
        if any(grad_idx)
            quiver(cor1(grad_idx,2), cor1(grad_idx,1), xu(grad_idx), yv(grad_idx), ...
                0, 'r', 'LineWidth',0.8);  % gradient = red
        end
        legend({'CAO orientation','Gradient fallback'},'Location','best');
    end
    title('Keypoint Orientations: Blue=CAO, Red=Gradient Fallback');
    print(fig3c, fullfile(outFig,[pair_id '_orientation_quiver.png']),'-dpng','-r150');
    close(fig3c);
catch; end

% Spatial distribution scatter plot
try
    fig3d = figure('Visible','off','Name','Keypoint Scatter');
    if ~isempty(cor1) && ~isempty(ori1) && length(ori1)==size(cor1,1)
        scatter(cor1(:,2), cor1(:,1), 20, ori1*180/pi, 'filled');
        colorbar; colormap(hsv);
        xlabel('Keypoint X (col)');  ylabel('Keypoint Y (row)');
        title('Thermal Keypoint Spatial Distribution (colour = orientation deg)');
        axis ij;  grid on;
    end
    print(fig3d, fullfile(outFig,[pair_id '_kp_scatter.png']),'-dpng','-r150');
    close(fig3d);
catch; end

% =========================================================================
%  STEP 4 — Facial landmark detection
% =========================================================================
fprintf('\n--- STEP 4: Facial landmark detection ---\n');
t_lm = tic;

I_vis_u8 = uint8(255*mat2gray(double(I_visible_raw)));
if size(I_vis_u8,3)==1, I_vis_u8 = repmat(I_vis_u8,[1 1 3]); end

[pts_vis, bbox, det_mode] = mm_detectFaceRegions(I_vis_u8);
fprintf('  Detection mode: %s\n', det_mode);
fprintf('  Face bbox: [%d %d %d %d]\n', bbox);
RT.landmark = toc(t_lm);

% Visualise landmarks on visible image
try
    fig4 = figure('Visible','off','Name','Facial Landmarks');
    imshow(I_vis_u8); hold on;
    if ~isempty(pts_vis)
        scatter(pts_vis(:,1), pts_vis(:,2), 30, 'r', 'filled');
        for lm = 1:size(pts_vis,1)
            text(pts_vis(lm,1)+2, pts_vis(lm,2), num2str(lm), ...
                'FontSize',6,'Color','y');
        end
    end
    title(sprintf('Facial Landmarks (%s detection)', det_mode));
    print(fig4, fullfile(outFig,[pair_id '_landmarks.png']),'-dpng','-r150');
    close(fig4);
catch; end

% Estimate landmark positions in thermal via H_coarse
pts_therm_est = [];
if ~isempty(pts_vis) && ~any(isnan(H_coarse(:)))
    pts_h = [pts_vis, ones(size(pts_vis,1),1)];
    H_inv = inv(H_coarse);
    pt_   = (H_inv * pts_h')';
    pts_therm_est = pt_(:,1:2) ./ pt_(:,3);
    fprintf('  Estimated %d landmark positions in thermal via coarse H.\n', size(pts_therm_est,1));
end

% =========================================================================
%  STEP 5 — Face segmentation into 7 regions
% =========================================================================
fprintf('\n--- STEP 5: Face segmentation (7 regions) ---\n');
t_seg = tic;
masks = [];
if ~isempty(pts_vis) && ~isempty(bbox)
    try
        masks = mm_buildFaceMasks(pts_vis, size(I_visible_d), bbox);
    catch ME5
        fprintf('  Mask building failed: %s\n', ME5.message);
    end
end
RT.segment = toc(t_seg);

try
    if ~isempty(masks)
        fnames  = fieldnames(masks);
        colours = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1; 1 0 1; 0.5 0.5 0];
        fig5 = figure('Visible','off','Name','Segmentation Masks');
        if size(I_visible_raw,3)==3, bg = im2double(I_visible_raw);
        else,                        bg = repmat(im2double(I_visible_raw),[1 1 3]); end
        overlay = bg;
        for k = 1:length(fnames)
            mask_k = masks.(fnames{k});
            for ch = 1:3
                layer = overlay(:,:,ch);
                layer(mask_k) = 0.5*layer(mask_k) + 0.5*colours(k,ch);
                overlay(:,:,ch) = layer;
            end
        end
        imshow(overlay);
        legend_entries = cellfun(@(n) strrep(n,'_',' '), fnames, 'UniformOutput',false);
        title('7-Region Face Segmentation');
        % Colour patches for legend
        hold on;
        h_leg = gobjects(length(fnames),1);
        for k = 1:length(fnames)
            h_leg(k) = patch(NaN,NaN,colours(k,:));
        end
        legend(h_leg, legend_entries, 'Location','bestoutside');
        print(fig5, fullfile(outFig,[pair_id '_segmentation.png']),'-dpng','-r150');
        close(fig5);
    end
catch; end

% =========================================================================
%  STEP 6 — Per-region NCC refinement
% =========================================================================
fprintf('\n--- STEP 6: Per-region NCC refinement ---\n');
t_xcorr = tic;
region_offsets = zeros(7,2);
region_conf    = zeros(7,1);
xcorr_maps     = {};

if ~isempty(masks)
    try
        [region_offsets, region_conf, xcorr_maps] = ...
            mm_regionXcorrRefine(I_coarse, I_visible_d, masks);
    catch ME6
        fprintf('  NCC refinement failed: %s\n', ME6.message);
    end
end
RT.xcorr = toc(t_xcorr);

fnames_mask = {};
if ~isempty(masks), fnames_mask = fieldnames(masks); end
for k = 1:length(fnames_mask)
    fprintf('  Region %-15s : offset=[%+.1f,%+.1f]  confidence=%.3f\n', ...
        fnames_mask{k}, region_offsets(k,2), region_offsets(k,1), region_conf(k));
end

% Per-region NCC figures
try
    for k = 1:min(length(fnames_mask), length(xcorr_maps))
        if isempty(xcorr_maps{k}), continue; end
        mask_k = masks.(fnames_mask{k});
        [rows_m,cols_m] = find(mask_k);
        if isempty(rows_m), continue; end
        r1=min(rows_m); r2=max(rows_m); c1=min(cols_m); c2=max(cols_m);

        fig6k = figure('Visible','off');
        subplot(1,3,1); imshow(mat2gray(I_coarse(r1:r2,c1:c2).*double(mask_k(r1:r2,c1:c2))));
        title(['Thermal: ' strrep(fnames_mask{k},'_',' ')]);
        subplot(1,3,2); imshow(mat2gray(I_visible_d(r1:r2,c1:c2).*double(mask_k(r1:r2,c1:c2))));
        title('Visible region');
        subplot(1,3,3);
        C = xcorr_maps{k};
        imshow(mat2gray(C));  hold on;
        [~,pk]=max(C(:)); [rp,cp]=ind2sub(size(C),pk);
        plot(cp,rp,'r+','MarkerSize',12,'LineWidth',2);
        % Draw offset arrow
        quiver(size(C,2)/2, size(C,1)/2, region_offsets(k,2), region_offsets(k,1), ...
            0,'g','LineWidth',2);
        title(sprintf('NCC map  conf=%.3f', region_conf(k)));

        fname6k = fullfile(outFig, sprintf('%s_region_%s_ncc.png', pair_id, fnames_mask{k}));
        print(fig6k, fname6k, '-dpng', '-r150');  close(fig6k);
    end
catch; end

% =========================================================================
%  STEP 7 — Piecewise linear blending
% =========================================================================
fprintf('\n--- STEP 7: Piecewise-linear blending ---\n');
t_blend = tic;
I_fine  = I_coarse;  % fallback
H_final = H_coarse;

if ~isempty(pts_vis) && size(pts_vis,1) >= 4
    try
        src_pts = pts_vis;
        dst_pts = pts_vis;
        for k = 1:length(fnames_mask)
            mask_k = masks.(fnames_mask{k});
            for lm = 1:size(pts_vis,1)
                px = max(1,min(size(mask_k,2),round(pts_vis(lm,1))));
                py = max(1,min(size(mask_k,1),round(pts_vis(lm,2))));
                if mask_k(py,px)
                    dst_pts(lm,:) = pts_vis(lm,:) + region_offsets(k,[2,1]);
                end
            end
        end
        tform_fine = fitgeotrans(src_pts, dst_pts, 'pwl');
        I_fine = imwarp(I_coarse, tform_fine, 'OutputView', refObj);
        fprintf('  PWL blending applied successfully.\n');

        % Approximate overall H_final from landmark pairs
        if ~isempty(pts_therm_est) && size(pts_therm_est,1) >= 4
            [H_final, ~] = mm_computeHomographyFromPoints(pts_therm_est, dst_pts);
        end
    catch ME7
        fprintf('  PWL blending failed (%s), using coarse.\n', ME7.message);
    end
else
    fprintf('  WARNING: < 4 landmarks available, using coarse homography.\n');
end
RT.blend = toc(t_blend);

fprintf('\n  H_final (effective, from landmark pairs):\n');
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_final(1,:));
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_final(2,:));
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_final(3,:));
fprintf('  H_final(1,3)=%.2f, H_final(2,3)=%.2f (translation)\n', H_final(1,3), H_final(2,3));

% =========================================================================
%  STEP 8 — Evaluation
% =========================================================================
fprintf('\n--- STEP 8: Evaluation ---\n');
results = struct();

%% 8a. Landmark RMSE
fprintf('\n  8a. Landmark RMSE (ground truth)\n');
landmark_rmse_vals = [];
gt_scale = 1;
try
    [gt_f1,gt_p1] = uigetfile({'*.csv','CSV files'},'Select THERMAL ground truth CSV (or Cancel to skip)');
    [gt_f2,gt_p2] = uigetfile({'*.csv','CSV files'},'Select VISIBLE ground truth CSV (or Cancel to skip)');
    if ~isequal(gt_f1,0) && ~isequal(gt_f2,0)
        pts_gt_therm = mm_loadGTcsv(fullfile(gt_p1,gt_f1), gt_scale);
        pts_gt_vis   = mm_loadGTcsv(fullfile(gt_p2,gt_f2), gt_scale);
        n_gt = min(size(pts_gt_therm,1), size(pts_gt_vis,1));
        pts_gt_therm = pts_gt_therm(1:n_gt,:);
        pts_gt_vis   = pts_gt_vis(1:n_gt,:);

        % Apply H_final to thermal GT points
        pts_h = [pts_gt_therm, ones(n_gt,1)];
        pts_trans = (H_final * pts_h')';
        pts_trans = pts_trans(:,1:2) ./ pts_trans(:,3);

        errs = sqrt(sum((pts_trans - pts_gt_vis).^2, 2));
        landmark_rmse_vals = errs;
        fprintf('    Mean RMSE: %.2f px  |  Min: %.2f  |  Max: %.2f  |  Std: %.2f\n', ...
            mean(errs), min(errs), max(errs), std(errs));

        fig8a = figure('Visible','off','Name','Landmark RMSE');
        stem(1:n_gt, errs, 'filled');  grid on;
        xlabel('Landmark Index');  ylabel('Reprojection Error (px)');
        title('Per-Landmark Registration Error');
        yline(mean(errs),'r--','Mean','LabelHorizontalAlignment','left');
        print(fig8a, fullfile(outFig,[pair_id '_landmark_rmse.png']),'-dpng','-r150');
        close(fig8a);
        results.landmark_rmse_mean = mean(errs);
        results.landmark_rmse_std  = std(errs);
    else
        fprintf('    Ground truth skipped.\n');
    end
catch ME8a
    fprintf('    Landmark RMSE error: %s\n', ME8a.message);
end

%% 8b. Per-region pixel RMSE
fprintf('\n  8b. Per-region pixel RMSE\n');
results.region_rmse_coarse = struct();
results.region_rmse_fine   = struct();
try
    if ~isempty(masks)
        rr_coarse = mm_computeRegionRMSE(I_coarse, I_visible_d, masks);
        rr_fine   = mm_computeRegionRMSE(I_fine,   I_visible_d, masks);
        fnames_m  = fieldnames(rr_coarse);
        fprintf('    %-15s  Coarse-RMSE  Fine-RMSE  Improvement\n', 'Region');
        for k = 1:length(fnames_m)
            c = rr_coarse.(fnames_m{k});
            f = rr_fine.(fnames_m{k});
            imp = (c-f)/max(c,1e-9)*100;
            fprintf('    %-15s  %10.3f  %9.3f  %+.1f%%\n', fnames_m{k}, c, f, imp);
        end
        results.region_rmse_coarse = rr_coarse;
        results.region_rmse_fine   = rr_fine;
    end
catch ME8b
    fprintf('    Region RMSE error: %s\n', ME8b.message);
end

%% 8c. Regional RMSE heatmap
fprintf('\n  8c. Regional RMSE heatmap\n');
try
    mm_saveRegistrationHeatmap(I_fine, I_visible_d, ...
        fullfile(outHeat, [pair_id '_heatmap_proposed.png']), ...
        'Registration Error Heatmap — My Proposed Method');
catch ME8c
    fprintf('    Heatmap error: %s\n', ME8c.message);
end

%% 8d. Global metrics
fprintf('\n  8d. Global metrics\n');
try
    [rmse_coarse, ncc_coarse, ssim_coarse] = mm_computeImageMetrics(I_coarse, I_visible_d);
    [rmse_fine,   ncc_fine,   ssim_fine  ] = mm_computeImageMetrics(I_fine,   I_visible_d);
    fprintf('    %30s  %8s  %8s  %8s\n','Method','RMSE','NCC','SSIM');
    fprintf('    %30s  %8.3f  %8.4f  %8.4f\n','Coarse only (Step 3)', rmse_coarse, ncc_coarse, ssim_coarse);
    fprintf('    %30s  %8.3f  %8.4f  %8.4f\n','Full pipeline (Step 7)', rmse_fine, ncc_fine, ssim_fine);
    results.rmse_coarse = rmse_coarse;  results.ncc_coarse = ncc_coarse;  results.ssim_coarse = ssim_coarse;
    results.rmse_fine   = rmse_fine;    results.ncc_fine   = ncc_fine;    results.ssim_fine   = ssim_fine;
catch ME8d
    fprintf('    Global metrics error: %s\n', ME8d.message);
end

%% 8e. Precision and Recall
fprintf('\n  8e. Precision / Recall / F1\n');
try
    [prec, rec, f1_sc] = mm_matchPrecisionRecall(P1, P2, extra.putative_pts1, H_coarse, 5);
    fprintf('    Precision=%.4f  Recall=%.4f  F1=%.4f  (threshold=5px)\n', prec, rec, f1_sc);
    results.precision = prec;  results.recall = rec;  results.f1 = f1_sc;
catch ME8e
    fprintf('    Precision/Recall error: %s\n', ME8e.message);
end

%% 8f. Runtime breakdown
fprintf('\n  8f. Runtime breakdown\n');
RT.total = RT.clahe + RT.coarse + RT.landmark + RT.segment + RT.xcorr + RT.blend;
fprintf('    %-30s  %.4f s\n', 'CLAHE preprocessing',       RT.clahe);
fprintf('    %-30s  %.4f s\n', 'Coarse registration',        RT.coarse);
fprintf('    %-30s  %.4f s\n', 'Landmark detection',         RT.landmark);
fprintf('    %-30s  %.4f s\n', 'Segmentation',               RT.segment);
fprintf('    %-30s  %.4f s\n', 'NCC refinement',             RT.xcorr);
fprintf('    %-30s  %.4f s\n', 'PWL blending',               RT.blend);
fprintf('    %-30s  %.4f s\n', 'TOTAL',                      RT.total);
fprintf('    (Real-time at 25 fps requires < 40 ms = %.1f s; achievable: %s)\n', ...
    0.04, mat2str(RT.total < 0.04));
results.runtime = RT;

%% 8g. Homography matrices with explanation
fprintf('\n  8g. Homography matrices\n');
fprintf('    H_coarse = [%.4f %.4f %.4f; %.4f %.4f %.4f; %.4f %.4f %.4f]\n', H_coarse');
fprintf('    H_final  = [%.4f %.4f %.4f; %.4f %.4f %.4f; %.4f %.4f %.4f]\n', H_final');
results.H_coarse = H_coarse;
results.H_final  = H_final;

%% 8h. Final summary figure
fprintf('\n  8h. Final summary figure\n');
try
    fig8h = figure('Visible','off','Name','Registration Summary','Position',[0 0 1200 350]);
    subplot(1,4,1); imshow(uint8(I_thermal_d));  title('1. Original Thermal');
    subplot(1,4,2); imshow(uint8(I_visible_d));  title('2. Visible Image');
    subplot(1,4,3); imshow(mat2gray(I_coarse));  title('3. Coarse Registered (Step 3)');
    subplot(1,4,4); imshow(mat2gray(I_fine));    title('4. Fine Registered (Step 7)');
    sgtitle(sprintf('Proposed Method Registration Summary | RMSE: coarse=%.2f fine=%.2f', ...
        results.rmse_coarse, results.rmse_fine));
    print(fig8h, fullfile(outMosaic,[pair_id '_summary.png']),'-dpng','-r150');
    close(fig8h);
catch ME8h
    fprintf('    Summary figure error: %s\n', ME8h.message);
end

%% Save results
fprintf('\n--- Saving results ---\n');
try
    save(fullfile(outTable,'my_method_results.mat'), 'results', 'H_coarse', 'H_final', 'RT');
    % CSV table
    metric_names = {'rmse_coarse','rmse_fine','ncc_coarse','ncc_fine',...
                    'ssim_coarse','ssim_fine','precision','recall','f1',...
                    'runtime_total'};
    metric_vals  = [results.rmse_coarse, results.rmse_fine, ...
                    results.ncc_coarse,  results.ncc_fine, ...
                    results.ssim_coarse, results.ssim_fine, ...
                    results.precision,   results.recall, results.f1, ...
                    RT.total];
    T_out = array2table(metric_vals, 'VariableNames', metric_names);
    writetable(T_out, fullfile(outTable,'my_method_results.csv'));
    fprintf('  Results saved to Performance Metriczz/tables/\n');
catch ME_save
    fprintf('  Save error: %s\n', ME_save.message);
end

fprintf('\n=== my_method.m COMPLETE ===\n');
fprintf('All outputs saved to: %s\n', outRoot);
