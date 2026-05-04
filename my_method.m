% MY_METHOD.M
% Standalone demonstration of the full proposed thermal-visible face registration
% pipeline for driver state monitoring.
%
% Dissertation: "Developing a Thermal-Visible Dual Camera System with Image
% Registration for Driver State Monitoring"
% Base method: CAO-C2F (Jiang et al., 2021)
% Area-based inspiration: TWMM (Meng et al., 2022)

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

Lc = 6;
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

pair_id = f1(1:max(1,numel(f1)-4));  % thermal filename without extension

% Auto-resize if height > 480 (inline to avoid mid-script function definition)
if size(I_thermal_raw,1) > 480
    I_thermal_raw = imresize(I_thermal_raw, 480/size(I_thermal_raw,1));
end
if size(I_visible_raw,1) > 480
    I_visible_raw = imresize(I_visible_raw, 480/size(I_visible_raw,1));
end

fprintf('Thermal image: %dx%d  (%s)\n', size(I_thermal_raw,2), size(I_thermal_raw,1), f1);
fprintf('Visible image: %dx%d  (%s)\n', size(I_visible_raw,2), size(I_visible_raw,1), f2);

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
I_clahe     = double(I_clahe_u8);
I_thermal_d = double(I_thermal_g);
I_visible_d = double(I_visible_g);
RT.clahe = toc(t_clahe);
fprintf('  CLAHE applied in %.3f s\n', RT.clahe);

try
    [c_pre,~,~]  = cp_cornerDetection(I_thermal_d,[],[],[],[],0,1,1,Lc,1);
    [c_post,~,~] = cp_cornerDetection(I_clahe,    [],[],[],[],0,1,1,Lc,1);
    fprintf('  Keypoints before CLAHE: %d  |  after CLAHE: %d\n', size(c_pre,1), size(c_post,1));
catch
    fprintf('  (keypoint preview skipped)\n');
end

try
    fig2a = figure('Visible','off');
    subplot(1,3,1); imshow(uint8(I_thermal_d)); title('Original Thermal');
    subplot(1,3,2); imshow(I_clahe_u8);          title('CLAHE Enhanced Thermal');
    subplot(1,3,3); imshow(I_visible_g);          title('Visible Image');
    sgtitle('CLAHE Preprocessing');
    print(fig2a, fullfile(outHist,[pair_id '_clahe_images.png']),'-dpng','-r150');
    close(fig2a);
catch ME2a; fprintf('  CLAHE image fig: %s\n', ME2a.message); end

try
    thresh = 200;
    fig2b = figure('Visible','off');
    sgtitle('Effect of CLAHE Preprocessing on Thermal Image Intensity Distribution');
    subplot(1,3,1); histogram(I_thermal_d(:),64,'FaceColor',[0.2 0.4 0.8]);
    title('Raw Thermal'); xlabel('Intensity'); ylabel('Count');
    text(0.05,0.9,sprintf('Mean=%.1f\nStd=%.1f\nAbove%d=%d', ...
        mean(I_thermal_d(:)),std(I_thermal_d(:)),thresh,sum(I_thermal_d(:)>thresh)), ...
        'Units','normalized','FontSize',8);
    subplot(1,3,2); histogram(I_clahe(:),64,'FaceColor',[0.8 0.4 0.2]);
    title('CLAHE Thermal'); xlabel('Intensity');
    text(0.05,0.9,sprintf('Mean=%.1f\nStd=%.1f\nAbove%d=%d', ...
        mean(I_clahe(:)),std(I_clahe(:)),thresh,sum(I_clahe(:)>thresh)), ...
        'Units','normalized','FontSize',8);
    subplot(1,3,3); histogram(I_visible_d(:),64,'FaceColor',[0.2 0.7 0.3]);
    title('Visible'); xlabel('Intensity');
    text(0.05,0.9,sprintf('Mean=%.1f\nStd=%.1f\nAbove%d=%d', ...
        mean(I_visible_d(:)),std(I_visible_d(:)),thresh,sum(I_visible_d(:)>thresh)), ...
        'Units','normalized','FontSize',8);
    print(fig2b, fullfile(outHist,[pair_id '_clahe_histograms.png']),'-dpng','-r150');
    close(fig2b);
catch ME2b; fprintf('  Histogram fig: %s\n', ME2b.message); end

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
cor1  = extra.cor1;
cor2  = extra.cor2;

fprintf('  Thermal keypoints: %d  |  Visible keypoints: %d\n', n_kp1, n_kp2);
fprintf('  Putative matches: %d  |  Final matches: %d\n', n_put, n_fin);
fprintf('  Runtime coarse: %.3f s\n', RT.coarse);

[H_coarse, tform_coarse] = mm_computeHomographyFromPoints(P1, P2);
refObj   = imref2d(size(I_visible_d));
I_coarse = imwarp(I_clahe, tform_coarse, 'OutputView', refObj);

fprintf('\n  H_coarse (3x3 projective homography):\n');
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_coarse(1,:));
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_coarse(2,:));
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_coarse(3,:));
fprintf('  Interpretation:\n');
fprintf('    H(1,3)=%.2f, H(2,3)=%.2f  → pixel translation\n', H_coarse(1,3), H_coarse(2,3));
fprintf('    H(1:2,1:2) = [%.3f %.3f; %.3f %.3f]  → rotation + scale\n', ...
    H_coarse(1,1),H_coarse(1,2),H_coarse(2,1),H_coarse(2,2));
fprintf('    H(3,1)=%.4f, H(3,2)=%.4f  → perspective distortion\n', H_coarse(3,1), H_coarse(3,2));

% --- STEP 3a visualisations ---
try
    if ~isempty(extra.putative_pts1)
        fig3a = figure('Visible','off');
        cp_showMatch(I_clahe, I_visible_d, extra.putative_pts1, extra.putative_pts2, [], 'Putative Matches');
        print(fig3a, fullfile(outFig,[pair_id '_putative_matches.png']),'-dpng','-r150'); close(fig3a);
    end
catch; end

try
    fig3b = figure('Visible','off');
    cp_showMatch(I_clahe, I_visible_d, P1, P2, [], 'Final Matches (post-RANSAC)');
    print(fig3b, fullfile(outFig,[pair_id '_final_matches.png']),'-dpng','-r150'); close(fig3b);
catch; end

try
    fig3c = figure('Visible','off');
    imshow(uint8(I_clahe)); hold on;
    if ~isempty(cor1) && ~isempty(ori1) && length(ori1)==size(cor1,1)
        xu = 8*cos(ori1);  yv = 8*sin(ori1);
        cao_idx  = om1 == 1;
        grad_idx = om1 == 0;
        if any(cao_idx)
            quiver(cor1(cao_idx,2),cor1(cao_idx,1),xu(cao_idx),yv(cao_idx),0,'b','LineWidth',0.8);
        end
        if any(grad_idx)
            quiver(cor1(grad_idx,2),cor1(grad_idx,1),xu(grad_idx),yv(grad_idx),0,'r','LineWidth',0.8);
        end
        legend({'CAO orientation','Gradient fallback'},'Location','best');
    end
    title('Keypoint Orientations: Blue=CAO, Red=Gradient Fallback');
    print(fig3c, fullfile(outFig,[pair_id '_orientation_quiver.png']),'-dpng','-r150'); close(fig3c);
catch; end

try
    fig3d = figure('Visible','off');
    if ~isempty(cor1) && ~isempty(ori1) && length(ori1)==size(cor1,1)
        scatter(cor1(:,2),cor1(:,1),20,ori1*180/pi,'filled');
        colorbar; colormap(hsv);
        xlabel('X (col)'); ylabel('Y (row)');
        title('Thermal Keypoint Spatial Distribution (colour=orientation deg)');
        axis ij; grid on;
    end
    print(fig3d, fullfile(outFig,[pair_id '_kp_scatter.png']),'-dpng','-r150'); close(fig3d);
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

try
    fig4 = figure('Visible','off');
    imshow(I_vis_u8); hold on;
    if ~isempty(pts_vis)
        scatter(pts_vis(:,1),pts_vis(:,2),30,'r','filled');
        for lm = 1:size(pts_vis,1)
            text(pts_vis(lm,1)+2,pts_vis(lm,2),num2str(lm),'FontSize',6,'Color','y');
        end
    end
    title(sprintf('Facial Landmarks (%s detection)', det_mode));
    print(fig4, fullfile(outFig,[pair_id '_landmarks.png']),'-dpng','-r150'); close(fig4);
catch; end

pts_therm_est = [];
if ~isempty(pts_vis) && ~any(isnan(H_coarse(:)))
    pts_h = [pts_vis, ones(size(pts_vis,1),1)];
    pt_   = (inv(H_coarse) * pts_h')';
    pts_therm_est = pt_(:,1:2) ./ pt_(:,3);
    fprintf('  Estimated %d landmark positions in thermal.\n', size(pts_therm_est,1));
end

% =========================================================================
%  STEP 5 — Face segmentation (7 regions)
% =========================================================================
fprintf('\n--- STEP 5: Face segmentation ---\n');
t_seg = tic;
masks = [];
if ~isempty(pts_vis) && ~isempty(bbox)
    try
        masks = mm_buildFaceMasks(pts_vis, size(I_visible_d), bbox);
    catch ME5; fprintf('  Mask building failed: %s\n', ME5.message); end
end
RT.segment = toc(t_seg);

try
    if ~isempty(masks)
        fnames  = fieldnames(masks);
        colours = [1 0 0;0 1 0;0 0 1;1 1 0;0 1 1;1 0 1;0.5 0.5 0];
        fig5 = figure('Visible','off');
        if size(I_visible_raw,3)==3, bg=im2double(I_visible_raw);
        else, bg=repmat(im2double(I_visible_raw),[1 1 3]); end
        overlay = bg;
        for k = 1:length(fnames)
            mask_k = masks.(fnames{k});
            for ch = 1:3
                layer = overlay(:,:,ch);
                layer(mask_k) = 0.5*layer(mask_k)+0.5*colours(k,ch);
                overlay(:,:,ch) = layer;
            end
        end
        imshow(overlay); hold on;
        h_leg = gobjects(length(fnames),1);
        for k=1:length(fnames), h_leg(k)=patch(NaN,NaN,colours(k,:)); end
        legend(h_leg, cellfun(@(n)strrep(n,'_',' '),fnames,'UniformOutput',false),'Location','bestoutside');
        title('7-Region Face Segmentation');
        print(fig5, fullfile(outFig,[pair_id '_segmentation.png']),'-dpng','-r150'); close(fig5);
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
        [region_offsets, region_conf, xcorr_maps] = mm_regionXcorrRefine(I_coarse, I_visible_d, masks);
    catch ME6; fprintf('  NCC failed: %s\n', ME6.message); end
end
RT.xcorr = toc(t_xcorr);

fnames_mask = {};
if ~isempty(masks), fnames_mask = fieldnames(masks); end
for k = 1:length(fnames_mask)
    fprintf('  %-15s : offset=[%+.1f,%+.1f]  conf=%.3f\n', ...
        fnames_mask{k}, region_offsets(k,2), region_offsets(k,1), region_conf(k));
end

try
    for k = 1:min(length(fnames_mask),length(xcorr_maps))
        if isempty(xcorr_maps{k}), continue; end
        mask_k = masks.(fnames_mask{k});
        [rm,cm] = find(mask_k);
        if isempty(rm), continue; end
        r1=min(rm);r2=max(rm);c1=min(cm);c2=max(cm);
        fig6k = figure('Visible','off');
        subplot(1,3,1); imshow(mat2gray(I_coarse(r1:r2,c1:c2).*double(mask_k(r1:r2,c1:c2))));
        title(['Thermal: ' strrep(fnames_mask{k},'_',' ')]);
        subplot(1,3,2); imshow(mat2gray(I_visible_d(r1:r2,c1:c2).*double(mask_k(r1:r2,c1:c2))));
        title('Visible');
        subplot(1,3,3); C=xcorr_maps{k}; imshow(mat2gray(C)); hold on;
        [~,pk]=max(C(:));[rp,cp]=ind2sub(size(C),pk);
        plot(cp,rp,'r+','MarkerSize',12,'LineWidth',2);
        quiver(size(C,2)/2,size(C,1)/2,region_offsets(k,2),region_offsets(k,1),0,'g','LineWidth',2);
        title(sprintf('NCC  conf=%.3f',region_conf(k)));
        print(fig6k, fullfile(outFig,sprintf('%s_region_%s_ncc.png',pair_id,fnames_mask{k})),'-dpng','-r150');
        close(fig6k);
    end
catch; end

% =========================================================================
%  STEP 7 — Piecewise linear blending
% =========================================================================
fprintf('\n--- STEP 7: Piecewise-linear blending ---\n');
t_blend = tic;
I_fine  = I_coarse;
H_final = H_coarse;

if ~isempty(pts_vis) && size(pts_vis,1) >= 4 && ~isempty(masks)
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
        fprintf('  PWL blending applied.\n');
        if ~isempty(pts_therm_est) && size(pts_therm_est,1) >= 4
            [H_final, ~] = mm_computeHomographyFromPoints(pts_therm_est, dst_pts);
        end
    catch ME7; fprintf('  PWL failed (%s), using coarse.\n', ME7.message); end
else
    fprintf('  WARNING: insufficient landmarks for PWL, using coarse H.\n');
end
RT.blend = toc(t_blend);

fprintf('\n  H_final:\n');
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_final(1,:));
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_final(2,:));
fprintf('  [ %8.4f  %8.4f  %8.4f ]\n', H_final(3,:));
fprintf('  H_final translation: [%.2f, %.2f] px\n', H_final(1,3), H_final(2,3));

% =========================================================================
%  STEP 8 — Evaluation
% =========================================================================
fprintf('\n--- STEP 8: Evaluation ---\n');
results = struct();

%% 8a. Landmark RMSE
fprintf('\n  8a. Landmark RMSE\n');
try
    [gt_f1,gt_p1] = uigetfile({'*.csv','CSV'},'Select THERMAL ground truth CSV (Cancel to skip)');
    [gt_f2,gt_p2] = uigetfile({'*.csv','CSV'},'Select VISIBLE  ground truth CSV (Cancel to skip)');
    if ~isequal(gt_f1,0) && ~isequal(gt_f2,0)
        pts_gt_t = mm_loadGTcsv(fullfile(gt_p1,gt_f1));
        pts_gt_v = mm_loadGTcsv(fullfile(gt_p2,gt_f2));
        n_gt = min(size(pts_gt_t,1),size(pts_gt_v,1));
        pts_gt_t = pts_gt_t(1:n_gt,:);  pts_gt_v = pts_gt_v(1:n_gt,:);
        pts_trans = (H_final * [pts_gt_t, ones(n_gt,1)]')';
        pts_trans = pts_trans(:,1:2) ./ pts_trans(:,3);
        errs = sqrt(sum((pts_trans-pts_gt_v).^2,2));
        fprintf('    Mean=%.2f px  Min=%.2f  Max=%.2f  Std=%.2f\n', ...
            mean(errs),min(errs),max(errs),std(errs));
        fig8a = figure('Visible','off');
        stem(1:n_gt,errs,'filled'); grid on;
        xlabel('Landmark'); ylabel('Error (px)'); title('Per-Landmark Registration Error');
        yline(mean(errs),'r--','Mean');
        print(fig8a,fullfile(outFig,[pair_id '_landmark_rmse.png']),'-dpng','-r150'); close(fig8a);
        results.landmark_rmse_mean = mean(errs); results.landmark_rmse_std = std(errs);
    else; fprintf('    GT skipped.\n'); end
catch ME8a; fprintf('    8a error: %s\n',ME8a.message); end

%% 8b. Per-region pixel RMSE
fprintf('\n  8b. Per-region RMSE\n');
try
    if ~isempty(masks)
        rr_c = mm_computeRegionRMSE(I_coarse, I_visible_d, masks);
        rr_f = mm_computeRegionRMSE(I_fine,   I_visible_d, masks);
        fns  = fieldnames(rr_c);
        fprintf('    %-15s  Coarse  Fine  Improvement\n','Region');
        for k=1:length(fns)
            c=rr_c.(fns{k}); f=rr_f.(fns{k});
            fprintf('    %-15s  %6.3f  %6.3f  %+.1f%%\n',fns{k},c,f,(c-f)/max(c,1e-9)*100);
        end
        results.region_rmse_coarse=rr_c; results.region_rmse_fine=rr_f;
    end
catch ME8b; fprintf('    8b error: %s\n',ME8b.message); end

%% 8c. RMSE heatmap
fprintf('\n  8c. RMSE heatmap\n');
try
    mm_saveRegistrationHeatmap(I_fine, I_visible_d, ...
        fullfile(outHeat,[pair_id '_heatmap_proposed.png']), ...
        'Registration Error Heatmap — My Proposed Method');
catch ME8c; fprintf('    8c error: %s\n',ME8c.message); end

%% 8d. Global metrics
fprintf('\n  8d. Global metrics\n');
try
    [rc,nc,sc] = mm_computeImageMetrics(I_coarse,I_visible_d);
    [rf,nf,sf] = mm_computeImageMetrics(I_fine,  I_visible_d);
    fprintf('    %-25s  RMSE=%7.3f  NCC=%7.4f  SSIM=%7.4f\n','Coarse only',rc,nc,sc);
    fprintf('    %-25s  RMSE=%7.3f  NCC=%7.4f  SSIM=%7.4f\n','Full pipeline',rf,nf,sf);
    results.rmse_coarse=rc; results.ncc_coarse=nc; results.ssim_coarse=sc;
    results.rmse_fine=rf;   results.ncc_fine=nf;   results.ssim_fine=sf;
catch ME8d; fprintf('    8d error: %s\n',ME8d.message); end

%% 8e. Precision / Recall / F1
fprintf('\n  8e. Precision/Recall/F1\n');
try
    [prec,rec,f1_sc] = mm_matchPrecisionRecall(P1,P2,extra.putative_pts1,H_coarse,5);
    fprintf('    Precision=%.4f  Recall=%.4f  F1=%.4f  (5px threshold)\n',prec,rec,f1_sc);
    results.precision=prec; results.recall=rec; results.f1=f1_sc;
catch ME8e; fprintf('    8e error: %s\n',ME8e.message); end

%% 8f. Runtime
fprintf('\n  8f. Runtime breakdown\n');
RT.total = RT.clahe+RT.coarse+RT.landmark+RT.segment+RT.xcorr+RT.blend;
fprintf('    %-28s  %.4fs\n','CLAHE',         RT.clahe);
fprintf('    %-28s  %.4fs\n','Coarse reg.',   RT.coarse);
fprintf('    %-28s  %.4fs\n','Landmark det.', RT.landmark);
fprintf('    %-28s  %.4fs\n','Segmentation',  RT.segment);
fprintf('    %-28s  %.4fs\n','NCC refinement',RT.xcorr);
fprintf('    %-28s  %.4fs\n','PWL blending',  RT.blend);
fprintf('    %-28s  %.4fs\n','TOTAL',         RT.total);
fprintf('    Real-time (25fps=40ms): %s\n', mat2str(RT.total<0.04));
results.runtime = RT;

%% 8g. Homography matrices
fprintf('\n  8g. Homographies\n');
fprintf('    H_coarse=[%.4f %.4f %.4f;%.4f %.4f %.4f;%.4f %.4f %.4f]\n',H_coarse');
fprintf('    H_final =[%.4f %.4f %.4f;%.4f %.4f %.4f;%.4f %.4f %.4f]\n',H_final');
results.H_coarse=H_coarse; results.H_final=H_final;

%% 8h. Summary figure
fprintf('\n  8h. Summary figure\n');
try
    fig8h = figure('Visible','off','Position',[0 0 1200 350]);
    subplot(1,4,1); imshow(uint8(I_thermal_d));  title('1. Original Thermal');
    subplot(1,4,2); imshow(uint8(I_visible_d));  title('2. Visible Image');
    subplot(1,4,3); imshow(mat2gray(I_coarse));  title('3. Coarse (Step 3)');
    subplot(1,4,4); imshow(mat2gray(I_fine));    title('4. Fine (Step 7)');
    sgtitle(sprintf('Registration Summary | RMSE coarse=%.2f fine=%.2f', ...
        results.rmse_coarse, results.rmse_fine));
    print(fig8h,fullfile(outMosaic,[pair_id '_summary.png']),'-dpng','-r150'); close(fig8h);
catch ME8h; fprintf('    Summary fig: %s\n',ME8h.message); end

%% Save
fprintf('\n--- Saving ---\n');
try
    save(fullfile(outTable,'my_method_results.mat'),'results','H_coarse','H_final','RT');
    m_names = {'rmse_coarse','rmse_fine','ncc_coarse','ncc_fine','ssim_coarse','ssim_fine',...
               'precision','recall','f1','runtime_total'};
    m_vals  = [results.rmse_coarse, results.rmse_fine, results.ncc_coarse, results.ncc_fine, ...
               results.ssim_coarse, results.ssim_fine, results.precision,  results.recall, ...
               results.f1, RT.total];
    writetable(array2table(m_vals,'VariableNames',m_names), ...
               fullfile(outTable,'my_method_results.csv'));
    fprintf('  Saved to Performance Metriczz/tables/\n');
catch ME_sv; fprintf('  Save error: %s\n',ME_sv.message); end

fprintf('\n=== my_method.m COMPLETE ===\n');
fprintf('Outputs in: %s\n', outRoot);
