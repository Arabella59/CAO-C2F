% COMPARISON_ALL_METHODS.M
% Batch comparison of all 4 registration methods on thermal-visible image pairs.
%
% Method A: Original CAO-C2F baseline
% Method B: CLAHE + original CAO orientation
% Method C: CLAHE + hybrid CAO orientation
% Method D: Full proposed method (CLAHE + Hybrid + Landmark NCC + PWL blend)
%
% Produces 12 output types: tables, bar charts, box plots, scatter plots,
% histograms, keypoint plots, heatmaps, homography grids, PR plot, runtime chart.

clear; close all; clc;
thisFolder = fileparts(mfilename('fullpath'));
addpath(thisFolder);
addpath(genpath(thisFolder));

%% System information
fprintf('=== COMPARISON_ALL_METHODS ===\n');
fprintf('MATLAB: %s\n', version);
fprintf('CPU cores: %d\n', feature('numcores'));
try; [~,s]=memory; fprintf('RAM: %.1fGB\n',s.PhysicalMemory.Total/1e9); catch; end
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

DATASET_FOLDER = ['C:\Users\nokor\OneDrive\Documents\Individual project-YEAR3' ...
    '\WHU-IIP Dataset\combo data\'];
Lc  = 6;
MAX_PAIRS = 15;
METHOD_NAMES = {'A: Original CAO-C2F','B: CLAHE+CAO','C: CLAHE+Hybrid','D: Full Proposed'};
METHOD_COLORS= [0 0.4 0.8; 0.9 0.5 0; 0.1 0.7 0.2; 0.8 0.1 0.1];  % B O G R

%% Ask user: auto or single pair?
mode = input('Process all images automatically or select a single pair?\n  1 = auto-detect pairs in dataset folder\n  2 = select single pair manually\nChoice: ');
if isempty(mode), mode = 1; end

pairs = struct('thermal_path',{}, 'visible_path',{});
if mode == 2
    [f1,p1] = uigetfile({'*.jpg;*.bmp;*.png;*.tif','Images'},'Select THERMAL image');
    if isequal(f1,0), error('Cancelled.'); end
    [f2,p2] = uigetfile({'*.jpg;*.bmp;*.png;*.tif','Images'},'Select VISIBLE image');
    if isequal(f2,0), error('Cancelled.'); end
    pairs(1).thermal_path = fullfile(p1,f1);
    pairs(1).visible_path = fullfile(p2,f2);
    fprintf('Processing single pair.\n');
else
    if ~isfolder(DATASET_FOLDER)
        fprintf('Dataset folder not found: %s\nPlease select it manually.\n', DATASET_FOLDER);
        DATASET_FOLDER = uigetdir('', 'Select dataset folder');
    end
    pairs = mm_autoPairImages(DATASET_FOLDER, MAX_PAIRS);
    if isempty(pairs)
        error('No image pairs found. Check dataset folder: %s', DATASET_FOLDER);
    end
end

N_pairs = length(pairs);
fprintf('\nFound %d image pair(s) to process.\n\n', N_pairs);

%% Pre-allocate results storage
% all_results{pair_idx, method_idx}
all_results = cell(N_pairs, 4);

%% =========================================================================
%  MAIN PROCESSING LOOP
%% =========================================================================
for pi = 1:N_pairs
    fprintf('\n[%d/%d] (%.0f%%) Processing pair: %s  +  %s\n', ...
        pi, N_pairs, 100*(pi-1)/N_pairs, ...
        fileparts_name(pairs(pi).thermal_path), ...
        fileparts_name(pairs(pi).visible_path));

    % Load and resize images
    try
        I_therm = imread(pairs(pi).thermal_path);
        I_vis   = imread(pairs(pi).visible_path);
        if size(I_therm,1)>480, sf=480/size(I_therm,1); I_therm=imresize(I_therm,sf); end
        if size(I_vis,1)>480,   sf=480/size(I_vis,1);   I_vis  =imresize(I_vis,sf);   end
    catch ME_load
        fprintf('  Load error: %s\n', ME_load.message);
        continue;
    end

    [~, pair_id, ~] = fileparts(pairs(pi).thermal_path);

    % Run all 4 methods
    method_runners = {@mm_runMethodA, @mm_runMethodB, @mm_runMethodC, @mm_runMethodD};
    for mi = 1:4
        fprintf('  Running Method %s...', char('A'+mi-1));
        try
            all_results{pi,mi} = method_runners{mi}(I_therm, I_vis, Lc);
            r = all_results{pi,mi};
            if r.success
                fprintf(' OK  (RMSE=%.2f NCC=%.4f SSIM=%.4f time=%.2fs)\n', ...
                    r.rmse, r.ncc, r.ssim, r.runtime);
            else
                fprintf(' FAILED: %s\n', r.error);
            end
        catch ME_run
            fprintf(' ERROR: %s\n', ME_run.message);
            all_results{pi,mi} = mm_empty_result();
        end
    end

    fprintf('  Pair %d/%d complete (%.0f%%)\n', pi, N_pairs, 100*pi/N_pairs);
end

%% =========================================================================
%  EXTRACT METRICS FROM all_results
%% =========================================================================
metric_fns = {'n_kp1','n_kp2','n_final','rmse','ncc','ssim','runtime','precision','recall','f1'};
M_data = struct();  % M_data.(field) = [N_pairs x 4] matrix
for fn = metric_fns
    M_data.(fn{1}) = NaN(N_pairs, 4);
end

for pi = 1:N_pairs
    for mi = 1:4
        r = all_results{pi,mi};
        if isempty(r), continue; end
        for fn = metric_fns
            fld = fn{1};
            if isfield(r,fld) && ~isempty(r.(fld)) && isnumeric(r.(fld))
                v = r.(fld);
                if isscalar(v)
                    M_data.(fld)(pi,mi) = v;
                end
            end
        end
    end
end

%% =========================================================================
%  OUTPUT 1: Main summary table (mean +/- std)
%% =========================================================================
fprintf('\n\n=== OUTPUT 1: Summary Table (mean +/- std across %d pairs) ===\n', N_pairs);
fprintf('%-20s  %s\n', 'Metric', strjoin(cellfun(@(s) sprintf('%20s',s), METHOD_NAMES,'UniformOutput',false),'  '));
for fn = metric_fns
    fld = fn{1};
    row_str = sprintf('%-20s', fld);
    for mi = 1:4
        vals = M_data.(fld)(:,mi);
        vals = vals(~isnan(vals));
        if isempty(vals)
            row_str = [row_str sprintf('  %20s', 'N/A')]; %#ok<AGROW>
        else
            row_str = [row_str sprintf('  %9.3f+/-%.3f', mean(vals), std(vals))]; %#ok<AGROW>
        end
    end
    fprintf('%s\n', row_str);
end

% Save summary table CSV
try
    rows = {};
    for fn = metric_fns
        fld = fn{1};
        row = {fld};
        for mi = 1:4
            vals = M_data.(fld)(:,mi);  vals = vals(~isnan(vals));
            if isempty(vals), row{end+1} = 'NaN'; %#ok<AGROW>
            else, row{end+1} = sprintf('%.4f+/-%.4f',mean(vals),std(vals)); end %#ok<AGROW>
        end
        rows{end+1} = row; %#ok<AGROW>
    end
    hdr = [{'Metric'}, METHOD_NAMES];
    T_sum = cell2table(rows, 'VariableNames', hdr);
    writetable(T_sum, fullfile(outTable,'summary_table.csv'));
    fprintf('Summary table saved.\n');
catch ME_t1
    fprintf('Summary table save error: %s\n', ME_t1.message);
end

%% =========================================================================
%  OUTPUT 2: Per-pair results table
%% =========================================================================
try
    rows2 = {};
    for pi = 1:N_pairs
        for mi = 1:4
            r = all_results{pi,mi};
            row = {sprintf('pair%02d',pi), METHOD_NAMES{mi}};
            for fn = metric_fns
                fld = fn{1};
                if ~isempty(r) && isfield(r,fld) && isscalar(r.(fld))
                    row{end+1} = r.(fld); %#ok<AGROW>
                else
                    row{end+1} = NaN; %#ok<AGROW>
                end
            end
            rows2{end+1} = row; %#ok<AGROW>
        end
    end
    hdr2 = [{'Pair','Method'}, metric_fns];
    T_pp = cell2table(rows2, 'VariableNames', hdr2);
    writetable(T_pp, fullfile(outTable,'per_pair_table.csv'));
    fprintf('Per-pair table saved.\n');
catch ME_t2
    fprintf('Per-pair table error: %s\n', ME_t2.message);
end

%% =========================================================================
%  OUTPUT 3: Bar charts (6 subplots)
%% =========================================================================
try
    bar_metrics = {'n_kp1','n_final','rmse','ncc','ssim','runtime'};
    bar_labels  = {'Keypoints Detected','Final Matches','Pixel RMSE','NCC','SSIM','Runtime (s)'};
    fig3 = figure('Visible','off','Name','Comparison Bar Charts','Position',[0 0 1400 800]);
    for k = 1:6
        subplot(2,3,k);
        fld = bar_metrics{k};
        means_ = nanmean(M_data.(fld), 1);
        stds_  = nanstd(M_data.(fld),  0, 1);
        b = bar(1:4, means_, 'FaceColor','flat');
        for mi = 1:4, b.CData(mi,:) = METHOD_COLORS(mi,:); end
        hold on;
        errorbar(1:4, means_, stds_, 'k.', 'LineWidth',1.2);
        set(gca,'XTick',1:4,'XTickLabel',{'A','B','C','D'});
        xlabel('Method');  ylabel(bar_labels{k});  title(bar_labels{k});
        grid on;
    end
    sgtitle(sprintf('Method Comparison (%d pairs)', N_pairs));
    print(fig3, fullfile(outFig,'comparison_barcharts.png'),'-dpng','-r150');
    close(fig3);
    fprintf('Bar charts saved.\n');
catch ME3
    fprintf('Bar chart error: %s\n', ME3.message);
end

%% =========================================================================
%  OUTPUT 4: Box plots
%% =========================================================================
try
    fig4 = figure('Visible','off','Name','Comparison Box Plots','Position',[0 0 1400 800]);
    for k = 1:6
        subplot(2,3,k);
        fld = bar_metrics{k};
        data_ = M_data.(fld);  % [N x 4]
        boxplot(data_, 'Labels',{'A','B','C','D'},'Colors',METHOD_COLORS);
        ylabel(bar_labels{k});  title(bar_labels{k});  grid on;
    end
    sgtitle(sprintf('Method Comparison — Box Plots (%d pairs)', N_pairs));
    print(fig4, fullfile(outFig,'comparison_boxplots.png'),'-dpng','-r150');
    close(fig4);
    fprintf('Box plots saved.\n');
catch ME4
    fprintf('Box plot error: %s\n', ME4.message);
end

%% =========================================================================
%  OUTPUT 5: Scatter plots (per-pair RMSE, NCC, SSIM: A vs D)
%% =========================================================================
try
    scatter_pairs = {{'rmse','Pixel RMSE'},{'ncc','NCC'},{'ssim','SSIM'}};
    fig5 = figure('Visible','off','Name','Scatter: A vs D','Position',[0 0 1200 400]);
    for k = 1:3
        subplot(1,3,k);
        fld   = scatter_pairs{k}{1};  lbl = scatter_pairs{k}{2};
        x_val = M_data.(fld)(:,1);   % Method A
        y_val = M_data.(fld)(:,4);   % Method D
        scatter(x_val, y_val, 60, 'filled','MarkerFaceColor',[0.6 0 0.8]);
        hold on;
        all_v = [x_val; y_val];  all_v = all_v(~isnan(all_v));
        ax_lim = [min(all_v)*0.95, max(all_v)*1.05];
        if diff(ax_lim)>0
            plot(ax_lim, ax_lim, 'k--', 'LineWidth',1.2);
            xlim(ax_lim); ylim(ax_lim);
        end
        xlabel(['Method A ' lbl]);  ylabel(['Method D ' lbl]);
        title(sprintf('%s: A vs D', lbl));
        grid on;
        if strcmp(fld,'rmse')
            text(0.05,0.92,'Points below line = improvement','Units','normalized','FontSize',8);
        end
    end
    sgtitle('Per-pair comparison: Original CAO-C2F vs Proposed Method');
    print(fig5, fullfile(outFig,'scatter_A_vs_D.png'),'-dpng','-r150');
    close(fig5);
    fprintf('Scatter plots saved.\n');
catch ME5
    fprintf('Scatter plot error: %s\n', ME5.message);
end

%% =========================================================================
%  OUTPUT 6: CLAHE histogram comparison (representative pair)
%% =========================================================================
try
    rep_pi = find(cellfun(@(r) ~isempty(r) && r.success, all_results(:,4)), 1);
    if isempty(rep_pi), rep_pi = 1; end
    I_therm_rep = imread(pairs(rep_pi).thermal_path);
    if size(I_therm_rep,1)>480, sf=480/size(I_therm_rep,1); I_therm_rep=imresize(I_therm_rep,sf); end
    if size(I_therm_rep,3)==3, I_therm_rep=rgb2gray(I_therm_rep); end
    I_clahe_rep = adapthisteq(uint8(I_therm_rep),'NumTiles',[8 8],'ClipLimit',0.02);

    fig6 = figure('Visible','off','Name','CLAHE Histogram Comparison');
    histogram(double(I_therm_rep(:)), 64,'Normalization','probability',...
        'FaceColor',[0.2 0.4 0.8],'FaceAlpha',0.7,'DisplayName','Raw thermal'); hold on;
    histogram(double(I_clahe_rep(:)), 64,'Normalization','probability',...
        'FaceColor',[0.9 0.4 0.1],'FaceAlpha',0.7,'DisplayName','After CLAHE');
    std_raw = std(double(I_therm_rep(:)));  std_cl = std(double(I_clahe_rep(:)));
    pct_inc = (std_cl-std_raw)/std_raw*100;
    legend('show');  grid on;
    xlabel('Pixel Intensity');  ylabel('Probability');
    title(sprintf('Thermal Histogram Before/After CLAHE\nStd: %.1f \rightarrow %.1f (+%.1f%%)', ...
        std_raw, std_cl, pct_inc));
    print(fig6, fullfile(outHist,'clahe_histogram_comparison.png'),'-dpng','-r150');
    close(fig6);
    fprintf('CLAHE histogram saved.\n');
catch ME6
    fprintf('CLAHE histogram error: %s\n', ME6.message);
end

%% =========================================================================
%  OUTPUT 7: Keypoint spatial distribution (representative pair, 4 methods)
%% =========================================================================
try
    fig7 = figure('Visible','off','Name','Keypoint Distributions','Position',[0 0 1400 350]);
    for mi = 1:4
        subplot(1,4,mi);
        r = all_results{rep_pi, mi};
        if ~isempty(r) && r.success && ~isempty(r.kp1) && ~isempty(r.orientation1)
            ori_d = r.orientation1 * 180/pi;
            sz    = 20*ones(size(r.kp1,1),1);
            scatter(r.kp1(:,2), r.kp1(:,1), sz, ori_d, 'filled');
            colormap(hsv);  colorbar;
            xlabel('X (col)');  ylabel('Y (row)');
            axis ij;  grid on;
        else
            text(0.5,0.5,'No data','Units','normalized','HorizontalAlignment','center');
        end
        title(sprintf('Method %s\nKP=%d', char('A'+mi-1), ...  
            ifelse_val(isfield(r,'n_kp1') && ~isnan(r.n_kp1), r.n_kp1, 0)));
    end
    sgtitle('Thermal Keypoint Spatial Distribution (colour = orientation)');
    print(fig7, fullfile(outFig,'keypoint_distributions.png'),'-dpng','-r150');
    close(fig7);
    fprintf('Keypoint distribution plot saved.\n');
catch ME7
    fprintf('Keypoint distribution error: %s\n', ME7.message);
end

%% =========================================================================
%  OUTPUT 8: Per-region RMSE heatmaps (all 4 methods, representative pair)
%% =========================================================================
try
    I_therm_r = imread(pairs(rep_pi).thermal_path);
    I_vis_r   = imread(pairs(rep_pi).visible_path);
    if size(I_therm_r,1)>480, sf=480/size(I_therm_r,1); I_therm_r=imresize(I_therm_r,sf); end
    if size(I_vis_r,1)>480,   sf=480/size(I_vis_r,1);   I_vis_r  =imresize(I_vis_r,sf);   end
    if size(I_vis_r,3)==3,    I_vis_g=double(rgb2gray(I_vis_r));
    else,                     I_vis_g=double(I_vis_r); end

    fig8 = figure('Visible','off','Name','Heatmaps All Methods','Position',[0 0 1600 450]);
    all_rmse_vals = [];
    reg_images = cell(1,4);
    for mi = 1:4
        r = all_results{rep_pi,mi};
        if ~isempty(r) && r.success && ~isempty(r.I_registered)
            reg_images{mi} = r.I_registered;
        end
    end

    % Compute per-pixel errors and find common colour scale
    err_maps = cell(1,4);
    for mi = 1:4
        if ~isempty(reg_images{mi})
            A = double(reg_images{mi});  B = I_vis_g;
            if ~isequal(size(A),size(B))
                rr=min(size(A,1),size(B,1)); cc=min(size(A,2),size(B,2));
                A=A(1:rr,1:cc); B=B(1:rr,1:cc);
            end
            h_g = fspecial('gaussian',15,5);
            em  = imfilter(abs(A-B), h_g, 'replicate');
            err_maps{mi} = em;
            all_rmse_vals(end+1) = sqrt(mean(em(:).^2)); %#ok<AGROW>
        end
    end
    c_max = max(all_rmse_vals) + 1e-6;

    for mi = 1:4
        subplot(1,4,mi);
        if ~isempty(err_maps{mi})
            imagesc(err_maps{mi}, [0, c_max]);  colormap(jet);  axis image off;
            r_gm = sqrt(mean(err_maps{mi}(:).^2));
            title(sprintf('Method %s\nRMSE=%.2f', char('A'+mi-1), r_gm));
        else
            axis off;  text(0.5,0.5,'FAILED','Units','normalized','HorizontalAlignment','center');
            title(sprintf('Method %s\nFailed', char('A'+mi-1)));
        end
    end
    sgtitle(sprintf('Registration Error Heatmaps (pair: %s)', fileparts_name(pairs(rep_pi).thermal_path)));
    colorbar;
    [~,rep_id,~] = fileparts(pairs(rep_pi).thermal_path);
    print(fig8, fullfile(outHeat,sprintf('comparison_heatmaps_%s.png',rep_id)),'-dpng','-r150');
    close(fig8);
    fprintf('Heatmap comparison saved.\n');
catch ME8
    fprintf('Heatmap comparison error: %s\n', ME8.message);
end

%% =========================================================================
%  OUTPUT 9: Homography grid visualisations
%% =========================================================================
try
    r_ref = all_results{rep_pi, 4};  % Method D
    if ~isempty(r_ref) && r_ref.success
        I_vis_r2 = imread(pairs(rep_pi).visible_path);
        if size(I_vis_r2,1)>480, sf=480/size(I_vis_r2,1); I_vis_r2=imresize(I_vis_r2,sf); end
        [Hv,Wv] = size(I_vis_r2(:,:,1));

        % Create grid of points
        gx = linspace(10, Wv-10, 12);  gy = linspace(10, Hv-10, 9);
        [GX,GY] = meshgrid(gx,gy);
        grid_pts = [GX(:), GY(:)];

        fig9 = figure('Visible','off','Name','Homography Grid','Position',[0 0 1600 400]);
        for mi = 1:4
            subplot(1,4,mi);
            r_m = all_results{rep_pi,mi};
            if size(I_vis_r2,3)==3, bg=im2double(I_vis_r2);
            else, bg=repmat(im2double(I_vis_r2),[1 1 3]); end
            imshow(bg); hold on;
            if ~isempty(r_m) && r_m.success && ~any(isnan(r_m.H(:)))
                H_m = r_m.H;
                pts_h = [grid_pts, ones(size(grid_pts,1),1)];
                mapped = (H_m * pts_h')';
                mapped = mapped(:,1:2)./mapped(:,3);
                % Draw source (grid) and mapped points
                plot(grid_pts(:,1), grid_pts(:,2), 'b.', 'MarkerSize',4);
                plot(mapped(:,1), mapped(:,2), 'r.', 'MarkerSize',4);
                for gp = 1:size(grid_pts,1)
                    line([grid_pts(gp,1),mapped(gp,1)],[grid_pts(gp,2),mapped(gp,2)],...
                        'Color',[0.4 0.8 0.4],'LineWidth',0.5);
                end
                title(sprintf('Method %s', char('A'+mi-1)));
            else
                title(sprintf('Method %s\n(no H)', char('A'+mi-1)));
            end
        end
        sgtitle('Homography Grid: Blue=source, Red=mapped, Green=displacement');
        print(fig9, fullfile(outFig,'homography_grids.png'),'-dpng','-r150');
        close(fig9);
        fprintf('Homography grid saved.\n');
    end
catch ME9
    fprintf('Homography grid error: %s\n', ME9.message);
end

%% =========================================================================
%  OUTPUT 10: Precision-Recall plot
%% =========================================================================
try
    fig10 = figure('Visible','off','Name','Precision-Recall');
    hold on;  grid on;
    for mi = 1:4
        prec_ = nanmean(M_data.precision(:,mi));
        rec_  = nanmean(M_data.recall(:,mi));
        f1_   = nanmean(M_data.f1(:,mi));
        if ~isnan(prec_) && ~isnan(rec_)
            plot(rec_, prec_, 'o', 'MarkerSize',14, ...
                'MarkerFaceColor', METHOD_COLORS(mi,:), ...
                'MarkerEdgeColor','k', 'LineWidth',1.5);
            text(rec_+0.01, prec_, sprintf('  %s\nF1=%.3f', char('A'+mi-1), f1_), ...
                'FontSize',9, 'Color', METHOD_COLORS(mi,:));
        end
    end
    xlim([0 1]);  ylim([0 1]);
    xlabel('Recall');  ylabel('Precision');
    title('Precision-Recall (mean across pairs, threshold=5px)');
    legend(METHOD_NAMES,'Location','southwest');
    print(fig10, fullfile(outFig,'precision_recall.png'),'-dpng','-r150');
    close(fig10);
    fprintf('Precision-recall plot saved.\n');
catch ME10
    fprintf('PR plot error: %s\n', ME10.message);
end

%% =========================================================================
%  OUTPUT 11: Runtime stacked bar chart (Method D)
%% =========================================================================
try
    stage_names = {'CLAHE','Coarse','Landmark','Segment','NCC xcorr','Blend'};
    stage_fields= {'clahe','coarse','landmark','segment','xcorr','blend'};
    stage_data  = NaN(N_pairs, 6);
    for pi = 1:N_pairs
        r = all_results{pi,4};
        if ~isempty(r) && r.success && isstruct(r.runtime_stages)
            for si = 1:6
                fld = stage_fields{si};
                if isfield(r.runtime_stages, fld)
                    stage_data(pi,si) = r.runtime_stages.(fld);
                end
            end
        end
    end
    valid_rows = any(~isnan(stage_data),2);
    if any(valid_rows)
        stage_data(isnan(stage_data)) = 0;
        fig11 = figure('Visible','off','Name','Runtime Breakdown Method D');
        bar(find(valid_rows), stage_data(valid_rows,:), 'stacked');
        yline(0.04,'r--','40ms (25fps)','LineWidth',2,'LabelHorizontalAlignment','right');
        xlabel('Image Pair Index');  ylabel('Time (s)');
        title('Method D Runtime Breakdown per Image Pair');
        legend(stage_names,'Location','northwest');
        grid on;
        % Print real-time analysis
        total_times = sum(stage_data(valid_rows,:),2);
        fprintf('  Method D runtime: mean=%.3fs, min=%.3fs, max=%.3fs\n',...
            mean(total_times), min(total_times), max(total_times));
        fprintf('  Real-time (40ms=25fps) achievable on %d/%d pairs.\n',...
            sum(total_times < 0.04), sum(valid_rows));
        print(fig11, fullfile(outFig,'runtime_breakdown_D.png'),'-dpng','-r150');
        close(fig11);
        fprintf('Runtime breakdown saved.\n');
    end
catch ME11
    fprintf('Runtime breakdown error: %s\n', ME11.message);
end

%% =========================================================================
%  OUTPUT 12: Save all results
%% =========================================================================
try
    save(fullfile(outTable,'all_results.mat'), 'all_results', 'pairs', 'M_data', 'METHOD_NAMES');
    fprintf('\nAll results saved to: %s\n', outRoot);
catch ME12
    fprintf('Save error: %s\n', ME12.message);
end

%% Final printed summary
fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('%-20s', 'Metric');
for mi=1:4, fprintf('  %18s', sprintf('M%s mean+/-std',char('A'+mi-1))); end
fprintf('\n');
for fn = metric_fns
    fld = fn{1};
    fprintf('%-20s', fld);
    for mi=1:4
        vals=M_data.(fld)(:,mi); vals=vals(~isnan(vals));
        if isempty(vals), fprintf('  %18s','N/A');
        else, fprintf('  %8.3f+/-%.3f', mean(vals), std(vals)); end
    end
    fprintf('\n');
end

fprintf('\n=== comparison_all_methods.m COMPLETE ===\n');

%% =========================================================================
%  LOCAL HELPERS
%% =========================================================================
function n = fileparts_name(p)
[~,n,e] = fileparts(p);  n = [n e];
end

function v = ifelse_val(cond, a, b)
if cond, v=a; else, v=b; end
end
