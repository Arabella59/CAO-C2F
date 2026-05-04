function result = mm_runMethodA(I_thermal, I_visible, Lc)
%MM_RUNMETHODA  Run original CAO-C2F baseline (Method A).
%
%   result = mm_runMethodA(I_thermal, I_visible, Lc)
%
%   No CLAHE, no hybrid orientation.  Uses cp_registration directly.
%   Returns standardised result struct.

if nargin < 3, Lc = 6; end

result = mm_empty_result();
maxRMSE = 4*ceil(size(I_visible,1)/300);

try
    t0 = tic;
    I_t = double(im2gray_safe(I_thermal));
    I_v = double(im2gray_safe(I_visible));

    [P1, P2, Rt, cor12] = cp_registration(I_t, I_v, 20, maxRMSE, 0, 1, 0, Lc, 0, I_v);

    result.runtime = toc(t0);
    result.pts1    = P1;
    result.pts2    = P2;

    % Separate cor12 into cor1 and cor2 (sep = [0 0] row)
    sep = find(cor12(:,1)==0 & cor12(:,2)==0, 1);
    if ~isempty(sep)
        result.kp1 = cor12(1:sep-1,:);
        result.kp2 = cor12(sep+1:end,:);
    end
    result.n_kp1     = size(result.kp1,1);
    result.n_kp2     = size(result.kp2,1);
    result.n_putative = size(P1,1);  % approximate (post-SITAC)
    result.n_final   = size(P1,1);

    % Compute homography and warp
    if size(P1,1) >= 3
        [result.H, tform] = mm_computeHomographyFromPoints(P1, P2);
        refObj = imref2d(size(I_v));
        I_reg  = imwarp(I_t, tform, 'OutputView', refObj);
        result.I_registered = I_reg;
        result.I_coarse     = I_reg;
    end

    % Global metrics
    [result.rmse, result.ncc, result.ssim] = ...
        mm_computeImageMetrics(result.I_registered, I_v);

    result.success = true;
catch ME
    fprintf('  Method A failed: %s\n', ME.message);
    result.error = ME.message;
end
end

%% -------------------------------------------------------
function I = im2gray_safe(I)
if size(I,3) == 3, I = rgb2gray(I); end
I = double(I);
end
