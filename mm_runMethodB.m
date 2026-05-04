function result = mm_runMethodB(I_thermal, I_visible, Lc)
%MM_RUNMETHODB  CLAHE preprocessing + original CAO orientation (Method B).
%
%   result = mm_runMethodB(I_thermal, I_visible, Lc)
%
%   Applies CLAHE to thermal, then calls cp_registration_v2 WITHOUT passing
%   I_clahe (so hybrid orientation is NOT triggered).  CLAHE improves edge
%   detection but orientation assignment remains standard CAO.

if nargin < 3, Lc = 6; end

result = mm_empty_result();
maxRMSE = 4*ceil(size(I_visible,1)/300);

try
    t0 = tic;
    I_t_raw = im2gray_safe(I_thermal);
    I_v     = double(im2gray_safe(I_visible));

    % CLAHE on thermal only
    I_clahe = adapthisteq(uint8(I_t_raw), 'NumTiles',[8 8], 'ClipLimit',0.02);
    I_t_in  = double(I_clahe);

    % cp_registration_v2 without I_clahe arg → no hybrid orientation
    [P1, P2, Rt, cor12, extra] = cp_registration_v2(I_t_in, I_v, 20, maxRMSE, 0, 1, 0, Lc, 0, I_v);

    result.runtime   = toc(t0);
    result.pts1      = P1;  result.pts2 = P2;
    result.n_kp1     = extra.n_kp1;
    result.n_kp2     = extra.n_kp2;
    result.n_putative= extra.n_putative;
    result.n_final   = extra.n_final;
    result.kp1       = extra.cor1;  result.kp2 = extra.cor2;
    result.orientation1 = extra.orientation1;
    result.orient_method1 = extra.orient_method1;

    if size(P1,1) >= 3
        [result.H, tform] = mm_computeHomographyFromPoints(P1, P2);
        refObj = imref2d(size(I_v));
        I_reg  = imwarp(I_t_in, tform, 'OutputView', refObj);
        result.I_registered = I_reg;
        result.I_coarse     = I_reg;
    end

    [result.rmse, result.ncc, result.ssim] = ...
        mm_computeImageMetrics(result.I_registered, I_v);

    result.success = true;
catch ME
    fprintf('  Method B failed: %s\n', ME.message);
    result.error = ME.message;
end
end

function I = im2gray_safe(I)
if size(I,3) == 3, I = rgb2gray(I); end
I = double(I);
end
