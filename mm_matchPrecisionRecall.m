function [precision, recall, f1] = mm_matchPrecisionRecall(pts1, pts2, putative_pts1, H, threshold_px)
%MM_MATCHPRECISIONRECALL  Precision, Recall and F1 for the matching stage.
%
%   [precision, recall, f1] = mm_matchPrecisionRecall(
%       pts1, pts2, putative_pts1, H, threshold_px)
%
%   pts1, pts2    : [N x 2] final matched points [u v] = [col row]
%   putative_pts1 : [M x 2] all putative thermal points
%   H             : 3x3 projective homography (thermal -> visible)
%   threshold_px  : reprojection error threshold (default 5)

if nargin < 5, threshold_px = 5; end

precision = NaN;  recall = NaN;  f1 = NaN;

if isempty(pts1) || isempty(H) || any(isnan(H(:)))
    return;
end

try
    N = size(pts1, 1);
    % Apply H to thermal (pts1) -> predicted visible location
    pts1h = [pts1, ones(N,1)];
    pred  = (H * pts1h')';          % [N x 3] homogeneous
    pred  = pred(:,1:2) ./ pred(:,3); % dehomogenise
    errs  = sqrt(sum((pred - pts2).^2, 2));
    n_correct = sum(errs < threshold_px);

    M = size(putative_pts1, 1);
    if M == 0, return; end

    precision = n_correct / max(1, N);
    recall    = n_correct / max(1, M);
    denom     = precision + recall;
    if denom < 1e-10, f1 = 0;
    else,             f1 = 2*precision*recall / denom;
    end
catch ME
    fprintf('  mm_matchPrecisionRecall error: %s\n', ME.message);
end
end
