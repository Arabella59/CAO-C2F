function [rmse, ncc, ssim_val] = mm_computeImageMetrics(I_registered, I_visible)
%MM_COMPUTEIMAGEMETRICS  Global pixel RMSE, NCC and SSIM between images.
%
%   [rmse, ncc, ssim_val] = mm_computeImageMetrics(I_registered, I_visible)
%
%   Both inputs should be the same size (double or uint8, grayscale).
%   Returns NaN for any metric that cannot be computed.

try
    A = double(I_registered);
    B = double(I_visible);
    if ~isequal(size(A), size(B))
        % Crop to smaller size
        r = min(size(A,1), size(B,1));
        c = min(size(A,2), size(B,2));
        A = A(1:r, 1:c);  B = B(1:r, 1:c);
    end
    diff  = A - B;
    rmse  = sqrt(mean(diff(:).^2));
catch
    rmse = NaN;
end

try
    A = double(I_registered);  B = double(I_visible);
    if ~isequal(size(A), size(B))
        r = min(size(A,1),size(B,1)); c = min(size(A,2),size(B,2));
        A = A(1:r,1:c); B = B(1:r,1:c);
    end
    A = A - mean(A(:));  B = B - mean(B(:));
    denom = norm(A(:)) * norm(B(:));
    if denom < 1e-10, ncc = NaN;
    else,             ncc = dot(A(:), B(:)) / denom;
    end
catch
    ncc = NaN;
end

try
    A = im2double(I_registered);  B = im2double(I_visible);
    if ~isequal(size(A), size(B))
        r = min(size(A,1),size(B,1)); c = min(size(A,2),size(B,2));
        A = A(1:r,1:c); B = B(1:r,1:c);
    end
    ssim_val = ssim(A, B);
catch
    ssim_val = NaN;
end
end
