function [H, tform] = mm_computeHomographyFromPoints(pts1_xy, pts2_xy)
%MM_COMPUTEHOMOGRAPHYFROMPOINTS  Fit projective homography from matched [x y] pairs.
%
%   [H, tform] = mm_computeHomographyFromPoints(pts1_xy, pts2_xy)
%
%   pts1_xy, pts2_xy : [N x 2] in [x y] = [col row] format
%   H     : 3x3 conventional homography s.t. p2 ~ H * p1 (homogeneous)
%   tform : MATLAB projective2d object (for use with imwarp)
%
%   Falls back to affine if fewer than 4 point pairs.

H    = eye(3);
tform = affine2d(eye(3));

if isempty(pts1_xy) || size(pts1_xy,1) < 3
    warning('mm_computeHomographyFromPoints: too few points (%d).', size(pts1_xy,1));
    return;
end

try
    if size(pts1_xy,1) >= 4
        tform = fitgeotrans(pts1_xy, pts2_xy, 'projective');
    else
        tform = fitgeotrans(pts1_xy, pts2_xy, 'affine');
    end
    % MATLAB stores T such that [x2 y2 w] = [x1 y1 1] * T
    % Conventional H: p2 = H * p1  =>  H = T'
    H = tform.T';
catch ME
    warning('mm_computeHomographyFromPoints: %s', ME.message);
    H    = eye(3);
    tform = affine2d(eye(3));
end
end
