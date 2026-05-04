function masks = mm_buildFaceMasks(pts, img_size, bbox)
%MM_BUILDFACEMASKS  Build 7 binary face region masks from 66 landmark points.
%
%   masks = mm_buildFaceMasks(pts, img_size, bbox)
%
%   pts      : [66 x 2] landmark [x y] = [col row] coordinates
%   img_size : [rows cols] of the target image
%   bbox     : [x y w h] face bounding box
%
%   Returns struct with fields (each field is a logical [rows x cols] mask):
%     left_eye, right_eye, nose, mouth, forehead, left_cheek, right_cheek
%
%   Landmark numbering (MATLAB 66-point model):
%   1-17 jawline, 18-22 R-brow, 23-27 L-brow, 28-31 nose bridge,
%   32-36 nose bottom, 37-42 R-eye, 43-48 L-eye,
%   49-60 outer lips, 61-66 inner lips.

H = img_size(1);  W = img_size(2);
bx = bbox(1); by = bbox(2); bw = bbox(3); bh = bbox(4);

function m = make_mask(px, py)
    px = max(1,min(W,px));  py = max(1,min(H,py));
    m  = poly2mask(px, py, H, W);
end

% ─ Right eye (pts 37–42) ─
masks.right_eye = make_mask(pts(37:42,1), pts(37:42,2));

% ─ Left eye (pts 43–48) ─
masks.left_eye  = make_mask(pts(43:48,1), pts(43:48,2));

% ─ Nose (pts 28–36, close polygon) ─
nose_idx = [28:36, 28];
masks.nose = make_mask(pts(nose_idx,1), pts(nose_idx,2));

% ─ Mouth (pts 49–60, close polygon) ─
mouth_idx = [49:60, 49];
masks.mouth = make_mask(pts(mouth_idx,1), pts(mouth_idx,2));

% ─ Forehead: above eyebrows, top 20% of face box ─
brow_y  = min(pts([18:22,23:27],2));      % top of brows
top_y   = by;
fore_poly_x = [bx, bx+bw, bx+bw, bx];
fore_poly_y = [top_y, top_y, brow_y, brow_y];
masks.forehead = make_mask(fore_poly_x, fore_poly_y);

% ─ Left cheek: pts 1–8 (left jaw), left eye outer, nose left ─
lcheek_x = [pts(1:8,1);  pts(37,1); pts(28,1); pts(1,1)];
lcheek_y = [pts(1:8,2);  pts(37,2); pts(28,2); pts(1,2)];
masks.left_cheek = make_mask(lcheek_x, lcheek_y);

% ─ Right cheek: pts 10–17 (right jaw), right eye outer, nose right ─
rcheek_x = [pts(10:17,1); pts(48,1); pts(36,1); pts(17,1)];
rcheek_y = [pts(10:17,2); pts(48,2); pts(36,2); pts(17,2)];
masks.right_cheek = make_mask(rcheek_x, rcheek_y);

% Ensure no overlap between cheeks and eye/nose/mouth masks
masks.left_cheek  = masks.left_cheek  & ~(masks.left_eye | masks.nose | masks.mouth);
masks.right_cheek = masks.right_cheek & ~(masks.right_eye| masks.nose | masks.mouth);

end
