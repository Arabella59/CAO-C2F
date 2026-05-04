function [pts, bbox, detection_mode] = mm_detectFaceRegions(I_vis)
%MM_DETECTFACEREGIONS  Detect face and 66 facial landmarks on a visible image.
%
%   [pts, bbox, detection_mode] = mm_detectFaceRegions(I_vis)
%
%   I_vis : visible image (grayscale or colour, uint8 or double)
%   pts   : [66 x 2] landmark [x y] coordinates (col, row)
%   bbox  : [1 x 4] face bounding box [x y w h]
%   detection_mode: 'landmark' | 'proportional'
%
%   Falls back to proportional region estimates if detection fails.

if size(I_vis,3) == 3
    I_gray = rgb2gray(I_vis);
else
    I_gray = I_vis;
end
I_u8 = im2uint8(I_gray);
[H, W] = size(I_u8);

%% Face bounding box
bbox = [];
detection_mode = 'proportional';
try
    faceDetector = vision.CascadeObjectDetector();
    bboxes = step(faceDetector, I_u8);
    if ~isempty(bboxes)
        % Use largest detected face
        [~, imax] = max(bboxes(:,3).*bboxes(:,4));
        bbox = bboxes(imax,:);
    end
catch
end

if isempty(bbox)
    % Proportional default: middle 60% of image
    margin_x = round(W*0.15);  margin_y = round(H*0.1);
    bbox = [margin_x, margin_y, W-2*margin_x, round(H*0.8)];
end

%% Landmark detection
pts = [];
try
    if exist('detectFacialLandmarks','file') || exist('detectFacialLandmarks','builtin')
        lmStruct = detectFacialLandmarks(I_u8, bbox, 'Robustness', 'High');
        if ~isempty(lmStruct)
            raw = lmStruct(1).Location;  % [66 x 2]
            if ndims(raw) == 3            %#ok<ISMAT>
                raw = squeeze(raw(1,:,:));
            end
            pts = double(raw);  % [66 x 2] [x y]
        end
    end
catch
    pts = [];
end

if size(pts,1) == 66
    detection_mode = 'landmark';
    return;
end

%% Proportional fallback — synthesise 66 landmark positions
detection_mode = 'proportional';
bx = bbox(1); by = bbox(2); bw = bbox(3); bh = bbox(4);
pts = zeros(66,2);

% Jawline: pts 1–17 (horizontal sweep along lower face)
for k = 1:17
    pts(k,:) = [bx + bw*(k-1)/16, by + bh*0.90];
end
% Right eyebrow: pts 18–22
for k = 1:5
    pts(17+k,:) = [bx + bw*(0.15+(k-1)*0.12), by + bh*0.25];
end
% Left eyebrow: pts 23–27
for k = 1:5
    pts(22+k,:) = [bx + bw*(0.45+(k-1)*0.10), by + bh*0.25];
end
% Nose bridge: pts 28–31
for k = 1:4
    pts(27+k,:) = [bx + bw*0.50, by + bh*(0.30+(k-1)*0.08)];
end
% Nose bottom: pts 32–36
for k = 1:5
    pts(31+k,:) = [bx + bw*(0.35+(k-1)*0.075), by + bh*0.60];
end
% Right eye: pts 37–42
for k = 1:6
    a = (k-1)*pi/5;
    pts(36+k,:) = [bx+bw*0.27+bw*0.09*cos(a), by+bh*0.37+bh*0.05*sin(a)];
end
% Left eye: pts 43–48
for k = 1:6
    a = (k-1)*pi/5;
    pts(42+k,:) = [bx+bw*0.68+bw*0.09*cos(a), by+bh*0.37+bh*0.05*sin(a)];
end
% Outer lips: pts 49–60
for k = 1:12
    a = (k-1)*2*pi/12;
    pts(48+k,:) = [bx+bw*0.50+bw*0.18*cos(a), by+bh*0.74+bh*0.06*sin(a)];
end
% Inner lips: pts 61–66
for k = 1:6
    a = (k-1)*pi/5;
    pts(60+k,:) = [bx+bw*0.50+bw*0.10*cos(a), by+bh*0.74+bh*0.04*sin(a)];
end

fprintf('  [mm_detectFaceRegions] Using proportional landmark fallback.\n');
end
