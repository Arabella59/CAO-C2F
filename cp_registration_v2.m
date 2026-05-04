function [regis_points1, regis_points2, Runtime, cor12, extra] = ...
        cp_registration_v2(I1, I2, maxtheta, maxErr, iteration, ...
                           zoomascend, zoomdescend, Lc, showflag, I2ori, I_clahe)
%CP_REGISTRATION_V2  Modified CAO-C2F registration with optional CLAHE/hybrid orientation.
%
%   [P1, P2, Runtime, cor12, extra] = cp_registration_v2(
%       I1, I2, maxtheta, maxErr, iteration,
%       zoomascend, zoomdescend, Lc, showflag, I2ori, I_clahe)
%
%   I_clahe (optional, 11th arg): when non-empty, cp_cornerDetection_hybrid is
%   called for I1 with I_clahe as the CLAHE-enhanced fallback image.  When
%   I_clahe is empty or omitted, the original cp_cornerDetection is used
%   (Method B behaviour: CLAHE preprocessing on I1, but standard CAO orientation).
%
%   extra: struct with fields
%     .n_kp1, .n_kp2           keypoint counts
%     .n_putative              putative match count (after BBF + SITAC)
%     .n_final                 final match count (after RANSAC)
%     .cor1, .cor2             all detected keypoints [row col]
%     .putative_pts1/2         putative match positions [u v]
%     .orientation1/2          keypoint orientations
%     .orient_method1          orientation method per kp in I1 (1=CAO,0=grad)

if nargin < 11, I_clahe = []; end

fprintf('\n[cp_registration_v2] Registrating (CLAHE=%s, hybrid=%s)...\n', ...
    mat2str(~isempty(I_clahe)), mat2str(~isempty(I_clahe)));

%% 1. Corner detection ────────────────────────────────────────────────────
tic;
use_hybrid = ~isempty(I_clahe);

if use_hybrid
    [cor1, orientation1, I1edge, orient_method1] = cp_cornerDetection_hybrid(...
        I1, I_clahe, [], [], [], [], 0, 1, 1, Lc, iteration==0);
else
    [cor1, orientation1, I1edge] = cp_cornerDetection(...
        I1, [], [], [], [], 0, 1, 1, Lc, iteration==0);
    orient_method1 = ones(size(cor1,1),1);  % all CAO
end
[cor2, orientation2, ~] = cp_cornerDetection(...
    I2, [], [], [], [], [], 1, 1, Lc, iteration==0);

CFO_time = toc;
cor12 = [cor1(:,2) cor1(:,1); 0 0; cor2(:,2) cor2(:,1)];

I1 = double(I1);
I2 = double(I2);
I2ori = double(I2ori);

if iteration==0 && showflag
    xu1 = 4*cos(orientation1);  yv1 = 4*sin(orientation1);
    xu2 = 4*cos(orientation2);  yv2 = 4*sin(orientation2);
    figure('Name','cp_registration_v2 — Keypoints');
    subplot(121); imshow(I1/255); hold on;
    plot(cor1(:,2),cor1(:,1),'y+');
    title('Thermal keypoints'); quiver(cor1(:,2),cor1(:,1),xu1,yv1,'r');
    subplot(122); imshow(I2/255); hold on;
    plot(cor2(:,2),cor2(:,1),'y+');
    title('Visible keypoints'); quiver(cor2(:,2),cor2(:,1),xu2,yv2,'r');
end

%% 2. Descriptors ─────────────────────────────────────────────────────────
tic;
scale12 = 36;
cols1 = cor1(:,1);  rows1 = cor1(:,2);
s1    = scale12 * ones(size(cols1));
o1    = (iteration>0) * zeros(size(cols1)) + (iteration==0) * orientation1;
key1  = [rows1'; cols1'; s1'; o1'];
des1  = cp_descriptor(I1, key1);

cols2 = cor2(:,1);  rows2 = cor2(:,2);
o2    = (iteration>0) * zeros(size(cols2)) + (iteration==0) * orientation2;
s2    = scale12 * ones(size(cols2));
key2  = [rows2'; cols2'; s2'; o2'];

zoomstepup   = 1;
zoomstepdown = 0.5;
des2 = zeros([128, size(cor2,1), zoomascend+zoomdescend+1]);
des2(:,:,1) = cp_descriptor(I2, key2);
level = 1;
scale = size(I2ori,1) / size(I2,1);
if iteration == 0
    for zi = 1:zoomascend
        level    = level+1;
        key2z    = key2;
        I2zoom   = imresize(I2ori, (1+zoomstepup*zi)/scale);
        key2z(1:2,:) = floor((1+zoomstepup*zi)*key2z(1:2,:));
        des2(:,:,level) = cp_descriptor(I2zoom, key2z);
    end
    for zi = 1:zoomdescend
        level    = level+1;
        key2z    = key2;
        I2zoom   = imresize(I2ori, (1-zoomstepdown*zi)/scale);
        key2z(1:2,:) = floor((1-zoomstepdown*zi)*key2z(1:2,:));
        des2(:,:,level) = cp_descriptor(I2zoom, key2z);
    end
end
descriptortoc = toc;

%% 3. Coarse BBF matching ─────────────────────────────────────────────────
[matchIndex1, matchIndex2, zoom] = cp_match(des1', des2, 0.97);
zoomscale = (zoom==1)*1 + (zoom>1 & zoom<=(1+zoomascend))*(1+(zoom-1)*zoomstepup) + ...
    (zoom>(1+zoomascend))*(1-(zoom-1-zoomascend)*zoomstepdown);

regis_points111 = [cor1(matchIndex1,2) cor1(matchIndex1,1) o1(matchIndex1)];
regis_points222 = [cor2(matchIndex2,2) cor2(matchIndex2,1) o2(matchIndex2)];

fprintf('  KP-thermal=%d  KP-visible=%d  putative=%d\n', ...
    size(cor1,1), size(cor2,1), size(regis_points111,1));

if size(regis_points111,1) < 2
    error('cp_registration_v2: insufficient putative matches (%d).', size(regis_points111,1));
end
if showflag
    cp_showMatch(I1, imresize(I2ori,zoomscale/scale), ...
        regis_points111(:,1:2), zoomscale*regis_points222(:,1:2), [], 'Putative matches');
end

%% 4. Scale-invariant tilt-angle consistency ─────────────────────────────
tic;
if iteration == 0
    delta0 = mod(round(180/pi*(regis_points222(:,3)-regis_points111(:,3))),360);
    dd = 5;  d_delta = 0:dd:360;
    n_delta = histc(delta0, d_delta);
    [~,nidx] = sort(n_delta,'descend');  n0 = nidx(1);
    nmat = [n0^2,n0,1; (n0-1)^2,n0-1,1; (n0+1)^2,n0+1,1] \ ...
           [n_delta(n0); n_delta(n0-1+360/dd*(n0==1)); n_delta(n0+1-(n0==360/dd))];
    Modetheta = -nmat(2)/(2*nmat(1)) * dd;
else
    Modetheta = 0;
end

trans222 = [regis_points222(:,1)-size(I2,2)/2, regis_points222(:,2)-size(I2,1)/2] * ...
           [cos(Modetheta*pi/180) -sin(Modetheta*pi/180); ...
            sin(Modetheta*pi/180)  cos(Modetheta*pi/180)];
trans222 = [trans222(:,1)+size(I2,2)/2, trans222(:,2)+size(I2,1)/2];

phi_uv = cp_atan(zoomscale*trans222(:,2)-regis_points111(:,2), ...
                 zoomscale*trans222(:,1)+size(I1,2)-regis_points111(:,1));

dd = 5;  d_phi = -90:dd:90;
n_phi = histc(phi_uv, d_phi);
[~,nidx] = sort(n_phi,'descend');  n0 = nidx(1);
interval_index = find(n_phi < n_phi(n0)*0.2) - n0;
left_phi  = interval_index(interval_index<0);
right_phi = interval_index(interval_index>0);
maxtheta1 = -dd*interval_index(left_phi(end));
maxtheta2 =  dd*interval_index(right_phi(1));
nmat = [n0^2,n0,1; (n0-1+180/dd*(n0==1))^2,n0-1+180/dd*(n0==1),1; ...
        (n0+1-(n0==180/dd))^2,n0+1-(n0==180/dd),1] \ ...
       [n_phi(n0); n_phi(n0-1+180/dd*(n0==1)); n_phi(n0+1-(n0==180/dd))];
ModePhi = (-nmat(2)/(2*nmat(1)) - 90/dd) * dd;
delta1 = ModePhi - maxtheta1;  delta2 = ModePhi + maxtheta2;
valid0 = find(phi_uv >= delta1 & phi_uv <= delta2);

Dist    = sqrt((zoomscale*trans222(valid0,2)-regis_points111(valid0,2)).^2 + ...
               (zoomscale*trans222(valid0,1)+size(I1,2)-regis_points111(valid0,1)).^2);
meandist = mean(Dist);
valid1   = find(Dist >= 0.5*meandist & Dist <= 1.5*meandist);

regis_points11 = regis_points111(valid0(valid1),:);
regis_points22 = regis_points222(valid0(valid1),:);

if showflag
    cp_showMatch(I1, imresize(imrotate(I2,Modetheta,'crop'),zoomscale), ...
        regis_points11(:,1:2), zoomscale*trans222(valid0(valid1),1:2), ...
        [], 'After SITAC consistency');
end
SITAC_TIME = toc;

%% 5. RANSAC mismatch removal ─────────────────────────────────────────────
correctindex = cp_mismatchRemoval(regis_points11, regis_points22, I1, I2, maxErr);
RANSAC_TIME  = toc;

regis_points1 = regis_points11(correctindex, 1:2);
regis_points2 = regis_points22(correctindex, 1:2);
Runtime = CFO_time + descriptortoc + RANSAC_TIME;

if showflag
    cp_showMatch(I1, I2, regis_points1, regis_points2, [], 'Final matches');
end

fprintf('[cp_registration_v2] Done. KP1=%d KP2=%d putative=%d final=%d\n', ...
    size(cor1,1), size(cor2,1), size(regis_points11,1), size(regis_points1,1));

%% 6. Pack extra outputs ───────────────────────────────────────────────────
if nargout >= 5
    extra.n_kp1          = size(cor1,1);
    extra.n_kp2          = size(cor2,1);
    extra.n_putative     = size(regis_points11,1);
    extra.n_final        = size(regis_points1,1);
    extra.cor1           = cor1;          % [row col]
    extra.cor2           = cor2;
    extra.orientation1   = orientation1;
    extra.orientation2   = orientation2;
    extra.orient_method1 = orient_method1;
    extra.putative_pts1  = regis_points11(:,1:2);
    extra.putative_pts2  = regis_points22(:,1:2);
    extra.use_hybrid     = use_hybrid;
end
end
