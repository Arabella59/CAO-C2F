function [cout, angle, edge_map, orient_method] = cp_cornerDetection_hybrid(I, I_clahe, C, T_angle, sig, H_thresh, L_thresh, Endpoint, Gap_size, maxlength, rflag)
%CP_CORNERDETECTION_HYBRID  Hybrid corner detection with CAO + gradient fallback.
%
%   [cout, angle, edge_map, orient_method] = cp_cornerDetection_hybrid(
%       I, I_clahe, C, T_angle, sig, H_thresh, L_thresh,
%       Endpoint, Gap_size, maxlength, rflag)
%
%   I           : grayscale thermal image (double or uint8)
%   I_clahe     : CLAHE-enhanced thermal image used for gradient fallback
%                 (pass [] to disable fallback, reverts to I)
%   C           : curvature threshold ratio          (default 1.5)
%   T_angle     : maximum corner obtuse angle        (default 170 deg)
%   sig         : Gaussian sigma for curvature       (default 3)
%   H_thresh    : Canny high threshold               (default 0.2)
%   L_thresh    : Canny low  threshold               (default 0)
%   Endpoint    : flag to add line endpoints         (default 1)
%   Gap_size    : contour gap fill size (pixels)     (default 1)
%   maxlength   : max curvature region of support    (default 2)
%   rflag       : compute orientations flag          (default 0)
%
%   cout         : detected corner positions [row col]
%   angle        : orientation per corner (radians, [0 2pi))
%   edge_map     : binary edge map
%   orient_method: 1 = CAO contour orientation, 0 = gradient histogram fallback
%
%   Contours with length >= MIN_CONTOUR use original CAO orientation.
%   Shorter endpoint-only contours use gradient direction histogram from I_clahe.

MIN_CONTOUR = 6;   % Named constant — minimum contour length for CAO orientation

% ── Default arguments ──────────────────────────────────────────────────────
if nargin < 3  || isempty(C),        C        = 1.5;  end
if nargin < 4  || isempty(T_angle),  T_angle  = 170;  end
if nargin < 5  || isempty(sig),      sig      = 3;    end
if nargin < 6  || isempty(H_thresh), H_thresh = 0.2;  end
if nargin < 7  || isempty(L_thresh), L_thresh = 0;    end
if nargin < 8  || isempty(Endpoint), Endpoint = 1;    end
if nargin < 9  || isempty(Gap_size), Gap_size = 1;    end
if nargin < 10 || isempty(maxlength),maxlength = 2;   end
if nargin < 11 || isempty(rflag),    rflag    = 0;    end

% ── Ensure grayscale double ──────────────────────────────────────────────
if size(I,3) > 1, I = rgb2gray(I); end
I = double(I);

if isempty(I_clahe)
    I_grad = I;
else
    if size(I_clahe,3) > 1, I_clahe = rgb2gray(I_clahe); end
    I_grad = double(I_clahe);
end

% ── Edge detection & curve extraction ───────────────────────────────────
BW = edge(I, 'canny', [L_thresh, H_thresh]);
[curve, curve_start, curve_end, curve_mode, curve_num, edge_map] = h_extract_curve(BW, Gap_size);

% ── Gradient of CLAHE image for fallback orientation ────────────────────
[Gx, Gy] = gradient(I_grad);

% ── Corner detection with hybrid orientation ─────────────────────────────
[cout, angle, orient_method] = h_get_corner(curve, curve_start, curve_end, ...
    curve_mode, curve_num, BW, sig, Endpoint, C, T_angle, maxlength, rflag, ...
    Gx, Gy, MIN_CONTOUR);

end  % main function

% ═══════════════════════════════════════════════════════════════════════════
%  LOCAL HELPER FUNCTIONS
% ═══════════════════════════════════════════════════════════════════════════

function [curve,curve_start,curve_end,curve_mode,cur_num,edge_map] = h_extract_curve(BW,Gap_size)
% Extract ordered contour curves from binary edge map (same logic as cp_cornerDetection).
[L,W] = size(BW);
BW1 = zeros(L+2*Gap_size, W+2*Gap_size);
BW_edge = zeros(L,W);
BW1(Gap_size+1:Gap_size+L, Gap_size+1:Gap_size+W) = BW;
[r,c] = find(BW1==1);
cur_num = 0;
curve = {};
curve_start = [];
curve_end   = [];
curve_mode  = [];
while ~isempty(r)
    point = [r(1), c(1)];
    cur = point;
    BW1(point(1),point(2)) = 0;
    [I_,J_] = find(BW1(point(1)-Gap_size:point(1)+Gap_size, ...
                       point(2)-Gap_size:point(2)+Gap_size)==1);
    while ~isempty(I_)
        dist = (I_-Gap_size-1).^2 + (J_-Gap_size-1).^2;
        [~,idx] = min(dist);
        point = point + [I_(idx), J_(idx)] - Gap_size - 1;
        cur = [cur; point]; %#ok<AGROW>
        BW1(point(1),point(2)) = 0;
        [I_,J_] = find(BW1(point(1)-Gap_size:point(1)+Gap_size, ...
                           point(2)-Gap_size:point(2)+Gap_size)==1);
    end
    % Trace the other direction from seed
    point = [r(1), c(1)];
    BW1(point(1),point(2)) = 0;
    [I_,J_] = find(BW1(point(1)-Gap_size:point(1)+Gap_size, ...
                       point(2)-Gap_size:point(2)+Gap_size)==1);
    while ~isempty(I_)
        dist = (I_-Gap_size-1).^2 + (J_-Gap_size-1).^2;
        [~,idx] = min(dist);
        point = point + [I_(idx), J_(idx)] - Gap_size - 1;
        cur = [point; cur]; %#ok<AGROW>
        BW1(point(1),point(2)) = 0;
        [I_,J_] = find(BW1(point(1)-Gap_size:point(1)+Gap_size, ...
                           point(2)-Gap_size:point(2)+Gap_size)==1);
    end
    if size(cur,1) > (size(BW,1)+size(BW,2))/60
        cur_num = cur_num + 1;
        curve{cur_num} = cur - Gap_size; %#ok<AGROW>
    end
    [r,c] = find(BW1==1);
end
for i = 1:cur_num
    curve_start(i,:) = curve{i}(1,:);  %#ok<AGROW>
    curve_end(i,:)   = curve{i}(end,:); %#ok<AGROW>
    if (curve_start(i,1)-curve_end(i,1))^2 + (curve_start(i,2)-curve_end(i,2))^2 <= 4
        curve_mode(i,:) = 'loop';
    else
        curve_mode(i,:) = 'line';
    end
    BW_edge(curve{i}(:,1) + (curve{i}(:,2)-1)*L) = 1;
end
edge_map = BW_edge;
end

% ─────────────────────────────────────────────────────────────────────────
function [cout, angle, orient_method] = h_get_corner(curve, curve_start, curve_end, ...
        curve_mode, curve_num, BW, sig, Endpoint, C, T_angle, maxlength, rflag, ...
        Gx, Gy, MIN_CONTOUR)
% Detect corners with hybrid orientation assignment.
% Uses sequential for-loop (not parfor) so orient_method can be tracked.
cout = [];
angle = [];
orient_method = [];  % 1=CAO, 0=gradient fallback

GaussianDieOff = 1e-4;
pw  = 1:30;
ssq = sig^2;
width = max(find(exp(-(pw.*pw)/(2*ssq)) > GaussianDieOff));
if isempty(width), width = 1; end
t   = (-width:width);
gau = exp(-(t.*t)/(2*ssq));
gau = gau / sum(gau);
sigmaLs = maxlength;
warning('off','all');

for i = 1:curve_num
    x  = curve{i}(:,1);
    y  = curve{i}(:,2);
    W_ = width;
    L_ = length(x);
    if L_ <= W_, continue; end

    % ── Smooth the curve ──────────────────────────────────────────────
    if strcmp(strtrim(curve_mode(i,:)),'loop')
        xL = [x(L_-W_+1:L_); x; x(1:W_)];
        yL = [y(L_-W_+1:L_); y; y(1:W_)];
    else
        xL = [ones(W_,1)*2*x(1)-x(W_+1:-1:2); x; ones(W_,1)*2*x(L_)-x(L_-1:-1:L_-W_)];
        yL = [ones(W_,1)*2*y(1)-y(W_+1:-1:2); y; ones(W_,1)*2*y(L_)-y(L_-1:-1:L_-W_)];
    end
    xx = conv(xL, gau); xx = xx(W_+1:L_+3*W_);
    yy = conv(yL, gau); yy = yy(W_+1:L_+3*W_);

    % ── Curvature ─────────────────────────────────────────────────────
    Xu  = [xx(2)-xx(1); (xx(3:end)-xx(1:end-2))/2; xx(end)-xx(end-1)];
    Yu  = [yy(2)-yy(1); (yy(3:end)-yy(1:end-2))/2; yy(end)-yy(end-1)];
    Xuu = [Xu(2)-Xu(1); (Xu(3:end)-Xu(1:end-2))/2; Xu(end)-Xu(end-1)];
    Yuu = [Yu(2)-Yu(1); (Yu(3:end)-Yu(1:end-2))/2; Yu(end)-Yu(end-1)];
    K   = abs((Xu.*Yuu - Xuu.*Yu) ./ ((Xu.^2 + Yu.^2).^1.5));
    K   = ceil(K*100)/100;

    % ── Local maxima of curvature ─────────────────────────────────────
    extremum = [];
    N_ = length(K);
    n_ = 0;
    Search = 1;
    for j = 1:N_-1
        if (K(j+1)-K(j))*Search > 0
            n_ = n_+1;
            extremum(n_) = j;  %#ok<AGROW>
            Search = -Search;
        end
    end
    if mod(length(extremum),2) == 0
        n_ = n_+1;
        extremum(n_) = N_;
    end
    if isempty(extremum), continue; end
    n_ = length(extremum);

    % ── Adaptive local threshold ──────────────────────────────────────
    flag_k = ones(1,n_);
    lambda_LR = [];
    for k = 2:2:n_
        if k > length(extremum) || k+1 > n_, break; end
        [~,idx1] = min(K(extremum(k):-1:extremum(k-1)));
        [~,idx2] = min(K(extremum(k):extremum(k+1)));
        ROS    = K(extremum(k)-idx1+1 : extremum(k)+idx2-1);
        K_thre = C * mean(ROS);
        if K(extremum(k)) < K_thre
            flag_k(k) = 0;
        else
            lambda_LR = [lambda_LR; extremum(k) idx1 idx2]; %#ok<AGROW>
        end
    end
    if n_ < 2, continue; end
    extremum  = extremum(2:2:n_);
    flag_k    = flag_k(2:2:n_);
    extremum  = extremum(flag_k==1);
    lambda_LR = lambda_LR(flag_k==1,:);
    if isempty(extremum), continue; end

    % ── Corner angle check ────────────────────────────────────────────
    smoothed_curve = [xx, yy];
    n_  = length(extremum);
    flag_a = ones(1,n_);
    for j = 1:n_
        if     j==1 && j==n_,   ang = h_curve_tangent(smoothed_curve(1:L_+2*W_,:), extremum(j));
        elseif j==1,             ang = h_curve_tangent(smoothed_curve(1:extremum(j+1),:), extremum(j));
        elseif j==n_,            ang = h_curve_tangent(smoothed_curve(extremum(j-1):L_+2*W_,:), extremum(j)-extremum(j-1)+1);
        else,                    ang = h_curve_tangent(smoothed_curve(extremum(j-1):extremum(j+1),:), extremum(j)-extremum(j-1)+1);
        end
        if ang > T_angle && ang < (360-T_angle)
            flag_a(j) = 0;
        end
    end
    extremum  = extremum(flag_a~=0);
    lambda_LR = lambda_LR(flag_a~=0,:);

    % ── Map back to original curve indices ────────────────────────────
    extremum = extremum - W_;
    valid    = extremum > 0 & extremum <= L_;
    extremum  = extremum(valid);
    lambda_LR = lambda_LR(valid,:);
    n_ = length(extremum);

    % ── Accumulate corners with CAO orientation ───────────────────────
    % All curvature corners come from curves with L_ > W_ >= MIN_CONTOUR
    % so orient_method = 1 (CAO) always for this block.
    for j = 1:n_
        cout = [cout; curve{i}(extremum(j),:)]; %#ok<AGROW>
        if rflag
            xcor = curve{i}(extremum(j),2);
            ycor = curve{i}(extremum(j),1);
            retail   = size(curve{i},1);
            lengthL  = min([lambda_LR(j,2), extremum(j)]);
            lengthR  = min([lambda_LR(j,3), retail-extremum(j)+1]);
            coefL    = exp(-((0:lengthL-1)/lengthL).^2/2);
            coefL    = coefL/sum(coefL);
            coefR    = exp(-((lengthR-1:-1:0)/lengthR).^2/2);
            coefR    = coefR/sum(coefR);
            xL_  = sum(coefL' .* curve{i}(extremum(j)+1-lengthL:extremum(j),2));
            yL_  = sum(coefL' .* curve{i}(extremum(j)+1-lengthL:extremum(j),1));
            xR_  = sum(coefR' .* curve{i}(extremum(j):extremum(j)+lengthR-1,2));
            yR_  = sum(coefR' .* curve{i}(extremum(j):extremum(j)+lengthR-1,1));
            vL_  = [xL_-xcor, yL_-ycor];  vR_ = [xR_-xcor, yR_-ycor];
            if norm(vL_) < 1e-9 || norm(vR_) < 1e-9
                ori = 0;
            else
                vm  = vL_/norm(vL_) + vR_/norm(vR_);
                ori = h_quadrant_angle(vm(1), vm(2));
            end
            angle        = [angle;        ori]; %#ok<AGROW>
            orient_method= [orient_method; 1];  %#ok<AGROW>  % CAO
        end
    end
end  % curve loop

% ── Add Endpoints ──────────────────────────────────────────────────────
if Endpoint
    for i = 1:curve_num
        retail = size(curve{i},1);
        if retail > 0 && strcmp(strtrim(curve_mode(i,:)),'line')

            % ─ Start endpoint ─
            if ~isempty(cout)
                dist2 = sum((cout - ones(size(cout,1),1)*curve_start(i,:)).^2, 2);
            else
                dist2 = Inf;
            end
            if min(dist2) > 25
                cout = [cout; curve_start(i,:)]; %#ok<AGROW>
                if rflag
                    if retail >= MIN_CONTOUR
                        % CAO orientation for start endpoint
                        xx_ep = curve{i}(1,2);  yy_ep = curve{i}(1,1);
                        n_pts = min(retail, maxlength);
                        coef2 = exp(-((n_pts-1:-1:0)/sigmaLs).^2/2);
                        coef2 = coef2/sum(coef2);
                        xL_ = sum(coef2' .* curve{i}(1:n_pts,2));
                        yL_ = sum(coef2' .* curve{i}(1:n_pts,1));
                        ori = h_quadrant_angle(xL_-xx_ep, yL_-yy_ep);
                        angle        = [angle;        ori]; %#ok<AGROW>
                        orient_method= [orient_method;  1]; %#ok<AGROW>  % CAO
                    else
                        % Gradient histogram fallback for short contour
                        r_ep = curve_start(i,1);  c_ep = curve_start(i,2);
                        ori  = h_gradient_orientation(Gx, Gy, r_ep, c_ep, 2);
                        angle        = [angle;        ori]; %#ok<AGROW>
                        orient_method= [orient_method;  0]; %#ok<AGROW>  % gradient
                    end
                end
            end

            % ─ End endpoint ─
            if ~isempty(cout)
                dist2 = sum((cout - ones(size(cout,1),1)*curve_end(i,:)).^2, 2);
            else
                dist2 = Inf;
            end
            if min(dist2) > 25
                cout = [cout; curve_end(i,:)]; %#ok<AGROW>
                if rflag
                    if retail >= MIN_CONTOUR
                        xx_ep = curve{i}(end,2);  yy_ep = curve{i}(end,1);
                        st    = max(1, retail-maxlength+1);
                        n_pts = retail - st;
                        coef1 = exp(-((0:n_pts)/sigmaLs).^2/2);
                        coef1 = coef1/sum(coef1);
                        xL_ = sum(coef1' .* curve{i}(st:retail,2));
                        yL_ = sum(coef1' .* curve{i}(st:retail,1));
                        ori = h_quadrant_angle(xL_-xx_ep, yL_-yy_ep);
                        angle        = [angle;        ori]; %#ok<AGROW>
                        orient_method= [orient_method;  1]; %#ok<AGROW>  % CAO
                    else
                        r_ep = curve_end(i,1);  c_ep = curve_end(i,2);
                        ori  = h_gradient_orientation(Gx, Gy, r_ep, c_ep, 2);
                        angle        = [angle;        ori]; %#ok<AGROW>
                        orient_method= [orient_method;  0]; %#ok<AGROW>  % gradient
                    end
                end
            end
        end
    end
end
end  % h_get_corner

% ─────────────────────────────────────────────────────────────────────────
function ori = h_gradient_orientation(Gx, Gy, r, c, half_win)
% Dominant gradient orientation in (2*half_win+1) x (2*half_win+1) window.
[nr, nc] = size(Gx);
r1 = max(1, r-half_win);  r2 = min(nr, r+half_win);
c1 = max(1, c-half_win);  c2 = min(nc, c+half_win);
gx = Gx(r1:r2, c1:c2);
gy = Gy(r1:r2, c1:c2);
angles = atan2(gy(:), gx(:));       % [-pi, pi]
angles = mod(angles, 2*pi);         % [0, 2*pi)
if isempty(angles)
    ori = 0;
    return;
end
n_bins = 36;
edges  = linspace(0, 2*pi, n_bins+1);
counts = histcounts(angles, edges);
[~, imax] = max(counts);
ori = (edges(imax) + edges(imax+1)) / 2;  % midpoint of peak bin
end

% ─────────────────────────────────────────────────────────────────────────
function orientation = h_quadrant_angle(deltax, deltay)
% Convert (deltax, deltay) direction vector to orientation in [0, 2*pi).
if isnan(deltay) || isnan(deltax) || (deltax==0 && deltay==0)
    orientation = 0;
elseif deltay >= 0 && deltax >= 0
    orientation = atan2(deltay, deltax);
elseif deltay < 0 && deltax >= 0
    orientation = atan2(deltay, deltax) + 2*pi;
else
    orientation = atan2(deltay, deltax) + pi;
end
end

% ─────────────────────────────────────────────────────────────────────────
function ang = h_curve_tangent(cur, center)
% Compute the angle between tangent directions at a curve point (degrees).
direction = zeros(1,2);
for ii = 1:2
    if ii==1, crv = cur(center:-1:1,:);
    else,     crv = cur(center:end,:);
    end
    Lc = size(crv,1);
    if Lc > 3
        if any(crv(1,:) ~= crv(end,:))
            M  = ceil(Lc/2);
            x1=crv(1,1); y1=crv(1,2); x2=crv(M,1); y2=crv(M,2); x3=crv(end,1); y3=crv(end,2);
        else
            M1=ceil(Lc/3); M2=ceil(2*Lc/3);
            x1=crv(1,1); y1=crv(1,2); x2=crv(M1,1); y2=crv(M1,2); x3=crv(M2,1); y3=crv(M2,2);
        end
        denom = -y1*x2+y1*x3+y3*x2+x1*y2-x1*y3-x3*y2;
        if abs(denom) < 1e-8 || abs((x1-x2)*(y1-y3)-(x1-x3)*(y1-y2)) < 1e-8
            td = angle(complex(crv(end,1)-crv(1,1), crv(end,2)-crv(1,2)));
        else
            x0 = 0.5*(-y1*x2^2+y3*x2^2-y3*y1^2-y3*x1^2-y2*y3^2+x3^2*y1 ...
                       +y2*y1^2-y2*x3^2-y2^2*y1+y2*x1^2+y3^2*y1+y2^2*y3)/denom;
            y0 =-0.5*(x1^2*x2-x1^2*x3+y1^2*x2-y1^2*x3+x1*x3^2-x1*x2^2 ...
                      -x3^2*x2-y3^2*x2+x3*y2^2+x1*y3^2-x1*y2^2+x3*x2^2)/denom;
            rad_dir  = angle(complex(x0-x1, y0-y1));
            adj_dir  = angle(complex(x2-x1, y2-y1));
            td = sign(sin(adj_dir-rad_dir))*pi/2 + rad_dir;
        end
    else
        td = angle(complex(crv(end,1)-crv(1,1), crv(end,2)-crv(1,2)));
    end
    direction(ii) = td * 180/pi;
end
ang = abs(direction(1)-direction(2));
end
