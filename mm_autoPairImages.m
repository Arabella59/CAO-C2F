function pairs = mm_autoPairImages(folder_path, max_pairs)
%MM_AUTOPAIRIMAGES  Auto-detect and pair thermal / visible images in a folder.
%
%   pairs = mm_autoPairImages(folder_path, max_pairs)
%
%   Scans folder_path for image files (jpg, bmp, png, tif).
%   Pairs files using thermal/visible filename indicators:
%     Thermal: 'IR','thermal','T_','tir','infrared'  (case-insensitive)
%     Visible: 'RGB','visible','V_','vis','color','colour' (case-insensitive)
%
%   Returns: pairs struct array with fields .thermal_path, .visible_path

if nargin < 2, max_pairs = 15; end

pairs = struct('thermal_path',{}, 'visible_path',{});

exts = {'*.jpg','*.jpeg','*.bmp','*.png','*.tif','*.tiff'};
files = {};
for e = 1:length(exts)
    d = dir(fullfile(folder_path, exts{e}));
    for k = 1:length(d)
        files{end+1} = fullfile(d(k).folder, d(k).name); %#ok<AGROW>
    end
end

if isempty(files)
    fprintf('  mm_autoPairImages: no image files found in %s\n', folder_path);
    return;
end

therm_kw = {'IR','thermal','T_','tir','infrared','LWIR'};
vis_kw   = {'RGB','visible','V_','vis','color','colour','VIS'};

thermal_files = {};
visible_files = {};
for k = 1:length(files)
    [~, fname, ~] = fileparts(files{k});
    is_therm = false;  is_vis = false;
    for t = 1:length(therm_kw)
        if contains(fname, therm_kw{t}, 'IgnoreCase', true)
            is_therm = true;  break;
        end
    end
    if ~is_therm
        for v = 1:length(vis_kw)
            if contains(fname, vis_kw{v}, 'IgnoreCase', true)
                is_vis = true;  break;
            end
        end
    end
    if is_therm,      thermal_files{end+1} = files{k}; %#ok<AGROW>
    elseif is_vis,    visible_files{end+1} = files{k}; %#ok<AGROW>
    end
end

fprintf('  Found %d thermal and %d visible files.\n', ...
    length(thermal_files), length(visible_files));

% Pair by matching base name after stripping thermal/visible keywords
used_vis = false(1, length(visible_files));
for t = 1:length(thermal_files)
    if length(pairs) >= max_pairs, break; end
    [~, tname, ~] = fileparts(thermal_files{t});
    % Strip known thermal keywords from name to get base
    base_t = tname;
    for k = 1:length(therm_kw)
        base_t = strrep(base_t, therm_kw{k}, '');
        base_t = strrep(base_t, lower(therm_kw{k}), '');
    end
    base_t = strtrim(regexprep(base_t,'[_\-\s]+',''));

    best_match = -1;  best_score = 0;
    for v = 1:length(visible_files)
        if used_vis(v), continue; end
        [~, vname, ~] = fileparts(visible_files{v});
        base_v = vname;
        for k = 1:length(vis_kw)
            base_v = strrep(base_v, vis_kw{k}, '');
            base_v = strrep(base_v, lower(vis_kw{k}), '');
        end
        base_v = strtrim(regexprep(base_v,'[_\-\s]+',''));

        % Score: length of longest common substring
        score = lcs_length(base_t, base_v);
        if score > best_score
            best_score = score;
            best_match = v;
        end
    end

    if best_match > 0 && best_score > 0
        pairs(end+1).thermal_path = thermal_files{t}; %#ok<AGROW>
        pairs(end).visible_path   = visible_files{best_match};
        used_vis(best_match)      = true;
        fprintf('  Pair %d: %s  <->  %s\n', length(pairs), ...
            fileparts_name(thermal_files{t}), fileparts_name(visible_files{best_match}));
    end
end

if isempty(pairs)
    fprintf('  WARNING: no pairs matched by name. Falling back to index pairing.\n');
    n = min(length(thermal_files), length(visible_files));
    n = min(n, max_pairs);
    for k = 1:n
        pairs(end+1).thermal_path = thermal_files{k}; %#ok<AGROW>
        pairs(end).visible_path   = visible_files{k};
    end
end
end

function n = lcs_length(a, b)
% Longest common substring length between two strings.
a = lower(a);  b = lower(b);
la = length(a); lb = length(b);
M  = zeros(la+1, lb+1);
n  = 0;
for i = 1:la
    for j = 1:lb
        if a(i) == b(j)
            M(i+1,j+1) = M(i,j)+1;
            if M(i+1,j+1) > n, n = M(i+1,j+1); end
        end
    end
end
end

function n = fileparts_name(p)
[~,n,e] = fileparts(p);  n = [n e];
end
