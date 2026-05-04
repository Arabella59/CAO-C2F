function [pts, scale_factor] = mm_loadGTcsv(csv_path, scale_factor)
%MM_LOADGTCSV  Load ImageJ Results CSV and return [x y] landmark coordinates.
%
%   [pts, scale_factor] = mm_loadGTcsv(csv_path, scale_factor)
%
%   ImageJ exports: col1=index/blank, col2=Area, col3=X, col4=Y.
%   Returns pts as [N x 2] array of [x y] = [col row] coordinates.
%   scale_factor: if supplied, coordinates are multiplied by it (to match
%   a resized image).  Returned for convenience.

if nargin < 2, scale_factor = 1; end

try
    raw = readmatrix(csv_path);
catch
    try
        raw = readmatrix(csv_path, 'NumHeaderLines', 1);
    catch ME
        error('mm_loadGTcsv: cannot read "%s": %s', csv_path, ME.message);
    end
end

% Remove rows that are entirely NaN (header artefacts from readmatrix)
raw = raw(~all(isnan(raw),2), :);

if size(raw,2) < 4
    error('mm_loadGTcsv: expected at least 4 columns, got %d in "%s"', ...
        size(raw,2), csv_path);
end

% ImageJ format: col 3 = X (horizontal), col 4 = Y (vertical)
x = raw(:,3);
y = raw(:,4);
pts = [x, y] * scale_factor;
end
