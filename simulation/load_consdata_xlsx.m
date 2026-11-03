function [wq_data, info] = load_consdata_xlsx(xlsxPath)
% LOAD_CONSDATA_XLSX Reads the original KSPCB workbook (consdata.xlsx) directly.
% One worksheet per monitoring station, wide format: parameters in rows, sampling
% dates in columns (one or more tables per sheet). Returns the table expected by
% create_periyar_workload_continuous: Location, Date, Temperature, pH,
% Conductivity, TDS, DO (full-precision workbook values, NaN where not numeric).
%
% Record rules (they give 1273 records from 10 stations):
%   - station name = text after "Sampling point:" on the sheet;
%   - date header dd.mm.yy or dd.mm.yyyy; only dates 01-Jan-2025 .. 31-Jul-2025 are kept
%     (headers with mistyped years, e.g. 31.05.2026 or 18.05.20285, are excluded and logged);
%   - a column is a record when at least one of the five parameters is numeric;
%   - a date repeated within a station (second table on the VETTUKADAVU sheet) keeps its
%     first occurrence; repeats are logged.
    if nargin < 1 || isempty(xlsxPath)
        cfg = wsn_config(); xlsxPath = cfg.workbook;
    end
    assert(isfile(xlsxPath), 'Workbook not found: %s', xlsxPath);
    labels = {'Temperature', 'pH', 'Conductivity', 'TDS', 'D.O.'};
    names  = {'Temperature', 'pH', 'Conductivity', 'TDS', 'DO'};
    window = [datetime(2025,1,1), datetime(2025,7,31)];

    sheets = sheetnames(xlsxPath);
    loc = {}; dt = NaT(0,1); vals = zeros(0,5);
    excluded = cell(0,4);   % station, sheet, header text, reason
    stationOfSheet = strings(numel(sheets),1);
    for k = 1:numel(sheets)
        C = readcell(xlsxPath, 'Sheet', sheets(k));
        nr = size(C,1); nc = size(C,2);
        col1 = arrayfun(@(r) txt(C{r,1}), 1:nr, 'UniformOutput', false);
        sp = find(startsWith(col1, 'Sampling point'), 1);
        assert(~isempty(sp), 'No "Sampling point" on sheet %s', sheets(k));
        station = strtrim(extractAfter(col1{sp}, ':'));
        stationOfSheet(k) = station;
        col2 = repmat({''}, 1, nr);
        if nc >= 2, col2 = arrayfun(@(r) txt(C{r,2}), 1:nr, 'UniformOutput', false); end
        hdr = find(startsWith(lower(col1), 'sl') | strcmp(col2, 'Parameter'));
        seen = NaT(0,1);
        for h = 1:numel(hdr)
            r0 = hdr(h); r1 = nr; if h < numel(hdr), r1 = hdr(h+1) - 1; end
            prow = zeros(1,5);
            for r = r0+1:r1
                j = find(strcmp(labels, col2{r}), 1);
                if ~isempty(j) && prow(j) == 0, prow(j) = r; end
            end
            for c = 4:nc
                raw = C{r0,c};
                if isMissingCell(raw), continue; end
                v = NaN(1,5);
                for j = 1:5
                    if prow(j) > 0, v(j) = numval(C{prow(j),c}); end
                end
                if all(isnan(v)), continue; end
                d = parseDate(raw);
                if isnat(d) || d < window(1) || d > window(2)
                    excluded(end+1,:) = {station, char(sheets(k)), txt(raw), 'date outside Jan-Jul 2025 or unreadable'}; %#ok<AGROW>
                    continue;
                end
                if any(seen == d)
                    excluded(end+1,:) = {station, char(sheets(k)), txt(raw), 'repeated date (first occurrence kept)'}; %#ok<AGROW>
                    continue;
                end
                seen(end+1,1) = d; %#ok<AGROW>
                loc{end+1,1} = station; dt(end+1,1) = d; vals(end+1,:) = v; %#ok<AGROW>
            end
        end
    end
    wq_data = table(loc, dt, vals(:,1), vals(:,2), vals(:,3), vals(:,4), vals(:,5), ...
        'VariableNames', [{'Location', 'Date'}, names]);
    wq_data = sortrows(wq_data, {'Date', 'Location'});

    info = struct();
    info.workbook = xlsxPath;
    info.sheets = cellstr(sheets);
    info.stationOfSheet = cellstr(stationOfSheet);
    info.records = height(wq_data);
    info.excluded = cell2table(excluded, 'VariableNames', {'Station', 'Sheet', 'Header', 'Reason'});
    info.rules = 'Jan-Jul 2025 window; record if any of 5 parameters numeric; first occurrence of a repeated date kept';
end

function s = txt(x)
    if ischar(x) || isstring(x), s = strtrim(char(x));
    elseif isdatetime(x), s = char(x, 'dd.MM.yyyy');
    elseif isnumeric(x) && isscalar(x) && ~isnan(x), s = num2str(x, 15);
    else, s = '';
    end
end

function tf = isMissingCell(x)
    tf = (isa(x, 'missing')) || ((ischar(x) || isstring(x)) && strlength(strtrim(string(x))) == 0);
end

function v = numval(x)
    v = NaN;
    if isnumeric(x) && isscalar(x), v = double(x);
    elseif (ischar(x) || isstring(x)) && ~isempty(regexp(strtrim(char(x)), '^-?\d+(\.\d+)?$', 'once'))
        v = str2double(strtrim(char(x)));
    end
end

function d = parseDate(x)
    d = NaT;
    if isdatetime(x), d = dateshift(x, 'start', 'day'); return; end
    if isnumeric(x) && isscalar(x) && x > 30000, d = datetime(x, 'ConvertFrom', 'excel'); return; end
    t = regexp(txt(x), '^(\d{1,2})[./-](\d{1,2})[./-](\d{2,5})$', 'tokens', 'once');
    if isempty(t), return; end
    dd = str2double(t{1}); mm = str2double(t{2}); yy = str2double(t{3});
    if yy < 100, yy = yy + 2000; end
    if mm < 1 || mm > 12 || dd < 1 || dd > eomday(yy, mm), return; end
    d = datetime(yy, mm, dd);
end
