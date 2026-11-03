function plot_v2_final_report_preserving(outDir)
% PLOT_V2_FINAL_REPORT_PRESERVING The six final V2 figures (layout, terminology and style of
% plot_v2_final_rlv5_corrected.m, whose figures are locked) for the common report-preserving energy model:
% a frame carrying m station reports is charged as m packets for every method.
% Data: results/v2_results_500rounds_report_preserving.mat (all 11 methods, seeds 42-46, rounds 1-500).
% Nothing is simulated here. Writes figures/FINAL_SUBMISSION_REPORT_PRESERVING/V2_G#_Final.png (300 dpi) and .fig.
    cfg = v2_config(); addpath(fullfile(cfg.root, 'simulation'));
    if nargin < 1, outDir = fullfile(cfg.root, 'figures', 'FINAL_SUBMISSION_REPORT_PRESERVING'); end
    S = load(fullfile(cfg.root, 'results', 'v2_results_500rounds_report_preserving.mat'), 'R'); R = S.R;
    keep = {'name', 'alive', 'energy', 'cumDelivered', 'cumMeanDelay', 'delivered', 'generated', 'usableResidual', 'fnd'};
    pick = @(Q) orderfields(rmfield(Q, setdiff(fieldnames(Q), keep)));
    P = pick(R.protocols); names = {P.name};
    Rn = R.config.numRounds; N = R.config.numNodes; eMax = N * 2;   % 2 J per node
    V = R.workload.available; r = 1:Rn;
    if ~isfolder(outDir), mkdir(outDir); end

    % Delivery ratio at round t = delivered reports in rounds 1..t / valid reports scheduled in rounds 1..t
    % (cumulative denominator; it reaches the 4,825 valid reports only at round 500).
    W = create_periyar_workload_continuous(load_consdata_xlsx(cfg.workbook), Rn, 'uniform', R.config.sensingNodes, N, R.config.scheduleRounds);
    validCum = cumsum(cellfun(@numel, W.round_obs));
    assert(validCum(end) == V && all(validCum > 0), 'workload does not match the results file');
    for k = 1:numel(P)                                       % in memory only
        P(k).cumPDR = 100 * P(k).cumDelivered ./ validCum;
    end

    groups = {
        'V2_G1', 'RL-Hybrid vs LEACH vs DEEC vs PEGASIS', {'RL-Hybrid', 'LEACH', 'DEEC', 'PEGASIS'}
        'V2_G2', 'LEACH vs LEACH-RL v5', {'LEACH', 'LEACH-RL v5'}
        'V2_G3', 'DEEC vs DEEC-RL v5', {'DEEC', 'DEEC-RL v5'}
        'V2_G4', 'PEGASIS vs PEGASIS-RL v5', {'PEGASIS', 'PEGASIS-RL v5'}
        'V2_G5', 'RL-Hybrid vs LEACH-RL v5 vs DEEC-RL v5 vs PEGASIS-RL v5', {'RL-Hybrid', 'LEACH-RL v5', 'DEEC-RL v5', 'PEGASIS-RL v5'}
        'V2_G6', 'RL-Hybrid: Q-Learning Method Comparison', {'Tabular Q', 'Double Q', 'Double DQN', 'Dueling Double DQN'}};
    for g = 1:size(groups, 1)
        f = fullfile(outDir, [groups{g, 1} '_Final']);
        assert(~isfile([f '.png']) && ~isfile([f '.fig']), '%s already exists; not overwriting', f);
    end
    ymaxT = 250 * ceil(maxUpper(P, 'cumDelivered') / 250);    % same y range for a metric in every figure
    ymaxD = 10 * ceil(maxUpper(P, 'delay') / 10);
    spec = {
        'alive', 'Operational nodes', '1. Operational Nodes vs Round', [0 N], [0 N]
        'energy', 'Total residual battery energy (J)', '2. Total Residual Battery Energy vs Round', [0 eMax], [0 eMax]
        'cumDelivered', 'Cumulative delivered reports', {'3. Cumulative Successfully Delivered', 'Reports vs Round'}, [0 ymaxT], [0 Inf]
        'cumPDR', 'Valid-report delivery ratio (%)', {'4. Cumulative Valid-Report', 'Delivery Ratio vs Round'}, [0 100], [0 100]
        'delay', 'Cumulative mean end-to-end delay (ms)', {'5. Cumulative Packet-Weighted Mean', 'End-to-End Delay vs Round'}, [0 ymaxD], [0 Inf]};
    noteL = {
        ['Lines: mean of seeds 42-46. Bands: ' char(177) '1 sample SD (not confidence intervals).']
        'Operational node: can still perform its protocol role (may retain battery energy).'
        'Panel 2 includes energy stranded in unavailable nodes; usable energy excludes it.'
        sprintf('FNF = first node to become operationally unavailable; >%d = none in %d rounds.', Rn, Rn)
        'Energy: first-order radio model; a frame carrying m reports is charged as m packets.'};
    noteR = {
        sprintf('Delivery ratio = delivered / valid reports scheduled (cumulative; %s by round %d).', commas(V), Rn)
        'Source-packet PDR = delivered / source packets actually generated.'
        'Delay = cumulative packet-weighted mean end-to-end delay.'
        'Panels 4-5: early cumulative values fluctuate strongly while few reports are delivered.'};

    for g = 1:size(groups, 1)
        mem = groups{g, 3}; nm = numel(mem);
        sp = spec; nR = noteR;
        if any(g == 2:4)   % pairwise figures: panel 3 scaled to the pair (both lines share it)
            sp{3, 4} = [0 250 * ceil(maxUpper(P(ismember(names, mem)), 'cumDelivered') / 250)];
            nR{end + 1} = sprintf('Panel 3 y-axis is scaled to this figure (G1, G5, G6: 0-%s).', commas(ymaxT)); %#ok<AGROW>
        else
            nR{end + 1} = sprintf('Panel 3 y-axis: common 0-%s scale shared by G1, G5 and G6.', commas(ymaxT)); %#ok<AGROW>
        end
        top = find(ismember(mem, {'RL-Hybrid', 'Tabular Q'}));
        order = [setdiff(1:nm, top, 'stable') top];          % proposed method drawn last (on top)
        alpha = 0.15 - 0.03 * (nm > 2);                      % V1: 0.15 (graphs 1-5), 0.12 (graph 6)
        fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [0.5 0.5 15 8.75]);
        if exist('theme', 'file'), theme(fig, 'light'); end
        tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        tl.OuterPosition = [0 0.115 1 0.885];                % strip for the definitions footnote
        title(tl, groups{g, 2}, 'FontWeight', 'bold', 'FontSize', 15);
        axT = nexttile(tl, 6);                               % created first so it is never drawn over panel 3's tick labels
        for p = 1:size(spec, 1)
            ax = nexttile(tl, p); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
            st = cell(1, nm);
            for k = 1:nm
                [mu, lo, hi] = seriesStats(P(strcmp(names, mem{k})), sp{p, 1}, sp{p, 5});
                v = [mu hi]; v = v(~isnan(v));
                assert(all(v >= sp{p, 4}(1) & v <= sp{p, 4}(2)), '%s / %s outside the axis range', mem{k}, sp{p, 1});
                st{k} = struct('mu', mu, 'lo', lo, 'hi', hi);
            end
            for k = order, band(ax, r, st{k}.lo, st{k}.hi, styleOf(mem{k}), alpha); end
            for k = order
                s = styleOf(mem{k}); plot(ax, r, st{k}.mu, 'Color', s{1}, 'LineStyle', s{2}, 'LineWidth', 2);
            end
            xlim(ax, [1 Rn]); xticks(ax, [1 100:100:Rn]); ylim(ax, sp{p, 4});
            xlabel(ax, 'Round'); ylabel(ax, sp{p, 2}); title(ax, sp{p, 3});
        end
        summaryTable(axT, P, names, mem, V, g == 6);
        axF = axes(fig, 'Position', [0 0 1 0.115], 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], 'HitTest', 'off');
        foot = {text(axF, 0.012, 0.97, noteL), text(axF, 0.512, 0.97, nR)};
        set([foot{:}], 'FontSize', 11, 'Color', [0.2 0.2 0.2], 'VerticalAlignment', 'top', 'Interpreter', 'none');
        checkText(fig, tl, axT, axF, groups{g, 1});
        base = fullfile(outDir, [groups{g, 1} '_Final']);
        % Rendered at 400 dpi and scaled to 300 dpi: a direct 300-dpi export overlapped the digits of one tick
        % label, and 600 dpi misplaced tick labels.
        tmp = [tempname '.png']; exportgraphics(fig, tmp, 'Resolution', 400);
        I = imresize(imread(tmp), 0.75); delete(tmp);
        imwrite(I, [base '.png'], 'ResolutionUnit', 'meter', 'XResolution', round(300 / 0.0254), 'YResolution', round(300 / 0.0254));
        fig.CreateFcn = 'set(gcbo, ''Visible'', ''on'')';    % .fig opens visible
        savefig(fig, [base '.fig']);
        close(fig);
    end
end

function summaryTable(ax, P, names, mem, V, isLearner)
% Five-seed summary at the last round (V1 tile 6); the line keys double as the legend.
    hold(ax, 'on'); axis(ax, 'off'); ax.Color = 'none'; xlim(ax, [0 1]); ylim(ax, [0 1]); fs = 9.5;
    text(ax, 0, 0.98, ['Five-seed summary at round 500 (mean ' char(177) ' SD, n = 5)'], 'FontWeight', 'bold', 'FontSize', fs + 1.5, 'VerticalAlignment', 'top', 'Interpreter', 'none');
    xs = [0.47 0.62 0.781 0.935];
    hdr = {{'Delivery', 'ratio (%)'}, {'Source-', 'packet', 'PDR (%)'}, {'Usable', 'energy (J)'}, {'FNF', '(round)'}};   % bottom-aligned
    for c = 1:4, text(ax, xs(c), 0.715, hdr{c}, 'FontSize', fs, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom'); end
    line(ax, [0 1], [0.70 0.70], 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8);
    pm = @(x, d) sprintf(['%.' num2str(d) 'f' char(177) '%.' num2str(d) 'f'], mean(x, 'omitnan'), std(x, 0, 'omitnan'));   % compact mean±SD
    for k = 1:numel(mem)
        q = P(strcmp(names, mem{k})); y = 0.635 - 0.085 * (k - 1); s = styleOf(mem{k});
        plot(ax, [0 0.045], [y y], 'Color', s{1}, 'LineStyle', s{2}, 'LineWidth', 2.5);
        text(ax, 0.057, y, mem{k}, 'FontSize', fs, 'Interpreter', 'none');
        if all(isnan(q.fnd)), fnf = '>500'; else, fnf = pm(q.fnd, 0); assert(~any(isnan(q.fnd)), 'partly censored FNF'); end   % NaN = no failure in 500 rounds
        vals = {pm(100 * q.delivered / V, 1), pm(100 * q.delivered ./ q.generated, 1), pm(q.usableResidual, 1), fnf};
        for c = 1:4, text(ax, xs(c), y, vals{c}, 'FontSize', fs, 'HorizontalAlignment', 'center', 'Interpreter', 'none'); end
    end
    y0 = 0.635 - 0.085 * numel(mem) - 0.02;                  % definitions are in the figure footnote
    if numel(mem) == 2
        a = P(strcmp(names, mem{1})); b = P(strcmp(names, mem{2})); d = 100 * (b.delivered - a.delivered) / V;
        note = sprintf('%s - %s, delivery ratio: %+.2f pp (higher in %d/5 seeds)', mem{2}, mem{1}, mean(d), nnz(d > 0));
    elseif isLearner
        note = 'Tabular Q is RL-Hybrid''s own learner (identical runs)';
    else
        note = '';
    end
    if ~isempty(note)
        text(ax, 0, y0, note, 'FontSize', fs - 1, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3], 'Interpreter', 'none');
    end
end

function checkText(fig, tl, axT, axF, tag)
% Fails before export if any text is clipped or overlaps: panel titles wider than their plot box, y labels
% taller than it, summary-table texts overlapping each other or leaving the tile, footnote lines leaving the strip.
    drawnow;
    tol = 1e-3;
    for ax = findobj(tl.Children, 'flat', 'Type', 'axes')'
        if ax == axT, continue; end
        e = ax.Title.Extent; xl = ax.XLim;
        assert(e(1) >= xl(1) - tol * diff(xl) && e(1) + e(3) <= xl(2) + tol * diff(xl), '%s: title "%s" is wider than its panel', tag, strjoin(cellstr(ax.Title.String), ' '));
        e = ax.YLabel.Extent; yl = ax.YLim;
        assert(e(4) <= diff(yl), '%s: y label "%s" is taller than its panel', tag, ax.YLabel.String);
    end
    T = findobj(axT, 'Type', 'text'); B = cell2mat(get(T, 'Extent'));
    assert(all(B(:, 1) >= -tol & B(:, 1) + B(:, 3) <= 1 + tol & B(:, 2) >= -tol & B(:, 2) + B(:, 4) <= 1 + tol), '%s: table text leaves its tile', tag);
    for i = 1:numel(T)
        for j = i + 1:numel(T)
            ov = min(B(i, 1) + B(i, 3), B(j, 1) + B(j, 3)) - max(B(i, 1), B(j, 1)) > tol && ...
                min(B(i, 2) + B(i, 4), B(j, 2) + B(j, 4)) - max(B(i, 2), B(j, 2)) > tol;
            assert(~ov, '%s: table texts "%s" and "%s" overlap', tag, strjoin(cellstr(T(i).String), ' '), strjoin(cellstr(T(j).String), ' '));
        end
    end
    F = findobj(axF, 'Type', 'text'); B = cell2mat(get(F, 'Extent')); [~, i] = sort(B(:, 1)); B = B(i, :);
    assert(all(B(:, 2) >= 0 & B(:, 2) + B(:, 4) <= 1) && B(1, 1) + B(1, 3) < B(2, 1) && B(2, 1) + B(2, 3) <= 1, '%s: footnote does not fit (extents %s)', tag, mat2str(B, 3));
    assert(fig.Position(3) == 15 && fig.Position(4) == 8.75);
end

function s = commas(n)
    s = regexprep(sprintf('%d', n), '\d(?=(\d{3})+$)', '$0,');
end

function v = maxUpper(P, key)
    v = 0;
    for k = 1:numel(P)
        [mu, ~, hi] = seriesStats(P(k), key, [0 Inf]);
        v = max([v mu hi], [], 'omitnan');                  % mean too: early rounds can have a mean but no SD
    end
end

function [mu, lo, hi] = seriesStats(p, key, range)
% Mean and +-1 sample SD across seeds. Delay (ms): per seed, the per-report mean delay of all reports
% delivered in rounds 1..r (stored cumulative values); undefined until a seed has delivered a report.
    if strcmp(key, 'delay')
        Y = 1000 * p.cumMeanDelay;
        k = sum(~isnan(Y), 1);
        mu = mean(Y, 1, 'omitnan'); mu(k == 0) = NaN;
        sd = std(Y, 0, 1, 'omitnan'); sd(k < 2) = NaN;      % sample SD needs at least two seeds
    else
        Y = p.(key); mu = mean(Y, 1); sd = std(Y, 0, 1);
    end
    lo = max(mu - sd, range(1), 'includenan'); hi = min(mu + sd, range(2), 'includenan');
end

function band(ax, x, lo, hi, s, alpha)
% +-1 SD band over each contiguous run of defined values (gaps stay empty, never filled with zeros).
    ok = ~isnan(lo) & ~isnan(hi);
    d = diff([false ok false]); a = find(d == 1); b = find(d == -1) - 1;
    for k = 1:numel(a)
        i = a(k):b(k);
        fill(ax, [x(i) fliplr(x(i))], [hi(i) fliplr(lo(i))], s{1}, 'FaceAlpha', alpha, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

function s = styleOf(name)
% Version 1 colours, one per algorithm in every figure; RL variants and learners use a second line style.
    switch name
        case {'RL-Hybrid', 'Tabular Q'}, s = {[0.80 0.15 0.45], '-'};
        case 'LEACH',                    s = {[0.12 0.47 0.71], '-'};
        case 'DEEC',                     s = {[0.17 0.63 0.17], '-'};
        case 'PEGASIS',                  s = {[0.58 0.40 0.74], '-'};
        case {'LEACH-RL', 'LEACH-RL v5'},     s = {[0.85 0.37 0.01], '--'};
        case {'DEEC-RL', 'DEEC-RL v5'},       s = {[0.46 0.44 0.70], '--'};
        case {'PEGASIS-RL', 'PEGASIS-RL v5'}, s = {[0.11 0.62 0.47], '--'};
        case 'Double Q',                 s = {[0.93 0.69 0.13], '--'};
        case 'Double DQN',               s = {[0.30 0.75 0.93], '-.'};
        case 'Dueling Double DQN',       s = {[0.35 0.35 0.35], ':'};
        otherwise, error('no style for %s', name);
    end
end
