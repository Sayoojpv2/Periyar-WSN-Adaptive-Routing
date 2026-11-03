function plot_v2_final_v1style(outDir)
% PLOT_V2_FINAL_V1STYLE Alternative V2 figure set in the Version 1 dashboard arrangement
% (RL_WSN_Periyar/run_final_project_dashboards.m, graphs 1-5, and
% RL_WSN_Periyar/figures/generate_rl_hybrid_learning_comparison.m, graph 6): 2 x 3 tiles with three line
% panels (operational nodes, cumulative delivery ratio, total residual energy; mean and +-1 sample SD band),
% two bar panels (energy per delivered report; operational nodes at the last round; mean bar, SD whisker
% and the five seed values as circles) and a five-seed summary table. V1 sizes, fonts and colours.
% Data: results/v2_results_500rounds_report_preserving.mat (seeds 42-46, rounds 1-500). Nothing is simulated.
% Writes figures/V1_STYLE_ALTERNATIVE/V2_G#_V1Style.png (300 dpi) and .fig; never overwrites.
    cfg = v2_config(); addpath(fullfile(cfg.root, 'simulation'));
    if nargin < 1, outDir = fullfile(cfg.root, 'figures', 'V1_STYLE_ALTERNATIVE'); end
    S = load(fullfile(cfg.root, 'results', 'v2_results_500rounds_report_preserving.mat'), 'R'); R = S.R;
    Rn = R.config.numRounds; V = R.workload.available; N = R.config.numNodes;
    W = create_periyar_workload_continuous(load_consdata_xlsx(cfg.workbook), Rn, 'uniform', R.config.sensingNodes, N, R.config.scheduleRounds);
    validCum = cumsum(cellfun(@numel, W.round_obs));
    assert(validCum(end) == V && all(validCum > 0), 'workload does not match the results file');
    P = R.protocols; names = {P.name};
    for k = 1:numel(P), P(k).cumDR = 100 * P(k).cumDelivered ./ validCum; end   % cumulative valid-report delivery ratio
    if ~isfolder(outDir), mkdir(outDir); end

    blue = [0.12 0.47 0.71]; orange = [0.85 0.37 0.01];
    note = sprintf(['Mean of seeds 42-46; bands and whiskers \\pm 1 sample SD; circles = seed values (descriptive). ' ...
        'Delivery ratio = delivered / valid reports (%s). Operational = protocol-duty availability. V2: 10 nodes, rounds 1-%d.'], commas(V), Rn);
    G = {   % file, title, members (V1 order), V1 colours, paired (RL, parent), footnote suffix, V1 template
        'V2_G1_V1Style', 'Conventional Protocols vs RL-Hybrid', {'LEACH', 'DEEC', 'PEGASIS', 'RL-Hybrid'}, ...
            {blue, [0.17 0.63 0.17], [0.58 0.40 0.74], [0.80 0.15 0.45]}, [], '', 1
        'V2_G2_V1Style', 'LEACH vs LEACH-RL v5', {'LEACH', 'LEACH-RL v5'}, {blue, orange}, [2 1], ' LEACH-RL: v5.', 1
        'V2_G3_V1Style', 'DEEC vs DEEC-RL v5', {'DEEC', 'DEEC-RL v5'}, {blue, [0.11 0.62 0.47]}, [2 1], ' DEEC-RL: v5.', 1
        'V2_G4_V1Style', 'PEGASIS vs PEGASIS-RL v5', {'PEGASIS', 'PEGASIS-RL v5'}, {blue, orange}, [2 1], ' PEGASIS-RL: v5.', 1
        'V2_G5_V1Style', 'RL-Hybrid vs LEACH-RL v5 vs DEEC-RL v5 vs PEGASIS-RL v5', {'RL-Hybrid', 'LEACH-RL v5', 'DEEC-RL v5', 'PEGASIS-RL v5'}, ...
            {[0.80 0.15 0.45], orange, [0.46 0.44 0.70], [0.11 0.62 0.47]}, [], ' Different architectures, not a paired comparison.', 1
        'V2_G6_V1Style', 'RL-Hybrid: Q-Learning Method Comparison', {'Tabular Q', 'Double Q', 'Double DQN', 'Dueling Double DQN'}, ...
            {[0.85 0.10 0.55], [0.00 0.40 0.85], [0.00 0.62 0.45], [0.95 0.50 0.00]}, [], ' Same RL-Hybrid problem for all four learners; Tabular Q = RL-Hybrid.', 2};
    for g = 1:size(G, 1)
        base = fullfile(outDir, G{g, 1});
        assert(~isfile([base '.png']) && ~isfile([base '.fig']), '%s already exists; not overwriting', base);
        M = P(cellfun(@(n) find(strcmp(names, n)), G{g, 3}));
        fig = dashboard(M, G{g, 4}, G{g, 2}, [note G{g, 6}], G{g, 5}, G{g, 7}, Rn, N, V);
        % Rendered about 3,750 px wide: at a full 300 dpi on V1's 1800-2300 px canvases the renderer truncated
        % or garbled text in the text-heavy figures. Saved with 300-dpi print metadata (about 12.5 in wide).
        dpi = round(3750 / fig.Position(3) * get(groot, 'ScreenPixelsPerInch'));
        tmp = [tempname '.png']; exportgraphics(fig, tmp, 'Resolution', dpi); I = imread(tmp); delete(tmp);
        imwrite(I, [base '.png'], 'ResolutionUnit', 'meter', 'XResolution', round(300 / 0.0254), 'YResolution', round(300 / 0.0254));
        fig.CreateFcn = 'set(gcbo, ''Visible'', ''on'')';    % .fig opens visible
        savefig(fig, [base '.fig']); close(fig);
    end
end

function fig = dashboard(M, cols, ttl, note, pairIdx, tpl, Rn, N, V)
% V1 template 1 (graphs 1-5) or 2 (graph 6, larger fonts).
    if tpl == 1
        z = struct('pos', [50 50 1800 1050], 'title', 15, 'ax', [], 'lab', [], 'sub', [], 'lw', 2, 'alpha', 0.15, ...
            'leg', {{'best', 'best', 'best'}}, 'legFs', [], 'table', 11, 'note', 9);
    else
        z = struct('pos', [30 30 2300 1350], 'title', 17, 'ax', 12, 'lab', 13, 'sub', 14, 'lw', 2.4, 'alpha', 0.12, ...
            'leg', {{'southwest', 'northeast', 'southwest'}}, 'legFs', 12, 'table', 13, 'note', 11);
    end
    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'pixels', 'Position', z.pos);
    if exist('theme', 'file'), theme(fig, 'light'); end
    tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    tl.OuterPosition = [0 0.05 1 0.95];   % thin strip for the one-line footnote (as V1)
    title(tl, ttl, 'FontWeight', 'bold', 'FontSize', z.title);
    r = 1:Rn;
    spec = {'alive', 'Operational nodes', '1. Operational nodes vs round', [0 N]
            'cumDR', 'Cumulative delivery ratio (%)', '2. Cumulative delivery ratio vs round', [0 100]
            'energy', 'Total residual energy (J)', '3. Total residual energy vs round', [0 2 * N]};
    for p = 1:3
        ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
        if ~isempty(z.ax), set(ax, 'FontSize', z.ax); end
        lo = inf; hi = -inf;
        for k = 1:numel(M)   % all bands first, so every mean line is drawn on top of every band
            X = M(k).(spec{p, 1}); mu = mean(X, 1); sd = std(X, 0, 1);
            fill(ax, [r fliplr(r)], [mu + sd fliplr(mu - sd)], cols{k}, 'FaceAlpha', z.alpha, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            lo = min(lo, min(mu - sd)); hi = max(hi, max(mu + sd));
        end
        for k = 1:numel(M)
            plot(ax, r, mean(M(k).(spec{p, 1}), 1), 'Color', cols{k}, 'LineWidth', z.lw, 'DisplayName', M(k).name);
        end
        yl = spec{p, 4}; ylim(ax, [min(yl(1), floor(lo)) max(yl(2), ceil(hi))]);   % never hides data (as V1)
        xlim(ax, [1 Rn]); xlabel(ax, 'Round'); ylabel(ax, spec{p, 2}); title(ax, spec{p, 3});
        if ~isempty(z.lab), ax.XLabel.FontSize = z.lab; ax.YLabel.FontSize = z.lab; ax.Title.FontSize = z.sub; end
        lg = legend(ax, 'Location', z.leg{p}, 'Interpreter', 'none'); if ~isempty(z.legFs), lg.FontSize = z.legFs; end
    end
    barPanel(nexttile(tl), M, cols, 'epr', 'Energy per delivered report (J/report)', '4. Energy per delivered report', z);
    barPanel(nexttile(tl), M, cols, 'finalAlive', sprintf('Operational nodes at round %d', Rn), '5. Final operational nodes', z);
    axT = nexttile(tl); axis(axT, 'off');
    summaryTable(axT, M, cols, pairIdx, z.table, V);
    axF = axes(fig, 'Position', [0 0 1 0.05], 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], 'HitTest', 'off');
    text(axF, 0.5, 0.5, note, 'FontSize', z.note, 'Interpreter', 'tex', 'Color', [0.25 0.25 0.25], 'HorizontalAlignment', 'center');
    checkText(tl, axT, axF, ttl);
end

function barPanel(ax, M, cols, field, ylab, ttl, z)
% V1 bar panel: mean bar, +-1 sample SD whisker, the five seed values as circles.
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on'); n = numel(M);
    if ~isempty(z.ax), set(ax, 'FontSize', z.ax); end
    for k = 1:n
        v = M(k).(field); mu = mean(v); sd = std(v, 0);
        bar(ax, k, mu, 0.6, 'FaceColor', cols{k}, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
        errorbar(ax, k, mu, sd, 'k', 'LineWidth', 1.2, 'CapSize', 12);
        plot(ax, k + linspace(-0.15, 0.15, numel(v)), v, 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 5);
    end
    set(ax, 'XTick', 1:n, 'XTickLabel', {M.name}, 'TickLabelInterpreter', 'none'); xlim(ax, [0.4 n + 0.6]);
    top = max(arrayfun(@(k) max([M(k).(field), mean(M(k).(field)) + std(M(k).(field), 0)]), 1:n));
    ylim(ax, [0 1.1 * top]);   % headroom so no seed circle sits on the frame (as V1)
    ylabel(ax, ylab); title(ax, ttl);
    if ~isempty(z.lab), ax.YLabel.FontSize = z.lab; ax.Title.FontSize = z.sub; end
end

function summaryTable(ax, M, cols, pairIdx, fs, V)
% V1 table layout (colour squares, aligned columns, paired note) with the V2 summary metrics.
    hold(ax, 'on'); set(ax, 'XLim', [0 1], 'YLim', [0 1]);
    text(ax, 0, 0.98, 'Five-seed summary at round 500 (mean \pm SD, n = 5)', 'FontWeight', 'bold', 'FontSize', fs + 1, 'VerticalAlignment', 'top');
    xs = [0.42 0.60 0.775 0.935];
    hdr = {{'Delivery', 'ratio (%)'}, {'Source-', 'packet', 'PDR (%)'}, {'Usable', 'energy (J)'}, {'FNF', '(round)'}};
    for c = 1:4, text(ax, xs(c), 0.775, hdr{c}, 'FontSize', fs, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom'); end
    line(ax, [0 1], [0.76 0.76], 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8);
    y = 0.69; dy = 0.105;
    for k = 1:numel(M)
        q = M(k);
        plot(ax, 0.012, y, 's', 'MarkerSize', fs - 1, 'MarkerFaceColor', cols{k}, 'MarkerEdgeColor', 'none');
        text(ax, 0.04, y, q.name, 'FontSize', fs, 'Interpreter', 'none');
        if all(isnan(q.fnd)), fnf = '>500'; else, assert(~any(isnan(q.fnd)), 'partly censored FNF'); fnf = ms(q.fnd, 0); end
        vals = {ms(100 * q.delivered / V, 1), ms(100 * q.delivered ./ q.generated, 1), ms(q.usableResidual, 1), fnf};
        for c = 1:4, text(ax, xs(c), y, vals{c}, 'FontSize', fs, 'HorizontalAlignment', 'center', 'Interpreter', 'none'); end
        y = y - dy;
    end
    if ~isempty(pairIdx)   % paired per-seed difference (same seed, topology, channel draws, workload)
        a = M(pairIdx(1)); b = M(pairIdx(2)); d = 100 * (a.delivered - b.delivered) / V;
        text(ax, 0, y - 0.01, {sprintf('Paired %s - %s:', a.name, b.name), sprintf('delivery ratio %+.2f \\pm %.2f pp,  higher in %d of 5 seeds', ...
            mean(d), std(d, 0), nnz(d > 0))}, 'FontSize', fs - 0.5, 'FontAngle', 'italic', 'Interpreter', 'tex', 'VerticalAlignment', 'top');
    end
end

function checkText(tl, axT, axF, tag)
% Fails before export if a panel title is wider than its panel, table texts overlap or leave the tile,
% or the footnote is wider than the figure.
    drawnow; tol = 1e-3;
    for ax = findobj(tl.Children, 'flat', 'Type', 'axes')'
        if ax == axT, continue; end
        e = ax.Title.Extent; xl = ax.XLim;
        assert(e(1) >= xl(1) - tol * diff(xl) && e(1) + e(3) <= xl(2) + tol * diff(xl), '%s: title "%s" is wider than its panel', tag, ax.Title.String);
    end
    T = findobj(axT, 'Type', 'text'); B = cell2mat(get(T, 'Extent'));
    out = find(~(B(:, 1) >= -tol & B(:, 1) + B(:, 3) <= 1 + tol & B(:, 2) >= -tol & B(:, 2) + B(:, 4) <= 1 + tol));
    assert(isempty(out), '%s: table text leaves its tile: %s', tag, strjoin(arrayfun(@(i) sprintf('"%s" [%s]', strjoin(cellstr(T(i).String), ' '), mat2str(B(i, :), 3)), out', 'UniformOutput', false), '; '));
    for i = 1:numel(T)
        for j = i + 1:numel(T)
            ov = min(B(i, 1) + B(i, 3), B(j, 1) + B(j, 3)) - max(B(i, 1), B(j, 1)) > tol && min(B(i, 2) + B(i, 4), B(j, 2) + B(j, 4)) - max(B(i, 2), B(j, 2)) > tol;
            assert(~ov, '%s: table texts "%s" and "%s" overlap', tag, strjoin(cellstr(T(i).String), ' '), strjoin(cellstr(T(j).String), ' '));
        end
    end
    e = findobj(axF, 'Type', 'text').Extent; assert(e(1) >= 0 && e(1) + e(3) <= 1, '%s: footnote wider than the figure', tag);
end

function s = ms(x, dp)
    s = sprintf(['%.' num2str(dp) 'f ' char(177) ' %.' num2str(dp) 'f'], mean(x), std(x, 0));
end

function s = commas(n)
    s = regexprep(sprintf('%d', n), '\d(?=(\d{3})+$)', '$0,');
end
