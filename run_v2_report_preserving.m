function R = run_v2_report_preserving()
% RUN_V2_REPORT_PRESERVING All 11 methods of the final figure set (seeds 42-46, rounds 1-500) under one common
% energy model in which a frame carrying m station reports is charged as m packets (the delivery metric
% counts every report). Only the frames that previously carried several reports in one packet change:
% LEACH/DEEC(-RL v5) cluster head -> BS (protocols/*_RP.m), PEGASIS(-RL v5) chain and leader -> BS
% (protocols/*_EA.m with reportPreservingPackets, plus the leader-aggregation double-count fix), and the
% RL-Hybrid / learner relay route (protocols/RL_Hybrid*_RP.m). Rewards, states, learning parameters,
% topology, workload, radio parameters, 2 J and the death rule are unchanged.
% Saves results/v2_results_500rounds_report_preserving.mat; never overwrites.
    cfg = v2_config(); cfg.numRounds = 500;
    addpath(fullfile(cfg.root, 'protocols'), fullfile(cfg.root, 'simulation'));
    out = fullfile(cfg.root, 'results', 'v2_results_500rounds_report_preserving.mat');
    assert(~isfile(out), '%s exists; not overwriting', out);
    B = load(fullfile(cfg.root, 'results', 'v2_results_500rounds_corrected_detailed.mat'), 'R'); base = B.R;
    W = create_periyar_workload_continuous(load_consdata_xlsx(cfg.workbook), cfg.numRounds, 'uniform', cfg.sensingNodes, cfg.numNodes, cfg.scheduleRounds);
    assert(sum(cellfun(@numel, W.round_obs)) == base.workload.available);
    ea = struct('chargeFailedRx', false, 'singleLeaderAggregation', true, 'reportPreservingPackets', true);
    names = {'LEACH', 'DEEC', 'PEGASIS', 'LEACH-RL v5', 'DEEC-RL v5', 'PEGASIS-RL v5', 'RL-Hybrid', ...
        'Tabular Q', 'Double Q', 'Double DQN', 'Dueling Double DQN'};
    runners = {
        @(e, lq, sh, sd) LEACH_RP(e, lq, sh, sd, W)
        @(e, lq, sh, sd) DEEC_RP(e, lq, sh, sd, W)
        @(e, lq, sh, sd) PEGASIS_EA(e, lq, sh, sd, W, ea)
        @(e, lq, sh, sd) RL_LEACH_v5_RP(e, [], sh, sd, W, 'active')
        @(e, lq, sh, sd) RL_DEEC_v5_RP(e, [], sh, sd, W, 'active', struct('beta', 16))
        @(e, lq, sh, sd) RL_PEGASIS_v5_EA(e, [], sh, sd, W, 'active', struct('energyGuard', 0.5), ea)
        @(e, lq, sh, sd) RL_Hybrid_RP(e, [], sh, sd, W)
        @(e, lq, sh, sd) RL_Hybrid_LearnerComparison_RP(e, [], sh, sd, W, 'tabularQ')
        @(e, lq, sh, sd) RL_Hybrid_LearnerComparison_RP(e, [], sh, sd, W, 'doubleQ')
        @(e, lq, sh, sd) RL_Hybrid_LearnerComparison_RP(e, [], sh, sd, W, 'doubleDQN')
        @(e, lq, sh, sd) RL_Hybrid_LearnerComparison_RP(e, [], sh, sd, W, 'duelingDoubleDQN')};
    series = {'alive', 'dead', 'energy', 'cumDelivered', 'cumPDR', 'delay', 'cumMeanDelay', ...
        'usableEnergy', 'generatedPerRound', 'attemptedPerRound', 'lostPerRound', 'unattemptedPerRound'};
    scal = {'pdr', 'delivered', 'scheduled', 'residual', 'epr', 'meanDelay', 'meanDelayPacket', 'fnd', 'hnd', 'lnd', 'finalAlive', 'energyClosureError', ...
        'generated', 'attempted', 'lost', 'unattempted', 'usableResidual'};
    seeds = cfg.seeds; S = numel(seeds); P = numel(names); Rn = cfg.numRounds; N = cfg.numNodes;
    for p = 1:P
        prot(p).name = names{p}; %#ok<AGROW>
        for f = series, prot(p).(f{1}) = zeros(S, Rn); end
        for f = scal, prot(p).(f{1}) = zeros(1, S); end
        prot(p).nodeEnergy = zeros(S, N, Rn); prot(p).nodeAlive = false(S, N, Rn); prot(p).runtime = zeros(1, S);
    end
    tabularEqualsHybrid = false(1, S);
    for s = 1:S
        seed = seeds(s); [env, lq, sh] = generate_scenario_v2(seed, cfg);
        assert(isequal(env.X, base.positions(s).X) && isequal(env.Y, base.positions(s).Y), 'topology differs');
        for p = 1:P
            t = tic; res = runners{p}(env, lq, sh, seed); prot(p).runtime(s) = toc(t);
            m = extract_v2_metrics(res, env);
            assert(m.energyClosureError < 1e-8, '%s seed %d: energy does not close', names{p}, seed);
            assert(all(m.nodeEnergy(:) >= 0) && all(diff(m.nodeEnergy, 1, 2) <= 1e-15, 'all') && all(diff(double(m.nodeAlive), 1, 2) <= 0, 'all'), '%s: energy/status monotonicity', names{p});
            for f = series, prot(p).(f{1})(s, :) = m.(f{1}); end
            for f = scal, prot(p).(f{1})(s) = m.(f{1}); end
            prot(p).nodeEnergy(s, :, :) = reshape(m.nodeEnergy, 1, N, Rn); prot(p).nodeAlive(s, :, :) = reshape(m.nodeAlive, 1, N, Rn);
            if strcmp(names{p}, 'RL-Hybrid'), hyb = m; elseif strcmp(names{p}, 'Tabular Q'), tabularEqualsHybrid(s) = isequal(m.cumDelivered, hyb.cumDelivered) && isequal(m.energy, hyb.energy); end
            fprintf('seed %d %-19s delivered %4d  operational r500 %2d  FNF %4g  residual %5.2f J  (%.1f s)\n', seed, names{p}, m.delivered, m.finalAlive, m.fnd, m.residual, prot(p).runtime(s));
        end
    end
    R = struct('config', cfg, 'protocols', prot, 'positions', base.positions, 'workload', base.workload, 'tabularQEqualsRLHybrid', tabularEqualsHybrid, ...
        'energyModel', 'first-order radio model; a frame carrying m station reports is charged as m packets (report-preserving); PEGASIS leader fusion charged once', ...
        'pegasisEnergyAudit', ea, 'created', datetime('now'));
    save(out, 'R', '-v7.3'); fprintf('saved %s\n', out);
end
