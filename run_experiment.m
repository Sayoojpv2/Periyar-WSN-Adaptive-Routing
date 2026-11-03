function R = run_experiment(outFile)
% RUN_EXPERIMENT All 11 methods (seeds 42-46, rounds 1-500) under one common energy model in which a
% frame carrying m station reports is charged as m packets (the delivery metric counts every report). This
% applies to the cluster head -> BS frames of LEACH, DEEC, RL-LEACH and RL-DEEC, the chain and leader -> BS
% frames of PEGASIS and RL-PEGASIS (reportPreservingPackets, with the leader fusion charged once), and the
% relay route of RL-Hybrid and its learners. Rewards, states, learning parameters, topology, workload, radio
% parameters, 2 J and the death rule are the same for all methods.
% outFile (default results/results_500rounds.mat) is never overwritten. Positions and workload are checked
% against results/reference_layout.mat.
    cfg = wsn_config(); cfg.numRounds = 500;
    addpath(fullfile(cfg.root, 'protocols'), fullfile(cfg.root, 'simulation'));
    out = fullfile(cfg.root, 'results', 'results_500rounds.mat'); if nargin > 0, out = outFile; end
    assert(~isfile(out), '%s exists; not overwriting', out);
    base = load(fullfile(cfg.root, 'results', 'reference_layout.mat'), 'positions', 'workload');
    W = create_periyar_workload_continuous(load_consdata_xlsx(cfg.workbook), cfg.numRounds, 'uniform', cfg.sensingNodes, cfg.numNodes, cfg.scheduleRounds);
    assert(sum(cellfun(@numel, W.round_obs)) == base.workload.available);
    ea = struct('chargeFailedRx', false, 'singleLeaderAggregation', true, 'reportPreservingPackets', true);
    names = {'LEACH', 'DEEC', 'PEGASIS', 'RL-LEACH', 'RL-DEEC', 'RL-PEGASIS', 'RL-Hybrid', ...
        'Tabular Q-learning', 'Double Q-learning', 'Double DQN', 'Dueling DDQN'};
    runners = {
        @(e, lq, sh, sd) LEACH(e, lq, sh, sd, W)
        @(e, lq, sh, sd) DEEC(e, lq, sh, sd, W)
        @(e, lq, sh, sd) PEGASIS(e, lq, sh, sd, W, ea)
        @(e, lq, sh, sd) RL_LEACH(e, [], sh, sd, W, 'active')
        @(e, lq, sh, sd) RL_DEEC(e, [], sh, sd, W, 'active', struct('beta', 16))
        @(e, lq, sh, sd) RL_PEGASIS(e, [], sh, sd, W, 'active', struct('energyGuard', 0.5), ea)
        @(e, lq, sh, sd) RL_Hybrid(e, [], sh, sd, W)
        @(e, lq, sh, sd) RL_Hybrid_LearnerComparison(e, [], sh, sd, W, 'tabularQ')
        @(e, lq, sh, sd) RL_Hybrid_LearnerComparison(e, [], sh, sd, W, 'doubleQ')
        @(e, lq, sh, sd) RL_Hybrid_LearnerComparison(e, [], sh, sd, W, 'doubleDQN')
        @(e, lq, sh, sd) RL_Hybrid_LearnerComparison(e, [], sh, sd, W, 'duelingDoubleDQN')};
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
        seed = seeds(s); [env, lq, sh] = generate_scenario(seed, cfg);
        assert(isequal(env.X, base.positions(s).X) && isequal(env.Y, base.positions(s).Y), 'topology differs');
        for p = 1:P
            t = tic; res = runners{p}(env, lq, sh, seed); prot(p).runtime(s) = toc(t);
            m = extract_metrics(res, env);
            assert(m.energyClosureError < 1e-8, '%s seed %d: energy does not close', names{p}, seed);
            assert(all(m.nodeEnergy(:) >= 0) && all(diff(m.nodeEnergy, 1, 2) <= 1e-15, 'all') && all(diff(double(m.nodeAlive), 1, 2) <= 0, 'all'), '%s: energy/status monotonicity', names{p});
            for f = series, prot(p).(f{1})(s, :) = m.(f{1}); end
            for f = scal, prot(p).(f{1})(s) = m.(f{1}); end
            prot(p).nodeEnergy(s, :, :) = reshape(m.nodeEnergy, 1, N, Rn); prot(p).nodeAlive(s, :, :) = reshape(m.nodeAlive, 1, N, Rn);
            if strcmp(names{p}, 'RL-Hybrid'), hyb = m; elseif strcmp(names{p}, 'Tabular Q-learning'), tabularEqualsHybrid(s) = isequal(m.cumDelivered, hyb.cumDelivered) && isequal(m.energy, hyb.energy); end
            fprintf('seed %d %-19s delivered %4d  operational r500 %2d  FNF %4g  residual %5.2f J  (%.1f s)\n', seed, names{p}, m.delivered, m.finalAlive, m.fnd, m.residual, prot(p).runtime(s));
        end
    end
    R = struct('config', cfg, 'protocols', prot, 'positions', base.positions, 'workload', base.workload, 'tabularQEqualsRLHybrid', tabularEqualsHybrid, ...
        'energyModel', 'first-order radio model; a frame carrying m station reports is charged as m packets (report-preserving); PEGASIS leader fusion charged once', ...
        'pegasisEnergyAudit', ea, 'created', datetime('now'));
    save(out, 'R', '-v7.3'); fprintf('saved %s\n', out);
end
