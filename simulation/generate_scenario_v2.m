function [wsn_env, lq_history, success_history] = generate_scenario_v2(seed, cfg, mode)
% GENERATE_SCENARIO_V2 V1's generate_scenario with the node set as an input.
% Field, base station, radio parameters and the channel law/stream are V1's.
%   mode 'v1Layout'           : V1 exactly (50 seeded random nodes) - used for regression checks
%   mode 'v1StationPositions' : the 10 station nodes of V1's seeded layout, relays removed
%   mode 'custom'             : cfg.customX / cfg.customY (metres)
    if nargin < 3 || isempty(mode), mode = cfg.topology; end
    rng(seed, 'twister');

    wsn_env.riverLength = 600;
    wsn_env.riverWidth = 50;
    wsn_env.BS_X = wsn_env.riverLength;
    wsn_env.BS_Y = wsn_env.riverWidth / 2;

    % V1 draws (same order and count as V1 generate_scenario)
    X = rand(1, 50) * wsn_env.riverLength;
    Y = rand(1, 50) * wsn_env.riverWidth;
    [X, sort_idx] = sort(X);
    Y = Y(sort_idx);
    switch mode
        case 'v1Layout'
            wsn_env.topologyNote = 'V1 seeded random layout, 50 nodes';
        case 'v1StationPositions'
            X = X(cfg.v1StationNodeIds); Y = Y(cfg.v1StationNodeIds);
            wsn_env.topologyNote = ['SIMULATION ASSUMPTION: positions of V1 station nodes 4,8,...,48 ' ...
                'in the seeded 600 m x 50 m layout; relays removed; not geographic coordinates'];
        case 'custom'
            X = cfg.customX(:)'; Y = cfg.customY(:)';
            assert(numel(X) == cfg.numNodes && numel(Y) == cfg.numNodes, 'custom positions must have numNodes entries');
            wsn_env.topologyNote = 'custom positions from v2_config';
        otherwise
            error('Unknown topology mode %s', mode);
    end
    wsn_env.numNodes = numel(X);
    wsn_env.X = X;
    wsn_env.Y = Y;
    wsn_env.topologyMode = mode;

    wsn_env.initialEnergy = 2.0; % 2.0 Joules
    wsn_env.packetSize = 4000; % bits
    wsn_env.numRounds = cfg.numRounds;  % simulation horizon (1000 in v2_config; run_v2 opts.numRounds may override)
    wsn_env.budgetHorizonRounds = cfg.budgetHorizonRounds;  % RL energy-planning horizon, fixed (not tied to the run length)

    wsn_env.E_elec = 50e-9;
    wsn_env.E_fs = 10e-12;
    wsn_env.E_mp = 0.0013e-12;
    wsn_env.E_DA = 5e-9;
    wsn_env.d0 = sqrt(wsn_env.E_fs / wsn_env.E_mp);

    wsn_env.topologySeed = seed;

    numNodes = wsn_env.numNodes;
    numRounds = wsn_env.numRounds;
    lq_history = zeros(numNodes, numNodes+1, numRounds);
    success_history = false(numNodes, numNodes+1, numRounds);

    for r = 1:numRounds
        rng(seed + r, 'twister');
        for i = 1:numNodes
            for j = 1:numNodes
                if i ~= j
                    d = sqrt((wsn_env.X(i) - wsn_env.X(j))^2 + (wsn_env.Y(i) - wsn_env.Y(j))^2);
                    base_lq = 1.0 - (d / (wsn_env.riverLength * 0.8));
                    noise = 0.3 * (rand() - 0.5);
                    lq_history(i,j,r) = max(0.1, min(0.95, base_lq + noise));
                end
            end
            d_bs = sqrt((wsn_env.X(i) - wsn_env.BS_X)^2 + (wsn_env.Y(i) - wsn_env.BS_Y)^2);
            base_lq_bs = 1.0 - (d_bs / (wsn_env.riverLength * 1.2));
            noise_bs = 0.3 * (rand() - 0.5);
            lq_history(i, numNodes+1, r) = max(0.1, min(0.95, base_lq_bs + noise_bs));
        end
        success_history(:,:,r) = rand(numNodes, numNodes+1) < lq_history(:,:,r);
    end
end
