function res = RL_PEGASIS(wsn_env, ~, success_history, seed, app_workload, rl_mode, opts, ea)
% Energy-audit options: behaviour is the plain protocol's unless ea.chargeFailedRx or
% ea.singleLeaderAggregation is true (see eaDefaults); adds a per-node energy ledger (res.energyLedger).
%RL_PEGASIS PEGASIS with RL leader selection without forced rotation.
% Parent: protocols/PEGASIS.m (unchanged). Everything is the parent's code except the leader:
%   PEGASIS: leader = chain(mod(r-1, chain length) + 1), i.e. every node leads once per cycle;
%   RL-PEGASIS: epsilon-greedy over the learned value V(s) of the live chain nodes (shared table, 36
%            states: BS-distance band x E/E_avg band x longer-branch length 1-2, 3-5, >=6 hops).
%            opts.energyGuard g > 0 restricts the choice to nodes with E >= g E_avg when any exist.
% Greedy chain construction, token passing with fusion at every hop, the single leader->BS frame and
% the death rule are PEGASIS's.
% Why: the leader->BS frame is 87-92 % of PEGASIS's energy; a leader 400-550 m from the BS pays
% 0.1-0.5 J per frame at 20-45 % link success. An RL that must still let every node lead once per
% cycle only reorders leaders (mean leader BS distance identical to PEGASIS) and its TD values,
% bootstrapped with gamma = 0.9 over aliased states, grew instead of converging.
% Reward (team, per round, after the outcome): R = (D/A)(1 - e/(e+E_b)); the leader's state value
% updates V(s) += alpha (R - V(s)) (contextual bandit, optimistic initial value).
    if nargin < 6 || isempty(rl_mode); rl_mode = 'active'; end
    if nargin < 7; opts = struct(); end
    assert(any(strcmpi(rl_mode, {'active', 'noop'})), 'Unknown rl_mode %s', rl_mode);
    rl_enabled = strcmpi(rl_mode, 'active');
    cfg = struct('variant', 'RL-PEGASIS', 'mode', rl_mode, 'alpha', 0.10, 'epsilonInitial', 0.20, 'epsilonDecayPerRound', 0.99, ...
        'epsilonMinimum', 0.02, 'vInit', 0.6, 'energyGuard', 0, 'useEnergyReward', true, 'rngSeed', 500000 + seed, ...
        'energyBudgetJ', wsn_env.numNodes * wsn_env.initialEnergy / (numel(app_workload.sensing_nodes) * wsn_env.budgetHorizonRounds));
    for f = fieldnames(opts)', cfg.(f{1}) = opts.(f{1}); end
    rl_stream = RandStream('mt19937ar', 'Seed', cfg.rngSeed);
    rl_V = cfg.vInit * ones(36, 1);
    rl_log = cell(1, wsn_env.numRounds);     % [round leader state explored nCandidates parentLeader reward VBefore]
    m_epsilon = zeros(1, wsn_env.numRounds);
    if nargin < 5; app_workload = []; end
    use_app_workload = ~isempty(app_workload);
    rng(seed, 'twister');
    
    numNodes = wsn_env.numNodes;
    E_residual = ones(1, numNodes) * wsn_env.initialEnergy;
    is_alive = true(1, numNodes);
    has_failed = false(1, numNodes);
    
    ar = wsn_env.numRounds;
    m_alive = zeros(1, ar);
    m_totalEnergy = zeros(1, ar);
    m_scheduled = zeros(1, ar);
    m_source_avail = zeros(1, ar);
    m_source_unavail = zeros(1, ar);
    m_transmitted = zeros(1, ar);
    m_del = zeros(1, ar);
    m_lost = zeros(1, ar);
    m_unattempted = zeros(1, ar);
    m_tx_packets = zeros(1, ar);
    m_tx_energy = zeros(1, ar); m_rx_energy = zeros(1, ar); m_agg_energy = zeros(1, ar);
    m_operational_energy = zeros(1, ar);
    E_residual_history = zeros(numNodes, ar);
    is_alive_history = false(numNodes, ar);
    sensing_nodes = app_workload.sensing_nodes(:)';
    sensing_source_available_history = false(length(sensing_nodes), ar);
    m_delay = zeros(1, ar);
    m_routingEnergy = zeros(1, ar);
    m_externalFailureEnergy = zeros(1, ar);
    m_energyStdDev = zeros(1, ar);
    
    R_bit = 250e3; T_proc = 0.005; c = 3e8;
    FND = NaN; HND = NaN; LND = NaN;
    
    if ~exist('ea', 'var') || isempty(ea), ea = struct(); end
    ea = eaDefaults(ea);
    led = struct('txChainJ', zeros(1, numNodes), 'txBSJ', zeros(1, numNodes), 'rxJ', zeros(1, numNodes), 'aggJ', zeros(1, numNodes), 'nTx', zeros(1, numNodes), 'nTxBS', zeros(1, numNodes), 'nRx', zeros(1, numNodes), 'nRxFailed', zeros(1, numNodes), 'nLinkFail', zeros(1, numNodes), 'reportsForwarded', zeros(1, numNodes), 'nLead', zeros(1, numNodes), 'nTxFail', zeros(1, numNodes), 'nBSFail', zeros(1, numNodes), 'nGenerated', zeros(1, numNodes)); ledHist = cell(1, wsn_env.numRounds);
    chain = buildChain(wsn_env, is_alive);
    last_alive = is_alive;
    
    for r = 1:ar
        % Natural baseline evaluation - no artificial node failure
        m_externalFailureEnergy(r) = 0;
        
        is_alive = is_alive & (E_residual > 0) & ~has_failed;
        num_alive = sum(is_alive);
        energy_start = sum(E_residual);
        
        if num_alive == 0
            if isnan(LND); LND = r-1; end
            m_scheduled(r) = length(app_workload.sensing_nodes);
            m_source_avail(r) = 0; m_source_unavail(r) = m_scheduled(r);
            m_transmitted(r) = 0; m_del(r) = 0; m_lost(r) = 0;
            m_alive(r) = 0; m_totalEnergy(r) = sum(E_residual);
            m_operational_energy(r) = 0; m_routingEnergy(r) = 0;
            E_residual_history(:,r) = E_residual(:); is_alive_history(:,r) = is_alive(:); ledHist{r} = led;
            continue;
        end
        
        if isnan(FND) && num_alive < numNodes; FND = r; end
        if isnan(HND) && num_alive <= numNodes/2; HND = r; end
        
        if any(is_alive ~= last_alive)
            chain = buildChain(wsn_env, is_alive);
            last_alive = is_alive;
        end
        
        % Fetch identical common environment realization (channel outcomes are read only after a frame is paid for)
        succ_matrix = success_history(:,:,r);
        
        % Workload determination & Lifecycle Accounting
        if use_app_workload
            scheduled_this_round = length(app_workload.sensing_nodes); % 10 scheduled opportunities
            obs_today = app_workload.round_obs{r};
            num_today = length(obs_today);
            
            node_reports = zeros(1, numNodes);
            node_unattempted = zeros(1, numNodes);
            source_avail_this_round = 0;
            dead_station_count = 0;
            for o_idx = obs_today
                s_node = app_workload.obs_src_nodes(o_idx);
                if is_alive(s_node)
                    node_reports(s_node) = node_reports(s_node) + 1; led.nGenerated(s_node) = led.nGenerated(s_node) + 1;
                    node_unattempted(s_node) = node_unattempted(s_node) + 1;
                    source_avail_this_round = source_avail_this_round + 1;
                    sensing_source_available_history(sensing_nodes == s_node, r) = true;
                else
                    dead_station_count = dead_station_count + 1;
                end
            end
            
            no_data_count = scheduled_this_round - num_today;
            source_unavail_this_round = no_data_count + dead_station_count;
        else
            scheduled_this_round = numNodes;
            node_reports = zeros(1, numNodes);
            node_reports(is_alive) = 1;
            node_unattempted = double(is_alive);
            source_avail_this_round = sum(is_alive);
            source_unavail_this_round = numNodes - source_avail_this_round;
        end
        
        % Conservation Check: scheduled = source_available + source_unavailable
        assert(scheduled_this_round == source_avail_this_round + source_unavail_this_round, ...
            'Scheduled conservation failed: scheduled != available + unavailable');
        
        m_scheduled(r) = scheduled_this_round;
        m_source_avail(r) = source_avail_this_round;
        m_source_unavail(r) = source_unavail_this_round;
        
        tx_attempts_this_round = 0; tx_packets_this_round = 0; lost_this_round = 0;
        tx_e_round = 0; rx_e_round = 0; agg_e_round = 0;
        
        if source_avail_this_round == 0
            m_alive(r) = num_alive;
            m_totalEnergy(r) = sum(E_residual);
            m_operational_energy(r) = sum(E_residual(is_alive));
            m_del(r) = 0;
            m_lost(r) = 0;
            m_delay(r) = NaN;
            m_routingEnergy(r) = 0;
            E_residual_history(:,r) = E_residual(:); is_alive_history(:,r) = is_alive(:); ledHist{r} = led;
            alive_energies = E_residual(is_alive);
            if length(alive_energies) > 1
                m_energyStdDev(r) = std(alive_energies);
            else
                m_energyStdDev(r) = 0;
            end
            continue;
        end
        
        leader_idx_in_chain = mod(r - 1, length(chain)) + 1;
        leader = chain(leader_idx_in_chain);
        parent_leader = leader;
        if rl_enabled
            epsilon = max(cfg.epsilonMinimum, cfg.epsilonInitial * cfg.epsilonDecayPerRound^(r-1)); m_epsilon(r) = epsilon;
            cands = chain;
            if cfg.energyGuard > 0
                ok = E_residual(chain) >= cfg.energyGuard * mean(E_residual(chain)); if any(ok), cands = chain(ok); end
            end
            cs = pegasisStates(chain, cands, wsn_env, E_residual); v = rl_V(cs)';
            explored = rand(rl_stream) < epsilon;
            if explored, k = randi(rl_stream, numel(cands)); else, best = find(v == max(v)); k = best(randi(rl_stream, numel(best))); end
            leader = cands(k); leader_idx_in_chain = find(chain == leader, 1);
            rl_log{r} = [r, leader, cs(k), double(explored), numel(cands), parent_leader, NaN, v(k)];
        end
        
        led.nLead(leader) = led.nLead(leader) + 1;
        report_delays = zeros(1, numNodes);
        
        for i = 1:(leader_idx_in_chain - 1)
            curr = chain(i);
            next_hop = chain(i+1);
            if is_alive(curr) && is_alive(next_hop)
                [E_residual, node_reports, node_unattempted, report_delays, is_alive, te, re, ae, att, packets, lost, led] = passToken(...
                    curr, next_hop, E_residual, node_reports, node_unattempted, report_delays, ...
                    wsn_env, succ_matrix, R_bit, T_proc, c, is_alive, ea, led);
                tx_e_round=tx_e_round+te; rx_e_round=rx_e_round+re; agg_e_round=agg_e_round+ae;
                tx_attempts_this_round=tx_attempts_this_round+att; tx_packets_this_round=tx_packets_this_round+packets;
                lost_this_round=lost_this_round+lost;
            end
        end
        
        for i = length(chain):-1:(leader_idx_in_chain + 1)
            curr = chain(i);
            next_hop = chain(i-1);
            if is_alive(curr) && is_alive(next_hop)
                [E_residual, node_reports, node_unattempted, report_delays, is_alive, te, re, ae, att, packets, lost, led] = passToken(...
                    curr, next_hop, E_residual, node_reports, node_unattempted, report_delays, ...
                    wsn_env, succ_matrix, R_bit, T_proc, c, is_alive, ea, led);
                tx_e_round=tx_e_round+te; rx_e_round=rx_e_round+re; agg_e_round=agg_e_round+ae;
                tx_attempts_this_round=tx_attempts_this_round+att; tx_packets_this_round=tx_packets_this_round+packets;
                lost_this_round=lost_this_round+lost;
            end
        end
        
        del_this_round = 0; delay_sum_this_round = 0;
        if is_alive(leader)
            reports_carried = node_reports(leader);
            if reports_carried > 0
                E_agg = wsn_env.E_DA * wsn_env.packetSize * reports_carried * ~ea.singleLeaderAggregation;   % fusion already charged at each reception
                if E_residual(leader) >= E_agg
                    E_residual(leader) = E_residual(leader) - E_agg;
                    agg_e_round = agg_e_round + E_agg; led.aggJ(leader) = led.aggJ(leader) + E_agg;
                    d_bs = sqrt((wsn_env.X(leader)-wsn_env.BS_X)^2 + (wsn_env.Y(leader)-wsn_env.BS_Y)^2);
                    E_tx = tx_energy(wsn_env.packetSize * (1 + ea.reportPreservingPackets * (reports_carried - 1)), d_bs, wsn_env);
                    if E_residual(leader) >= E_tx
                        E_residual(leader) = E_residual(leader) - E_tx;
                        tx_e_round = tx_e_round + E_tx; tx_packets_this_round = tx_packets_this_round + 1; led.txBSJ(leader) = led.txBSJ(leader) + E_tx; led.nTxBS(leader) = led.nTxBS(leader) + 1;
                        tx_attempts_this_round = tx_attempts_this_round + node_unattempted(leader);
                        node_unattempted(leader) = 0;
                        led.nBSFail(leader) = led.nBSFail(leader) + ~succ_matrix(leader, numNodes+1);   % counts failed BS frames
                        if succ_matrix(leader, numNodes+1)
                            del_this_round = del_this_round + reports_carried;
                            hop_delay = (wsn_env.packetSize / R_bit) + T_proc + (d_bs / c);
                            delay_sum_this_round = delay_sum_this_round + report_delays(leader) + reports_carried * hop_delay;
                        else
                            lost_this_round = lost_this_round + reports_carried;
                        end
                        node_reports(leader) = 0;
                        % If BS link fails, reports are lost in transit
                    else
                        % Leader cannot afford BS TX: retains energy, operationally dead
                        is_alive(leader) = false;
                        lost_this_round = lost_this_round + reports_carried - node_unattempted(leader);
                        node_reports(leader) = node_unattempted(leader);
                    end
                else
                    % Leader cannot afford Aggregation: retains energy, operationally dead
                    is_alive(leader) = false;
                    lost_this_round = lost_this_round + reports_carried - node_unattempted(leader);
                    node_reports(leader) = node_unattempted(leader);
                end
            end
        end
        
        % Reports still held by a live node at route end have already been
        % attempted upstream, except that node's own unattempted source data.
        for i = 1:numNodes
            if node_reports(i) > node_unattempted(i)
                lost_this_round = lost_this_round + node_reports(i) - node_unattempted(i);
                node_reports(i) = node_unattempted(i);
            end
        end

        % Count distinct application reports after their first actual RF transmission.
        m_transmitted(r) = tx_attempts_this_round;
        m_tx_packets(r) = tx_packets_this_round;
        assert(lost_this_round >= 0 && tx_attempts_this_round == del_this_round + lost_this_round, ...
            'Report-attempt conservation failed');
        m_unattempted(r) = source_avail_this_round - tx_attempts_this_round;
        assert(m_unattempted(r) >= 0 && tx_attempts_this_round <= source_avail_this_round, ...
            'Attempted reports exceed source-available reports');
        m_del(r) = del_this_round;
        m_lost(r) = lost_this_round;
        
        % Assertions on energy bounds
        assert(all(E_residual >= 0), 'Residual energy cannot be negative');
        assert(all(E_residual <= wsn_env.initialEnergy + 1e-12), 'Residual energy cannot exceed initial energy');
        
        is_alive = is_alive & (E_residual > 0) & ~has_failed;
        energy_end = sum(E_residual);
        actual_expended = tx_e_round + rx_e_round + agg_e_round;
        assert(actual_expended >= 0 && abs((energy_start - energy_end) - actual_expended) <= 1e-10, ...
            'Physical energy accounting does not close');
        assert(energy_end <= energy_start + 1e-12, 'Total physical residual energy cannot increase');
        if rl_enabled
            R_round = rlRoundReward(del_this_round, source_avail_this_round, actual_expended, cfg);
            s = rl_log{r}(3); rl_V(s) = rl_V(s) + cfg.alpha * (R_round - rl_V(s)); rl_log{r}(7) = R_round;
        end
        
        m_alive(r) = sum(is_alive);
        m_totalEnergy(r) = energy_end;
        m_operational_energy(r) = sum(E_residual(is_alive));
        if isnan(FND) && m_alive(r) < numNodes; FND = r; end
        if isnan(HND) && m_alive(r) <= numNodes/2; HND = r; end
        if del_this_round > 0
            m_delay(r) = delay_sum_this_round / del_this_round;
        else
            m_delay(r) = NaN;
        end
        m_routingEnergy(r) = actual_expended;
        m_tx_energy(r) = tx_e_round; m_rx_energy(r) = rx_e_round; m_agg_energy(r) = agg_e_round;
        E_residual_history(:,r) = E_residual(:); is_alive_history(:,r) = is_alive(:); ledHist{r} = led;
        alive_energies = E_residual(is_alive);
        if length(alive_energies) > 1
            m_energyStdDev(r) = std(alive_energies);
        else
            m_energyStdDev(r) = 0;
        end
    end
    
    res = struct();
    res.configuredRounds = wsn_env.numRounds;
    res.actualRoundsSimulated = find(~isnan(m_alive), 1, 'last');
    if isempty(res.actualRoundsSimulated); res.actualRoundsSimulated = ar; end
    res.wsn_env = wsn_env;
    res.aliveNodes = m_alive;
    res.totalResidualEnergy = m_totalEnergy;
    res.operationalResidualEnergy = m_operational_energy;
    res.E_residual_history = E_residual_history;
    res.is_alive_history = is_alive_history;
    res.sensingNodes = sensing_nodes;
    res.sensingSourceAvailableHistory = sensing_source_available_history;
    
    % Explicit Report Lifecycle Counters
    res.scheduledReports = m_scheduled;
    res.sourceAvailableReports = m_source_avail;
    res.sourceUnavailableReports = m_source_unavail;
    res.transmittedReports = m_transmitted;
    res.deliveredReports = m_del;
    res.lostInTransitReports = m_lost;
    res.availableButUnattemptedReports = m_unattempted;
    res.actualTxPacketAttempts = m_tx_packets;
    res.txEnergyExpended = m_tx_energy; res.rxEnergyExpended = m_rx_energy; res.aggregationEnergyExpended = m_agg_energy;
    res.actualEnergyExpendedThisRound = m_routingEnergy;
    
    res.totalScheduledReports = sum(m_scheduled(~isnan(m_scheduled)));
    res.totalSourceAvailableReports = sum(m_source_avail(~isnan(m_source_avail)));
    res.totalSourceUnavailableReports = sum(m_source_unavail(~isnan(m_source_unavail)));
    res.totalTransmittedReports = sum(m_transmitted(~isnan(m_transmitted)));
    res.totalDeliveredReports = sum(m_del(~isnan(m_del)));
    res.totalLostInTransitReports = sum(m_lost(~isnan(m_lost)));
    
    % Metric Definitions (Prompt Section 13)
    % A. Scheduled PDR: delivered / scheduled
    res.scheduledPDR = (res.totalDeliveredReports / max(1, res.totalScheduledReports)) * 100;
    % B. Source-available delivery ratio: delivered / source_available
    res.sourceAvailableDeliveryRatio = (res.totalDeliveredReports / max(1, res.totalSourceAvailableReports)) * 100;
    % C. Transmission success ratio: delivered / transmitted
    res.transmissionSuccessRatio = (res.totalDeliveredReports / max(1, res.totalTransmittedReports)) * 100;
    
    % Backward compatibility mappings
    res.generatedSourceReports = m_source_avail;
    res.deliveredSourceReports = m_del;
    m_pdr = m_del ./ max(1, m_source_avail);
    m_pdr(isnan(m_alive)) = NaN;
    res.sourceReportPDR = m_pdr;
    res.totalGeneratedReports = res.totalSourceAvailableReports;
    res.applicationPDR = res.scheduledPDR;
    
    res.averageEndToEndDelay = m_delay;
    res.routingEnergyConsumedThisRound = m_routingEnergy;
    res.externalFailureEnergy = m_externalFailureEnergy;
    res.energyStdDev = m_energyStdDev;
    res.FND = FND; res.HND = HND; res.LND = LND;
    res.finalNodeEnergies = E_residual;
    res.minResidualEnergy = min(E_residual);
    res.maxResidualEnergy = max(E_residual);
    res.energyLedger = led; res.energyAuditOptions = ea; res.energyLedgerHistory = ledHist;
    res.rlConfig = cfg; res.rlV = rl_V; res.epsilonHistory = m_epsilon;
    res.rlDecisionLog = vertcat(rl_log{:});
    res.rlDecisionLogColumns = {'round', 'leader', 'state', 'explored', 'nCandidates', 'parentLeader', 'reward', 'VBefore'};
end

function s = pegasisStates(chain, cands, env, E)
% Decision-time state of each candidate: BS-distance band, E/E_avg band (chain nodes), longer-branch hops.
    L = numel(chain); s = zeros(size(cands)); Eavg = max(1e-12, mean(E(chain)));
    for q = 1:numel(cands)
        nd = cands(q); pos = find(chain == nd, 1);
        dbs = hypot(env.X(nd) - env.BS_X, env.Y(nd) - env.BS_Y);
        f1 = 1 + (dbs >= env.d0) + (dbs >= 2 * env.d0) + (dbs >= 3 * env.d0);
        ratio = E(nd) / Eavg; f2 = 1 + (ratio >= 0.8) + (ratio > 1.2);
        br = max(pos - 1, L - pos); f3 = 1 + (br >= 3) + (br >= 6);
        s(q) = f1 + 4 * (f2 - 1) + 12 * (f3 - 1);
    end
end

function R = rlRoundReward(delivered, available, energyJ, cfg)
% Valid-report delivery of the round discounted by its energy per available report relative to the
% sustainable budget E_b (R in [0, 1]); without the energy term R = delivered / available.
    if available == 0, R = 0; return; end
    e = energyJ / available; R = delivered / available;
    if cfg.useEnergyReward, R = R * (1 - e / (e + cfg.energyBudgetJ)); end
end

function chain = buildChain(wsn_env, is_alive)
    alive_idx = find(is_alive);
    if isempty(alive_idx)
        chain = [];
        return;
    end
    max_d = -1; start_node = alive_idx(1);
    for i = 1:length(alive_idx)
        n = alive_idx(i);
        d = sqrt((wsn_env.X(n)-wsn_env.BS_X)^2 + (wsn_env.Y(n)-wsn_env.BS_Y)^2);
        if d > max_d
            max_d = d; start_node = n;
        end
    end
    chain = start_node;
    unvisited = alive_idx(alive_idx ~= start_node);
    curr = start_node;
    while ~isempty(unvisited)
        min_d = inf; next_node = unvisited(1);
        for i = 1:length(unvisited)
            n = unvisited(i);
            d = sqrt((wsn_env.X(curr)-wsn_env.X(n))^2 + (wsn_env.Y(curr)-wsn_env.Y(n))^2);
            if d < min_d
                min_d = d; next_node = n;
            end
        end
        chain = [chain, next_node];
        curr = next_node;
        unvisited = unvisited(unvisited ~= next_node);
    end
end

function [E_residual, node_reports, node_unattempted, report_delays, is_alive, tx_e, rx_e, agg_e, attempts, packets, lost_reports, led] = passToken(curr, next_hop, E_residual, node_reports, node_unattempted, report_delays, wsn_env, succ_matrix, R_bit, T_proc, c, is_alive, ea, led)
% PEGASIS token pass with a per-node energy ledger (led). Behaviour is the original's unless
% ea.chargeFailedRx: the receiver's radio is on for the whole frame, so a frame lost on the link still
% costs it E_rx (nothing is fused); a receiver that cannot pay this required reception is unavailable.
    tx_e = 0; rx_e = 0; agg_e = 0; attempts = 0; packets = 0; lost_reports = 0;
    reports_carried = node_reports(curr);
    if reports_carried == 0; return; end
    d = sqrt((wsn_env.X(curr)-wsn_env.X(next_hop))^2 + (wsn_env.Y(curr)-wsn_env.Y(next_hop))^2);
    bits = wsn_env.packetSize * (1 + ea.reportPreservingPackets * (reports_carried - 1));
    E_tx = tx_energy(bits, d, wsn_env);
    if E_residual(curr) >= E_tx
        E_residual(curr) = E_residual(curr) - E_tx;
        led.txChainJ(curr) = led.txChainJ(curr) + E_tx; led.nTx(curr) = led.nTx(curr) + 1;
        led.reportsForwarded(curr) = led.reportsForwarded(curr) + reports_carried - node_unattempted(curr);
        tx_e = E_tx; packets = 1; attempts = node_unattempted(curr); node_unattempted(curr) = 0;
        E_rx = rx_energy(bits, wsn_env);
        if succ_matrix(curr, next_hop)
            % Intermediate node pays RX + aggregation cost (PEGASIS paper: fusion at every hop)
            E_agg = wsn_env.E_DA * wsn_env.packetSize * reports_carried;
            if E_residual(next_hop) >= (E_rx + E_agg)
                E_residual(next_hop) = E_residual(next_hop) - E_rx - E_agg;
                rx_e = E_rx; agg_e = E_agg;
                led.rxJ(next_hop) = led.rxJ(next_hop) + E_rx; led.aggJ(next_hop) = led.aggJ(next_hop) + E_agg; led.nRx(next_hop) = led.nRx(next_hop) + 1;
                node_reports(next_hop) = node_reports(next_hop) + reports_carried;
                hop_delay = (wsn_env.packetSize / R_bit) + T_proc + (d / c);
                report_delays(next_hop) = report_delays(next_hop) + report_delays(curr) + reports_carried * hop_delay;
            else
                % next_hop retains residual energy; cannot afford RX+agg -> operationally dead
                is_alive(next_hop) = false;
                lost_reports = reports_carried;
            end
        else
            lost_reports = reports_carried;
            led.nLinkFail(next_hop) = led.nLinkFail(next_hop) + 1; led.nTxFail(curr) = led.nTxFail(curr) + 1;
            if ea.chargeFailedRx
                if E_residual(next_hop) >= E_rx
                    E_residual(next_hop) = E_residual(next_hop) - E_rx; rx_e = E_rx;
                    led.rxJ(next_hop) = led.rxJ(next_hop) + E_rx; led.nRxFailed(next_hop) = led.nRxFailed(next_hop) + 1;
                else
                    is_alive(next_hop) = false;
                end
            end
        end
        node_reports(curr) = 0; report_delays(curr) = 0;
    else
        % curr retains residual energy; cannot afford TX -> operationally dead
        is_alive(curr) = false;
        lost_reports = reports_carried - node_unattempted(curr);
        node_reports(curr) = node_unattempted(curr); report_delays(curr) = 0;
    end
end

function ea = eaDefaults(ea)
% Energy-audit options; both false = the original accounting (reproduced exactly).
%   chargeFailedRx          a frame lost on a chain link still costs the receiver E_rx
%   singleLeaderAggregation the leader's fusion is charged once, at reception (not again before the BS frame)
%   reportPreservingPackets a frame carrying m station reports is m packets long (each report keeps its
%                           packetSize bits, as RL-Hybrid already charges); otherwise one packet per frame
    d = struct('chargeFailedRx', false, 'singleLeaderAggregation', false, 'reportPreservingPackets', false);
    for f = fieldnames(d)', if ~isfield(ea, f{1}), ea.(f{1}) = d.(f{1}); end, end
end
