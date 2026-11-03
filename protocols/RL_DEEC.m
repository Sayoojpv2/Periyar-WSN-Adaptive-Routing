function results = RL_DEEC(wsn_env, ~, success_history, seed, app_workload, rl_mode, opts)
% Report-preserving energy accounting: a frame carrying m station reports is charged as
% m packets (radio TX/RX energy for m x packetSize bits), as the per-report delivery metric requires.
%RL_DEEC DEEC with an RL communication-cost correction of the CH probability.
% Parent: protocols/DEEC.m (unchanged). Everything is the parent's code except p_i:
%   DEEC:  p_i = p E_i / E_avg
%   RL-DEEC: p_i = p (E_i / E_avg) w_i,  w_i = c exp(beta (V(s_i) - mean V over operational nodes)),
%            c chosen so that sum_i p_i equals DEEC's sum (same expected number of CHs).
% DEEC's energy proportionality, threshold, epoch and eligibility set G are kept; the zero-CH
% fallback takes the operational node with the largest p_i instead of a uniformly random node.
% V(s) is the learned value of a round headed by a node in state s (shared table over 36 states:
% BS-distance band x E/E_avg band x BS frames the node can afford: <10, 10-100, >100).
% Why: DEEC weights only energy, so a node 400-550 m from the BS (0.1-0.5 J per BS frame, 20-45 %
% success) is as likely to lead as one next to the BS. Replacing DEEC's draw by an RL choice of the CH
% identity (an earlier design, initially random among ties) ignored E_i / E_avg and picked below-average-
% energy CHs (E/E_avg 0.74-0.87 in two development seeds), shortening lifetime.
% Reward (team, per round, after the outcome): R = (D/A)(1 - e/(e+E_b)); every CH of the round
% updates V(s) += alpha (R - V(s)). Exploration is the stochastic election itself (no epsilon).
    if nargin < 6 || isempty(rl_mode); rl_mode = 'active'; end
    if nargin < 7; opts = struct(); end
    assert(any(strcmpi(rl_mode, {'active', 'noop'})), 'Unknown rl_mode %s', rl_mode);
    rl_enabled = strcmpi(rl_mode, 'active');
    use_app_workload = ~isempty(app_workload);
    rng(seed, 'twister');
    cfg = struct('variant', 'RL-DEEC', 'mode', rl_mode, 'alpha', 0.10, 'beta', 8, 'vInit', 0.25, 'useEnergyReward', true, ...
        'energyBudgetJ', wsn_env.numNodes * wsn_env.initialEnergy / (numel(app_workload.sensing_nodes) * wsn_env.budgetHorizonRounds));
    for f = fieldnames(opts)', cfg.(f{1}) = opts.(f{1}); end
    rl_V = cfg.vInit * ones(36, 1);
    rl_log = cell(1, wsn_env.numRounds);     % [round node state pDEEC pRL draw electedDEEC electedRL]
    rl_upd = cell(1, wsn_env.numRounds);     % [round node state reward VBefore]
    rl_ch = false(wsn_env.numNodes, wsn_env.numRounds);
    
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
    
    p = 0.1;
    epoch_len = round(1/p);
    G = true(1, numNodes);
    m_chCount = zeros(1, ar);
    
    R_bit = 250e3; T_proc = 0.005; c = 3e8;
    FND = NaN; HND = NaN; LND = NaN;
    
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
            m_operational_energy(r) = 0; m_routingEnergy(r) = 0; m_chCount(r) = 0;
            E_residual_history(:,r) = E_residual(:); is_alive_history(:,r) = is_alive(:);
            continue;
        end
        
        if isnan(FND) && num_alive < numNodes; FND = r; end
        if isnan(HND) && num_alive <= numNodes/2; HND = r; end
        
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
                    node_reports(s_node) = node_reports(s_node) + 1;
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
            E_residual_history(:,r) = E_residual(:); is_alive_history(:,r) = is_alive(:);
            m_chCount(r) = 0;
            alive_energies = E_residual(is_alive);
            if length(alive_energies) > 1
                m_energyStdDev(r) = std(alive_energies);
            else
                m_energyStdDev(r) = 0;
            end
            continue;
        end
        
        % Canonical DEEC CH Election (Qing et al. 2006)
        r_prime = mod(r - 1, epoch_len);
        if r_prime == 0
            % Epoch reset: all surviving nodes re-enter eligibility set G
            G = is_alive;
        else
            % Nodes that died since epoch start are removed from G
            G = G & is_alive;
        end
        
        E_avg = max(1e-6, sum(E_residual(is_alive)) / num_alive);
        is_CH = false(1, numNodes);

        % RL cost correction w_i (w = 1 in noop mode): same expected CH count as DEEC.
        w = ones(1, numNodes);
        if rl_enabled
            rl_s = deecStates(wsn_env, E_residual, is_alive);
            Vs = rl_V(rl_s)'; Va = Vs(is_alive);
            w = exp(cfg.beta * (Vs - mean(Va)));
            w = w * sum(E_residual(is_alive)) / sum(E_residual(is_alive) .* w(is_alive));
        end
        for i = 1:numNodes
            if is_alive(i) && G(i)
                % Dynamic DEEC probability proportional to residual energy
                p_i = p * (E_residual(i) / E_avg);
                p_rl = p_i * w(i);

                % Robust threshold calculation preventing negative denominator
                denom = 1 - p_rl * r_prime;
                if denom > 1e-4
                    thresh = min(1.0, p_rl / denom);
                else
                    thresh = 1.0;
                end

                u = rand();
                if u <= thresh
                    is_CH(i) = true;
                    G(i) = false; % Node has been CH in this epoch, excluded until next epoch
                end
                if rl_enabled
                    denom0 = 1 - p_i * r_prime; thresh0 = 1.0; if denom0 > 1e-4, thresh0 = min(1.0, p_i / denom0); end
                    rl_log{r}(end+1,:) = [r, i, rl_s(i), p_i, p_rl, u, double(u <= thresh0), double(u <= thresh)];
                end
            end
        end

        % Implementation robustness fallback if 0 CHs elected
        if sum(is_CH) == 0
            alive_idx = find(is_alive);
            if ~isempty(alive_idx)
                chosen = alive_idx(randi(length(alive_idx)));
                if rl_enabled
                    % Node with the largest corrected probability p (E_i/E_avg) w_i.
                    sc = E_residual(alive_idx) .* w(alive_idx); chosen = alive_idx(find(sc == max(sc), 1));
                end
                is_CH(chosen) = true;
                G(chosen) = false;
            end
        end
        rl_ch(:, r) = is_CH(:);
        m_chCount(r) = sum(is_CH);
        
        ch_idx = find(is_CH);
        report_delays = zeros(1, numNodes);
        
        % Member Phase
        for i = 1:numNodes
            if is_alive(i) && ~is_CH(i) && node_reports(i) > 0
                min_d = inf; best_ch = ch_idx(1);
                for j = 1:length(ch_idx)
                    ch = ch_idx(j);
                    d = sqrt((wsn_env.X(i)-wsn_env.X(ch))^2 + (wsn_env.Y(i)-wsn_env.Y(ch))^2);
                    if d < min_d
                        min_d = d; best_ch = ch;
                    end
                end
                
                d = min_d;
                reps_i = node_reports(i);
                E_tx = tx_energy(wsn_env.packetSize, d, wsn_env) * reps_i;
                if E_residual(i) >= E_tx
                    E_residual(i) = E_residual(i) - E_tx;
                    tx_e_round = tx_e_round + E_tx; tx_packets_this_round = tx_packets_this_round + reps_i;
                    tx_attempts_this_round = tx_attempts_this_round + node_unattempted(i);
                    node_unattempted(i) = 0;
                    if succ_matrix(i, best_ch)
                        E_rx = rx_energy(wsn_env.packetSize, wsn_env) * reps_i;
                        if E_residual(best_ch) >= E_rx
                            E_residual(best_ch) = E_residual(best_ch) - E_rx;
                            rx_e_round = rx_e_round + E_rx;
                            node_reports(best_ch) = node_reports(best_ch) + reps_i;
                            hop_delay = (wsn_env.packetSize / R_bit) + T_proc + (d / c);
                            report_delays(best_ch) = report_delays(best_ch) + reps_i * hop_delay;
                        else
                            % CH unable to afford reception: retains energy, operationally dead
                            is_alive(best_ch) = false;
                            lost_this_round = lost_this_round + reps_i;
                        end
                        % If best_ch cannot afford E_rx, best_ch retains energy, reports fail to be received
                    else
                        lost_this_round = lost_this_round + reps_i;
                    end
                    % If link fails, reports fail
                else
                    % Member unable to afford transmission: retains energy, operationally dead
                    is_alive(i) = false;
                end
                node_reports(i) = 0;
            end
        end

        % Reports already received by a CH that became unavailable during
        % the member phase are stranded in transit.
        for i = 1:numNodes
            if ~is_alive(i) && node_reports(i) > node_unattempted(i)
                lost_this_round = lost_this_round + node_reports(i) - node_unattempted(i);
                node_reports(i) = node_unattempted(i);
            end
        end
        
        % CH Phase
        del_this_round = 0; delay_sum_this_round = 0;
        
        for i = 1:numNodes
            if is_alive(i) && is_CH(i)
                reports_carried = node_reports(i);
                if reports_carried > 0
                    E_agg = wsn_env.E_DA * wsn_env.packetSize * reports_carried;
                    if E_residual(i) >= E_agg
                        E_residual(i) = E_residual(i) - E_agg;
                        agg_e_round = agg_e_round + E_agg;
                        d_bs = sqrt((wsn_env.X(i)-wsn_env.BS_X)^2 + (wsn_env.Y(i)-wsn_env.BS_Y)^2);
                        E_tx = tx_energy(wsn_env.packetSize * reports_carried, d_bs, wsn_env);   % RP: m reports = m packets
                        if E_residual(i) >= E_tx
                            E_residual(i) = E_residual(i) - E_tx;
                            tx_e_round = tx_e_round + E_tx; tx_packets_this_round = tx_packets_this_round + 1;
                            tx_attempts_this_round = tx_attempts_this_round + node_unattempted(i);
                            node_unattempted(i) = 0;
                            if succ_matrix(i, numNodes+1)
                                del_this_round = del_this_round + reports_carried;
                                hop_delay = (wsn_env.packetSize / R_bit) + T_proc + (d_bs / c);
                                delay_sum_this_round = delay_sum_this_round + report_delays(i) + reports_carried * hop_delay;
                            else
                                lost_this_round = lost_this_round + reports_carried;
                            end
                            node_reports(i) = 0;
                            % If BS link fails, reports are lost in transit
                        else
                            % CH cannot afford BS TX: retains energy, operationally dead
                            is_alive(i) = false;
                            lost_this_round = lost_this_round + reports_carried - node_unattempted(i);
                            node_reports(i) = node_unattempted(i);
                        end
                    else
                        % CH cannot afford Aggregation: retains energy, operationally dead
                        is_alive(i) = false;
                        lost_this_round = lost_this_round + reports_carried - node_unattempted(i);
                        node_reports(i) = node_unattempted(i);
                    end
                end
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
            % Team reward of the round; every node that headed a cluster this round updates its state value.
            R_round = rlRoundReward(del_this_round, source_avail_this_round, actual_expended, cfg);
            for i = find(rl_ch(:, r))'
                s = rl_s(i); rl_upd{r}(end+1,:) = [r, i, s, R_round, rl_V(s)];
                rl_V(s) = rl_V(s) + cfg.alpha * (R_round - rl_V(s));
            end
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
        E_residual_history(:,r) = E_residual(:); is_alive_history(:,r) = is_alive(:);
        alive_energies = E_residual(is_alive);
        if length(alive_energies) > 1
            m_energyStdDev(r) = std(alive_energies);
        else
            m_energyStdDev(r) = 0;
        end
    end
    
    results = struct();
    results.configuredRounds = wsn_env.numRounds;
    results.actualRoundsSimulated = find(~isnan(m_alive), 1, 'last');
    if isempty(results.actualRoundsSimulated); results.actualRoundsSimulated = ar; end
    results.wsn_env = wsn_env;
    results.aliveNodes = m_alive;
    results.totalResidualEnergy = m_totalEnergy;
    results.operationalResidualEnergy = m_operational_energy;
    results.E_residual_history = E_residual_history;
    results.is_alive_history = is_alive_history;
    results.sensingNodes = sensing_nodes;
    results.sensingSourceAvailableHistory = sensing_source_available_history;
    
    % Explicit Report Lifecycle Counters
    results.scheduledReports = m_scheduled;
    results.sourceAvailableReports = m_source_avail;
    results.sourceUnavailableReports = m_source_unavail;
    results.transmittedReports = m_transmitted;
    results.deliveredReports = m_del;
    results.lostInTransitReports = m_lost;
    results.availableButUnattemptedReports = m_unattempted;
    results.actualTxPacketAttempts = m_tx_packets;
    results.txEnergyExpended = m_tx_energy; results.rxEnergyExpended = m_rx_energy; results.aggregationEnergyExpended = m_agg_energy;
    results.actualEnergyExpendedThisRound = m_routingEnergy;
    
    results.totalScheduledReports = sum(m_scheduled(~isnan(m_scheduled)));
    results.totalSourceAvailableReports = sum(m_source_avail(~isnan(m_source_avail)));
    results.totalSourceUnavailableReports = sum(m_source_unavail(~isnan(m_source_unavail)));
    results.totalTransmittedReports = sum(m_transmitted(~isnan(m_transmitted)));
    results.totalDeliveredReports = sum(m_del(~isnan(m_del)));
    results.totalLostInTransitReports = sum(m_lost(~isnan(m_lost)));
    
    % Metric Definitions (Prompt Section 13)
    % A. Scheduled PDR: delivered / scheduled
    results.scheduledPDR = (results.totalDeliveredReports / max(1, results.totalScheduledReports)) * 100;
    % B. Source-available delivery ratio: delivered / source_available
    results.sourceAvailableDeliveryRatio = (results.totalDeliveredReports / max(1, results.totalSourceAvailableReports)) * 100;
    % C. Transmission success ratio: delivered / transmitted
    results.transmissionSuccessRatio = (results.totalDeliveredReports / max(1, results.totalTransmittedReports)) * 100;
    
    % Backward compatibility mappings
    results.generatedSourceReports = m_source_avail;
    results.deliveredSourceReports = m_del;
    m_pdr = m_del ./ max(1, m_source_avail);
    m_pdr(isnan(m_alive)) = NaN;
    results.sourceReportPDR = m_pdr;
    results.totalGeneratedReports = results.totalSourceAvailableReports;
    results.applicationPDR = results.scheduledPDR;
    
    results.averageEndToEndDelay = m_delay;
    results.routingEnergyConsumedThisRound = m_routingEnergy;
    results.externalFailureEnergy = m_externalFailureEnergy;
    results.energyStdDev = m_energyStdDev;
    results.chCountHistory = m_chCount;
    valid_ch = m_chCount(~isnan(m_chCount));
    if ~isempty(valid_ch)
        results.meanCHCount = mean(valid_ch);
        results.stdCHCount = std(valid_ch);
        results.minCHCount = min(valid_ch);
        results.maxCHCount = max(valid_ch);
    else
        results.meanCHCount = NaN; results.stdCHCount = NaN; results.minCHCount = NaN; results.maxCHCount = NaN;
    end
    results.FND = FND; results.HND = HND; results.LND = LND;
    results.finalNodeEnergies = E_residual;
    results.minResidualEnergy = min(E_residual);
    results.maxResidualEnergy = max(E_residual);
    results.rlConfig = cfg; results.rlV = rl_V; results.rlCHHistory = rl_ch;
    results.rlElectionLog = vertcat(rl_log{:});
    results.rlElectionLogColumns = {'round', 'node', 'state', 'pDEEC', 'pRL', 'draw', 'electedUnderDEEC', 'electedUnderRL'};
    results.rlUpdateLog = vertcat(rl_upd{:});
    results.rlUpdateLogColumns = {'round', 'node', 'state', 'reward', 'VBefore'};
end

function s = deecStates(env, E, alive)
% Decision-time state of every node: BS-distance band, E/E_avg band, BS frames its energy can pay for.
    dbs = hypot(env.X - env.BS_X, env.Y - env.BS_Y); ratio = E / max(1e-12, mean(E(alive)));
    f1 = 1 + (dbs >= env.d0) + (dbs >= 2 * env.d0) + (dbs >= 3 * env.d0);
    f2 = 1 + (ratio >= 0.8) + (ratio > 1.2);
    frames = E ./ arrayfun(@(d) tx_energy(env.packetSize, d, env), dbs); f3 = 1 + (frames >= 10) + (frames > 100);
    s = f1 + 4 * (f2 - 1) + 12 * (f3 - 1);
end

function R = rlRoundReward(delivered, available, energyJ, cfg)
% Valid-report delivery of the round discounted by its energy per available report relative to the
% sustainable budget E_b (R in [0, 1]); without the energy term R = delivered / available.
    if available == 0, R = 0; return; end
    e = energyJ / available; R = delivered / available;
    if cfg.useEnergyReward, R = R * (1 - e / (e + cfg.energyBudgetJ)); end
end
