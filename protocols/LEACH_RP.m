% protocols/LEACH.m
function res = LEACH_RP(wsn_env, lq_history, success_history, seed, app_workload)
% REPORT-PRESERVING COPY of LEACH.m: identical except that a frame carrying m station reports is charged as
% m packets (radio TX/RX energy for m x packetSize bits), as the per-report delivery metric requires.
    if nargin < 5; app_workload = []; end
    use_app_workload = ~isempty(app_workload);
    rng(seed, 'twister'); % Use reproducible seed for protocol-specific probabilistic decisions
    
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
        
        % Fetch identical common environment realization
        lq_matrix = lq_history(:,:,r);
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
        
        % Canonical LEACH CH Election (Heinzelman et al. 2002)
        r_prime = mod(r - 1, epoch_len);
        if r_prime == 0
            % Epoch reset: all surviving nodes re-enter eligibility set G
            G = is_alive;
        else
            % Nodes that died since epoch start are removed from G
            G = G & is_alive;
        end
        
        thresh = p / (1 - p * r_prime);
        is_CH = false(1, numNodes);
        for i = 1:numNodes
            if is_alive(i) && G(i)
                if rand() <= thresh
                    is_CH(i) = true;
                    G(i) = false; % Node has been CH in this epoch, excluded until next epoch
                end
            end
        end
        
        % Implementation robustness fallback if 0 CHs elected (<0.5% probability)
        if sum(is_CH) == 0
            alive_idx = find(is_alive);
            if ~isempty(alive_idx)
                chosen = alive_idx(randi(length(alive_idx)));
                is_CH(chosen) = true;
                G(chosen) = false;
            end
        end
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
    res.chCountHistory = m_chCount;
    valid_ch = m_chCount(~isnan(m_chCount));
    if ~isempty(valid_ch)
        res.meanCHCount = mean(valid_ch);
        res.stdCHCount = std(valid_ch);
        res.minCHCount = min(valid_ch);
        res.maxCHCount = max(valid_ch);
    else
        res.meanCHCount = NaN; res.stdCHCount = NaN; res.minCHCount = NaN; res.maxCHCount = NaN;
    end
    res.FND = FND; res.HND = HND; res.LND = LND;
    res.finalNodeEnergies = E_residual;
    res.minResidualEnergy = min(E_residual);
    res.maxResidualEnergy = max(E_residual);
end
