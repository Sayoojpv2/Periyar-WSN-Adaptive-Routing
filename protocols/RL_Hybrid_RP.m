function res = RL_Hybrid_RP(wsn_env, ~, success_history, seed, app_workload)
% REPORT-PRESERVING COPY of RL_Hybrid.m: identical except that a frame carrying m station reports is charged as
% m packets (radio TX/RX energy for m x packetSize bits), as the per-report delivery metric requires.
%RL_HYBRID Online tabular Q-learning for cluster assignment and multi-hop routing.
% Causal information rule: before an action is chosen the agent may use only
% (a) geometry and the deterministic distance-based link prior of the channel
% model, (b) the acting node's OWN previously observed link outcomes, (c) the
% current residual energies and loads. The realised per-round link quality
% (lq_history, 2nd argument) is deliberately not bound to any variable and
% cannot be read. success_history(i,j,r) is read only AFTER a transmission
% has been paid for, to decide whether it succeeded. Each run learns online
% from its own seed; no cross-seed transfer or future-round data is used.
%
% Death rule (same semantics as LEACH/DEEC/PEGASIS): a node becomes
% unavailable when it must perform an operation it cannot afford. Additionally
% a node holding less energy than the cheapest radio operation of the frozen
% energy model (E_elec*packetSize = one packet reception) is unavailable at
% the end of the round, because it can no longer perform any radio operation.

    if nargin < 5 || isempty(app_workload)
        error('RL_Hybrid requires the validated continuous Periyar workload.');
    end
    % Input guards only (no effect on the algorithm): the 1000-round case keeps the authoritative
    % workload check; other horizons must use a workload built for the same horizon.
    assert(wsn_env.numNodes == numel(wsn_env.X) && wsn_env.numRounds == app_workload.numRounds, ...
        'RL-Hybrid expects an N-node scenario whose horizon matches the workload.');
    assert(app_workload.totalScheduledReports == wsn_env.numRounds * numel(app_workload.sensing_nodes) && ...
        app_workload.totalAvailableDataReports + app_workload.totalUnavailableDataReports == app_workload.totalScheduledReports && ...
        (wsn_env.numRounds ~= 1000 || (app_workload.totalAvailableDataReports == 9825 && app_workload.totalUnavailableDataReports == 175)), ...
        'Unexpected workload; refusing to run under non-authoritative conditions.');

    rng(seed, 'twister');
    n = wsn_env.numNodes; rounds = wsn_env.numRounds; bs = n + 1;
    sensingNodes = app_workload.sensing_nodes(:)';
    assert(all(ismember(sensingNodes, 1:n)) && numel(unique(sensingNodes)) == numel(sensingNodes));
    assert(isequal(size(success_history), [n n+1 rounds]));

    % Fixed hyperparameters across all seeds. Q(s,a) starts unbiased at zero.
    cfg.alpha = 0.10;
    cfg.gamma = 0.90;
    cfg.epsilonInitial = 0.25;
    cfg.epsilonMinimum = 0.05;
    cfg.epsilonDecayPerRound = 0.9985;
    cfg.numStateBins = 3;
    cfg.numStates = 3^8;
    cfg.numActions = n + 1; % nodes 1..n and BS = n+1
    cfg.qTableMode = 'independent per-node Q table; no cross-node value sharing';
    % The shared channel has no hard radio radius. Candidate links are
    % discovered from a CAUSAL link-quality estimate and must reach 0.50:
    %   qHat(i,j) = (successes_ij + K*p0_ij) / (attempts_ij + K),  K = 2
    % p0_ij = expected link quality at the i-j distance (closed form of the
    % scenario's clipped distance model, geometry only); successes/attempts are
    % node i's OWN past outcomes on that link. No realised per-round LQ is used.
    cfg.minimumLinkQuality = 0.50;
    cfg.linkPriorStrength = 2;
    cfg.minRadioOperationEnergyJ = wsn_env.E_elec * wsn_env.packetSize;
    cfg.maxHops = 15;
    cfg.stateDefinition = ['6561 states, 8 ternary action-context features: current energy, candidate energy, ' ...
        'causal link-quality estimate qHat, link distance relative to d0, progress toward BS, past observed link success, ' ...
        'candidate forwarding burden, and remaining-hop bin.'];
    cfg.actionDefinition = ['Choose one eligible cluster head for each live sensing source, ' ...
        'then one eligible strictly-forward next hop for each aggregate packet; action 51 is BS.'];
    cfg.rewardWeights = struct('delivery', 2.0, 'failure', 1.0, ...
        'linkQuality', 0.15, 'progress', 0.15, 'relayEnergy', 0.10, ...
        'loadBalance', 0.10, 'energyCost', 0.10, 'hopCost', 0.05);
    cfg.energyNormalizationJ = 1e-3; % order-of-magnitude packet radio operation
    cfg.rewardDefinition = ['Terminal delivery +2.0; failed/stranded report -1.0; successful relay: ' ...
        '+0.15*qHat +0.15*progress +0.10*relay-energy-fraction + ' ...
        '0.10*(1-normalized forwarding burden) -0.10*min(1,operation energy/0.001J) -0.05 per hop.'];
    cfg.learningMode = 'online per-node tabular learning within each seed; no cross-node Q sharing';
    cfg.seedPolicy = 'MATLAB twister seeded with the scenario seed';

    E = repmat(wsn_env.initialEnergy, 1, n);
    alive = true(1, n);
    firstUnavailableRound = nan(1, n);
    unavailabilityReason = repmat({''}, 1, n);
    Q = sparse(cfg.numStates, cfg.numActions*n);
    epsilon = cfg.epsilonInitial;
    neighborRange = Inf; % shared channel model has no hard geographic cutoff
    priorLQ = distancePriorLQ(wsn_env); % geometry-only prior, n x (n+1)
    decisionLog = cell(1, rounds);

    scheduled = zeros(1, rounds); workloadAvailable = zeros(1, rounds);
    sourceAvailable = zeros(1, rounds); sourceUnavailable = zeros(1, rounds);
    routingEntered = zeros(1, rounds); attempts = zeros(1, rounds);
    physicalFrames = zeros(1, rounds); delivered = zeros(1, rounds);
    successfulStageFrames = zeros(1, rounds); failedStageFrames = zeros(1, rounds);
    stageLoss = zeros(1, rounds); unattempted = zeros(1, rounds);
    aliveCount = zeros(1, rounds); physicalEnergy = zeros(1, rounds);
    residualEnergyHistory = zeros(n, rounds); isAliveHistory = false(n, rounds);
    sourceAvailabilityHistory = false(length(sensingNodes), rounds);
    epsilonHistory = zeros(1, rounds); rewardHistory = zeros(1, rounds);
    qUpdateCount = zeros(1, rounds); explorationCount = zeros(1, rounds);
    exploitationCount = zeros(1, rounds); fallbackCount = zeros(1, rounds);
    invalidActionCount = 0; selfAssignmentCount = 0; loopViolationCount = 0;
    actionSelectionCounts = zeros(1, cfg.numActions);
    clusterHeadSelectionCounts = zeros(1, n);
    txExpenditure = zeros(1, rounds); rxExpenditure = zeros(1, rounds);
    aggregationExpenditure = zeros(1, rounds); delayDelivered = zeros(1, rounds);
    deliveredHopTotal = zeros(1, rounds); deliveredHopReports = zeros(1, rounds);
    txEnergyByNode=zeros(1,n); rxEnergyByNode=zeros(1,n); aggEnergyByNode=zeros(1,n);
    txFramesByNode=zeros(1,n); rxFramesByNode=zeros(1,n);
    txReportLoadByNode=zeros(1,n); rxReportLoadByNode=zeros(1,n);
    relaySelectionsByNode=zeros(1,n);
    linkSuccessObservations=zeros(n,n+1);
    linkAttemptObservations=zeros(n,n+1);
    routeFailureNoCandidate=zeros(1,rounds);
    routeFailureChannel=zeros(1,rounds);
    routeFailureEnergy=zeros(1,rounds);
    routeFailureTTL=zeros(1,rounds);
    deliveredReportHopCounts=zeros(1,0);
    forwardingLoadEWMA=zeros(1,n);

    Rbit = 250e3; Tproc = 0.005; c = 3e8;
    nominalHopDelay = wsn_env.packetSize/Rbit + Tproc + wsn_env.riverLength/c;

    for r = 1:rounds
        energyStart = sum(E); % all 50 physical batteries, including disabled nodes
        aliveAtStart = alive;
        scheduled(r) = length(sensingNodes);
        obsToday = app_workload.round_obs{r};
        workloadAvailable(r) = length(obsToday);
        sourceNodes = app_workload.obs_src_nodes(obsToday);
        reportCounts = zeros(1, n);
        for oi = 1:length(sourceNodes)
            reportCounts(sourceNodes(oi)) = reportCounts(sourceNodes(oi)) + 1;
        end
        for si = 1:length(sensingNodes)
            sourceAvailabilityHistory(si,r) = aliveAtStart(sensingNodes(si)) && ...
                reportCounts(sensingNodes(si)) > 0;
        end
        sourceAvailable(r) = sum(reportCounts(aliveAtStart));
        sourceUnavailable(r) = scheduled(r) - sourceAvailable(r);
        assert(sourceAvailable(r) <= workloadAvailable(r));

        nodeLoad = zeros(1,n);
        clusterReports = zeros(1, n);
        clusterUnattempted = zeros(1, n);
        clusterDelay = zeros(1, n);
        clusterMemberNodes = cell(1,n);
        clusterIngressHopReports = zeros(1,n);
        qEst = currentLinkEstimate(); % causal: prior + own past outcomes
        rewardSum = 0; rewardCount = 0;

        % RL cluster assignment: each available sensing source selects one
        % reachable live head (self is always a no-radio-cost option).
        for si = 1:length(sensingNodes)
            src = sensingNodes(si); reps = reportCounts(src);
            if reps == 0 || ~alive(src), continue; end
            cands = src;
            for j = 1:n
                if j == src || ~alive(j), continue; end
                d = hypot(wsn_env.X(src)-wsn_env.X(j), wsn_env.Y(src)-wsn_env.Y(j));
                eTx = tx_energy(wsn_env.packetSize, d, wsn_env) * reps;
                eRx = rx_energy(wsn_env.packetSize, wsn_env) * reps;
                if qEst(src,j) >= cfg.minimumLinkQuality && ...
                        E(src) >= eTx && E(j) >= eRx
                    cands(end+1) = j; %#ok<AGROW>
                end
            end
            localMeanLoad=neighborLoadMean(cands,nodeLoad,forwardingLoadEWMA);
            [chStates, ch] = candidateStates(src, cands, E, alive, nodeLoad, ...
                wsn_env, qEst, bs, linkSuccessObservations, linkAttemptObservations, 0, ...
                forwardingLoadEWMA);
            [ch, chosenState, explored] = selectAction(Q, chStates, cands, epsilon,src,cfg.numActions);
            assert(ismember(ch,cands) && alive(ch));
            if ch==src, selfAssignmentCount=selfAssignmentCount+1; end
            recordAction(ch, explored);
            clusterHeadSelectionCounts(ch) = clusterHeadSelectionCounts(ch) + reps;
            routingEntered(r) = routingEntered(r) + reps;

            if ch == src
                clusterReports(ch) = clusterReports(ch) + reps;
                clusterUnattempted(ch) = clusterUnattempted(ch) + reps;
                clusterMemberNodes{ch}(end+1)=src;
                logDecision(0, src, src, reps, [1, 1, 0, 0, E(src), E(src)], NaN);
                % No radio operation has occurred yet. A positive immediate
                % reward here would encourage self-selection even if routing
                % later strands the report; delivery is rewarded by routing.
                reward = 0;
                Q = qUpdate(Q, chosenState, ch+(src-1)*cfg.numActions, reward, 0, cfg);
                recordReward(reward);
            else
                d = hypot(wsn_env.X(src)-wsn_env.X(ch), wsn_env.Y(src)-wsn_env.Y(ch));
                eTx = tx_energy(wsn_env.packetSize, d, wsn_env) * reps;
                if E(src) + 1e-15 < eTx
                    alive(src) = false;
                    setUnavailable(src, r, 'insufficient_tx_energy');
                    continue;
                end
                % Pre-decision snapshot (what the agent knew when it chose ch).
                ctx = [qEst(src,ch), priorLQ(src,ch), linkAttemptObservations(src,ch), ...
                    linkSuccessObservations(src,ch), E(src), E(ch)];
                E(src) = max(0, E(src)-eTx); txExpenditure(r) = txExpenditure(r)+eTx;
                txEnergyByNode(src)=txEnergyByNode(src)+eTx;
                txFramesByNode(src)=txFramesByNode(src)+reps;
                txReportLoadByNode(src)=txReportLoadByNode(src)+reps;
                physicalFrames(r) = physicalFrames(r) + reps;
                attempts(r) = attempts(r) + reps;
                if E(src) == 0
                    alive(src) = false; setUnavailable(src, r, 'battery_depleted');
                end
                linkAttemptObservations(src,ch)=linkAttemptObservations(src,ch)+1;
                % Channel outcome is read only now, after the frame was paid for.
                linkOK = logical(success_history(src,ch,r));
                logDecision(1, src, ch, reps, ctx, double(linkOK));
                lq = ctx(1); % causal estimate that was visible at decision time
                hopDelay = wsn_env.packetSize/Rbit + Tproc + d/c;
                if linkOK
                    linkSuccessObservations(src,ch)=linkSuccessObservations(src,ch)+1;
                    eRx = rx_energy(wsn_env.packetSize, wsn_env) * reps;
                    if alive(ch) && E(ch) + 1e-15 >= eRx
                        successfulStageFrames(r)=successfulStageFrames(r)+reps;
                        E(ch) = max(0,E(ch)-eRx); rxExpenditure(r)=rxExpenditure(r)+eRx;
                        rxEnergyByNode(ch)=rxEnergyByNode(ch)+eRx;
                        rxFramesByNode(ch)=rxFramesByNode(ch)+reps;
                        rxReportLoadByNode(ch)=rxReportLoadByNode(ch)+reps;
                        clusterReports(ch) = clusterReports(ch) + reps;
                        clusterMemberNodes{ch}(end+1)=src;
                        clusterIngressHopReports(ch)=clusterIngressHopReports(ch)+reps;
                        clusterDelay(ch) = clusterDelay(ch) + reps*hopDelay;
                        nodeLoad(ch) = nodeLoad(ch) + reps;
                        if E(ch) == 0
                            alive(ch) = false; setUnavailable(ch,r,'battery_depleted');
                            stranded = clusterReports(ch)-clusterUnattempted(ch);
                            stageLoss(r)=stageLoss(r)+stranded;
                            routeFailureEnergy(r)=routeFailureEnergy(r)+stranded;
                            clusterReports(ch)=clusterUnattempted(ch);
                        end
                        progress = hypot(wsn_env.X(src)-wsn_env.BS_X,wsn_env.Y(src)-wsn_env.BS_Y) - ...
                            hypot(wsn_env.X(ch)-wsn_env.BS_X,wsn_env.Y(ch)-wsn_env.BS_Y);
                        reward = transitionReward(true,false,lq, eTx+eRx, ...
                            E(ch)/wsn_env.initialEnergy, progress, wsn_env.riverLength, ...
                            nodeLoad(ch), localMeanLoad, hopDelay, nominalHopDelay);
                        Q=qUpdate(Q,chosenState,ch+(src-1)*cfg.numActions,reward,0,cfg); recordReward(reward);
                    else
                        routeFailureEnergy(r)=routeFailureEnergy(r)+reps;
                        failedStageFrames(r)=failedStageFrames(r)+reps;
                        if alive(ch)
                            alive(ch)=false; setUnavailable(ch,r,'insufficient_rx_energy');
                        end
                        stageLoss(r)=stageLoss(r)+reps;
                        stranded = clusterReports(ch)-clusterUnattempted(ch);
                        stageLoss(r)=stageLoss(r)+max(0,stranded);
                        clusterReports(ch)=clusterUnattempted(ch);
                        reward=transitionReward(false,false,lq,eTx,0,d, ...
                            wsn_env.riverLength,nodeLoad(ch),localMeanLoad, ...
                            hopDelay,nominalHopDelay);
                        Q=qUpdate(Q,chosenState,ch+(src-1)*cfg.numActions,reward,0,cfg); recordReward(reward);
                    end
                else
                    routeFailureChannel(r)=routeFailureChannel(r)+reps;
                    failedStageFrames(r)=failedStageFrames(r)+reps;
                    stageLoss(r)=stageLoss(r)+reps;
                    reward=transitionReward(false,false,lq,eTx,0,d, ...
                        wsn_env.riverLength,nodeLoad(ch),localMeanLoad, ...
                        hopDelay,nominalHopDelay);
                    Q=qUpdate(Q,chosenState,ch+(src-1)*cfg.numActions,reward,0,cfg); recordReward(reward);
                end
            end
        end

        % RL multi-hop forwarding. Each cluster payload is a real aggregated
        % frame; relays are selected from current-round feasible forward links.
        for ch = 1:n
            reports = clusterReports(ch);
            pendingOwn = clusterUnattempted(ch);
            if reports == 0, continue; end
            if ~alive(ch)
                % Unavailable before its own forwarding turn (e.g. died while relaying
                % another payload): reports it had received are stranded; its own
                % never-transmitted reports stay unattempted.
                strandedReports = max(0, reports-pendingOwn);
                stageLoss(r) = stageLoss(r) + strandedReports;
                routeFailureEnergy(r) = routeFailureEnergy(r) + strandedReports;
                clusterReports(ch) = pendingOwn;
                continue;
            end
            eAgg = wsn_env.E_DA * wsn_env.packetSize * reports;
            if E(ch) + 1e-15 < eAgg
                alive(ch)=false; setUnavailable(ch,r,'insufficient_aggregation_energy');
                stageLoss(r)=stageLoss(r)+max(0,reports-pendingOwn);
                routeFailureEnergy(r)=routeFailureEnergy(r)+max(0,reports-pendingOwn);
                continue;
            end
            E(ch)=max(0,E(ch)-eAgg); aggregationExpenditure(r)=aggregationExpenditure(r)+eAgg;
            aggEnergyByNode(ch)=aggEnergyByNode(ch)+eAgg;
            if E(ch)==0
                alive(ch)=false; setUnavailable(ch,r,'battery_depleted');
                stageLoss(r)=stageLoss(r)+max(0,reports-pendingOwn);
                routeFailureEnergy(r)=routeFailureEnergy(r)+max(0,reports-pendingOwn);
                continue;
            end
            curr=ch; visited=false(1,n); visited(curr)=true;
            visited(clusterMemberNodes{ch})=true;
            pendingState=0; pendingAction=0; pendingReward=0;
            hops=0; carriedDelay=clusterDelay(ch); routeDone=false;
            while ~routeDone && hops < cfg.maxHops
                qEst = currentLinkEstimate(); % causal snapshot taken before this hop's decision
                cands = routeCandidates(curr,reports,E,alive,visited,wsn_env, ...
                    qEst,bs,neighborRange);
                if isempty(cands)
                    fallbackCount(r)=fallbackCount(r)+1;
                    strandedReports = max(0,reports-pendingOwn);
                    stageLoss(r)=stageLoss(r)+strandedReports;
                    if E(curr)+1e-15 < cheapestAllowedTx(curr,visited,alive,wsn_env)
                        % The operation the protocol requires of this node (forwarding
                        % its payload to any strictly-forward live node or the BS) is
                        % unaffordable: same rule as a LEACH/DEEC CH that cannot afford
                        % its BS transmission. The node is unavailable from now on.
                        alive(curr)=false; setUnavailable(curr,r,'insufficient_tx_energy');
                        routeFailureEnergy(r)=routeFailureEnergy(r)+strandedReports;
                    else
                        routeFailureNoCandidate(r)=routeFailureNoCandidate(r)+strandedReports;
                    end
                    if pendingState>0
                        terminalReturn=pendingReward-cfg.gamma*cfg.rewardWeights.failure;
                        Q=qUpdate(Q,pendingState,pendingAction,terminalReturn,0,cfg);
                        recordReward(terminalReturn);
                    end
                    break;
                end
                [states,~]=candidateStates(curr,cands,E,alive,nodeLoad,wsn_env,qEst,bs, ...
                    linkSuccessObservations,linkAttemptObservations,hops,forwardingLoadEWMA);
                localMeanLoad=neighborLoadMean(cands,nodeLoad,forwardingLoadEWMA);
                if pendingState>0
                    qValues=zeros(size(cands));
                    for qi=1:numel(cands)
                        qValues(qi)=Q(states(qi),cands(qi)+(curr-1)*cfg.numActions);
                    end
                    Q=qUpdate(Q,pendingState,pendingAction,pendingReward,max(qValues),cfg);
                    recordReward(pendingReward);
                    pendingState=0;
                end
                [nexthop,st,explored]=selectAction(Q,states,cands,epsilon,curr,cfg.numActions);
                isVisited = nexthop<=n && visited(nexthop);
                if ~ismember(nexthop,cands) || nexthop==curr || isVisited || ~alive(curr) || (nexthop<=n && ~alive(nexthop))
                    invalidActionCount=invalidActionCount+1;
                    error('RL selected an ineligible, looping, or dead-node route action.');
                end
                if nexthop<=n && hypot(wsn_env.X(nexthop)-wsn_env.BS_X, ...
                        wsn_env.Y(nexthop)-wsn_env.BS_Y) >= ...
                        hypot(wsn_env.X(curr)-wsn_env.BS_X,wsn_env.Y(curr)-wsn_env.BS_Y)
                    loopViolationCount=loopViolationCount+1;
                    error('RL route action does not strictly progress toward the BS.');
                end
                recordAction(nexthop,explored);
                if nexthop<=n, relaySelectionsByNode(nexthop)=relaySelectionsByNode(nexthop)+1; end
                d=distanceTo(curr,nexthop,wsn_env,bs);
                eTx=tx_energy(wsn_env.packetSize*reports,d,wsn_env);   % RP: m reports = m packets
                if E(curr)+1e-15 < eTx
                    routeFailureEnergy(r)=routeFailureEnergy(r)+max(0,reports-pendingOwn);
                    alive(curr)=false; setUnavailable(curr,r,'insufficient_tx_energy');
                    fallbackCount(r)=fallbackCount(r)+1;
                    stageLoss(r)=stageLoss(r)+max(0,reports-pendingOwn);
                    break;
                end
                % Pre-decision snapshot (what the agent knew when it chose nexthop).
                if nexthop==bs, ctxEnergyTarget=NaN; else, ctxEnergyTarget=E(nexthop); end
                ctx=[qEst(curr,nexthop), priorLQ(curr,nexthop), linkAttemptObservations(curr,nexthop), ...
                    linkSuccessObservations(curr,nexthop), E(curr), ctxEnergyTarget];
                E(curr)=max(0,E(curr)-eTx); txExpenditure(r)=txExpenditure(r)+eTx;
                txEnergyByNode(curr)=txEnergyByNode(curr)+eTx;
                txFramesByNode(curr)=txFramesByNode(curr)+1;
                txReportLoadByNode(curr)=txReportLoadByNode(curr)+reports;
                physicalFrames(r)=physicalFrames(r)+1;
                attempts(r)=attempts(r)+pendingOwn; pendingOwn=0;
                if E(curr)==0
                    alive(curr)=false; setUnavailable(curr,r,'battery_depleted');
                end
                lq=ctx(1); % causal estimate that was visible at decision time
                hopDelay=wsn_env.packetSize/Rbit+Tproc+d/c;
                hops=hops+1;
                linkAttemptObservations(curr,nexthop)=linkAttemptObservations(curr,nexthop)+1;
                % Channel outcome is read only now, after the frame was paid for.
                chanOK = logical(success_history(curr,nexthop,r));
                logDecision(2+(nexthop==bs), curr, nexthop, reports, ctx, double(chanOK));
                if ~chanOK
                    routeFailureChannel(r)=routeFailureChannel(r)+reports;
                    failedStageFrames(r)=failedStageFrames(r)+1;
                    stageLoss(r)=stageLoss(r)+reports;
                    reward=transitionReward(false,nexthop==bs,lq,eTx,0,d, ...
                        wsn_env.riverLength,0,localMeanLoad,hopDelay,nominalHopDelay);
                    Q=qUpdate(Q,st,nexthop+(curr-1)*cfg.numActions,reward,0,cfg); recordReward(reward);
                    routeDone=true;
                elseif nexthop==bs
                    linkSuccessObservations(curr,nexthop)=linkSuccessObservations(curr,nexthop)+1;
                    successfulStageFrames(r)=successfulStageFrames(r)+1;
                    delivered(r)=delivered(r)+reports;
                    carriedDelay=carriedDelay+reports*hopDelay;
                    delayDelivered(r)=delayDelivered(r)+carriedDelay;
                    deliveredHopTotal(r)=deliveredHopTotal(r)+reports*hops+clusterIngressHopReports(ch);
                    deliveredHopReports(r)=deliveredHopReports(r)+reports;
                    ownPathReports=max(0,reports-clusterIngressHopReports(ch));
                    deliveredReportHopCounts=[deliveredReportHopCounts ...
                        repmat(hops+1,1,clusterIngressHopReports(ch)) ...
                        repmat(hops,1,ownPathReports)]; %#ok<AGROW>
                    reward=transitionReward(true,true,lq,eTx,1,d, ...
                        wsn_env.riverLength,0,localMeanLoad,hopDelay,nominalHopDelay);
                    Q=qUpdate(Q,st,nexthop+(curr-1)*cfg.numActions,reward,0,cfg); recordReward(reward);
                    routeDone=true;
                else
                    linkSuccessObservations(curr,nexthop)=linkSuccessObservations(curr,nexthop)+1;
                    eRx=rx_energy(wsn_env.packetSize*reports,wsn_env);   % RP
                    eAggRelay=wsn_env.E_DA*wsn_env.packetSize*reports;
                    needed=eRx+eAggRelay;
                    if ~alive(nexthop) || E(nexthop)+1e-15 < needed
                        routeFailureEnergy(r)=routeFailureEnergy(r)+reports;
                        failedStageFrames(r)=failedStageFrames(r)+1;
                        if alive(nexthop)
                            alive(nexthop)=false;
                            if E(nexthop)+1e-15 < eRx
                                setUnavailable(nexthop,r,'insufficient_rx_energy');
                            else
                                setUnavailable(nexthop,r,'insufficient_aggregation_energy');
                            end
                        end
                        stageLoss(r)=stageLoss(r)+reports;
                        reward=transitionReward(false,false,lq,eTx,0,d, ...
                            wsn_env.riverLength,nodeLoad(nexthop),localMeanLoad, ...
                            hopDelay,nominalHopDelay);
                        Q=qUpdate(Q,st,nexthop+(curr-1)*cfg.numActions,reward,0,cfg); recordReward(reward);
                        routeDone=true;
                    else
                        successfulStageFrames(r)=successfulStageFrames(r)+1;
                        E(nexthop)=E(nexthop)-needed;
                        rxExpenditure(r)=rxExpenditure(r)+eRx;
                        aggregationExpenditure(r)=aggregationExpenditure(r)+eAggRelay;
                        rxEnergyByNode(nexthop)=rxEnergyByNode(nexthop)+eRx;
                        rxFramesByNode(nexthop)=rxFramesByNode(nexthop)+1;
                        rxReportLoadByNode(nexthop)=rxReportLoadByNode(nexthop)+reports;
                        aggEnergyByNode(nexthop)=aggEnergyByNode(nexthop)+eAggRelay;
                        nodeLoad(nexthop)=nodeLoad(nexthop)+reports;
                        if E(nexthop)==0
                            alive(nexthop)=false; setUnavailable(nexthop,r,'battery_depleted');
                            stageLoss(r)=stageLoss(r)+reports;
                            routeFailureEnergy(r)=routeFailureEnergy(r)+reports;
                            routeDone=true;
                            reward=transitionReward(false,false,lq,eTx+needed,0,d, ...
                                wsn_env.riverLength,nodeLoad(nexthop),localMeanLoad, ...
                                hopDelay,nominalHopDelay);
                            Q=qUpdate(Q,st,nexthop+(curr-1)*cfg.numActions,reward,0,cfg); recordReward(reward);
                        else
                            carriedDelay=carriedDelay+reports*hopDelay;
                            progress=hypot(wsn_env.X(curr)-wsn_env.BS_X,wsn_env.Y(curr)-wsn_env.BS_Y)- ...
                                hypot(wsn_env.X(nexthop)-wsn_env.BS_X,wsn_env.Y(nexthop)-wsn_env.BS_Y);
                            reward=transitionReward(true,false,lq,eTx+needed, ...
                                E(nexthop)/wsn_env.initialEnergy,progress,wsn_env.riverLength, ...
                                nodeLoad(nexthop),localMeanLoad,hopDelay,nominalHopDelay);
                            pendingState=st; pendingAction=nexthop+(curr-1)*cfg.numActions; pendingReward=reward;
                            curr=nexthop; visited(curr)=true;
                        end
                    end
                end
            end
            if ~routeDone && hops >= cfg.maxHops
                fallbackCount(r)=fallbackCount(r)+1;
                routeFailureTTL(r)=routeFailureTTL(r)+max(0,reports-pendingOwn);
                stageLoss(r)=stageLoss(r)+max(0,reports-pendingOwn);
                if pendingState>0
                    terminalReturn=pendingReward-cfg.gamma*cfg.rewardWeights.failure;
                    Q=qUpdate(Q,pendingState,pendingAction,terminalReturn,0,cfg);
                    recordReward(terminalReturn);
                end
            end
        end

        epsilonHistory(r)=epsilon;
        epsilon=max(cfg.epsilonMinimum,epsilon*cfg.epsilonDecayPerRound);
        unattempted(r)=sourceAvailable(r)-attempts(r);
        assert(scheduled(r)==sourceUnavailable(r)+sourceAvailable(r));
        assert(sourceAvailable(r)==unattempted(r)+attempts(r));
        assert(attempts(r)==delivered(r)+stageLoss(r), ...
            'RL report attempt/loss conservation failed; loss events are stage traced.');
        assert(physicalFrames(r)==successfulStageFrames(r)+failedStageFrames(r), ...
            'Physical frame attempts do not match stage outcomes.');
        assert(all([scheduled(r),workloadAvailable(r),sourceAvailable(r), ...
            routingEntered(r),attempts(r),delivered(r),stageLoss(r),unattempted(r)]>=0));
        assert(all(E>=0) && all(E<=wsn_env.initialEnergy+1e-12));
        forwardingLoadEWMA=0.90*forwardingLoadEWMA+0.10*nodeLoad;

        % Energy consumed only at successful physical operation debits.
        energyEnd=sum(E);
        spent=txExpenditure(r)+rxExpenditure(r)+aggregationExpenditure(r);
        assert(abs((energyStart-energyEnd)-spent)<1e-10, ...
            'Physical battery-energy conservation failed in RL-Hybrid.');
        assert(energyEnd<=energyStart+1e-12);
        % A node that cannot afford even the cheapest radio operation of the
        % frozen energy model (one packet reception, E_elec*k) can take part in
        % no protocol function and is unavailable from the next round on.
        for i=find(alive & (E < cfg.minRadioOperationEnergyJ))
            alive(i)=false;
            if E(i)==0, setUnavailable(i,r,'battery_depleted');
            else, setUnavailable(i,r,'below_minimum_radio_operation_energy'); end
        end
        aliveCount(r)=sum(alive);
        physicalEnergy(r)=sum(E);
        residualEnergyHistory(:,r)=E(:);
        isAliveHistory(:,r)=alive(:);
        if rewardCount>0, rewardHistory(r)=rewardSum/rewardCount; end
    end

    fnd=find(aliveCount< n,1,'first'); if isempty(fnd), fnd=NaN; end
    hnd=find(aliveCount<= n/2,1,'first'); if isempty(hnd), hnd=NaN; end
    lnd=find(aliveCount==0,1,'first'); if isempty(lnd), lnd=NaN; end
    cumPDR=100*cumsum(delivered)./cumsum(scheduled);
    assert(abs(sum(txEnergyByNode)-sum(txExpenditure))<1e-10);
    assert(abs(sum(rxEnergyByNode)-sum(rxExpenditure))<1e-10);
    assert(abs(sum(aggEnergyByNode)-sum(aggregationExpenditure))<1e-10);

    res=struct();
    res.protocol='RL-Hybrid'; res.seed=seed; res.topologySeed=seed;
    res.channelSeedRule='twister(seed + round), regenerated by generate_scenario';
    res.rlConfig=cfg; res.finalQ=Q;
    res.sensingNodeIds=sensingNodes;
    res.scheduledPerRound=scheduled;
    res.workloadAvailablePerRound=workloadAvailable;
    res.sourceAvailablePerRound=sourceAvailable;
    res.sourceUnavailablePerRound=sourceUnavailable;
    res.routingEnteredPerRound=routingEntered;
    res.actualTxAttemptReportsPerRound=attempts;
    res.physicalTransmissionFramesPerRound=physicalFrames;
    res.successfulStageFramesPerRound=successfulStageFrames;
    res.failedStageFramesPerRound=failedStageFrames;
    res.deliveredPerRound=delivered; res.stageLossPerRound=stageLoss;
    res.unattemptedPerRound=unattempted;
    res.aliveCountPerRound=aliveCount;
    res.physicalResidualEnergyPerRound=physicalEnergy;
    res.residualEnergyHistory=residualEnergyHistory;
    res.isAliveHistory=isAliveHistory;
    res.sensingSourceAvailabilityHistory=sourceAvailabilityHistory;
    res.firstUnavailableRound=firstUnavailableRound;
    res.unavailabilityReason=unavailabilityReason;
    res.epsilonHistory=epsilonHistory; res.rewardHistory=rewardHistory;
    res.qUpdateCountPerRound=qUpdateCount;
    res.explorationActionCountPerRound=explorationCount;
    res.exploitationActionCountPerRound=exploitationCount;
    res.actionSelectionCounts=actionSelectionCounts;
    res.clusterHeadSelectionCounts=clusterHeadSelectionCounts;
    res.routingFallbackPerRound=fallbackCount;
    res.invalidActionCount=invalidActionCount;
    res.selfAssignmentCount=selfAssignmentCount;
    res.routingLoopViolationCount=loopViolationCount;
    res.txEnergyExpended=txExpenditure; res.rxEnergyExpended=rxExpenditure;
    res.aggregationEnergyExpended=aggregationExpenditure;
    res.txEnergyByNode=txEnergyByNode; res.rxEnergyByNode=rxEnergyByNode;
    res.aggregationEnergyByNode=aggEnergyByNode;
    res.txFramesByNode=txFramesByNode; res.rxFramesByNode=rxFramesByNode;
    res.txReportLoadByNode=txReportLoadByNode; res.rxReportLoadByNode=rxReportLoadByNode;
    res.relaySelectionsByNode=relaySelectionsByNode;
    res.routeFailureNoCandidatePerRound=routeFailureNoCandidate;
    res.routeFailureChannelPerRound=routeFailureChannel;
    res.routeFailureEnergyPerRound=routeFailureEnergy;
    res.routeFailureTTLPerRound=routeFailureTTL;
    res.linkSuccessObservations=linkSuccessObservations;
    res.linkAttemptObservations=linkAttemptObservations;
    res.recentForwardingLoad=forwardingLoadEWMA;
    res.deliveredReportHopCounts=deliveredReportHopCounts;
    res.actualEnergyExpendedPerRound=txExpenditure+rxExpenditure+aggregationExpenditure;
    res.meanEndToEndDelayPerRound=delayDelivered./max(1,delivered);
    res.meanHopsPerDeliveredReport=deliveredHopTotal./max(1,deliveredHopReports);
    res.deliveredDelayTotalPerRound=delayDelivered;
    res.deliveredHopTotalPerRound=deliveredHopTotal;
    res.deliveredHopReportsPerRound=deliveredHopReports;
    res.cumulativeScheduledPDR=cumPDR;
    res.totalScheduledReports=sum(scheduled);
    res.totalWorkloadAvailableReports=sum(workloadAvailable);
    res.totalSourceAvailableReports=sum(sourceAvailable);
    res.totalDeliveredReports=sum(delivered);
    res.totalTransmissionAttemptReports=sum(attempts);
    res.totalStageLossReports=sum(stageLoss);
    res.totalSuccessfulStageFrames=sum(successfulStageFrames);
    res.totalFailedStageFrames=sum(failedStageFrames);
    res.totalUnattemptedReports=sum(unattempted);
    res.applicationPDR=100*sum(delivered)/sum(scheduled);
    res.sourceAvailableDeliveryRatio=100*sum(delivered)/max(1,sum(sourceAvailable));
    res.energyPerDeliveredReport=sum(res.actualEnergyExpendedPerRound)/max(1,sum(delivered));
    res.FND=fnd; res.HND=hnd; res.LND=lnd;
    res.decisionLog=vertcat(decisionLog{:});
    res.decisionLogColumns={'round','hopType(0self,1member,2relay,3BS)','actor','target','reports', ...
        'qHatBeforeAction','distancePrior','ownAttemptsBefore','ownSuccessesBefore','actorEnergyBefore','targetEnergyBefore','channelOutcome'};
    res.distancePriorLQ=priorLQ;
    res.finalNodeEnergies=E; res.finalAliveCount=aliveCount(end);
    res.finalPhysicalResidualEnergy=physicalEnergy(end);
    res.wsn_env=wsn_env;

    function Qh = currentLinkEstimate()
        % Causal link-quality estimate: distance prior combined with the acting
        % node's own past outcomes. Uses only state accumulated before now.
        Qh = (linkSuccessObservations + cfg.linkPriorStrength*priorLQ) ./ ...
             (linkAttemptObservations + cfg.linkPriorStrength);
    end
    function logDecision(hopType, actor, target, nReports, ctxRow, outcome)
        % One row per action: [round hopType actor target reports qHat p0 attemptsBefore
        % successesBefore actorEnergy targetEnergy channelOutcome]; hopType 0=self head,
        % 1=member->head, 2=relay hop, 3=hop to BS. Outcome is appended after the action.
        decisionLog{r}(end+1,:) = [r, hopType, actor, target, nReports, ctxRow, outcome];
    end
    function recordAction(action, explored)
        actionSelectionCounts(action)=actionSelectionCounts(action)+1;
        if explored, explorationCount(r)=explorationCount(r)+1;
        else, exploitationCount(r)=exploitationCount(r)+1; end
    end
    function recordReward(value)
        rewardSum=rewardSum+value; rewardCount=rewardCount+1;
        qUpdateCount(r)=qUpdateCount(r)+1;
    end
    function setUnavailable(node, roundNow, reason)
        if isnan(firstUnavailableRound(node))
            firstUnavailableRound(node)=roundNow+1;
            unavailabilityReason{node}=reason;
        end
    end
end

function [states,candidates] = candidateStates(curr,candidates,E,alive,load,env,qEst,bs,linkSuccess,linkAttempts,hops,forwardingLoadEWMA)
    % Candidate-conditioned state: Q values distinguish each action's actual link.
    states=zeros(size(candidates));
    for k=1:numel(candidates)
        states(k)=hybridState(curr,candidates(k),E,alive,load+forwardingLoadEWMA,env,qEst,bs, ...
            linkSuccess,linkAttempts,hops);
    end
end

function state=hybridState(curr,candidate,E,alive,load,env,qEst,bs,linkSuccess,linkAttempts,hops)
    if curr<=env.numNodes, currentEnergy=E(curr)/env.initialEnergy; else, currentEnergy=1; end
    if candidate==bs, candidateEnergy=1; distance=0; candidateLoad=0;
        dCandidateBS=0; linkLQ=qEst(curr,bs);
    elseif candidate==curr
        candidateEnergy=E(candidate)/env.initialEnergy; distance=0; candidateLoad=load(candidate);
        dCandidateBS=hypot(env.X(candidate)-env.BS_X,env.Y(candidate)-env.BS_Y);
        linkLQ=1; % local self-head action requires no radio link
    else
        candidateEnergy=E(candidate)/env.initialEnergy;
        distance=hypot(env.X(curr)-env.X(candidate),env.Y(curr)-env.Y(candidate));
        candidateLoad=load(candidate);
        dCandidateBS=hypot(env.X(candidate)-env.BS_X,env.Y(candidate)-env.BS_Y);
        linkLQ=qEst(curr,candidate);
    end
    if curr==bs, dCurrentBS=0;
    else, dCurrentBS=hypot(env.X(curr)-env.BS_X,env.Y(curr)-env.BS_Y); end
    progress=max(0,(dCurrentBS-dCandidateBS)/max(1,dCurrentBS));
    if linkAttempts(curr,candidate)==0, pastReliability=0.5;
    else, pastReliability=(linkSuccess(curr,candidate)+1)/(linkAttempts(curr,candidate)+2); end
    b=[energyBin(currentEnergy), energyBin(candidateEnergy), ...
        ternaryBin(linkLQ,0.65,0.80), ternaryBin(distance/max(env.d0,eps),1,2), ...
        ternaryBin(progress,0.05,0.25), ternaryBin(pastReliability,0.50,0.75), ...
        ternaryBin(candidateLoad,1,4), ternaryBin(max(0,15-hops),4,10)];
    state=1;
    for k=1:numel(b), state=state+(b(k)-1)*3^(numel(b)-k); end
end

function b=ternaryBin(v,t1,t2)
    if v<=t1, b=1; elseif v<=t2, b=2; else, b=3; end
end

function value=neighborLoadMean(candidates,roundLoad,loadEWMA)
    % Only currently eligible local candidates contribute; BS has zero relay load.
    values=zeros(size(candidates)); isNode=candidates<=numel(roundLoad);
    nodes=candidates(isNode);
    values(isNode)=roundLoad(nodes)+loadEWMA(nodes);
    value=mean(values)+1;
end

function b=energyBin(ratio)
    if ratio<=0.33, b=1; elseif ratio<=0.66, b=2; else, b=3; end
end

function [action,state,explored]=selectAction(Q,states,candidates,epsilon,actingNode,numActions)
    if isempty(candidates), error('RL action set unexpectedly empty.'); end
    explored=rand()<epsilon;
    if explored
        k=randi(length(candidates));
    else
        values=zeros(1,length(candidates));
        for j=1:length(candidates)
            values(j)=Q(states(j),candidates(j)+(actingNode-1)*numActions);
        end
        best=find(abs(values-max(values))<1e-12);
        k=best(randi(length(best)));
    end
    action=candidates(k); state=states(k);
end

function cands=routeCandidates(curr,reports,E,alive,visited,env,qEst,bs,range)
    cands=[];
    dcur=hypot(env.X(curr)-env.BS_X,env.Y(curr)-env.BS_Y);
    dBS=dcur; eBS=tx_energy(env.packetSize,dBS,env);
    if qEst(curr,bs)>=0.50 && E(curr)+1e-15>=eBS, cands(end+1)=bs; end
    for j=1:env.numNodes
        if j==curr || visited(j) || ~alive(j), continue; end
        d=hypot(env.X(curr)-env.X(j),env.Y(curr)-env.Y(j));
        djBS=hypot(env.X(j)-env.BS_X,env.Y(j)-env.BS_Y);
        if djBS>=dcur || qEst(curr,j)<0.50, continue; end
        eTx=tx_energy(env.packetSize,d,env);
        eRx=rx_energy(env.packetSize,env)+env.E_DA*env.packetSize*reports;
        if E(curr)+1e-15>=eTx && E(j)+1e-15>=eRx
            cands(end+1)=j; %#ok<AGROW>
        end
    end
end

function reward=transitionReward(success,terminal,qEst,energySpent,relayEnergy, ...
        distance,riverLength,load,meanLoad,hopDelay,nominalDelay)
    energyTerm=min(1,max(0,energySpent/1e-3));
    if ~success
        reward=-1.0-0.10*energyTerm; return;
    end
    if terminal
        reward=2.0+0.15*min(1,max(0,qEst))-0.10*energyTerm-0.05;
        return;
    end
    progress=min(1,max(-1,distance/max(1,riverLength)));
    loadRatio=min(1,max(0,load/max(1,meanLoad)));
    reward=0.15*min(1,max(0,qEst))+0.15*progress+ ...
        0.10*min(1,max(0,relayEnergy))+0.10*(1-loadRatio)- ...
        0.10*energyTerm-0.05;
end

function Q=qUpdate(Q,state,action,reward,nextMax,cfg)
    Q(state,action)=Q(state,action)+cfg.alpha*(reward+cfg.gamma*nextMax-Q(state,action));
end

function d=distanceTo(i,j,env,bs)
    if j==bs, d=hypot(env.X(i)-env.BS_X,env.Y(i)-env.BS_Y);
    else, d=hypot(env.X(i)-env.X(j),env.Y(i)-env.Y(j)); end
end

function e = cheapestAllowedTx(curr,visited,alive,env)
    % Cheapest transmission the protocol could require of curr: the BS or any
    % live, unvisited, strictly-forward node. Link quality and the receiver's
    % energy are deliberately ignored here: they are not the sender's affordability.
    dcur = hypot(env.X(curr)-env.BS_X, env.Y(curr)-env.BS_Y);
    e = tx_energy(env.packetSize, dcur, env);
    for j = 1:env.numNodes
        if j==curr || visited(j) || ~alive(j), continue; end
        if hypot(env.X(j)-env.BS_X, env.Y(j)-env.BS_Y) >= dcur, continue; end
        e = min(e, tx_energy(env.packetSize, hypot(env.X(curr)-env.X(j), env.Y(curr)-env.Y(j)), env));
    end
end

function P = distancePriorLQ(env)
    % E[LQ | distance] for the scenario's channel model
    % LQ = clip(base + 0.3*(xi-0.5), 0.1, 0.95), xi ~ U(0,1), computed in closed
    % form from geometry only. Node-node base = 1-d/(0.8*L); node-BS base = 1-d/(1.2*L).
    n = env.numNodes; P = ones(n, n+1);
    for i = 1:n
        for j = 1:n+1
            if j == i, continue; end
            if j == n+1
                d = hypot(env.X(i)-env.BS_X, env.Y(i)-env.BS_Y); base = 1 - d/(env.riverLength*1.2);
            else
                d = hypot(env.X(i)-env.X(j), env.Y(i)-env.Y(j)); base = 1 - d/(env.riverLength*0.8);
            end
            lo = 0.10; hi = 0.95; a = base-0.15; b = base+0.15; w = b-a;
            pLo = min(1,max(0,(lo-a)/w)); pHi = min(1,max(0,(b-hi)/w));
            m0 = max(a,lo); m1 = min(b,hi);
            if m1 > m0, mid = (m0+m1)/2*(m1-m0)/w; else, mid = 0; end
            P(i,j) = lo*pLo + hi*pHi + mid;
        end
    end
end
