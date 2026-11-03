function res = RL_Hybrid_LearnerComparison(wsn_env, ~, success_history, seed, app_workload, learner)
% Report-preserving energy accounting: a frame carrying m station reports is charged as
% m packets (radio TX/RX energy for m x packetSize bits), as the per-report delivery metric requires.
%RL_HYBRID_LEARNERCOMPARISON RL-Hybrid with a selectable learning method.
% Built on protocols/RL_Hybrid.m. Only the action-value lookup used by action selection and the
% Q-update call sites are routed through learner functions; environment, candidates, constraints, reward,
% lifecycle and accounting are unchanged. learner: 'tabularQ' (reproduces RL_Hybrid exactly), 'doubleQ',
% 'doubleDQN', 'duelingDoubleDQN'. RL_Hybrid_LearnerComparison('selftest') runs the learner unit tests.
    if ischar(wsn_env) && strcmp(wsn_env, 'selftest'), res = learnerSelfTest(); return; end
    if nargin < 6 || isempty(learner), learner = 'tabularQ'; end
    assert(any(strcmp(learner, {'tabularQ','doubleQ','doubleDQN','duelingDoubleDQN'})), 'Unknown learner %s', learner);
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
% a node holding less energy than the cheapest radio operation of the
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
    % ---- learner state (protocol section 3)
    isDQN = any(strcmp(learner, {'doubleDQN','duelingDoubleDQN'})); isDueling = strcmp(learner, 'duelingDoubleDQN');
    QA = []; QB = []; coinStream = []; net = []; tnet = []; adamM = []; adamV = []; adamT = 0; dq = [];
    replayStream = []; bufX = []; bufX2 = []; bufA = []; bufR = []; bufTerm = []; bufN = 0; bufPtr = 0;
    gradSteps = 0; targetSyncs = 0; lossSum = zeros(1, rounds); lossCount = zeros(1, rounds); transitions = 0;
    if strcmp(learner, 'doubleQ')
        QA = Q; QB = Q; coinStream = RandStream('mt19937ar', 'Seed', 600000 + seed);
    end
    if isDQN
        dq = dqnConfig(n);
        net = dqnInit(dq, isDueling, RandStream('mt19937ar', 'Seed', 700000 + seed)); tnet = net;
        fnames = fieldnames(net);
        for fi = 1:numel(fnames), adamM.(fnames{fi}) = zeros(size(net.(fnames{fi}))); adamV.(fnames{fi}) = zeros(size(net.(fnames{fi}))); end
        replayStream = RandStream('mt19937ar', 'Seed', 800000 + seed);
        bufX = zeros(dq.capacity, dq.nIn, 'single'); bufX2 = zeros(dq.capacity, dq.nIn, 'single');
        bufA = zeros(dq.capacity, 1); bufR = zeros(dq.capacity, 1); bufTerm = false(dq.capacity, 1);
    end
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
            [ch, chosenState, explored, xMember] = selectActionL(chStates, cands, epsilon, src);
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
                lrnTerminal(src, chosenState, ch, reward, xMember);
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
                        lrnTerminal(src, chosenState, ch, reward, xMember); recordReward(reward);
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
                        lrnTerminal(src, chosenState, ch, reward, xMember); recordReward(reward);
                    end
                else
                    routeFailureChannel(r)=routeFailureChannel(r)+reps;
                    failedStageFrames(r)=failedStageFrames(r)+reps;
                    stageLoss(r)=stageLoss(r)+reps;
                    reward=transitionReward(false,false,lq,eTx,0,d, ...
                        wsn_env.riverLength,nodeLoad(ch),localMeanLoad, ...
                        hopDelay,nominalHopDelay);
                    lrnTerminal(src, chosenState, ch, reward, xMember); recordReward(reward);
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
            pendingState=0; pendingAction=0; pendingReward=0; pendingActor=0; pendingNode=0; pendingX=[]; %#ok<NASGU>
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
                        lrnTerminal(pendingActor, pendingState, pendingNode, terminalReturn, pendingX);
                        recordReward(terminalReturn);
                    end
                    break;
                end
                [states,~]=candidateStates(curr,cands,E,alive,nodeLoad,wsn_env,qEst,bs, ...
                    linkSuccessObservations,linkAttemptObservations,hops,forwardingLoadEWMA);
                localMeanLoad=neighborLoadMean(cands,nodeLoad,forwardingLoadEWMA);
                if pendingState>0
                    lrnBootstrap(pendingActor, pendingState, pendingNode, pendingReward, pendingX, curr, states, cands);
                    recordReward(pendingReward);
                    pendingState=0;
                end
                [nexthop,st,explored,xCur]=selectActionL(states,cands,epsilon,curr);
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
                    lrnTerminal(curr, st, nexthop, reward, xCur); recordReward(reward);
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
                    lrnTerminal(curr, st, nexthop, reward, xCur); recordReward(reward);
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
                        lrnTerminal(curr, st, nexthop, reward, xCur); recordReward(reward);
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
                            lrnTerminal(curr, st, nexthop, reward, xCur); recordReward(reward);
                        else
                            carriedDelay=carriedDelay+reports*hopDelay;
                            progress=hypot(wsn_env.X(curr)-wsn_env.BS_X,wsn_env.Y(curr)-wsn_env.BS_Y)- ...
                                hypot(wsn_env.X(nexthop)-wsn_env.BS_X,wsn_env.Y(nexthop)-wsn_env.BS_Y);
                            reward=transitionReward(true,false,lq,eTx+needed, ...
                                E(nexthop)/wsn_env.initialEnergy,progress,wsn_env.riverLength, ...
                                nodeLoad(nexthop),localMeanLoad,hopDelay,nominalHopDelay);
                            pendingState=st; pendingAction=nexthop+(curr-1)*cfg.numActions; pendingReward=reward; %#ok<NASGU>
                            pendingActor=curr; pendingNode=nexthop; pendingX=xCur;
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
                    lrnTerminal(pendingActor, pendingState, pendingNode, terminalReturn, pendingX);
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
        % energy model (one packet reception, E_elec*k) can take part in
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
    res.learner=learner; res.learnerTransitions=transitions;
    if strcmp(learner,'doubleQ'), res.finalQA=QA; res.finalQB=QB; end
    if isDQN
        res.dqnConfig=dq; res.finalNet=net; res.gradientSteps=gradSteps; res.targetSyncs=targetSyncs;
        res.meanLossPerRound=lossSum./max(1,lossCount); res.replayStored=bufN;
    end
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

    function [action,state,explored,x]=selectActionL(states,candidates,epsilon,actingNode)
        % Same draws and tie-break as RL_Hybrid's selectAction; only the value lookup is learner-specific.
        if isempty(candidates), error('RL action set unexpectedly empty.'); end
        x=[]; if isDQN, x=encodeInput(states,candidates,actingNode,n); end
        explored=rand()<epsilon;
        if explored
            Lk=randi(length(candidates));
        else
            Lvalues=actionValues(states,candidates,actingNode,x);
            Lbest=find(abs(Lvalues-max(Lvalues))<1e-12);
            Lk=Lbest(randi(length(Lbest)));
        end
        action=candidates(Lk); state=states(Lk);
    end
    function Lvalues=actionValues(states,candidates,actingNode,x)
        Lvalues=zeros(1,length(candidates));
        switch learner
            case 'tabularQ'
                for Lj=1:length(candidates), Lvalues(Lj)=Q(states(Lj),candidates(Lj)+(actingNode-1)*cfg.numActions); end
            case 'doubleQ'
                for Lj=1:length(candidates)
                    Lidx=candidates(Lj)+(actingNode-1)*cfg.numActions; Lvalues(Lj)=QA(states(Lj),Lidx)+QB(states(Lj),Lidx);
                end
            otherwise
                LqAll=dqnForward(net,x,isDueling); Lvalues=LqAll(candidates);   % only valid candidates are selectable
        end
    end
    function lrnTerminal(actor,state,actionNode,rew,x)
        transitions=transitions+1;
        switch learner
            case 'tabularQ'
                Q=qUpdate(Q,state,actionNode+(actor-1)*cfg.numActions,rew,0,cfg);
            case 'doubleQ'
                Lidx=actionNode+(actor-1)*cfg.numActions;
                if rand(coinStream)<0.5, QA(state,Lidx)=QA(state,Lidx)+cfg.alpha*(rew+cfg.gamma*0-QA(state,Lidx));
                else, QB(state,Lidx)=QB(state,Lidx)+cfg.alpha*(rew+cfg.gamma*0-QB(state,Lidx)); end
            otherwise
                storeAndTrain(x,actionNode,rew,[],true);
        end
    end
    function lrnBootstrap(actor,state,actionNode,rew,x,nextActor,nextStates,nextCands)
        transitions=transitions+1;
        switch learner
            case 'tabularQ'
                LqValues=zeros(size(nextCands));
                for Lqi=1:numel(nextCands)
                    LqValues(Lqi)=Q(nextStates(Lqi),nextCands(Lqi)+(nextActor-1)*cfg.numActions);
                end
                Q=qUpdate(Q,state,actionNode+(actor-1)*cfg.numActions,rew,max(LqValues),cfg);
            case 'doubleQ'
                Lidx=actionNode+(actor-1)*cfg.numActions;
                LnIdx=nextCands+(nextActor-1)*cfg.numActions; Lqa=zeros(size(nextCands)); Lqb=Lqa;
                for Lqi=1:numel(nextCands), Lqa(Lqi)=QA(nextStates(Lqi),LnIdx(Lqi)); Lqb(Lqi)=QB(nextStates(Lqi),LnIdx(Lqi)); end
                if rand(coinStream)<0.5
                    [~,Lks]=max(Lqa); QA(state,Lidx)=QA(state,Lidx)+cfg.alpha*(rew+cfg.gamma*Lqb(Lks)-QA(state,Lidx));   % select QA, evaluate QB
                else
                    [~,Lks]=max(Lqb); QB(state,Lidx)=QB(state,Lidx)+cfg.alpha*(rew+cfg.gamma*Lqa(Lks)-QB(state,Lidx));   % select QB, evaluate QA
                end
            otherwise
                storeAndTrain(x,actionNode,rew,encodeInput(nextStates,nextCands,nextActor,n),false);
        end
    end
    function storeAndTrain(x,a,rew,x2,term)
        bufPtr=mod(bufPtr,dq.capacity)+1; bufN=min(bufN+1,dq.capacity);
        bufX(bufPtr,:)=x; bufA(bufPtr)=a; bufR(bufPtr)=rew; bufTerm(bufPtr)=term;
        if term, bufX2(bufPtr,:)=0; else, bufX2(bufPtr,:)=x2; end
        if bufN<dq.warmup, return; end
        Lbi=randi(replayStream,bufN,dq.batch,1);
        Ly=ddqnTargets(net,tnet,bufX2(Lbi,:),bufR(Lbi),bufTerm(Lbi),cfg.gamma,isDueling);
        [LQo,Lcache]=dqnForward(net,bufX(Lbi,:),isDueling);
        Llin=sub2ind(size(LQo),(1:dq.batch)',bufA(Lbi));
        Lerr=LQo(Llin)-Ly; LdQ=zeros(size(LQo)); LdQ(Llin)=max(-dq.huber,min(dq.huber,Lerr))/dq.batch;
        Lg=dqnBackward(net,Lcache,LdQ,isDueling); Lgf=fieldnames(Lg);
        Lnrm=0; for Lgi=1:numel(Lgf), Lnrm=Lnrm+sum(Lg.(Lgf{Lgi})(:).^2); end; Lnrm=sqrt(Lnrm);
        Lscl=1; if Lnrm>dq.clip, Lscl=dq.clip/Lnrm; end
        adamT=adamT+1;
        for Lgi=1:numel(Lgf)
            Lf=Lgf{Lgi}; Lgg=Lg.(Lf)*Lscl;
            adamM.(Lf)=dq.beta1*adamM.(Lf)+(1-dq.beta1)*Lgg; adamV.(Lf)=dq.beta2*adamV.(Lf)+(1-dq.beta2)*Lgg.^2;
            net.(Lf)=net.(Lf)-dq.lr*(adamM.(Lf)/(1-dq.beta1^adamT))./(sqrt(adamV.(Lf)/(1-dq.beta2^adamT))+dq.epsAdam);
        end
        gradSteps=gradSteps+1;
        if mod(gradSteps,dq.targetEvery)==0, tnet=net; targetSyncs=targetSyncs+1; end
        Lae=abs(Lerr); Lhl=(Lae<=dq.huber).*0.5.*Lerr.^2+(Lae>dq.huber).*(dq.huber*(Lae-0.5*dq.huber));
        lossSum(r)=lossSum(r)+mean(Lhl); lossCount(r)=lossCount(r)+1;
    end
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

function x=encodeInput(states,candidates,actingNode,n)
% (n+1) action slots x 8 state bins scaled to {0,0.5,1}; (n+1) validity bits; one-hot acting node (protocol 3C).
    A=n+1; x=zeros(1,A*8+A+n,'single');
    for k=1:numel(candidates)
        c=candidates(k); sv=states(k)-1; b=zeros(1,8);
        for j=8:-1:1, b(j)=mod(sv,3); sv=floor(sv/3); end
        x((c-1)*8+(1:8))=b/2; x(8*A+c)=1;
    end
    x(9*A+actingNode)=1;
end

function dq=dqnConfig(n)
    dq=struct('nIn',9*(n+1)+n,'nH',128,'nOut',n+1,'lr',1e-3,'beta1',0.9,'beta2',0.999,'epsAdam',1e-8,'clip',10, ...
        'capacity',50000,'batch',32,'warmup',500,'targetEvery',500,'huber',1,'activation','ReLU','optimizer','Adam', ...
        'init','He-uniform weights, zero biases','loss','Huber on taken action');
end

function net=dqnInit(dq,dueling,stream)
    he=@(fi,fo) (rand(stream,fi,fo)*2-1)*sqrt(6/fi);
    net.W1=he(dq.nIn,dq.nH); net.b1=zeros(1,dq.nH);
    net.W2=he(dq.nH,dq.nH); net.b2=zeros(1,dq.nH);
    if dueling
        net.Wv=he(dq.nH,1); net.bv=0; net.Wa=he(dq.nH,dq.nOut); net.ba=zeros(1,dq.nOut);
    else
        net.Wo=he(dq.nH,dq.nOut); net.bo=zeros(1,dq.nOut);
    end
end

function [Qv,c]=dqnForward(net,X,dueling)
    X=double(X); z1=X*net.W1+net.b1; h1=max(z1,0); z2=h1*net.W2+net.b2; h2=max(z2,0);
    nA=numActionsOf(net); M=X(:,8*nA+1:9*nA); cnt=max(1,sum(M,2));
    if dueling
        V=h2*net.Wv+net.bv; A=h2*net.Wa+net.ba;
        Qv=V+A-sum(A.*M,2)./cnt;              % Q = V + (A - mean of A over valid actions)
    else
        Qv=h2*net.Wo+net.bo;
    end
    c=struct('X',X,'z1',z1,'h1',h1,'z2',z2,'h2',h2,'M',M,'cnt',cnt);
end

function g=dqnBackward(net,c,dQ,dueling)
    if dueling
        dV=sum(dQ,2); dA=dQ-(sum(dQ,2)./c.cnt).*c.M;
        g.Wv=c.h2'*dV; g.bv=sum(dV,1); g.Wa=c.h2'*dA; g.ba=sum(dA,1);
        dh2=dV*net.Wv'+dA*net.Wa';
    else
        g.Wo=c.h2'*dQ; g.bo=sum(dQ,1); dh2=dQ*net.Wo';
    end
    dz2=dh2.*(c.z2>0); g.W2=c.h1'*dz2; g.b2=sum(dz2,1);
    dz1=(dz2*net.W2').*(c.z1>0); g.W1=c.X'*dz1; g.b1=sum(dz1,1);
end

function [y,aStar,qOn]=ddqnTargets(net,tnet,X2,rew,term,gamma,dueling)
% Double-DQN target: select a* with the ONLINE network over valid actions, evaluate it with the TARGET network.
    qOn=dqnForward(net,X2,dueling); nA=numActionsOf(net); M=double(X2(:,8*nA+1:9*nA))>0; qOn(~M)=-Inf;
    [~,aStar]=max(qOn,[],2);
    qT=dqnForward(tnet,X2,dueling); qa=qT(sub2ind(size(qT),(1:size(qT,1))',aStar));
    y=rew(:)+gamma*qa.*double(~term(:));
end

function T=learnerSelfTest()
% Unit tests for the DQN core (protocol section 5). Pure functions only; no simulation.
    dq=dqnConfig(50); T=struct();
    % 1. Double-DQN target uses online argmax and target evaluation, over valid actions only
    net=dqnInit(dq,false,RandStream('mt19937ar','Seed',1)); tnet=dqnInit(dq,false,RandStream('mt19937ar','Seed',2));
    x=zeros(1,dq.nIn,'single'); x(408+[3 7 51])=1; x(459+5)=1;
    net.Wo(:)=0; net.bo(:)=0; net.bo([3 7 51 20])=[1 5 2 100];      % online prefers 20 (invalid) then 7
    tnet.Wo(:)=0; tnet.bo(:)=0; tnet.bo([3 7 51 20])=[9 4 1 50];     % target prefers 3 among valid
    [y,aStar]=ddqnTargets(net,tnet,x,0.5,false,0.9,false);
    T.ddqnSelectsOnlineArgmaxOverValid = aStar==7;
    T.ddqnEvaluatesWithTarget = abs(y-(0.5+0.9*4))<1e-12;            % not 0.5+0.9*9 (target max) nor 0.5+0.9*5 (online value)
    T.terminalTargetIsReward = abs(ddqnTargets(net,tnet,zeros(1,dq.nIn,'single'),0.7,true,0.9,false)-0.7)<1e-12;
    % 2. invalid-action mask: an invalid action can never be the target argmax even with the largest value
    T.invalidActionMasked = ~ismember(aStar,20);
    % 3. dueling: separate streams and Q = V + A - mean_valid(A)
    dn=dqnInit(dq,true,RandStream('mt19937ar','Seed',3));
    T.duelingHasValueStream = isfield(dn,'Wv') && isequal(size(dn.Wv),[dq.nH 1]);
    T.duelingHasAdvantageStream = isfield(dn,'Wa') && isequal(size(dn.Wa),[dq.nH dq.nOut]) && ~isfield(dn,'Wo');
    X=single(rand(RandStream('mt19937ar','Seed',4),4,dq.nIn)>0.6); [Qd,c]=dqnForward(dn,X,true);
    V=c.h2*dn.Wv+dn.bv; A=c.h2*dn.Wa+dn.ba; M=double(X(:,409:459));
    T.duelingAggregation = max(abs(Qd-(V+A-sum(A.*M,2)./max(1,sum(M,2)))),[],'all')<1e-12;
    % 4. finite-difference gradient checks (both architectures)
    for du=[false true]
        nt=dqnInit(dq,du,RandStream('mt19937ar','Seed',5)); Xg=double(rand(RandStream('mt19937ar','Seed',6),3,dq.nIn)>0.5);
        W=rand(RandStream('mt19937ar','Seed',7),3,dq.nOut);
        [~,cc]=dqnForward(nt,Xg,du); g=dqnBackward(nt,cc,W,du); f=fieldnames(nt); worst=0;
        for fi=1:numel(f)
            for t=1:3
                k=1+mod(t*7919,numel(nt.(f{fi}))); h=1e-6; p=nt; p.(f{fi})(k)=p.(f{fi})(k)+h; m=nt; m.(f{fi})(k)=m.(f{fi})(k)-h;
                num=(sum(dqnForward(p,Xg,du).*W,'all')-sum(dqnForward(m,Xg,du).*W,'all'))/(2*h);
                worst=max(worst,abs(num-g.(f{fi})(k))/max(1,abs(num)));
            end
        end
        if du, T.gradCheckDueling=worst<1e-5; T.gradCheckDuelingErr=worst; else, T.gradCheckDDQN=worst<1e-5; T.gradCheckDDQNErr=worst; end
    end
    % 5. input encoding: bins and validity bits round-trip
    st=[1+2*3^7 6561]; xe=encodeInput(st,[4 51],9,50);
    T.encodingRoundTrip = isequal(double(xe(25:32)),[1 0 0 0 0 0 0 0]) && isequal(double(xe(401:408)),ones(1,8)) && ...
        xe(412)==1 && xe(459)==1 && xe(468)==1 && nnz(xe(409:459))==2;
end

function A=numActionsOf(net)
% number of action slots (n+1); 51 for a 50-node network
    if isfield(net,'ba'), A=numel(net.ba); else, A=numel(net.bo); end
end
