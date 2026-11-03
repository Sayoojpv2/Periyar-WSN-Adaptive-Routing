function m = extract_metrics(res, env)
% EXTRACT_METRICS Per-round series and summary values from one protocol run.
% Works for the RL-Hybrid/learner result format and the LEACH/DEEC/PEGASIS(-RL) format.
    N = env.numNodes; E0 = N * env.initialEnergy;
    % Report accounting per round (application reports, the unit every protocol carries):
    % gen = reports of operational stations that had data; att = generated reports actually sent;
    % lost = sent but not delivered; unatt = generated but never sent.
    if isfield(res, 'isAliveHistory')
        A = res.isAliveHistory; del = res.deliveredPerRound; sch = res.scheduledPerRound;
        E = res.residualEnergyHistory; ex = res.actualEnergyExpendedPerRound;
        delay = res.meanEndToEndDelayPerRound; delay(del == 0) = NaN;
        gen = res.sourceAvailablePerRound; att = res.actualTxAttemptReportsPerRound;
        lost = res.stageLossPerRound; unatt = res.unattemptedPerRound;
    else
        A = res.is_alive_history; del = res.deliveredReports; sch = res.scheduledReports;
        E = res.E_residual_history; ex = res.actualEnergyExpendedThisRound;
        delay = res.averageEndToEndDelay; delay(del == 0) = NaN;
        gen = res.sourceAvailableReports; att = res.transmittedReports;
        lost = res.lostInTransitReports; unatt = res.availableButUnattemptedReports;
    end
    m.generatedPerRound = gen(:)'; m.attemptedPerRound = att(:)'; m.lostPerRound = lost(:)'; m.unattemptedPerRound = unatt(:)';
    m.generated = sum(gen, 'omitnan'); m.attempted = sum(att, 'omitnan'); m.lost = sum(lost, 'omitnan'); m.unattempted = sum(unatt, 'omitnan');
    m.nodeEnergy = E; m.nodeAlive = logical(A);      % per-node remaining energy (J) and operational status, end of each round
    m.usableEnergy = sum(E .* A, 1);                  % energy held by operational nodes at the end of each round (J)
    m.usableResidual = m.usableEnergy(end);
    alive = sum(A, 1);
    m.alive = alive;                                  % 1. alive (operational) nodes
    m.dead = N - alive;                               % 2. dead (unavailable) nodes
    m.energy = sum(E, 1);                             % 3. total residual energy (J)
    m.cumDelivered = cumsum(del);                     % 4. cumulative throughput (reports at BS)
    m.cumPDR = 100 * cumsum(del) ./ max(1, cumsum(sch)); % 5. cumulative PDR (%)
    m.delay = delay;                                  % 6. mean end-to-end delay of delivered reports (s)
    dsum = cumsum(del .* max(0, delay), 'omitnan'); dn = cumsum(del);   % per-round mean x count = summed report delay
    m.cumMeanDelay = dsum ./ dn; m.cumMeanDelay(dn == 0) = NaN;          % packet-weighted mean over all reports delivered so far
    m.meanDelayPacket = m.cumMeanDelay(end);                            % packet-weighted (meanDelay below is round-weighted)
    m.energyClosureError = abs(sum(ex) + m.energy(end) - E0);
    m.delivered = sum(del); m.scheduled = sum(sch);
    m.pdr = 100 * sum(del) / sum(sch);
    m.residual = m.energy(end);
    m.epr = sum(ex) / max(1, sum(del));
    m.meanDelay = mean(delay, 'omitnan');
    first = @(v) firstOrNaN(v);
    m.fnd = first(alive < N); m.hnd = first(alive <= N/2); m.lnd = first(alive == 0);
    m.finalAlive = alive(end);
end

function v = firstOrNaN(mask)
    v = find(mask, 1); if isempty(v), v = NaN; end
end
