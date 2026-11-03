% simulation/create_periyar_workload_continuous.m
function workload = create_periyar_workload_continuous(wq_data, numRounds, mapping_method, node_ids, numNodes, scheduleRounds)
% CREATE_PERIYAR_WORKLOAD_CONTINUOUS Constructs a continuous WSN communication
% workload (10,000 total reports across 1000 rounds) using the real Periyar
% River water-quality dataset under a causal sample-and-hold (Last Observation
% Carried Forward - LOCF) application state model.
%
% V2: node_ids (station s -> node id, in the station order below) and numNodes
% are inputs. Defaults reproduce V1 (50 nodes, stations on nodes 4, 8, ..., 48);
% V2 uses 10 nodes with station s on node s.
% scheduleRounds (default numRounds): length of the date-to-round schedule. A run shorter
% than the schedule (e.g. numRounds 500, scheduleRounds 1000) reports exactly what rounds
% 1..numRounds of the full schedule report, so the per-round workload does not depend on
% where the run stops.
%
% Concept:
%   Separates environmental data from network communication.
%   - numNodes sensor nodes total (V1: 50; V2: 10)
%   - 10 fixed sensing stations (node_ids)
%   - numNodes - 10 relay/routing nodes (V2: none)
%   - 1000 communication rounds
%   - Exactly 10 application reports per round (1 from each sensing station)
%   - Exactly 10,000 communication reports total
%   - Packet size: 4000 bits
%
% Causal Sample-and-Hold (LOCF) Model:
%   At each round r, each station reports its latest available physical
%   measurement from the real Periyar dataset.
%   If measurement A arrives at milestone t1 and measurement B arrives at t2,
%   all rounds in [t1, t2-1] report state A. At t2, state switches to B.
%   No future observations are leaked; no data is interpolated or fabricated.
%   Before the first field reading, the station holds its initial baseline
%   field calibration observation.

    if nargin < 1 || isempty(wq_data)
        wq_data = load_consdata_xlsx();
    end
    if nargin < 2 || isempty(numRounds)
        numRounds = 1000;
    end
    if nargin < 3 || isempty(mapping_method)
        mapping_method = 'uniform'; % 'uniform' or 'sparse_aligned'
    end
    if nargin < 4 || isempty(node_ids)
        node_ids = [4, 8, 12, 18, 22, 28, 32, 38, 42, 48];
    end
    if nargin < 5 || isempty(numNodes)
        numNodes = 50;
    end
    if nargin < 6 || isempty(scheduleRounds)
        scheduleRounds = numRounds;
    end
    assert(scheduleRounds >= numRounds, 'The run cannot be longer than the reporting schedule.');
    assert(numel(node_ids) == 10 && all(ismember(node_ids, 1:numNodes)) && numel(unique(node_ids)) == 10, ...
        'node_ids must map the 10 stations to distinct nodes in 1..numNodes.');

    raw_total = height(wq_data);
    assert(raw_total == 1273, 'Expected exactly 1273 raw Periyar observations.');

    % Filter to clean 4-parameter observations (Temp, pH, Conductivity, TDS)
    is_valid = false(raw_total, 1);
    for i = 1:raw_total
        t = wq_data.Temperature(i);
        ph = wq_data.pH(i);
        c = wq_data.Conductivity(i);
        tds = wq_data.TDS(i);

        t_ok = ~isnan(t) && isnumeric(t) && t >= 15.0 && t <= 50.0;
        ph_ok = ~isnan(ph) && isnumeric(ph) && ph >= 0.0 && ph <= 14.0;
        c_ok = ~isnan(c) && isnumeric(c) && c > 0.0;
        tds_ok = ~isnan(tds) && isnumeric(tds) && tds > 0.0;

        if t_ok && ph_ok && c_ok && tds_ok
            is_valid(i) = true;
        end
    end

    total_valid = sum(is_valid);
    assert(total_valid == 1251, sprintf('Expected exactly 1251 valid observations, got %d', total_valid));
    clean_wq_data = wq_data(is_valid, :);

    station_names = { ...
        'ALUVA - MANAPPURAM', ...
        'PURAPILLIKADAVU US', ...
        'PURAPILLIKADAVU BUND NAVIGATION', ...
        'PURAPILLIKADAVU BUND DS', ...
        'PATHALAM BUND U/S', ...
        'PATHALAM BUND D/S', ...
        'PATHALAM BRIDGE', ...
        'PUTHALAM KADAVU', ...
        'MANJUMMAL BUND', ...
        'VETTUKADAVU' ...
    };
    node_ids = node_ids(:)';
    num_stations = length(station_names);

    unique_dates = unique(clean_wq_data.Date);
    num_dates = length(unique_dates);
    assert(num_dates == 212, sprintf('Expected exactly 212 unique dates, got %d', num_dates));

    % Deterministic chronological date-to-round mapping
    switch lower(mapping_method)
        case 'uniform'
            % Uniform linear mapping from 212 dates to 1000 communication rounds
            d_idx = 1:num_dates;
            r_events = 1 + round((d_idx - 1) * ((scheduleRounds - 1) / (num_dates - 1)));
        case 'sparse_aligned'
            % Exact piecewise alignment with the sparse baseline milestones
            d1 = 1:106;
            r1 = round(1 + (d1 - 1) * (499 / 105));
            d2 = 107:212;
            r2 = round(500 + (d2 - 106) * (500 / 106));
            r_events = [r1, r2];
        otherwise
            error('Unknown mapping method: %s', mapping_method);
    end

    assert(r_events(1) == 1, 'First event milestone must map to round 1.');
    assert(r_events(end) == scheduleRounds, sprintf('Last milestone must map to round %d.', scheduleRounds));
    assert(length(unique(r_events)) == num_dates, 'Milestone rounds must be strictly unique.');

    date_to_round_map = containers.Map(cellstr(datestr(unique_dates, 'yyyy-mm-dd')), num2cell(r_events));

    % For each station, extract its chronological sequence of valid observations
    station_obs = cell(num_stations, 1);
    for s = 1:num_stations
        st_name = station_names{s};
        st_mask = strcmp(clean_wq_data.Location, st_name);
        st_rows = find(st_mask);
        [~, sort_order] = sort(clean_wq_data.Date(st_rows));
        st_rows = st_rows(sort_order);

        st_dates = clean_wq_data.Date(st_rows);
        st_round_milestones = zeros(length(st_rows), 1);
        for k = 1:length(st_rows)
            d_str = datestr(st_dates(k), 'yyyy-mm-dd');
            st_round_milestones(k) = date_to_round_map(d_str);
        end

        st_struct = struct();
        st_struct.row_indices = st_rows;
        st_struct.dates = st_dates;
        st_struct.round_milestones = st_round_milestones;
        st_struct.temperature = clean_wq_data.Temperature(st_rows);
        st_struct.pH = clean_wq_data.pH(st_rows);
        st_struct.conductivity = clean_wq_data.Conductivity(st_rows);
        st_struct.TDS = clean_wq_data.TDS(st_rows);

        station_obs{s} = st_struct;
    end

    % Construct the continuous communication workload without future-data leakage
    total_scheduled_reports = numRounds * num_stations; % 10,000 scheduled report opportunities

    round_obs = cell(1, numRounds);
    obs_src_nodes = zeros(total_scheduled_reports, 1);
    sourceDates = cell(total_scheduled_reports, 1);
    sourceStations = cell(total_scheduled_reports, 1);
    temperature = zeros(total_scheduled_reports, 1);
    pH = zeros(total_scheduled_reports, 1);
    conductivity = zeros(total_scheduled_reports, 1);
    TDS = zeros(total_scheduled_reports, 1);

    round_state = zeros(numRounds, num_stations);
    node_gen_count = zeros(numRounds, numNodes);

    rep_idx = 0;
    for r = 1:numRounds
        round_rep_indices = [];

        for s = 1:num_stations
            st_data = station_obs{s};
            milestones = st_data.round_milestones;

            % Strict Causal Rule:
            % BEFORE first real observation: NO environmental report is available.
            % AT / AFTER first real observation: use latest observation available on or before round r (LOCF).
            valid_avail = find(milestones <= r);
            if isempty(valid_avail)
                % Before first observation date: NO report available yet (no future-data leakage)
                round_state(r, s) = 0;
            else
                active_k = valid_avail(end);
                clean_row_idx = st_data.row_indices(active_k);
                round_state(r, s) = clean_row_idx;

                rep_idx = rep_idx + 1;
                round_rep_indices = [round_rep_indices, rep_idx];

                obs_src_nodes(rep_idx) = node_ids(s);
                sourceDates{rep_idx} = datestr(st_data.dates(active_k), 'yyyy-mm-dd');
                sourceStations{rep_idx} = station_names{s};
                temperature(rep_idx) = st_data.temperature(active_k);
                pH(rep_idx) = st_data.pH(active_k);
                conductivity(rep_idx) = st_data.conductivity(active_k);
                TDS(rep_idx) = st_data.TDS(active_k);
                node_gen_count(r, node_ids(s)) = 1;
            end
        end

        round_obs{r} = round_rep_indices;
    end

    % Truncate pre-allocated arrays to exact count of real available reports
    obs_src_nodes = obs_src_nodes(1:rep_idx);
    sourceDates = sourceDates(1:rep_idx);
    sourceStations = sourceStations(1:rep_idx);
    temperature = temperature(1:rep_idx);
    pH = pH(1:rep_idx);
    conductivity = conductivity(1:rep_idx);
    TDS = TDS(1:rep_idx);

    total_data_reports = rep_idx;
    total_unavailable_data = total_scheduled_reports - total_data_reports;

    % Assemble comprehensive workload struct
    workload = struct();
    workload.type = 'causal_periyar_locf_continuous';
    workload.totalScheduledReports = total_scheduled_reports;
    workload.totalAvailableDataReports = total_data_reports;
    workload.totalUnavailableDataReports = total_unavailable_data;
    workload.totalGeneratedReports = total_data_reports;
    workload.totalCommunicationReports = total_data_reports;
    workload.numRounds = numRounds;
    workload.numStations = num_stations;
    workload.round_obs = round_obs;
    workload.obs_src_nodes = obs_src_nodes;
    workload.station_names = station_names;
    workload.sensing_nodes = node_ids;
    workload.sourceDates = sourceDates;
    workload.sourceStations = sourceStations;
    workload.temperature = temperature;
    workload.pH = pH;
    workload.conductivity = conductivity;
    workload.TDS = TDS;
    workload.round_state = round_state;
    workload.round_measurement_reference = round_state;
    workload.node_gen_count = node_gen_count;
    workload.clean_wq_data = clean_wq_data;
    workload.unique_dates = unique_dates;
    workload.r_events = r_events;

    % Metadata
    metadata = struct();
    metadata.originalObservationCount = raw_total;
    metadata.validObservationCount = total_valid;
    metadata.rejectedObservationCount = raw_total - total_valid;
    metadata.numStations = num_stations;
    metadata.simulationRounds = numRounds;
    metadata.totalScheduledReports = total_scheduled_reports;
    metadata.totalAvailableDataReports = total_data_reports;
    metadata.totalUnavailableDataReports = total_unavailable_data;
    metadata.temporalMappingMethod = sprintf('Deterministic %s mapping (212 dates to %d rounds), simulated rounds 1-%d', mapping_method, scheduleRounds, numRounds);
    metadata.scheduleRounds = scheduleRounds;
    metadata.sampleAndHoldRule = 'Strictly causal Last-Observation-Carried-Forward (LOCF); zero future leakage';
    metadata.selectedParameters = {'Temperature', 'pH', 'Conductivity', 'TDS'};
    metadata.isContinuousWorkload = true;

    workload.metadata = metadata;
end
