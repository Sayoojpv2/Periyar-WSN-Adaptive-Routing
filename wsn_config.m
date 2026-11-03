function cfg = wsn_config()
% WSN_CONFIG Experiment settings: 10 sensing nodes that also forward,
% no relay-only nodes, one base station.
    cfg.root = fileparts(mfilename('fullpath'));
    cfg.workbook = fullfile(cfg.root, 'data', 'consdata.xlsx');   % original KSPCB workbook (copy in data/), read directly
    cfg.workbookMD5 = '6100b72899bd76df5a393fbbc5fb898a';
    cfg.numNodes = 10;
    cfg.numRounds = 1000;
    % Fixed experiment design, independent of where a run stops (run_experiment sets numRounds = 500):
    cfg.scheduleRounds = 1000;        % sampling-date -> round schedule: 212 dates over 1000 rounds (~4.7 rounds per date)
    cfg.budgetHorizonRounds = 1000;   % energy-planning horizon of the LEACH-RL / DEEC-RL / PEGASIS-RL budgets
    cfg.seeds = 42:46;
    % Station s (station order upstream -> downstream) is node s.
    cfg.stationNames = {'ALUVA - MANAPPURAM', 'PURAPILLIKADAVU US', 'PURAPILLIKADAVU BUND NAVIGATION', ...
        'PURAPILLIKADAVU BUND DS', 'PATHALAM BUND U/S', 'PATHALAM BUND D/S', 'PATHALAM BRIDGE', ...
        'PUTHALAM KADAVU', 'MANJUMMAL BUND', 'VETTUKADAVU'};
    cfg.sensingNodes = 1:10;
    % TOPOLOGY = SIMULATION ASSUMPTION, not geographic coordinates.
    % 'stationPositions': each station takes the position of node 4, 8, ..., 48 of the seeded
    % 50-node 600 m x 50 m reference layout for the same seed; the other 40 positions are not used.
    % 'custom': cfg.customX / cfg.customY in metres inside the 600 m x 50 m section
    % (for verified station coordinates once they are available).
    cfg.topology = 'stationPositions';
    cfg.stationNodeIds = [4 8 12 18 22 28 32 38 42 48];
    cfg.customX = []; cfg.customY = [];
    cfg.learners = {'tabularQ', 'doubleQ', 'doubleDQN', 'duelingDoubleDQN'};
    cfg.learnerLabels = {'Tabular Q-learning', 'Double Q-learning', 'Double DQN', 'Dueling DDQN'};
end
