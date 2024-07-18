from enum import Enum

class SimulationPhase(Enum):
    START_OF_TICK = 0
    END_OF_TICK = 7
    INITIALIZING = 8

    BEFORE_TRAINING = 1
    TRAINING = 2
    AFTER_TRAINING = 3

    BEFORE_AVERAGING = 4
    AVERAGING = 5
    AFTER_AVERAGING = 6


class RuntimeParameters:
    def __init__(self):
        self.max_tick = None
        self.current_tick = None
        self.node_container = None
        self.dataset_label = None
        self.phase = SimulationPhase.INITIALIZING
        self.topology = None

        self.service_container = {}
        self.mpi_enabled = None
