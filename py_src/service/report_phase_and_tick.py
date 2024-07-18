import os
import logging
from py_src import internal_names, util
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase

logger = logging.getLogger(f"{internal_names.logger_simulator_base_name}.{util.basename_without_extension(__file__)}")

class ServiceReportPhaseTick(Service):
    def __init__(self) -> None:
        import os
        super().__init__()

    @staticmethod
    def get_service_name() -> str:
        return "report_phase_and_tick"

    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        logger.info(f"current tick: {parameters.current_tick}, phase: {parameters.phase}")

