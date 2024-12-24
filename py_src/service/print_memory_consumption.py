import os
import torch
import resource
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase

class PrintMemoryConsumption(Service):
    def __init__(self, interval, save_file_name="memory_consumption.txt", phase_to_record=(SimulationPhase.END_OF_TICK,)) -> None:
        super().__init__()
        self.interval = interval
        self.phase_to_record = phase_to_record
        self.save_file_name = save_file_name
        self.save_file = None

    @staticmethod
    def get_service_name() -> str:
        return "print_memory_consumption"

    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        self.initialize_without_runtime_parameters(output_path)

    def initialize_without_runtime_parameters(self, output_path):
        self.save_file = open(os.path.join(output_path, f"{self.save_file_name}"), "w+")

    def trigger_without_runtime_parameters(self, tick, phase_str=None):
        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        gpu_memory_usage = torch.cuda.memory_allocated() / 1024 ** 2
        self.save_file.write(f"tick: {tick}, phase:{phase_str}, memory usage: {memory_usage}, gpu memory usage: {gpu_memory_usage}\n")
        self.save_file.flush()

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.current_tick % self.interval != 0:
            return
        if parameters.phase in self.phase_to_record:
            self.trigger_without_runtime_parameters(parameters.current_tick, parameters.phase)

    def __del__(self):
        if self.save_file is not None:
            self.save_file.flush()
            self.save_file.close()
