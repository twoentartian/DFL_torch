import os
import torch
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase


class ModelStatRecorder(Service):
    def __init__(self, interval, phase=SimulationPhase.END_OF_TICK, record_node=None) -> None:
        super().__init__()
        self.save_path = None
        self.save_path_for_each_node = None
        self.record_node = record_node
        self.known_nodes_to_record = set()
        self.interval = interval
        self.record_phase = phase

    @staticmethod
    def get_service_name() -> str:
        return "model_stat_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING

        node_names = []
        for node_name, target_node in parameters.node_container.items():
            if self._is_current_node_recorded(node_name):
                self.known_nodes_to_record.add(node_name)
                node_names.append(node_name)
        self.initialize_without_runtime_parameters(node_names, output_path)

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.current_tick % self.interval != 0:
            return      # skip is not time yet

        if parameters.phase == self.record_phase:
            node_names = []
            model_stats = []
            for node_name in self.known_nodes_to_record:
                model_stat = parameters.node_container[node_name].get_model_stat()
                node_names.append(node_name)
                model_stats.append(model_stat)
            self.trigger_without_runtime_parameters(parameters.current_tick, node_names, model_stats)

    def initialize_without_runtime_parameters(self, node_names, output_path):
        self.save_path = os.path.join(output_path, "model_stat")
        os.mkdir(self.save_path)
        self.save_path_for_each_node = {}
        for node_name in node_names:
            save_path_for_this_node = os.path.join(self.save_path, str(node_name))
            self.save_path_for_each_node[node_name] = save_path_for_this_node
            os.mkdir(save_path_for_this_node)

    def trigger_without_runtime_parameters(self, tick, node_names, model_stats):
        assert len(node_names) == len(model_stats)
        for index, node_name in enumerate(node_names):
            assert node_name in self.save_path_for_each_node.keys()
            save_path_for_this_node = self.save_path_for_each_node[node_name]
            model_stat = model_stats[index]
            current_node_output_path = os.path.join(save_path_for_this_node, f"{tick}.pt")
            torch.save(model_stat, current_node_output_path)

    def _is_current_node_recorded(self, node_name) -> bool:
        record_current_node = True
        if (self.record_node is not None) and (node_name not in self.record_node):
            record_current_node = False
        return record_current_node

    def __del__(self):
        pass
