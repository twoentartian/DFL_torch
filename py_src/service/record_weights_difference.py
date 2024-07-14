import os
import torch
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase

class ServiceWeightsDifferenceRecorder(Service):
    def __init__(self, interval, l1_save_file_name="weight_difference_l1.csv", l2_save_file_name="weight_difference_l2.csv"):
        super().__init__()
        self.l1_save_file = None
        self.l2_save_file = None
        self.layer_order = None
        self.l1_save_file_name = l1_save_file_name
        self.l2_save_file_name = l2_save_file_name
        self.interval = interval

    @staticmethod
    def get_service_name() -> str:
        return "weight_difference_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        self.l1_save_file = open(os.path.join(output_path, f"{self.l1_save_file_name}"), "w+")
        self.l2_save_file = open(os.path.join(output_path, f"{self.l2_save_file_name}"), "w+")
        self.layer_order = []
        for node_name, target_node in parameters.node_container.items():
            model_stat = target_node.get_model_stat()
            for layer_name, _ in model_stat.items():
                self.layer_order.append(layer_name)
            break
        node_order_str = [str(i) for i in self.layer_order]
        header = ",".join(["tick", *node_order_str])
        self.l1_save_file.write(header + "\n")
        self.l2_save_file.write(header + "\n")

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if (parameters.phase == SimulationPhase.END_OF_TICK) and (parameters.current_tick % self.interval == 0):
            model_stats = []
            for node_name, target_node in parameters.node_container.items():
                model_stats.append(target_node.get_model_stat())
            l1_distances = []
            l2_distances = []
            for layer_name in self.layer_order:
                weights = [model_stat[layer_name].float() for model_stat in model_stats]
                stacked_weights = torch.stack(weights)
                mean_weight = torch.mean(stacked_weights, dim=0)
                l1_distance = torch.mean(torch.abs(stacked_weights - mean_weight))
                l2_distance = torch.mean((stacked_weights - mean_weight) ** 2)
                l1_distances.append(f'{l1_distance.item():e}')
                l2_distances.append(f'{l2_distance.item():e}')
            l1_row = ",".join([str(parameters.current_tick), *l1_distances])
            self.l1_save_file.write(l1_row + "\n")
            self.l1_save_file.flush()
            l2_row = ",".join([str(parameters.current_tick), *l2_distances])
            self.l2_save_file.write(l2_row + "\n")
            self.l2_save_file.flush()

    def __del__(self):
        self.l1_save_file.flush()
        self.l1_save_file.close()
        self.l2_save_file.flush()
        self.l2_save_file.close()
