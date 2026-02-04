import os
import torch
from typing import List
from glob import glob
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

        self.last_l1_distance = None
        self.last_l2_distance = None

        self.logger = None

    @staticmethod
    def get_service_name() -> str:
        return "weight_difference_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        model_stats = []
        for node_name, target_node in parameters.node_container.items():
            model_stats.append(target_node.get_model_stat())
            break
        self.initialize_without_runtime_parameters(model_stats, output_path)

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if (parameters.phase == SimulationPhase.END_OF_TICK) and (parameters.current_tick % self.interval == 0):
            model_stats = []
            for node_name, target_node in parameters.node_container.items():
                model_stats.append(target_node.get_model_stat())
            self.trigger_without_runtime_parameters(parameters.current_tick, model_stats)

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        infile_path = os.path.join(checkpoint_folder_path, self.l1_save_file_name)
        with open(infile_path, 'r', newline='') as infile:
            next(infile)
            for line in infile:
                row_tick = int(line.split(",", 1)[0])
                if row_tick < restore_until_tick:
                    self.l1_save_file.write(line)
        self.l1_save_file.flush()
        infile_path = os.path.join(checkpoint_folder_path, self.l2_save_file_name)
        with open(infile_path, 'r', newline='') as infile:
            next(infile)
            for line in infile:
                row_tick = int(line.split(",", 1)[0])
                if row_tick < restore_until_tick:
                    self.l2_save_file.write(line)
        self.l2_save_file.flush()

    def get_last_distance(self):
        return self.last_l1_distance, self.last_l2_distance

    # for manual use, not for simulator use
    def initialize_without_runtime_parameters(self, model_stats, output_path, logger=None):
        self.logger = logger
        self.l1_save_file = open(os.path.join(output_path, f"{self.l1_save_file_name}"), "w+")
        self.l2_save_file = open(os.path.join(output_path, f"{self.l2_save_file_name}"), "w+")
        self.layer_order = []
        model_stat = model_stats[0]
        for layer_name, _ in model_stat.items():
            self.layer_order.append(layer_name)
        layer_order_str = [str(i) for i in self.layer_order]
        header = ",".join(["tick", *layer_order_str])
        self.l1_save_file.write(header + "\n")
        self.l2_save_file.write(header + "\n")

    def trigger_without_runtime_parameters(self, tick, model_stats):
        l1_distances = []
        l2_distances = []
        for layer_name in self.layer_order:
            weights = [model_stat[layer_name].float() for model_stat in model_stats]
            stacked_weights = torch.stack(weights)
            mean_weight = torch.mean(stacked_weights, dim=0)
            l1_distance = torch.sum(torch.abs(stacked_weights - mean_weight))
            l2_distance = torch.sum((stacked_weights - mean_weight) ** 2)
            l1_distances.append(f'{l1_distance.item():.4e}')
            l2_distances.append(f'{l2_distance.item():.4e}')
        l1_row = ",".join([str(tick), *l1_distances])
        self.l1_save_file.write(l1_row + "\n")
        self.l1_save_file.flush()
        l2_row = ",".join([str(tick), *l2_distances])
        self.l2_save_file.write(l2_row + "\n")
        self.l2_save_file.flush()
        self.last_l1_distance = l1_distances
        self.last_l2_distance = l2_distances

    def __del__(self):
        self.l1_save_file.flush()
        self.l1_save_file.close()
        self.l2_save_file.flush()
        self.l2_save_file.close()


class ServiceDistanceToOriginRecorder(Service):
    def __init__(self, interval, nodes_to_record: List[int], l1_save_file_name="distance_to_origin_l1.csv", l2_save_file_name="distance_to_origin_l2.csv"):
        super().__init__()
        self.l1_save_file = None
        self.l2_save_file = None
        self.layer_order = None
        self.l1_save_file_name = l1_save_file_name
        self.l2_save_file_name = l2_save_file_name
        self.interval = interval
        self.nodes_to_record = nodes_to_record

        self.logger = None

    @staticmethod
    def get_service_name() -> str:
        return "weight_difference_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        node_name_and_model_stats = {}
        for node_name, target_node in parameters.node_container.items():
            if node_name in self.nodes_to_record:
                node_name_and_model_stats[node_name] = target_node.get_model_stat()
        self.initialize_without_runtime_parameters(node_name_and_model_stats, output_path)

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if (parameters.phase == SimulationPhase.END_OF_TICK) and (parameters.current_tick % self.interval == 0):
            model_stats = []
            for node_name, target_node in parameters.node_container.items():
                model_stats.append(target_node.get_model_stat())
            self.trigger_without_runtime_parameters(parameters.current_tick, model_stats)

    # for manual use, not for simulator use
    def initialize_without_runtime_parameters(self, node_name_and_model_stat, output_path, logger=None):
        self.logger = logger
        self.layer_order = []
        model_stat = node_name_and_model_stat[next(iter(node_name_and_model_stat))]
        for layer_name, _ in model_stat.items():
            self.layer_order.append(layer_name)
        layer_order_str = [str(i) for i in self.layer_order]
        header = ",".join(["tick", *layer_order_str])
        self.l1_save_file = {}
        self.l2_save_file = {}
        for node_name, _ in node_name_and_model_stat.items():
            l1_save_file = open(os.path.join(output_path, f"{node_name}__{self.l1_save_file_name}"), "w+")
            l2_save_file = open(os.path.join(output_path, f"{node_name}__{self.l2_save_file_name}"), "w+")
            l1_save_file.write(header + "\n")
            l2_save_file.write(header + "\n")
            self.l1_save_file[node_name] = l1_save_file
            self.l2_save_file[node_name] = l2_save_file

    def trigger_without_runtime_parameters(self, tick, node_name_and_model_stat):
        l1_distances = []
        l2_distances = []
        for node_name, model_stat in node_name_and_model_stat.items():
            for layer_name in self.layer_order:
                layer_tensor = model_stat[layer_name]
                l1_distance = torch.sum(torch.abs(layer_tensor)).item()
                l2_distance = torch.sqrt(torch.sum(layer_tensor ** 2)).item()
                l1_distances.append(f'{l1_distance:.4e}')
                l2_distances.append(f'{l2_distance:.4e}')
            l1_row = ",".join([str(tick), *l1_distances])
            self.l1_save_file[node_name].write(l1_row + "\n")
            self.l1_save_file[node_name].flush()
            l2_row = ",".join([str(tick), *l2_distances])
            self.l2_save_file[node_name].write(l2_row + "\n")
            self.l2_save_file[node_name].flush()

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        files = glob(f"{checkpoint_folder_path}/*{self.l1_save_file_name}", recursive=False)
        for file in files:
            node_name = int(os.path.basename(file).split("__")[0])
            with open(file, 'r', newline='') as infile:
                next(infile)
                for line in infile:
                    row_tick = int(line.split(",", 1)[0])
                    if row_tick < restore_until_tick:
                        self.l1_save_file[node_name].write(line)
            self.l1_save_file[node_name].flush()
        files = glob(f"{checkpoint_folder_path}/*{self.l2_save_file_name}", recursive=False)
        for file in files:
            node_name = int(os.path.basename(file).split("__")[0])
            with open(file, 'r', newline='') as infile:
                next(infile)
                for line in infile:
                    row_tick = int(line.split(",", 1)[0])
                    if row_tick < restore_until_tick:
                        self.l2_save_file[node_name].write(line)
            self.l2_save_file[node_name].flush()

    def __del__(self):
        if self.l1_save_file is not None:
            for node_name, file in self.l1_save_file.items():
                file.flush()
                file.close()
        if self.l2_save_file is not None:
            for node_name, file in self.l2_save_file.items():
                file.flush()
                file.close()
