import os
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from py_src.cuda import CudaDevice
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
from py_src.node import Node

class ServiceTestAccuracyLossRecorder(Service):
    def __init__(self, interval, test_batch_size, phase_to_record=(SimulationPhase.END_OF_TICK,), use_fixed_testing_dataset=True, accuracy_file_name="accuracy.csv", loss_file_name="loss.csv", test_whole_dataset=False):
        super().__init__()
        self.accuracy_file = None
        self.loss_file = None
        self.node_order = None
        self.accuracy_file_name = accuracy_file_name
        self.loss_file_name = loss_file_name
        self.interval = interval
        self.phase_to_record = phase_to_record

        self.test_model = None
        self.criterion = None
        self.use_fixed_testing_dataset = use_fixed_testing_dataset
        self.test_whole_dataset = test_whole_dataset
        self.test_batch_size = test_batch_size
        self.test_dataset = None
        self.allocated_gpu = None

    @staticmethod
    def get_service_name() -> str:
        return "test_accuracy_loss_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, ml_setup=None, gpu: CudaDevice=None, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        assert ml_setup is not None

        node_names = []
        pre_allocated_model = None
        for node_name, target_node in parameters.node_container.items():
            target_node: Node
            node_names.append(node_name)
            # if the node is using model stat, then we re-use the model to save gpu memory.
            if pre_allocated_model is None:
                if target_node.is_using_model_stat:
                    gpu = target_node.allocated_gpu
                    pre_allocated_model = gpu.model

        self.initialize_without_runtime_parameters(output_path, node_names, ml_setup.model, ml_setup.criterion, ml_setup.testing_data, gpu=gpu, existing_model_for_testing=pre_allocated_model)

    def initialize_without_runtime_parameters(self, output_path, node_names, model, criterion, test_dataset, gpu: CudaDevice=None, existing_model_for_testing=None, num_workers=None):
        self.accuracy_file = open(os.path.join(output_path, f"{self.accuracy_file_name}"), "w+")
        self.loss_file = open(os.path.join(output_path, f"{self.loss_file_name}"), "w+")
        self.node_order = node_names
        node_order_str = [str(i) for i in self.node_order]
        header = ",".join(["tick", "phase", *node_order_str])
        self.accuracy_file.write(header + "\n")
        self.loss_file.write(header + "\n")
        self.criterion = criterion

        # set testing dataset
        if self.test_whole_dataset:
            if num_workers is None:
                self.test_dataset = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True)
            else:
                self.test_dataset = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
        else:
            if self.use_fixed_testing_dataset:
                """we should iterate whole dataset"""
                labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
                unique_labels = set(labels)
                n_labels = len(unique_labels)
                assert self.test_batch_size % n_labels == 0, f"test batch size({self.test_batch_size}) must be divisible by number of labels({n_labels})"
                samples_per_label = self.test_batch_size // n_labels
                label_indices = {label: np.where(labels == label)[0] for label in unique_labels}
                balanced_indices = []
                for label in unique_labels:
                    indices = label_indices[label]
                    sampled_indices = np.random.choice(indices, samples_per_label, replace=False)
                    balanced_indices.extend(sampled_indices)
                balanced_subset = Subset(test_dataset, balanced_indices)
                batch_size = 100 if self.test_batch_size > 100 else self.test_batch_size
                if num_workers is None:
                    balanced_loader = DataLoader(balanced_subset, batch_size=batch_size, shuffle=True)
                else:
                    balanced_loader = DataLoader(balanced_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
                self.test_dataset = balanced_loader
            else:
                """we should only iterate the first batch of test data"""
                if num_workers is None:
                    self.test_dataset = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True)
                else:
                    self.test_dataset = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)

        # set model
        if existing_model_for_testing is None:
            self.test_model = copy.deepcopy(model)
            # move to cuda?
            if gpu is not None:
                self.allocated_gpu = gpu
            if self.allocated_gpu is not None:
                self.test_model = self.test_model.to(gpu.device)
        else:
            self.allocated_gpu = gpu
            self.test_model = existing_model_for_testing

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.current_tick % self.interval != 0:
            return
        if parameters.phase in self.phase_to_record:
            node_names_and_model_stats = {}
            for node_name in self.node_order:
                target_node = parameters.node_container[node_name]
                model_stat = target_node.get_model_stat()
                node_names_and_model_stats[node_name] = model_stat
            self.trigger_without_runtime_parameters(parameters.current_tick, node_names_and_model_stats, parameters.phase.name)

    def trigger_without_runtime_parameters(self, tick, node_names_and_model_stats, phase_str=None):
        row_accuracy = []
        row_loss = []
        for node_name in self.node_order:
            if self.test_whole_dataset:
                model_stat = node_names_and_model_stats[node_name]
                self.test_model.load_state_dict(model_stat)
                if self.allocated_gpu is not None:
                    self.test_model.to(self.allocated_gpu.device)
                self.test_model.eval()
                total_loss, correct, total = 0, 0, 0
                for d, l in self.test_dataset:
                    test_data = d
                    test_labels = l
                    if self.allocated_gpu is not None:
                        test_data, test_labels = test_data.to(self.allocated_gpu.device), test_labels.to(self.allocated_gpu.device)
                    outputs = self.test_model(test_data)
                    total_loss += self.criterion(outputs, test_labels).item() * test_labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == test_labels).sum().item()
                    total += test_labels.size(0)
                loss = total_loss / total
                accuracy = correct / total
            else:
                if self.use_fixed_testing_dataset:
                    total_loss, correct, total = 0, 0, 0
                    for test_data, test_labels in self.test_dataset:
                        if self.allocated_gpu is not None:
                            test_data, test_labels = test_data.to(self.allocated_gpu.device), test_labels.to(self.allocated_gpu.device)
                        model_stat = node_names_and_model_stats[node_name]
                        self.test_model.load_state_dict(model_stat)
                        if self.allocated_gpu is not None:
                            self.test_model.to(self.allocated_gpu.device)
                        self.test_model.eval()
                        outputs = self.test_model(test_data)
                        total_loss += self.criterion(outputs, test_labels).item()
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == test_labels).sum().item()
                        total += test_labels.size(0)
                    loss = total_loss / total
                    accuracy = correct / total
                else:
                    test_data = None
                    test_labels = None
                    for d, l in self.test_dataset:
                        test_data = d
                        test_labels = l
                        break
                    if self.allocated_gpu is not None:
                        test_data, test_labels = test_data.to(self.allocated_gpu.device), test_labels.to(self.allocated_gpu.device)
                    model_stat = node_names_and_model_stats[node_name]
                    self.test_model.load_state_dict(model_stat)
                    if self.allocated_gpu is not None:
                        self.test_model.to(self.allocated_gpu.device)
                    self.test_model.eval()
                    outputs = self.test_model(test_data)
                    loss = self.criterion(outputs, test_labels).item()
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions = (predicted == test_labels).sum().item()
                    accuracy = correct_predictions / len(test_labels)
            row_accuracy.append(str(accuracy))
            row_loss.append('%.4f' % loss)
        row_accuracy_str = ",".join([str(tick), str(phase_str), *row_accuracy])
        row_loss_str = ",".join([str(tick), str(phase_str), *row_loss])
        self.accuracy_file.write(row_accuracy_str + "\n")
        self.accuracy_file.flush()
        self.loss_file.write(row_loss_str + "\n")
        self.loss_file.flush()


    def __del__(self):
        self.accuracy_file.flush()
        self.accuracy_file.close()
        self.loss_file.flush()
        self.loss_file.close()
