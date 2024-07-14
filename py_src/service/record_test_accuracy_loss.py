import os
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase

class ServiceTestAccuracyLossRecorder(Service):
    def __init__(self, interval, test_batch_size, phase_to_record=(SimulationPhase.END_OF_TICK,), use_fixed_testing_dataset=True, accuracy_file_name="accuracy.csv", loss_file_name="loss.csv"):
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
        self.test_batch_size = test_batch_size
        self.test_dataset = None
        self.is_using_cuda = None

    @staticmethod
    def get_service_name() -> str:
        return "test_accuracy_loss_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, config_file=None, ml_setup=None, cuda_env=None, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        assert ml_setup is not None
        self.accuracy_file = open(os.path.join(output_path, f"{self.accuracy_file_name}"), "w+")
        self.loss_file = open(os.path.join(output_path, f"{self.loss_file_name}"), "w+")
        self.node_order = []
        for node_name, target_node in parameters.node_container.items():
            self.node_order.append(node_name)
        node_order_str = [str(i) for i in self.node_order]
        header = ",".join(["tick", "phase", *node_order_str])
        self.accuracy_file.write(header + "\n")
        self.loss_file.write(header + "\n")
        # set model
        self.test_model = copy.deepcopy(ml_setup.model)
        self.criterion = ml_setup.criterion

        # set testing dataset
        test_dataset = ml_setup.testing_data
        labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
        unique_labels = set(labels)
        n_labels = len(unique_labels)
        assert self.test_batch_size % n_labels == 0, f"test batch size({self.test_batch_size}) must be divisible by number of labels({n_labels})"
        if self.use_fixed_testing_dataset:
            samples_per_label = self.test_batch_size // n_labels
            label_indices = {label: np.where(labels == label)[0] for label in unique_labels}
            balanced_indices = []
            for label in unique_labels:
                indices = label_indices[label]
                sampled_indices = np.random.choice(indices, samples_per_label, replace=False)
                balanced_indices.extend(sampled_indices)
            balanced_subset = Subset(test_dataset, balanced_indices)
            balanced_loader = DataLoader(balanced_subset, batch_size=self.test_batch_size, shuffle=True)
            self.test_dataset = balanced_loader
        else:
            self.test_dataset = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True)
        # move to cuda?
        self.is_using_cuda = (config_file is not None) and (not config_file.force_use_cpu) and (cuda_env is not None) and cuda_env.cuda_available
        if self.is_using_cuda:
            self.test_model = self.test_model.cuda()  # use default CUDA device

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.current_tick % self.interval != 0:
            return
        row_accuracy = []
        row_loss = []
        if parameters.phase in self.phase_to_record:
            for node_name in self.node_order:
                test_data = None
                test_labels = None
                for d, l in self.test_dataset:
                    test_data = d
                    test_labels = l
                    break
                if self.is_using_cuda:
                    test_data, test_labels = test_data.cuda(), test_labels.cuda()
                target_node = parameters.node_container[node_name]
                model_stat = target_node.get_model_stat()
                self.test_model.load_state_dict(model_stat)
                self.test_model.eval()
                outputs = self.test_model(test_data)
                loss = self.criterion(outputs, test_labels)
                _, predicted = torch.max(outputs, 1)
                correct_predictions = (predicted == test_labels).sum().item()
                accuracy = correct_predictions / len(test_labels)
                row_accuracy.append(str(accuracy))
                row_loss.append('%.4f' % loss.item())
            row_accuracy_str = ",".join([str(parameters.current_tick), parameters.phase.name, *row_accuracy])
            row_loss_str = ",".join([str(parameters.current_tick), parameters.phase.name, *row_loss])
            self.accuracy_file.write(row_accuracy_str + "\n")
            self.accuracy_file.flush()
            self.loss_file.write(row_loss_str + "\n")
            self.loss_file.flush()

    def __del__(self):
        self.accuracy_file.flush()
        self.accuracy_file.close()
        self.loss_file.flush()
        self.loss_file.close()