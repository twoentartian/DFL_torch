import os
import copy

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from py_src.cuda import CudaDevice
from py_src.ml_setup_base.base import MlSetup
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase

class ServiceConsecutiveLinearInterpolationRecorder(Service):
    def __init__(self, interval, batch_size, consecutive_linear_interpolation_dataset_size, consecutive_linear_interpolation_points_size,
                 recorded_node_name,
                 consecutive_linear_interpolation_loss_filename="consec_linear_interpolation_loss.csv",
                 consecutive_linear_interpolation_accuracy_filename="consec_linear_interpolation_accuracy.csv"):
        super().__init__()
        self.points_size = consecutive_linear_interpolation_points_size
        self.dataset_size = consecutive_linear_interpolation_dataset_size
        self.dataloader = None
        self.loss_file_name = consecutive_linear_interpolation_loss_filename
        self.loss_file = None
        self.accuracy_file_name = consecutive_linear_interpolation_accuracy_filename
        self.accuracy_file = None

        self.interval = interval
        self.batch_size = batch_size
        self.test_model = None
        self.criterion = None
        self.allocated_gpu = None
        self.recorded_node_name = recorded_node_name

        self.cache_state_model_stat = None

    @staticmethod
    def get_service_name() -> str:
        return "consecutive_linear_interpolation_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, ml_setup: MlSetup=None, gpu: CudaDevice=None, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        assert ml_setup is not None

        pre_allocated_model = None
        assert self.recorded_node_name in parameters.node_container.keys()
        target_node = parameters.node_container[self.recorded_node_name]
        # if the node is using model stat, then we re-use the model to save gpu memory.
        if pre_allocated_model is None:
            if target_node.is_using_model_stat:
                gpu = target_node.allocated_gpu
                pre_allocated_model = gpu.model

        self.initialize_without_runtime_parameters(output_path, ml_setup.model, ml_setup.criterion, ml_setup.training_data, gpu=gpu, existing_model_for_testing=pre_allocated_model)

    def initialize_without_runtime_parameters(self, output_path, model, criterion, train_dataset, gpu: CudaDevice=None, existing_model_for_testing=None, num_workers=None):
        self.criterion = criterion

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

        # set dataset for measuring loss/accuracy of consecutive linear interpolation
        if self.points_size != 0:
            subset_indices = np.random.choice(range(len(train_dataset)), self.dataset_size)
            subset = Subset(train_dataset, subset_indices)
            batch_size = 100 if self.batch_size > 100 else self.batch_size
            if num_workers is None:
                loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
            else:
                loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
            self.dataloader = loader

            self.accuracy_file = open(os.path.join(output_path, f"{self.accuracy_file_name}"), "w+")
            self.loss_file = open(os.path.join(output_path, f"{self.loss_file_name}"), "w+")
            points_names = list(range(1, self.points_size))
            first_row = ",".join(["tick", "phase", *points_names])
            self.accuracy_file.write(first_row + "\n")
            self.loss_file.write(first_row + "\n")

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.phase in [SimulationPhase.START_OF_TICK, SimulationPhase.END_OF_TICK]:
            target_node = parameters.node_container[self.recorded_node_name]
            target_model_state = target_node.get_model_stat()
            self.trigger_without_runtime_parameters(parameters.current_tick, parameters.phase, target_model_state)

    def trigger_without_runtime_parameters(self, tick, phase:SimulationPhase, model_state):
        if phase == SimulationPhase.START_OF_TICK:
            assert self.cache_state_model_stat is None
            self.cache_state_model_stat = {k: v.detach().clone() for k, v in model_state.items()}
        if phase == SimulationPhase.END_OF_TICK:
            assert self.cache_state_model_stat is not None
            start_mode_state = self.cache_state_model_stat
            end_model_state = model_state
            points_size =  self.points_size
            loss_results = []
            accuracy_results = []
            for i in range(1, points_size):
                alpha = i / points_size
                model_stat = {k: (1 - alpha) * start_mode_state[k] + alpha * end_model_state[k] for k in start_mode_state.keys()}

                self.test_model.load_state_dict(model_stat)
                if self.allocated_gpu is not None:
                    self.test_model.to(self.allocated_gpu.device)
                self.test_model.eval()
                total_loss, correct, total = 0, 0.0, 0
                for d, l in self.dataloader:
                    test_data = d
                    test_labels = l
                    if self.allocated_gpu is not None:
                        test_data, test_labels = d.to(self.allocated_gpu.device), l.to(self.allocated_gpu.device)
                    outputs = self.test_model(test_data)
                    total_loss += self.criterion(outputs, test_labels).item() * test_labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == test_labels).sum().item()
                    total += test_labels.size(0)
                loss = total_loss / total
                accuracy = correct / total
                loss_results.append(loss)
                accuracy_results.append(accuracy)
            row_accuracy_str = ",".join([str(tick), str(phase.name), *accuracy_results])
            row_loss_str = ",".join([str(tick), str(phase.name), *loss_results])
            self.accuracy_file.write(row_accuracy_str + "\n")
            self.accuracy_file.flush()
            self.loss_file.write(row_loss_str + "\n")
            self.loss_file.flush()
            self.cache_state_model_stat = None

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        def copy_file(file_name, file):
            infile_path = os.path.join(checkpoint_folder_path, file_name)
            with open(infile_path, 'r', newline='') as infile:
                next(infile)
                for line in infile:
                    row_tick = int(line.split(",", 1)[0])
                    if row_tick < restore_until_tick:
                        file.write(line)
            file.flush()
        # accuracy file
        copy_file(self.accuracy_file_name, self.accuracy_file)
        # loss file
        copy_file(self.loss_file_name, self.loss_file)

    def __del__(self):
        self.accuracy_file.flush()
        self.accuracy_file.close()
        self.loss_file.flush()
        self.loss_file.close()