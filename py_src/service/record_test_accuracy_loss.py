import os
import copy
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset

from py_src.cuda import CudaDevice
from py_src.ml_setup_base.base import MlSetup
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
from py_src.node import Node
import py_src.util as util

class ServiceTestAccuracyLossRecorder(Service):
    def __init__(self, interval, test_batch_size, model_name, dataset_name, phase_to_record=(SimulationPhase.END_OF_TICK,), use_fixed_testing_dataset=True, store_top_accuracy_model_count = 0,
                 accuracy_file_name="accuracy.csv", loss_file_name="loss.csv", output_var_file_name="output_var.csv", test_whole_dataset=False,
                 test_val_split=0.5, test_accuracy_file_name="accuracy_test.csv", test_loss_file_name="loss_test.csv", val_accuracy_file_name="accuracy_val.csv", val_loss_file_name="loss_val.csv"):
        """
        test_val_split=0.5: split the dataset into two parts: test and val, the value indicates the portion of test
        """
        super().__init__()
        self.accuracy_file = None
        self.loss_file = None
        self.output_var_file = None
        self.node_order = None
        self.accuracy_file_name = accuracy_file_name
        self.loss_file_name = loss_file_name
        self.output_var_file_name = output_var_file_name
        self.interval = interval
        self.phase_to_record = phase_to_record
        self.store_top_accuracy_model_count = store_top_accuracy_model_count
        self.store_top_accuracy_model = self.store_top_accuracy_model_count != 0
        self.store_top_accuracy_model_path = None
        self.store_top_accuracy_model_buffer = None
        self.model_name = model_name
        self.dataset_name = dataset_name

        self.test_model = None
        self.criterion = None
        self.use_fixed_testing_dataset = use_fixed_testing_dataset
        self.test_whole_dataset = test_whole_dataset
        self.test_batch_size = test_batch_size
        self.test_dataset = None
        self.val_dataset = None
        self.allocated_gpu = None

        if test_whole_dataset:
            self.test_val_split = test_val_split
        else:
            if test_val_split is not None:
                raise ValueError(f"cannot use test_val_split({test_val_split}) if test_whole_dataset is set to {test_whole_dataset}")
        if self.test_val_split is not None:
            self.test_idx = None
            self.val_idx = None
            self.test_loss_file = None
            self.test_accuracy_file = None
            self.val_loss_file = None
            self.val_accuracy_file = None
            self.test_loss_file_name = test_loss_file_name
            self.test_accuracy_file_name = test_accuracy_file_name
            self.val_loss_file_name = val_loss_file_name
            self.val_accuracy_file_name = val_accuracy_file_name

    @staticmethod
    def get_service_name() -> str:
        return "test_accuracy_loss_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, ml_setup: MlSetup=None, gpu: CudaDevice=None, *args, **kwargs):
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
        self.output_var_file = open(os.path.join(output_path, f"{self.output_var_file_name}"), "w+")
        if self.test_val_split is not None:
            self.test_accuracy_file = open(os.path.join(output_path, f"{self.test_accuracy_file_name}"), "w+")
            self.test_loss_file = open(os.path.join(output_path, f"{self.test_loss_file_name}"), "w+")
            self.val_accuracy_file = open(os.path.join(output_path, f"{self.val_accuracy_file_name}"), "w+")
            self.val_loss_file = open(os.path.join(output_path, f"{self.val_loss_file_name}"), "w+")

        self.node_order = node_names
        node_order_str = [str(i) for i in self.node_order]
        header = ",".join(["tick", "phase", *node_order_str])
        self.accuracy_file.write(header + "\n")
        self.loss_file.write(header + "\n")
        self.output_var_file.write(header + "\n")
        if self.test_val_split is not None:
            self.test_accuracy_file.write(header + "\n")
            self.test_loss_file.write(header + "\n")
            self.val_accuracy_file.write(header + "\n")
            self.val_loss_file.write(header + "\n")
        self.criterion = criterion

        # set testing dataset
        if self.test_whole_dataset:
            if self.test_val_split is not None:
                assert 0.0 < self.test_val_split < 1.0
                perm = torch.randperm(len(test_dataset)).tolist()
                val_n = int(round(len(test_dataset) * self.test_val_split))
                self.test_idx = perm[:val_n]
                self.val_idx = perm[val_n:]
                test_ds = Subset(test_dataset, self.test_idx)
                val_ds = Subset(test_dataset, self.val_idx)
                if num_workers is None:
                    self.test_dataset = DataLoader(test_ds, batch_size=self.test_batch_size, shuffle=True, pin_memory=True)
                    self.val_dataset = DataLoader(val_ds, batch_size=self.test_batch_size, shuffle=True, pin_memory=True)
                else:
                    self.test_dataset = DataLoader(test_ds, batch_size=self.test_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=True)
                    self.val_dataset = DataLoader(val_ds, batch_size=self.test_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=True)
            else:
                if num_workers is None:
                    self.test_dataset = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True, pin_memory=True)
                else:
                    self.test_dataset = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=True)
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

        # store_top_accuracy_model
        self.store_top_accuracy_model_buffer = {}
        if self.store_top_accuracy_model:
            self.store_top_accuracy_model_path = os.path.join(output_path, f"top_accuracy_models")
            os.makedirs(self.store_top_accuracy_model_path, exist_ok=True)
            for node in node_names:
                self.store_top_accuracy_model_buffer[node] = OrderedDict()

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.phase in self.phase_to_record:
            node_names_and_model_stats = {}
            for node_name in self.node_order:
                target_node = parameters.node_container[node_name]
                model_stat = target_node.get_model_stat()
                node_names_and_model_stats[node_name] = model_stat
            self.trigger_without_runtime_parameters(parameters.current_tick, node_names_and_model_stats, parameters.phase.name)

    def trigger_without_runtime_parameters(self, tick, node_names_and_model_stats, phase_str=None):
        if tick % self.interval != 0:
            return
        row_accuracy = []
        row_loss = []
        row_output_var = []
        final_accuracy = {}
        final_model = {}

        # for test/val split
        row_test_accuracy = []
        row_test_loss = []
        row_val_accuracy = []
        row_val_loss = []
        for node_name in self.node_order:
            loss_test, accuracy_test, loss_val, accuracy_val = None, None, None, None,
            if self.test_whole_dataset:
                model_stat = node_names_and_model_stats[node_name]
                self.test_model.load_state_dict(model_stat)
                if self.allocated_gpu is not None:
                    self.test_model.to(self.allocated_gpu.device)
                self.test_model.eval()
                test_loss, test_correct, test_var, test_count = 0, 0, 0.0, 0
                val_loss, val_correct, val_var, val_count = 0, 0, 0.0, 0
                for d, l in self.test_dataset:
                    test_data = d
                    test_labels = l
                    if self.allocated_gpu is not None:
                        test_data, test_labels = test_data.to(self.allocated_gpu.device), test_labels.to(self.allocated_gpu.device)
                    outputs = self.test_model(test_data)
                    test_loss += self.criterion(outputs, test_labels).item() * test_labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    test_correct += (predicted == test_labels).sum().item()
                    test_count += test_labels.size(0)
                    test_var += outputs.var(dim=0, unbiased=False).mean().item()

                if self.test_val_split is not None:
                    for d, l in self.val_dataset:
                        val_data = d
                        val_labels = l
                        if self.allocated_gpu is not None:
                            val_data, val_labels = val_data.to(self.allocated_gpu.device), val_labels.to(self.allocated_gpu.device)
                        outputs = self.test_model(val_data)
                        val_loss += self.criterion(outputs, val_labels).item() * val_labels.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == val_labels).sum().item()
                        val_count += val_labels.size(0)
                        val_var += outputs.var(dim=0, unbiased=False).mean().item()
                loss = (test_loss+val_loss) / (test_count+val_count)
                accuracy = (test_correct+val_correct) / (test_count+val_count)
                var = (test_var+val_var) / (test_count+val_count)
                # for test/val split
                loss_test = test_loss / test_count
                accuracy_test = test_correct / test_count
                loss_val = val_loss / val_count
                accuracy_val = val_correct / val_count
            else:
                if self.use_fixed_testing_dataset:
                    total_loss, correct, var_acc, total = 0, 0, 0.0, 0
                    model_stat = node_names_and_model_stats[node_name]
                    self.test_model.load_state_dict(model_stat)
                    for test_data, test_labels in self.test_dataset:
                        if self.allocated_gpu is not None:
                            test_data, test_labels = test_data.to(self.allocated_gpu.device), test_labels.to(self.allocated_gpu.device)
                        if self.allocated_gpu is not None:
                            self.test_model.to(self.allocated_gpu.device)
                        self.test_model.eval()
                        outputs = self.test_model(test_data)
                        total_loss += self.criterion(outputs, test_labels).item()
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == test_labels).sum().item()
                        total += test_labels.size(0)
                        var_acc += outputs.var(dim=0, unbiased=False).mean().item()
                    loss = total_loss / total
                    accuracy = correct / total
                    var = var_acc / total
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
                    var = outputs.var(dim=0, unbiased=False).mean().item()
            row_accuracy.append('%.4E' % accuracy)
            row_loss.append('%.4E' % loss)
            row_output_var.append('%.4E' % var)
            if self.test_val_split is not None:
                row_test_accuracy.append('%.4E' % accuracy_test)
                row_test_loss.append('%.4E' % loss_test)
                row_val_accuracy.append('%.4E' % accuracy_val)
                row_val_loss.append('%.4E' % loss_val)
            final_accuracy[node_name] = accuracy
            final_model[node_name] = model_stat
        row_accuracy_str = ",".join([str(tick), str(phase_str), *row_accuracy])
        row_loss_str = ",".join([str(tick), str(phase_str), *row_loss])
        row_output_var_str = ",".join([str(tick), str(phase_str), *row_output_var])
        self.accuracy_file.write(row_accuracy_str + "\n")
        self.accuracy_file.flush()
        self.loss_file.write(row_loss_str + "\n")
        self.loss_file.flush()
        self.output_var_file.write(row_output_var_str + "\n")
        self.output_var_file.flush()
        if self.test_val_split is not None:
            row_test_accuracy_str = ",".join([str(tick), str(phase_str), *row_test_accuracy])
            row_test_loss_str = ",".join([str(tick), str(phase_str), *row_test_loss])
            row_val_accuracy_str = ",".join([str(tick), str(phase_str), *row_val_accuracy])
            row_val_loss_str = ",".join([str(tick), str(phase_str), *row_val_loss])
            self.test_accuracy_file.write(row_test_accuracy_str + "\n")
            self.test_accuracy_file.flush()
            self.test_loss_file.write(row_test_loss_str + "\n")
            self.test_loss_file.flush()
            self.val_accuracy_file.write(row_val_accuracy_str + "\n")
            self.val_accuracy_file.flush()
            self.val_loss_file.write(row_val_loss_str + "\n")
            self.val_loss_file.flush()

        # store_top_accuracy_model
        self._check_store_top_accuracy_model(final_accuracy, final_model, tick)

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
        # output variance file
        copy_file(self.output_var_file_name, self.output_var_file)

        if self.test_val_split is not None:
            copy_file(self.test_accuracy_file_name, self.test_accuracy_file)
            copy_file(self.test_loss_file_name, self.test_loss_file)
            copy_file(self.val_accuracy_file_name, self.val_accuracy_file)
            copy_file(self.val_loss_file_name, self.val_loss_file)

    def _check_store_top_accuracy_model(self, final_accuracy, final_model, tick):
        if self.store_top_accuracy_model:
            for node_name in self.node_order:
                accuracy = final_accuracy[node_name]
                model = final_model[node_name]
                buffer = self.store_top_accuracy_model_buffer[node_name]
                save_name = f"name_{node_name}_tick_{tick}_acc_{accuracy}.model.pt"
                save_path = os.path.join(self.store_top_accuracy_model_path, save_name)
                buffer_changed = False
                if accuracy not in buffer:
                    if len(buffer) < self.store_top_accuracy_model_count:
                        buffer[accuracy] = save_path
                        util.save_model_state(save_path, model, model_name=self.model_name, dataset_name=self.dataset_name)
                        buffer_changed = True
                    else:
                        smallest_accuracy, smallest_accuracy_path = next(iter(buffer.items()))
                        if smallest_accuracy < accuracy:
                            # new top accuracy models
                            buffer.pop(smallest_accuracy)
                            os.remove(smallest_accuracy_path)
                            buffer[accuracy] = save_path
                            util.save_model_state(save_path, model, model_name=self.model_name, dataset_name=self.dataset_name)
                            buffer_changed = True
                if buffer_changed:
                    buffer = OrderedDict(sorted(buffer.items()))
                    self.store_top_accuracy_model_buffer[node_name] = buffer

    def __del__(self):
        self.accuracy_file.flush()
        self.accuracy_file.close()
        self.loss_file.flush()
        self.loss_file.close()
        self.output_var_file.flush()
        self.output_var_file.close()
        if self.test_val_split is not None:
            self.test_accuracy_file.flush()
            self.test_loss_file.flush()
            self.val_accuracy_file.flush()
            self.val_loss_file.flush()

import unittest
class TestStoreTopAccuracyModel(unittest.TestCase):
    def test_1(self):
        service = ServiceTestAccuracyLossRecorder(10, 100, model_name="test", store_top_accuracy_model_count=5)
        service.node_order = ["0"]
        service.store_top_accuracy_model_path = "."
        service.store_top_accuracy_model_buffer = {"0": OrderedDict()}
        final_model = {"0": 1}
        final_accuracy = {"0": 0.1}
        service._check_store_top_accuracy_model(final_accuracy, final_model, 1)
        final_accuracy = {"0": 0.9}
        service._check_store_top_accuracy_model(final_accuracy, final_model, 1)
        final_accuracy = {"0": 0.2}
        service._check_store_top_accuracy_model(final_accuracy, final_model, 1)
        final_accuracy = {"0": 0.8}
        service._check_store_top_accuracy_model(final_accuracy, final_model, 1)
        final_accuracy = {"0": 0.7}
        service._check_store_top_accuracy_model(final_accuracy, final_model, 1)
        final_accuracy = {"0": 0.3}
        service._check_store_top_accuracy_model(final_accuracy, final_model, 1)
        final_accuracy = {"0": 0.4}
        service._check_store_top_accuracy_model(final_accuracy, final_model, 1)
        final_accuracy = {"0": 0.5}
        service._check_store_top_accuracy_model(final_accuracy, final_model, 1)


