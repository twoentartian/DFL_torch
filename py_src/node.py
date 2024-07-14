import copy
import torch
import logging
import os
import random
import numpy as np

from torch.utils.data import DataLoader

from py_src import dataset, internal_names, util, model_average
from py_src.ml_setup import MlSetup
from py_src.cuda import CudaDevice

logger = logging.getLogger(f"{internal_names.logger_simulator_base_name}.{util.basename_without_extension(__file__)}")


def re_initialize_model(model):
    random_data = os.urandom(4)
    seed = int.from_bytes(random_data, byteorder="big")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class Node:
    name: int
    is_using_model_stat: bool
    next_training_tick: int
    ml_setup: MlSetup
    train_loader: DataLoader
    model_averager: model_average.ModelAverager
    model_buffer_size: int
    use_cpu: bool
    send_model_after_P_training: int

    def __init__(self, name: int, ml_setup: MlSetup, use_model_stat: bool|None=None, allocated_gpu: CudaDevice=None, optimizer: None | torch.optim.Optimizer=None, use_cpu: bool=False):
        """
        for use_model_stat == True, the optimizer should be an optimizer attached to the model owned by gpu
        for use_model_stat == False, the optimizer should be set by "set_optimizer", user need to attach the model parameter to the optimizer externally
        """
        model = ml_setup.model
        self.name = name
        self.is_using_model_stat = use_model_stat
        self.use_cpu = use_cpu

        re_initialize_model(model)
        if use_cpu:
            self.model = copy.deepcopy(model)
            self.optimizer = None
        else:
            assert allocated_gpu is not None
            self.allocated_gpu = allocated_gpu
            if use_model_stat:
                assert optimizer is not None
                self.model_status = copy.deepcopy(model.state_dict())
                self.optimizer_status = copy.deepcopy(optimizer.state_dict())
            else:
                self.model = copy.deepcopy(model)
                self.model = self.model.to(self.allocated_gpu.device)
                self.optimizer = None

        self.next_training_tick = 0
        self.normalized_dataset_label_distribution = None
        self.ml_setup = None
        self.train_loader = None

        self.__dataset_label_distribution = None
        self.__dataset_with_fast_label = None

        """average buffer (average)"""
        self.model_averager = None
        self.model_buffer_size = None

        """status"""
        self.is_training_this_tick = False
        self.is_averaging_this_tick = False

        """initial state"""
        self.send_model_after_P_training = 1
        self._send_model_counter = 0
        self.most_recent_loss = 0

    def is_sending_model(self) -> bool:
        self._send_model_counter += 1
        if self._send_model_counter >= self.send_model_after_P_training:
            self._send_model_counter = 0
            return True
        else:
            return False

    def reset_statu_flags(self):
        self.is_training_this_tick = False
        self.is_averaging_this_tick = False

    def set_average_algorithm(self, average_algorithm: model_average.ModelAverager):
        self.model_averager = average_algorithm

    def set_average_buffer_size(self, average_buffer_size: int):
        self.model_buffer_size = average_buffer_size

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def set_ml_setup(self, setup: MlSetup):
        self.ml_setup = setup
        if self.__dataset_label_distribution is not None:
            self.set_label_distribution(self.__dataset_label_distribution, self.__dataset_with_fast_label)

    def set_batch_size(self, batch_size: int):
        assert self.ml_setup is not None
        new_ml_setup = copy.copy(self.ml_setup)
        new_ml_setup.training_batch_size = batch_size
        self.set_ml_setup(new_ml_setup)

    def set_next_training_tick(self, tick):
        self.next_training_tick = tick

    def set_label_distribution(self, dataset_label_distribution, dataset_with_fast_label: dataset.DatasetWithFastLabelSelection):
        self.__dataset_label_distribution = dataset_label_distribution
        self.__dataset_with_fast_label = dataset_with_fast_label
        self.normalized_dataset_label_distribution = dataset_label_distribution / dataset_label_distribution.sum()
        self.train_loader = dataset_with_fast_label.get_train_loader_by_label_prob(self.normalized_dataset_label_distribution, self.ml_setup.training_batch_size)

    def set_model_stat(self, model_stat):
        """warning: model_stat is shallow copied"""
        if self.use_cpu:
            self.model.load_state_dict(model_stat)
        else:
            if self.is_using_model_stat:
                self.model_status = model_stat
            else:
                self.model.load_state_dict(model_stat)

    def get_dataset_label_distribution(self):
        return self.normalized_dataset_label_distribution

    def get_data_loader(self):
        return self.train_loader

    def get_model_stat(self):
        if self.is_using_model_stat:
            return self.model_status
        else:
            return self.model.state_dict()

