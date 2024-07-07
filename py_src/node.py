import copy
import torch
import logging

from torch.utils.data import DataLoader
from py_src import dataset, internal_names, util, ml_setup

logger = logging.getLogger(f"{internal_names.logger_simulator_base_name}.{util.basename_without_extension(__file__)}")

class Node:
    def __init__(self, name: int, model: torch.nn.Module):
        self.name = name
        self.model = copy.deepcopy(model)
        self.next_training_tick = 0
        self.normalized_dataset_label_distribution = None
        self.ml_setup = None
        self.train_loader = None
        self.optimizer = None
        self.__dataset_label_distribution = None
        self.__dataset_with_fast_label = None

    def set_ml_setup(self, setup: ml_setup.MlSetup):
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

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def get_dataset_label_distribution(self):
        return self.normalized_dataset_label_distribution

    def get_data_loader(self):
        return self.train_loader
