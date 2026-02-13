import os
import torch
import random
from typing import Optional
from collections.abc import Callable
import numpy as np
from enum import Enum, auto
import torch.nn as nn
from torch.utils.data import DataLoader

# CriterionType: indicate the criterion function for diffusion models, etc
class CriterionType(Enum):
    Diffusion = auto()
    # Others = the criterion function

class DatasetSetup:
    def __init__(self, name, dataset_type, training_data, testing_data, labels=None):
        self.training_data = training_data
        self.testing_data = testing_data
        self.dataset_name = name
        self.dataset_type = dataset_type

        if labels is None:
            self.labels = self._get_dataset_labels(self.testing_data)
        else:
            self.labels = labels
        if hasattr(self.training_data, "__getitem__"):
            sample_data = self.training_data[0][0]
        elif hasattr(self.training_data, "get_first_data_tensor"):
            sample_data = self.training_data.get_first_data_tensor()
        else:
            raise NotImplementedError("cannot get the first element from dataset")
        self.tensor_size = sample_data.shape

        self.is_masked_dataset = False


    def _get_dataset_labels(self, dataset):
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        labels_set = set()
        for _, labels in dataloader:
            labels_set.update(labels.tolist())
        return labels_set

class MlSetup:
    def __init__(self):
        self.model: torch.nn.Module = None
        self.model_name: str = None
        self.model_type = None
        self.dataset_name: str = None
        self.dataset_type = None
        self.training_data = None
        self.testing_data = None
        self.criterion = None
        self.mixup_fn = None
        self.model_ema = None
        self.collate_fn = None
        self.collate_fn_val = None
        self.sampler_fn = None
        self.training_batch_size = None
        self.dataset_label = None
        self.dataset_tensor_size = None
        self.weights_init_func = None
        self.get_lr_scheduler_func = None
        self.clip_grad_norm = None

        self.has_normalization_layer = None

        self.func_handler_post_training = []

        """these dataset loader will be used during training and finding paths if they are set"""
        self.override_training_dataset_loader = None
        self.override_testing_dataset_loader = None

        self.override_train_step_function: Optional[Callable[..., TrainStepOutput]] = None
        self.override_evaluation_step_function: Optional[Callable[..., TrainStepOutput]] = None

    def self_validate(self):
        pass  # do nothing for now

    def get_info_from_dataset(self, dataset: DatasetSetup):
        self.training_data = dataset.training_data
        self.testing_data = dataset.testing_data
        self.dataset_name = dataset.dataset_name
        self.dataset_type = dataset.dataset_type
        self.dataset_label = dataset.labels
        self.dataset_tensor_size = dataset.tensor_size

    def assign_names_to_layers(self):
        for name, module in self.model.named_modules():
            if not hasattr(module, '_module_name'):
                module._module_name = name

    def re_initialize_model(self, model):
        self.assign_names_to_layers()

        # Set random seeds
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        def reset_parameters_recursively(module):
            for submodule in module.children():
                if hasattr(submodule, 'reset_parameters'):
                    submodule.reset_parameters()
                else:
                    reset_parameters_recursively(submodule)
        if self.weights_init_func is None:
            reset_parameters_recursively(model)
        else:
            model.apply(self.weights_init_func)

    def get_brief_description(self) -> str:
        return f"{self.model_name}@{self.dataset_name}"


def replace_bn_with_ln(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Replace BatchNorm2d with LayerNorm
            layer_norm = nn.LayerNorm(module.num_features, elementwise_affine=True)
            setattr(model, name, layer_norm)
        else:
            # Recursively replace in submodules
            replace_bn_with_ln(module)

class TrainStepOutput:
    def __init__(self):
        self.sample_count = None
        self.correct_count = None
        self.loss_value = None