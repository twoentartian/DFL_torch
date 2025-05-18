import torch.nn as nn
from py_src.models import simplenet
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType

def simplenet_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10()

    output_ml_setup.model = simplenet.__dict__["simplenet_cifar_5m"](num_classes=10)
    output_ml_setup.model_name = str(ModelType.simplenet)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def simplenet_cifar100():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10()

    output_ml_setup.model = simplenet.__dict__["simplenet_cifar_5m"](num_classes=100)
    output_ml_setup.model_name = str(ModelType.simplenet)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup