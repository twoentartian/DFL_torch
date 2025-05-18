import torch.nn as nn
from py_src.models import lenet

import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType

""" MNIST + LeNet """
def lenet4_mnist():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_mnist()

    output_ml_setup.model = lenet.lenet4()
    output_ml_setup.model_name = str(ModelType.lenet4)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.weights_init_func = lenet.weights_init_xavier
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

def lenet5_mnist():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_mnist()

    output_ml_setup.model = lenet.lenet5()
    output_ml_setup.model_name = str(ModelType.lenet5)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.weights_init_func = lenet.weights_init_xavier
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

def lenet5_random_mnist():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_random_mnist()

    output_ml_setup.model = lenet.lenet5()
    output_ml_setup.model_name = str(ModelType.lenet5)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.weights_init_func = lenet.weights_init_xavier
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

def lenet5_large_fc_mnist():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_mnist()

    output_ml_setup.model = lenet.lenet5(large_fc=True)
    output_ml_setup.model_name = str(ModelType.lenet5_large_fc)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.weights_init_func = lenet.weights_init_xavier
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup