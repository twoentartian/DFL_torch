import torch.nn as nn

import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType
import py_src.ml_setup_base.dla_cifar as dla_cifar

def dla_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10()
    output_ml_setup.model = dla_cifar.DLA(num_classes=10)
    output_ml_setup.model_name = str(ModelType.dla.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    return output_ml_setup

