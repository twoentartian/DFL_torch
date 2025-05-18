import torch.nn as nn
from torchvision import models
from py_src.ml_setup_base.base import MlSetup
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.model import ModelType




def vit_b_16_imagenet100():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet100(2)

    vit_b_16 = models.vit_b_16(weights=None)
    output_ml_setup.model_name = str(ModelType.vit_b_16.name)
    output_ml_setup.model = vit_b_16
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

def vit_b_16_imagenet1k():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k(2)

    vit_b_16 = models.vit_b_16(weights=None)
    output_ml_setup.model_name = str(ModelType.vit_b_16.name)
    output_ml_setup.model = vit_b_16
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup