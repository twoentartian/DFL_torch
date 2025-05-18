import torch.nn as nn
import py_src.third_party.compact_transformers.src.cct as cct
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType



def cct7_3x1_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10()

    output_ml_setup.model = cct.cct_7_3x1_32()
    output_ml_setup.model_name = str(ModelType.cct7.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct7_7x2_imagenet1k():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k(1)

    output_ml_setup.model = cct.cct_7_7x2_224()
    output_ml_setup.model_name = str(ModelType.cct7.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct7_7x2_imagenet100():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet100(1)

    output_ml_setup.model = cct.cct_7_7x2_224()
    output_ml_setup.model_name = str(ModelType.cct7.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct7_7x2_imagenet10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet10(1)

    output_ml_setup.model = cct.cct_7_7x2_224()
    output_ml_setup.model_name = str(ModelType.cct7.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup