import torch.nn as nn
from torchvision import models

from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.other_setup import get_pytorch_training_imagenet
import py_src.ml_setup_base.regnet_cifar as regnet_cifar


def regnet_y_400mf_imagenet1k(pytorch_preset_version=2):
    output_ml_setup = MlSetup()
    if pytorch_preset_version == 1:
        dataset = ml_setup_dataset.dataset_imagenet1k_custom()
    elif pytorch_preset_version == 2:
        dataset = ml_setup_dataset.dataset_imagenet1k_custom(auto_augment_policy='ta_wide', random_erase_prob=0.1, val_resize_size=232)
    else:
        raise NotImplementedError

    output_ml_setup.model = models.regnet_y_400mf(progress=False, num_classes=1000)
    output_ml_setup.model_name = str(ModelType.regnet_y_400mf.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn = get_pytorch_training_imagenet(pytorch_preset_version)
    output_ml_setup.criterion = loss_fn
    output_ml_setup.collate_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)
    output_ml_setup.sampler_fn = sampler_fn
    return output_ml_setup

def regnet_x_200mf_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10()
    output_ml_setup.model = regnet_cifar.RegNetX_200MF()
    output_ml_setup.model_name = str(ModelType.regnet_x_200mf.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    return output_ml_setup

def regnet_x_200mf_cifar100():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar100()
    output_ml_setup.model = regnet_cifar.RegNetX_200MF(num_classes=100)
    output_ml_setup.model_name = str(ModelType.regnet_x_200mf.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    return output_ml_setup