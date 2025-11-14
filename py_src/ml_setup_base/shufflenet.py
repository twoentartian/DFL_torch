import torch.nn as nn
from torchvision import models
from py_src.models import shufflenet
from py_src.ml_setup_base.base import MlSetup
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.model import ModelType
from py_src.ml_setup_base.other_setup import get_pytorch_training_imagenet

def shufflenet_v2_cifar10():
    output_ml_setup = MlSetup()

    dataset = ml_setup_dataset.dataset_cifar10()

    model_ft = shufflenet.ShuffleNet(10, g=1, scale_factor=1)
    output_ml_setup.model_name = str(ModelType.shufflenet_v2.name)
    output_ml_setup.model = model_ft
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def shufflenet_v2_cifar100():
    output_ml_setup = MlSetup()

    dataset = ml_setup_dataset.dataset_cifar100()

    model_ft = shufflenet.ShuffleNet(100, g=1, scale_factor=1)
    output_ml_setup.model_name = str(ModelType.shufflenet_v2.name)
    output_ml_setup.model = model_ft
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def shufflenet_v2_x2_0_imagenet1k(pytorch_preset_version=2):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k(pytorch_preset_version)

    output_ml_setup.model = models.shufflenet_v2_x2_0(progress=False, num_classes=1000)
    output_ml_setup.model_name = str(ModelType.shufflenet_v2_x2_0.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn = get_pytorch_training_imagenet(pytorch_preset_version)
    output_ml_setup.criterion = loss_fn
    output_ml_setup.collate_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)
    output_ml_setup.sampler_fn = sampler_fn
    return output_ml_setup