import torch.nn as nn
from torchvision import transforms, models, datasets
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.other_setup import get_pytorch_training_imagenet

class GroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(GroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels, eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x

def resnet18_cifar10(enable_replace_bn_with_group_norm=False):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10()

    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=GroupNorm)
        output_ml_setup.model_name = str(ModelType.resnet18_gn)
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
        output_ml_setup.model_name = str(ModelType.resnet18_bn)
    output_ml_setup.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # change for cifar10 resolution
    output_ml_setup.model.maxpool = nn.Identity()
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def resnet18_cifar100(enable_replace_bn_with_group_norm=False):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar100()

    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=GroupNorm)
        output_ml_setup.model_name = str(ModelType.resnet18_gn)
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
        output_ml_setup.model_name = str(ModelType.resnet18_bn)
    output_ml_setup.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # change for cifar100 resolution
    output_ml_setup.model.maxpool = nn.Identity()
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def resnet18_imagenet100(pytorch_preset_version=2, enable_replace_bn_with_group_norm=False):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet100(pytorch_preset_version)

    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=GroupNorm)
        output_ml_setup.model_name = str(ModelType.resnet18_gn)
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
        output_ml_setup.model_name = str(ModelType.resnet18_bn)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    loss_fn, collate_fn, model_ema_decay, model_ema_steps = get_pytorch_training_imagenet(pytorch_preset_version)
    output_ml_setup.criterion = loss_fn
    output_ml_setup.collect_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)
    return output_ml_setup

def resnet18_imagenet1k(pytorch_preset_version=2, enable_replace_bn_with_group_norm=False):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k(pytorch_preset_version)

    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=1000, norm_layer=GroupNorm)
        output_ml_setup.model_name = str(ModelType.resnet18_gn)
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=1000)
        output_ml_setup.model_name = str(ModelType.resnet18_bn)
    output_ml_setup.get_info_from_dataset(dataset)

    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True

    loss_fn, collate_fn, model_ema_decay, model_ema_steps = get_pytorch_training_imagenet(pytorch_preset_version)
    output_ml_setup.criterion = loss_fn
    output_ml_setup.collect_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)

    return output_ml_setup



def resnet50_imagenet1k(pytorch_preset_version=2):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k(pytorch_preset_version)

    output_ml_setup.model = models.resnet50(progress=False, num_classes=1000)
    output_ml_setup.model_name = str(ModelType.resnet50)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True

    loss_fn, collate_fn, model_ema_decay, model_ema_steps = get_pytorch_training_imagenet(pytorch_preset_version)
    output_ml_setup.criterion = loss_fn
    output_ml_setup.collect_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)
    return output_ml_setup