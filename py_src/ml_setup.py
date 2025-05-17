import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import random
import numpy as np
from enum import Enum, auto
from torchvision import transforms, models, datasets
from py_src.models import simplenet, lenet, vgg, mobilenet, shufflenet

import py_src.third_party.compact_transformers.src.cct as cct

import py_src.ml_setup_base.dataset as ml_setup_dataset

def replace_bn_with_ln(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Replace BatchNorm2d with LayerNorm
            layer_norm = nn.LayerNorm(module.num_features, elementwise_affine=True)
            setattr(model, name, layer_norm)
        else:
            # Recursively replace in submodules
            replace_bn_with_ln(module)


class MlSetup:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.dataset_name = None
        self.training_data = None
        self.testing_data = None
        self.criterion = None
        self.training_batch_size = None
        self.dataset_label = None
        self.dataset_tensor_size = None
        self.weights_init_func = None
        self.get_lr_scheduler_func = None

        self.has_normalization_layer = None

    def self_validate(self):
        pass  # do nothing for now

    def get_info_from_dataset(self, dataset: ml_setup_dataset.DatasetSetup):
        self.training_data = dataset.training_data
        self.testing_data = dataset.testing_data
        self.dataset_name = dataset.dataset_name
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



""" MNIST + LeNet """
def lenet4_mnist():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_mnist()

    output_ml_setup.model = lenet.lenet4()
    output_ml_setup.model_name = "lenet4"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.weights_init_func = lenet.weights_init_xavier
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

def lenet5_mnist():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_mnist()

    output_ml_setup.model = lenet.lenet5()
    output_ml_setup.model_name = "lenet5"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.weights_init_func = lenet.weights_init_xavier
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

def lenet5_random_mnist():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_random_mnist()

    output_ml_setup.model = lenet.lenet5()
    output_ml_setup.model_name = "lenet5"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.weights_init_func = lenet.weights_init_xavier
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

def lenet5_large_fc_mnist():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_mnist()

    output_ml_setup.model = lenet.lenet5(large_fc=True)
    output_ml_setup.model_name = "lenet5_large_fc"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.weights_init_func = lenet.weights_init_xavier
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

""" CIFAR10 + CCT7/3x1 """
def cct7_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10_32()

    output_ml_setup.model = cct.cct_7_3x1_32()
    output_ml_setup.model_name = "cct7"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct7_imagenet100():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet100()

    output_ml_setup.model = cct.cct_7_7x2_224()
    output_ml_setup.model_name = "cct7"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct7_imagenet10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet10()

    output_ml_setup.model = cct.cct_7_7x2_224()
    output_ml_setup.model_name = "cct7"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup


""" CIFAR10 + ResNet18 """
class GroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(GroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels, eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x

def resnet18_cifar10(enable_replace_bn_with_group_norm=False):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10_32()

    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=GroupNorm)
        output_ml_setup.model_name = "resnet18_gn"
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
        output_ml_setup.model_name = "resnet18_bn"
    output_ml_setup.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # change for cifar10 resolution
    output_ml_setup.model.maxpool = nn.Identity()
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def resnet18_cifar100(enable_replace_bn_with_group_norm=False):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar100_32()

    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=GroupNorm)
        output_ml_setup.model_name = "resnet18_gn"
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
        output_ml_setup.model_name = "resnet18_bn"
    output_ml_setup.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # change for cifar100 resolution
    output_ml_setup.model.maxpool = nn.Identity()
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def resnet18_imagenet100(enable_replace_bn_with_group_norm=False):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet100()

    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=GroupNorm)
        output_ml_setup.model_name = "resnet18_gn"
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
        output_ml_setup.model_name = "resnet18_bn"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def resnet18_imagenet1k(enable_replace_bn_with_group_norm=False):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k()

    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=1000, norm_layer=GroupNorm)
        output_ml_setup.model_name = "resnet18_gn"
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=1000)
        output_ml_setup.model_name = "resnet18_bn"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

""" CIFAR10 + MobileNet V3 small """
def mobilenet_v3_small_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10_32()

    output_ml_setup.model = models.mobilenet_v3_small(progress=False, num_classes=10)
    output_ml_setup.model.classifier[-1] = torch.nn.Linear(in_features=1024, out_features=10)
    output_ml_setup.model_name = "mobilenet_v3_small"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup


""" CIFAR10 + MobileNet V2 (for CIFAR10 dataset) """
def mobilenet_v2_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10_32()

    output_ml_setup.model = mobilenet.MobileNetV2(10)
    output_ml_setup.model_name = "mobilenet_v2"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

""" ImageNet + MobileNet V3 large """
def mobilenet_v3_large_imagenet1k():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k()

    output_ml_setup.model = models.mobilenet_v3_large(pretrained=False)
    output_ml_setup.model.classifier[-1] = nn.Linear(output_ml_setup.model.classifier[-1].in_features, 1000)
    output_ml_setup.model_name = "mobilenet_v3_large"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

""" CIFAR10 + SimpleNet """
def simplenet_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10_32()

    output_ml_setup.model = simplenet.__dict__["simplenet_cifar_5m"](num_classes=10)
    output_ml_setup.model_name = "simplenet"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def simplenet_cifar100():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10_32()

    output_ml_setup.model = simplenet.__dict__["simplenet_cifar_5m"](num_classes=100)
    output_ml_setup.model_name = "simplenet"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

""" MNIST + vgg11 """
def vgg11_mnist():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_mnist_224()

    vgg11 = vgg.VGG11_no_bn(in_channels=1, num_classes=10)
    output_ml_setup.model_name = "vgg11_mnist_no_bn"
    output_ml_setup.model = vgg11
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

""" CIFAR10 + vgg11 """
def vgg11_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10_224()

    vgg11 = vgg.VGG11_no_bn(in_channels=3, num_classes=10)
    output_ml_setup.model_name = "vgg11_cifar10_no_bn"
    output_ml_setup.model = vgg11
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

""" ViT + ImageNet """
def vit_b_16_imagenet100():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet100()

    vit_b_16 = models.vit_b_16(weights=None)
    output_ml_setup.model_name = "vit_b_16"
    output_ml_setup.model = vit_b_16
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

""" EfficientNet + CIFAR100 """
def efficientnet_cifar100():
    output_ml_setup = MlSetup()
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img_size = 224
    crop_size = 224
    transform_train = transforms.Compose([
            transforms.Resize(img_size),  # , interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
        ])

    transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    dataset = ml_setup_dataset.dataset_cifar100_224(transforms_training=transform_train, transforms_testing=transform_test)

    model_ft = models.efficientnet_v2_l(weights=None)
    in_features = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = nn.Linear(in_features, 100)

    output_ml_setup.model_name = "efficientnet_v2"
    output_ml_setup.model = model_ft
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

""" ShuffleNet + CIFAR10 """
def shufflenet_v2_cifar10():
    output_ml_setup = MlSetup()
    mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    dataset = ml_setup_dataset.dataset_cifar10_32(transforms_training=transform_train, transforms_testing=transform_test)

    model_ft = shufflenet.ShuffleNet(10, g=1, scale_factor=1)
    output_ml_setup.model_name = "shufflenet_v2"
    output_ml_setup.model = model_ft
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup


""" Helper function """
class ModelType(Enum):
    lenet5 = auto()
    resnet18_bn = auto()
    resnet18_gn = auto()
    simplenet = auto()
    cct7 = auto()
    vit = auto()
    lenet5_large_fc = auto()
    mobilenet_v3_small = auto()
    mobilenet_v3_large = auto()
    mobilenet_v2 = auto()
    lenet4 = auto()
    vgg11_no_bn = auto()
    efficientnet_v2 = auto()
    shufflenet_v2 = auto()

class DatasetType(Enum):
    default = auto()
    mnist = auto()
    random_mnist = auto()
    cifar10 = auto()
    cifar100 = auto()
    imagenet10 = auto()
    imagenet100 = auto()
    imagenet1k = auto()

def get_ml_setup_from_config(model_type: str, dataset_type: str = 'default'):
    model_type = ModelType[model_type]
    dataset_type_enum = DatasetType[dataset_type]
    output_ml_setup = get_ml_setup_from_model_type(model_type, dataset_type=dataset_type_enum)
    return output_ml_setup

def get_ml_setup_from_model_type(model_name, dataset_type=DatasetType.default):
    if model_name == ModelType.lenet5:
        if dataset_type in [dataset_type.default, dataset_type.mnist]:
            output_ml_setup = lenet5_mnist()
        elif dataset_type in [dataset_type.random_mnist]:
            output_ml_setup = lenet5_random_mnist()
        else:
            raise NotImplemented
    elif model_name == ModelType.lenet4:
        assert dataset_type in [dataset_type.default, dataset_type.mnist]
        output_ml_setup = lenet4_mnist()
    elif model_name == ModelType.resnet18_bn or model_name == ModelType.resnet18_gn:
        enable_replace_bn_with_group_norm = model_name == ModelType.resnet18_gn
        if dataset_type in [dataset_type.default, dataset_type.cifar10]:
            output_ml_setup = resnet18_cifar10(enable_replace_bn_with_group_norm=enable_replace_bn_with_group_norm)
        elif dataset_type in [dataset_type.cifar100]:
            output_ml_setup = resnet18_cifar100(enable_replace_bn_with_group_norm=enable_replace_bn_with_group_norm)
        elif dataset_type in [dataset_type.imagenet100]:
            output_ml_setup = resnet18_imagenet100(enable_replace_bn_with_group_norm=enable_replace_bn_with_group_norm)
        elif dataset_type in [dataset_type.imagenet1k]:
            output_ml_setup = resnet18_imagenet1k(enable_replace_bn_with_group_norm=enable_replace_bn_with_group_norm)
        else:
            raise NotImplemented
    elif model_name == ModelType.simplenet:
        assert dataset_type in [dataset_type.default, dataset_type.cifar10]
        output_ml_setup = simplenet_cifar10()
    elif model_name == ModelType.cct7:
        if dataset_type in [dataset_type.default, dataset_type.cifar10]:
            output_ml_setup = cct7_cifar10()
        elif dataset_type in [dataset_type.imagenet100]:
            output_ml_setup = cct7_imagenet100()
        else:
            raise NotImplemented
    elif model_name == ModelType.lenet5_large_fc:
        assert dataset_type in [dataset_type.default, dataset_type.mnist]
        output_ml_setup = lenet5_large_fc_mnist()
    elif model_name == ModelType.mobilenet_v3_small:
        assert dataset_type in [dataset_type.default, dataset_type.cifar10]
        output_ml_setup = mobilenet_v3_small_cifar10()
    elif model_name == ModelType.mobilenet_v3_large:
        assert dataset_type in [dataset_type.default, dataset_type.cifar10]
        output_ml_setup = mobilenet_v3_large_imagenet1k()
    elif model_name == ModelType.mobilenet_v2:
        if dataset_type in [DatasetType.default, DatasetType.cifar10]:
            output_ml_setup = mobilenet_v2_cifar10()
        else:
            raise NotImplemented
    elif model_name == ModelType.vgg11_no_bn:
        if dataset_type in [DatasetType.default, DatasetType.mnist]:
            output_ml_setup = vgg11_mnist()
        elif dataset_type in [DatasetType.cifar10]:
            output_ml_setup = vgg11_cifar10()
        else:
            raise NotImplemented
    elif model_name == ModelType.vit:
        if dataset_type in [DatasetType.default, DatasetType.imagenet100]:
            output_ml_setup = vit_b_16_imagenet100()
        else:
            raise NotImplemented
    elif model_name == ModelType.efficientnet_v2:
        if dataset_type in [DatasetType.default, DatasetType.cifar100]:
            output_ml_setup = efficientnet_cifar100()
        else:
            raise NotImplemented
    elif model_name == ModelType.shufflenet_v2:
        if dataset_type in [DatasetType.default, DatasetType.cifar10]:
            output_ml_setup = shufflenet_v2_cifar10()
        else:
            raise NotImplemented
    else:
        raise ValueError(f'Invalid model type: {model_name}')
    return output_ml_setup

