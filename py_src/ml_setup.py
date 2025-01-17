import torch
import os
import random
import numpy as np
from enum import Enum
import torch.nn as nn
from torchvision import transforms, models, datasets
from py_src.models import simple_net, lenet, vgg
import py_src.third_party.compact_transformers.src.cct as cct

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
        self.training_data = None
        self.testing_data = None
        self.criterion = None
        self.training_batch_size = None
        self.dataset_label = None
        self.weights_init_func = None
        self.get_lr_scheduler_func = None

        self.has_normalization_layer = None

    def self_validate(self):
        pass  # do nothing for now

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


""" MNIST """
def dataset_mnist():
    dataset_path = './data/mnist'
    mnist_train = datasets.MNIST(root=dataset_path, train=True, download=True)
    mean = mnist_train.data.float().mean() / 255
    std = mnist_train.data.float().std() / 255
    transforms_mnist_train = transforms.Compose([transforms.RandomRotation(5, fill=(0,)), transforms.RandomCrop(28, padding=2), transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])])
    transforms_mnist_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])])
    train_data = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transforms_mnist_train)
    test_data = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transforms_mnist_test)
    mnist_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return train_data, test_data, mnist_labels

def dataset_mnist_224():
    dataset_path = './data/mnist'
    mnist_train = datasets.MNIST(root=dataset_path, train=True, download=True)
    mean = mnist_train.data.float().mean() / 255
    std = mnist_train.data.float().std() / 255
    transforms_mnist_train = transforms.Compose(
        [transforms.RandomRotation(5, fill=(0,)),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[mean], std=[std])])
    transforms_mnist_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize(mean=[mean], std=[std])])
    train_data = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transforms_mnist_train)
    test_data = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transforms_mnist_test)
    mnist_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return train_data, test_data, mnist_labels

""" CIFAR10 """
def dataset_cifar10_32(transforms_training=None, transforms_testing=None):
    dataset_path = './data/cifar10'
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))

    if transforms_training is not None:
        transforms_cifar_train = transforms_training
    else:
        transforms_cifar_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.Normalize(*stats)])
    if transforms_testing is not None:
        transforms_cifar_test = transforms_testing
    else:
        transforms_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms_cifar_train)
    cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms_cifar_test)
    cifar10_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return cifar10_train, cifar10_test, cifar10_labels

def dataset_cifar10_224(transforms_training=None, transforms_testing=None):
    dataset_path = './data/cifar10'
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))

    if transforms_training is not None:
        transforms_cifar_train = transforms_training
    else:
        transforms_cifar_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((224, 224)),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.Normalize(*stats)])
    if transforms_testing is not None:
        transforms_cifar_test = transforms_testing
    else:
        transforms_cifar_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((224, 224)),
             transforms.Normalize(*stats)])
    cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms_cifar_train)
    cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms_cifar_test)
    cifar10_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return cifar10_train, cifar10_test, cifar10_labels

def dataset_cifar100_32(transforms_training=None, transforms_testing=None):
    dataset_path = './data/cifar100'
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    if transforms_training is not None:
        transforms_cifar_train = transforms_training
    else:
        transforms_cifar_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.Normalize(*stats)])
    if transforms_testing is not None:
        transforms_cifar_test = transforms_testing
    else:
        transforms_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    cifar100_train = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms_cifar_train)
    cifar100_test = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transforms_cifar_test)
    cifar100_labels = set(range(0, 100, 1))
    return cifar100_train, cifar100_test, cifar100_labels

""" ImageNet """
def dataset_imagenet(transforms_training=None, transforms_testing=None):
    dataset_path = './data/imagenet'

    standard_transform = transforms.Compose([
        transforms.Resize(256),  # Resize images to 256 pixels on the shorter side
        transforms.CenterCrop(224),  # Crop to 224x224 (standard for ImageNet)
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(  # Normalize with ImageNet's mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    if transforms_training is not None:
        transforms_cifar_train = transforms_training
    else:
        transforms_cifar_train = standard_transform
    if transforms_testing is not None:
        transforms_cifar_test = transforms_testing
    else:
        transforms_cifar_test = standard_transform

    imagenet_train = datasets.ImageNet(root=dataset_path, split='train', transform=transforms_cifar_train)
    imagenet_test = datasets.ImageNet(root=dataset_path, split='val', transform=transforms_cifar_test)
    image_labels = tuple(range(0, 1000))
    return imagenet_train, imagenet_test, image_labels


""" MNIST + LeNet """
def lenet4_mnist():
    output_ml_setup = MlSetup()
    output_ml_setup.model = lenet.lenet4()
    output_ml_setup.model_name = "lenet4"
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_mnist()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.weights_init_func = lenet.weights_init_xavier
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

def lenet5_mnist():
    output_ml_setup = MlSetup()
    output_ml_setup.model = lenet.lenet5()
    output_ml_setup.model_name = "lenet5"
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_mnist()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.weights_init_func = lenet.weights_init_xavier
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

def lenet5_large_fc_mnist():
    output_ml_setup = MlSetup()
    output_ml_setup.model = lenet.lenet5(large_fc=True)
    output_ml_setup.model_name = "lenet5_large_fc"
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_mnist()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.weights_init_func = lenet.weights_init_xavier
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

""" CIFAR10 + CCT7/3x1 """
def cct7_cifar10():
    output_ml_setup = MlSetup()
    output_ml_setup.model = cct.cct_7_3x1_32()
    output_ml_setup.model_name = "cct7"
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_cifar10_32()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
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
    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=GroupNorm)
        output_ml_setup.model_name = "resnet18_gn"
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
        output_ml_setup.model_name = "resnet18_bn"
    from torchvision.models.resnet import BasicBlock
    output_ml_setup.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # change for cifar10 resolution
    output_ml_setup.model.maxpool = nn.Identity()
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_cifar10_32()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def resnet18_cifar100(enable_replace_bn_with_group_norm=False):
    output_ml_setup = MlSetup()
    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=GroupNorm)
        output_ml_setup.model_name = "resnet18_gn"
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
        output_ml_setup.model_name = "resnet18_bn"
    from torchvision.models.resnet import BasicBlock
    output_ml_setup.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # change for cifar10 resolution
    output_ml_setup.model.maxpool = nn.Identity()
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_cifar100_32()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup


""" CIFAR10 + MobileNet V3 small """
def mobilenet_v3_small_cifar10():
    output_ml_setup = MlSetup()
    output_ml_setup.model = models.mobilenet_v3_small(progress=False, num_classes=10)
    output_ml_setup.model.classifier[-1] = torch.nn.Linear(in_features=1024, out_features=10)
    output_ml_setup.model_name = "mobilenet_v3_small"
    train_transforms = transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.RandomCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transforms = transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_cifar10_32(transforms_training=train_transforms, transforms_testing=test_transforms)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

""" ImageNet + MobileNet V3 large """
def mobilenet_v3_large_imagenet():
    output_ml_setup = MlSetup()
    output_ml_setup.model = models.mobilenet_v3_large(pretrained=False)
    output_ml_setup.model.classifier[-1] = nn.Linear(output_ml_setup.model.classifier[-1].in_features, 1000)
    output_ml_setup.model_name = "mobilenet_v3_large"
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_imagenet()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

""" CIFAR10 + SimpleNet """
def simplenet_cifar10():
    output_ml_setup = MlSetup()
    output_ml_setup.model = simple_net.__dict__["simplenet_cifar_5m"](num_classes=10)
    output_ml_setup.model_name = "simplenet"
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_cifar10_32()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

""" MNIST + vgg11 """
def vgg11_mnist():
    output_ml_setup = MlSetup()
    vgg11 = vgg.VGG11_no_bn(in_channels=1, num_classes=10)
    output_ml_setup.model_name = "vgg11_mnist_no_bn"
    output_ml_setup.model = vgg11
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_mnist_224()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

""" CIFAR10 + vgg11 """
def vgg11_cifar10():
    output_ml_setup = MlSetup()
    vgg11 = vgg.VGG11_no_bn(in_channels=3, num_classes=10)
    output_ml_setup.model_name = "vgg11_cifar10_no_bn"
    output_ml_setup.model = vgg11
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_cifar10_224()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

""" Helper function """
class ModelType(Enum):
    lenet5 = 0
    resnet18 = 1
    simplenet = 2
    cct7 = 3
    lenet5_large_fc = 4
    mobilenet_v3_small = 5
    mobilenet_v3_large = 6
    lenet4 = 7
    vgg11 = 8

class DatasetType(Enum):
    default = 0
    mnist = 1
    cifar10 = 2
    cifar100 = 3

class NormType(Enum):
    auto = 0
    bn = 1
    gn = 2

def get_ml_setup_from_config(model_type: str, norm_type: str = 'auto', dataset_type: str = 'default'):
    model_type = ModelType[model_type]
    norm_type = NormType[norm_type]
    dataset_type = DatasetType[dataset_type]
    if model_type == ModelType.lenet5:
        output_ml_setup = lenet5_mnist()
    elif model_type == ModelType.lenet4:
        output_ml_setup = lenet4_mnist()
    elif model_type == ModelType.resnet18:
        if dataset_type in [DatasetType.default, DatasetType.cifar10]:
            if norm_type == NormType.auto:
                output_ml_setup = resnet18_cifar10()
            elif norm_type == NormType.gn:
                output_ml_setup = resnet18_cifar10(enable_replace_bn_with_group_norm=True)
            else:
                raise NotImplementedError(f'{norm_type} is not implemented for {model_type} yet')
        if dataset_type in [DatasetType.cifar100]:
            if norm_type == NormType.auto:
                output_ml_setup = resnet18_cifar100()
            elif norm_type == NormType.gn:
                output_ml_setup = resnet18_cifar100(enable_replace_bn_with_group_norm=True)
            else:
                raise NotImplementedError(f'{norm_type} is not implemented for {model_type} yet')
    elif model_type == ModelType.simplenet:
        output_ml_setup = simplenet_cifar10()
    elif model_type == ModelType.cct7:
        output_ml_setup = cct7_cifar10()
    elif model_type == ModelType.lenet5_large_fc:
        output_ml_setup = lenet5_large_fc_mnist()
    elif model_type == ModelType.mobilenet_v3_small:
        output_ml_setup = mobilenet_v3_small_cifar10()
    elif model_type == ModelType.mobilenet_v3_large:
        output_ml_setup = mobilenet_v3_large_imagenet()
    elif model_type == ModelType.vgg11:
        if dataset_type in [DatasetType.default, DatasetType.mnist]:
            output_ml_setup = vgg11_mnist()
        if dataset_type in [DatasetType.default, DatasetType.cifar10]:
            output_ml_setup = vgg11_cifar10()
    else:
        raise ValueError(f'Invalid model type: {model_type}')
    return output_ml_setup


def get_ml_setup_from_model_type(model_name, dataset_type=DatasetType.default):
    if model_name == 'lenet5':
        assert dataset_type in [dataset_type.default, dataset_type.mnist]
        output_ml_setup = lenet5_mnist()
    elif model_name == 'lenet4':
        assert dataset_type in [dataset_type.default, dataset_type.mnist]
        output_ml_setup = lenet4_mnist()
    elif model_name == 'resnet18_bn':
        assert dataset_type in [dataset_type.default, dataset_type.cifar10, dataset_type.cifar100]
        if dataset_type in [dataset_type.default, dataset_type.cifar10]:
            output_ml_setup = resnet18_cifar10()
        if dataset_type in [dataset_type.cifar100]:
            output_ml_setup = resnet18_cifar100()
    elif model_name == 'resnet18_gn':
        assert dataset_type in [dataset_type.default, dataset_type.cifar10, dataset_type.cifar100]
        if dataset_type in [dataset_type.default, dataset_type.cifar10]:
            output_ml_setup = resnet18_cifar10(enable_replace_bn_with_group_norm=True)
        if dataset_type in [dataset_type.cifar100]:
            output_ml_setup = resnet18_cifar100(enable_replace_bn_with_group_norm=True)
    elif model_name == 'simplenet':
        assert dataset_type in [dataset_type.default, dataset_type.cifar10]
        output_ml_setup = simplenet_cifar10()
    elif model_name == 'cct7':
        assert dataset_type in [dataset_type.default, dataset_type.cifar10]
        output_ml_setup = cct7_cifar10()
    elif model_name == 'lenet5_large_fc':
        assert dataset_type in [dataset_type.default, dataset_type.mnist]
        output_ml_setup = lenet5_large_fc_mnist()
    elif model_name == 'mobilenet_v3_small':
        assert dataset_type in [dataset_type.default, dataset_type.cifar10]
        output_ml_setup = mobilenet_v3_small_cifar10()
    elif model_name == 'mobilenet_v3_large':
        assert dataset_type in [dataset_type.default, dataset_type.cifar10]
        output_ml_setup = mobilenet_v3_large_imagenet()
    elif model_name == 'vgg11_mnist_no_bn':
        assert dataset_type in [dataset_type.default, dataset_type.mnist]
        output_ml_setup = vgg11_mnist()
    elif model_name == 'vgg11_cifar10_no_bn':
        assert dataset_type in [dataset_type.default, dataset_type.cifar10]
        output_ml_setup = vgg11_cifar10()
    else:
        raise ValueError(f'Invalid model type: {model_name}')
    return output_ml_setup

