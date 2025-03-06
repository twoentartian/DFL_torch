import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import random
import numpy as np
from enum import Enum, auto
from torchvision import transforms, models, datasets
from py_src.models import simplenet, lenet, vgg, mobilenet
from py_src.dataset import DatasetWithCachedOutputInSharedMem, DatasetWithCachedOutputInMem, ImageDatasetWithCachedInputInSharedMem
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


class DatasetSetup:
    def __init__(self, name, training_data, testing_data):
        self.training_data = training_data
        self.testing_data = testing_data
        self.dataset_name = name

        self.labels = self._get_dataset_labels(self.testing_data)
        sample_data = self.testing_data[0][0]
        self.tensor_size = sample_data.shape

    def _get_dataset_labels(self, dataset):
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        labels_set = set()
        for _, labels in dataloader:
            labels_set.update(labels.tolist())
        return labels_set


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

    def get_info_from_dataset(self, dataset: DatasetSetup):
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


""" MNIST """
def dataset_mnist():
    dataset_path = '~/dataset/mnist'
    dataset_name = "mnist"
    mnist_train = datasets.MNIST(root=dataset_path, train=True, download=True)
    mean = mnist_train.data.float().mean() / 255
    std = mnist_train.data.float().std() / 255
    transforms_mnist_train = transforms.Compose([transforms.RandomRotation(5, fill=(0,)), transforms.RandomCrop(28, padding=2), transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])])
    transforms_mnist_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])])
    train_data = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transforms_mnist_train)
    test_data = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transforms_mnist_test)
    return DatasetSetup(dataset_name, train_data, test_data)

def dataset_mnist_224():
    dataset_path = '~/dataset/mnist'
    dataset_name = "mnist_224"
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
    return DatasetSetup(dataset_name, train_data, test_data)

""" CIFAR10 """
def dataset_cifar10_32(transforms_training=None, transforms_testing=None, mean_std = None):
    dataset_path = '~/dataset/cifar10'
    dataset_name = "cifar10_32"
    if mean_std is not None:
        stats = mean_std
    else:
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
    return DatasetSetup(dataset_name, cifar10_train, cifar10_test)

def dataset_cifar10_224(transforms_training=None, transforms_testing=None, mean_std = None):
    dataset_path = '~/dataset/cifar10'
    dataset_name = "cifar10_224"
    if mean_std is not None:
        stats = mean_std
    else:
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
    return DatasetSetup(dataset_name, cifar10_train, cifar10_test)

def dataset_cifar100_32(transforms_training=None, transforms_testing=None, mean_std = None):
    dataset_path = '~/dataset/cifar100'
    dataset_name = "cifar100_32"
    if mean_std is not None:
        stats = mean_std
    else:
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
    return DatasetSetup(dataset_name, cifar100_train, cifar100_test)

def dataset_cifar100_224(transforms_training=None, transforms_testing=None, mean_std = None):
    dataset_path = '~/dataset/cifar100'
    dataset_name = "cifar100_224"
    if mean_std is not None:
        stats = mean_std
    else:
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

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
    cifar100_train = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms_cifar_train)
    cifar100_test = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transforms_cifar_test)
    return DatasetSetup(dataset_name, cifar100_train, cifar100_test)

""" ImageNet """
def dataset_imagenet1k(transforms_training=None, transforms_testing=None, enable_memory_cache=False):
    dataset_path = '~/dataset/imagenet1k'
    dataset_name = "imagenet1k_224"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if transforms_training is not None:
        final_transforms_train = transforms_training
    else:
        final_transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    if transforms_testing is not None:
        final_transforms_test = transforms_testing
    else:
        final_transforms_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

    if enable_memory_cache:
        imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "train"), "imagenet1k_train", transform=final_transforms_train)
        imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "val"), "imagenet1k_test", transform=final_transforms_test)
    else:
        imagenet_train = datasets.ImageNet(root=dataset_path, split='train', transform=final_transforms_train)
        imagenet_test = datasets.ImageNet(root=dataset_path, split='val', transform=final_transforms_test)
    return DatasetSetup(dataset_name, imagenet_train, imagenet_test)

def dataset_imagenet100(transforms_training=None, transforms_testing=None):
    dataset_path = '~/dataset/imagenet100'
    dataset_name = "imagenet100_224"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if transforms_training is not None:
        final_transforms_train = transforms_training
    else:
        final_transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    if transforms_testing is not None:
        final_transforms_test = transforms_testing
    else:
        final_transforms_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

    imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "train"), "imagenet100_train", transform = final_transforms_train)
    imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "val"), "imagenet100_test", transform = final_transforms_test)

    return DatasetSetup(dataset_name, imagenet_train, imagenet_test)

def dataset_imagenet10(transforms_training=None, transforms_testing=None):
    dataset_path = '~/dataset/imagenet10'
    dataset_name = "imagenet10_224"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if transforms_training is not None:
        final_transforms_train = transforms_training
    else:
        final_transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    if transforms_testing is not None:
        final_transforms_test = transforms_testing
    else:
        final_transforms_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

    imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "train"), "imagenet10_train", transform = final_transforms_train)
    imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "val"), "imagenet10_test", transform = final_transforms_test)

    return DatasetSetup(dataset_name, imagenet_train, imagenet_test)

""" MNIST + LeNet """
def lenet4_mnist():
    output_ml_setup = MlSetup()
    dataset = dataset_mnist()

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
    dataset = dataset_mnist()

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
    dataset = dataset_mnist()

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
    dataset = dataset_cifar10_32()

    output_ml_setup.model = cct.cct_7_3x1_32()
    output_ml_setup.model_name = "cct7"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct7_imagenet100():
    output_ml_setup = MlSetup()
    dataset = dataset_imagenet100()

    output_ml_setup.model = cct.cct_7_7x2_224()
    output_ml_setup.model_name = "cct7"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct7_imagenet10():
    output_ml_setup = MlSetup()
    dataset = dataset_imagenet10()

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
    dataset = dataset_cifar10_32()

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
    dataset = dataset_cifar100_32()

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
    dataset = dataset_imagenet100()

    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=GroupNorm)
        output_ml_setup.model_name = "resnet18_gn"
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
        output_ml_setup.model_name = "resnet18_bn"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def resnet18_imagenet1k(enable_replace_bn_with_group_norm=False):
    output_ml_setup = MlSetup()
    dataset = dataset_imagenet1k()

    if enable_replace_bn_with_group_norm:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=1000, norm_layer=GroupNorm)
        output_ml_setup.model_name = "resnet18_gn"
    else:
        output_ml_setup.model = models.resnet18(progress=False, num_classes=1000)
        output_ml_setup.model_name = "resnet18_bn"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

""" CIFAR10 + MobileNet V3 small """
def mobilenet_v3_small_cifar10():
    output_ml_setup = MlSetup()
    dataset = dataset_cifar10_32()

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
    dataset = dataset_cifar10_32()

    output_ml_setup.model = mobilenet.MobileNetV2(10)
    output_ml_setup.model_name = "mobilenet_v2_cifar10"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

""" ImageNet + MobileNet V3 large """
def mobilenet_v3_large_imagenet():
    output_ml_setup = MlSetup()
    dataset = dataset_imagenet1k()

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
    dataset = dataset_cifar10_32()

    output_ml_setup.model = simplenet.__dict__["simplenet_cifar_5m"](num_classes=10)
    output_ml_setup.model_name = "simplenet"
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def simplenet_cifar100():
    output_ml_setup = MlSetup()
    dataset = dataset_cifar10_32()

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
    dataset = dataset_mnist_224()

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
    dataset = dataset_cifar10_224()

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
    dataset = dataset_imagenet100()

    vit_b_16 = models.vit_b_16(weights=None)
    output_ml_setup.model_name = "vit_b_16"
    output_ml_setup.model = vit_b_16
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup

""" EfficientNet + CIFAR100 """
def efficient_net_cifar100():
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

    dataset = dataset_cifar100_224(transforms_training=transform_train, transforms_testing=transform_test)

    model_ft = models.efficientnet_v2_l(weights=None)
    in_features = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = nn.Linear(in_features, 100)

    output_ml_setup.model_name = "efficient_net_v2"
    output_ml_setup.model = model_ft
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
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
    efficient_net_v2 = auto()

class DatasetType(Enum):
    default = auto()
    mnist = auto()
    cifar10 = auto()
    cifar100 = auto()
    imagenet100 = auto()
    imagenet1k = auto()

def get_ml_setup_from_config(model_type: str, dataset_type: str = 'default'):
    model_type = ModelType[model_type]
    dataset_type_enum = DatasetType[dataset_type]
    output_ml_setup = get_ml_setup_from_model_type(model_type, dataset_type=dataset_type_enum)
    return output_ml_setup

def get_ml_setup_from_model_type(model_name, dataset_type=DatasetType.default):
    if model_name == ModelType.lenet5:
        assert dataset_type in [dataset_type.default, dataset_type.mnist]
        output_ml_setup = lenet5_mnist()
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
        output_ml_setup = mobilenet_v3_large_imagenet()
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
    elif model_name == ModelType.efficient_net_v2:
        if dataset_type in [DatasetType.default, DatasetType.cifar100]:
            output_ml_setup = efficient_net_cifar100()
        else:
            raise NotImplemented
    else:
        raise ValueError(f'Invalid model type: {model_name}')
    return output_ml_setup

