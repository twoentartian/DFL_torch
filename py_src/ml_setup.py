import torch
from enum import Enum
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np
import torchvision.transforms.functional as visionF
from torchvision import transforms, models, datasets

def replace_bn_with_ln(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Replace BatchNorm2d with LayerNorm
            layer_norm = nn.LayerNorm(module.num_features, elementwise_affine=True)
            setattr(model, name, layer_norm)
        else:
            # Recursively replace in submodules
            replace_bn_with_ln(module)

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


class MlSetup:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.training_data = None
        self.training_data_for_rebuilding_normalization = None
        self.testing_data = None
        self.criterion = None
        self.training_batch_size = None
        self.learning_rate = None
        self.dataset_label = None
        self.weights_init_func = None
        self.get_lr_scheduler_func = None

        self.has_normalization_layer = None

    def self_validate(self):
        pass  # do nothing for now


""" CIFAR10 """
def dataset_cifar10(random_crop_flip=True):
    dataset_path = './data/cifar10'
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    if random_crop_flip:
        transforms_cifar_train = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(32, padding=4, padding_mode='reflect'), transforms.RandomHorizontalFlip(p=0.5), transforms.Normalize(*stats)])
    else:
        transforms_cifar_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    transforms_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms_cifar_train)
    cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms_cifar_test)
    cifar10_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return cifar10_train, cifar10_test, cifar10_labels


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


""" LeNet5 """
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = nnF.max_pool2d(nnF.relu(self.conv1(x)), (2, 2))
        x = nnF.max_pool2d(nnF.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = nnF.relu(self.fc1(x))
        x = nnF.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        return np.prod(size)

def lenet5():
    return LeNet5()


""" MNIST + LeNet5 """
def lenet5_mnist():
    lenet5_mnist = MlSetup()
    lenet5_mnist.model = lenet5()
    lenet5_mnist.model_name = "lenet5"
    lenet5_mnist.training_data, lenet5_mnist.testing_data, lenet5_mnist.dataset_label = dataset_mnist()
    lenet5_mnist.training_data_for_rebuilding_normalization = None
    lenet5_mnist.criterion = torch.nn.CrossEntropyLoss()
    lenet5_mnist.training_batch_size = 64
    lenet5_mnist.learning_rate = 0.001
    lenet5_mnist.weights_init_func = weights_init_xavier
    lenet5_mnist.has_normalization_layer = False
    return lenet5_mnist


""" CIFAR10 + ResNet18 """


class GroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(GroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels, eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x

def resnet18_cifar10(enable_replace_bn_with_group_norm=False):
    output_resnet18_cifar10 = MlSetup()
    if enable_replace_bn_with_group_norm:
        output_resnet18_cifar10.model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=GroupNorm)
        output_resnet18_cifar10.model_name = "resnet18_gn"
    else:
        output_resnet18_cifar10.model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
        output_resnet18_cifar10.model_name = "resnet18_bn"
    output_resnet18_cifar10.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # change for cifar10 dataset
    output_resnet18_cifar10.model.maxpool = nn.Identity()
    output_resnet18_cifar10.training_data, output_resnet18_cifar10.testing_data, output_resnet18_cifar10.dataset_label = dataset_cifar10(random_crop_flip=True)
    output_resnet18_cifar10.training_data_for_rebuilding_normalization, _, _ = dataset_cifar10(random_crop_flip=False)
    output_resnet18_cifar10.criterion = torch.nn.CrossEntropyLoss()
    output_resnet18_cifar10.training_batch_size = 256
    output_resnet18_cifar10.learning_rate = 0.001
    output_resnet18_cifar10.has_normalization_layer = True
    return output_resnet18_cifar10




""" Helper function """
class ModelType(Enum):
    lenet5 = 0
    resnet18 = 1

class NormType(Enum):
    auto = 0
    bn = 1
    gn = 2

def get_ml_setup_from_config(model_type: str, norm_type: str = 'auto'):
    model_type = ModelType[model_type]
    norm_type = NormType[norm_type]
    if model_type == ModelType.lenet5:
        output_ml_setup = lenet5_mnist()
    elif model_type == ModelType.resnet18:
        if norm_type == NormType.auto:
            output_ml_setup = resnet18_cifar10()
        elif norm_type == NormType.gn:
            output_ml_setup = resnet18_cifar10(enable_replace_bn_with_group_norm=True)
        else:
            raise NotImplementedError(f'{norm_type} is not implemented for {model_type} yet')
    else:
        raise ValueError(f'Invalid model type: {model_type}')
    return output_ml_setup


def get_ml_setup_from_model_type(model_name):
    if model_name == 'lenet5':
        output_ml_setup = lenet5_mnist()
    elif model_name == 'resnet18_bn':
        output_ml_setup = resnet18_cifar10()
    elif model_name == 'resnet18_gn':
        output_ml_setup = resnet18_cifar10(enable_replace_bn_with_group_norm=True)
    else:
        raise ValueError(f'Invalid model type: {model_name}')
    return output_ml_setup

