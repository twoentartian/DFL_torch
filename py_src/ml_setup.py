import torch
from enum import Enum
import torch.nn as nn
from torchvision import transforms, models, datasets
from py_src.models import simple_net, lenet
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

""" CIFAR10 """
def dataset_cifar10(transforms_training=None, transforms_testing=None):
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

""" ImageNet """
def dataset_imagenet(transforms_training=None, transforms_testing=None):
    dataset_path = './data/imagenet'
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))

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
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_cifar10()
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
    output_ml_setup.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # change for cifar10 dataset
    output_ml_setup.model.maxpool = nn.Identity()
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_cifar10()
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
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_cifar10(transforms_training=train_transforms, transforms_testing=test_transforms)
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
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_cifar10()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

""" CIFAR10 + vgg11 """
def vgg11_cifar10(enable_bn=False):
    output_ml_setup = MlSetup()
    if enable_bn:
        vgg11 = models.vgg11_bn(weights=None)
        output_ml_setup.model_name = "vgg11_bn"
    else:
        vgg11 = models.vgg11(weights=None)
        output_ml_setup.model_name = "vgg11_no_bn"
    vgg11.classifier[6] = nn.Linear(4096, 10)
    output_ml_setup.model = vgg11
    output_ml_setup.training_data, output_ml_setup.testing_data, output_ml_setup.dataset_label = dataset_cifar10()
    output_ml_setup.criterion = torch.nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
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
    vgg11_no_bn = 8
    vgg11_bn = 9

class NormType(Enum):
    auto = 0
    bn = 1
    gn = 2

def get_ml_setup_from_config(model_type: str, norm_type: str = 'auto'):
    model_type = ModelType[model_type]
    norm_type = NormType[norm_type]
    if model_type == ModelType.lenet5:
        output_ml_setup = lenet5_mnist()
    elif model_type == ModelType.lenet4:
        output_ml_setup = lenet4_mnist()
    elif model_type == ModelType.resnet18:
        if norm_type == NormType.auto:
            output_ml_setup = resnet18_cifar10()
        elif norm_type == NormType.gn:
            output_ml_setup = resnet18_cifar10(enable_replace_bn_with_group_norm=True)
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
    elif model_type == ModelType.vgg11_no_bn:
        output_ml_setup = vgg11_cifar10(enable_bn=False)
    elif model_type == ModelType.vgg11_bn:
        output_ml_setup = vgg11_cifar10(enable_bn=True)
    else:
        raise ValueError(f'Invalid model type: {model_type}')
    return output_ml_setup


def get_ml_setup_from_model_type(model_name):
    if model_name == 'lenet5':
        output_ml_setup = lenet5_mnist()
    elif model_name == 'lenet4':
        output_ml_setup = lenet4_mnist()
    elif model_name == 'resnet18_bn':
        output_ml_setup = resnet18_cifar10()
    elif model_name == 'resnet18_gn':
        output_ml_setup = resnet18_cifar10(enable_replace_bn_with_group_norm=True)
    elif model_name == 'simplenet':
        output_ml_setup = simplenet_cifar10()
    elif model_name == 'cct7':
        output_ml_setup = cct7_cifar10()
    elif model_name == 'lenet5_large_fc':
        output_ml_setup = lenet5_large_fc_mnist()
    elif model_name == 'mobilenet_v3_small':
        output_ml_setup = mobilenet_v3_small_cifar10()
    elif model_name == 'mobilenet_v3_large':
        output_ml_setup = mobilenet_v3_large_imagenet()
    elif model_name == 'vgg11_no_bn':
        output_ml_setup = vgg11_cifar10(enable_bn=False)
    elif model_name == 'vgg11_bn':
        output_ml_setup = vgg11_cifar10(enable_bn=True)
    else:
        raise ValueError(f'Invalid model type: {model_name}')
    return output_ml_setup

