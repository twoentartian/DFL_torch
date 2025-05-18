import torch.nn as nn
from torchvision.models import vgg11_bn
from py_src.models import vgg
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType

def vgg11_mnist():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_mnist(rescale_to_224=True)

    vgg11 = vgg.VGG11_no_bn(in_channels=1, num_classes=10)
    output_ml_setup.model_name = str(ModelType.vgg11_no_bn)
    output_ml_setup.model = vgg11
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup


def vgg11_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10(rescale_to_224=True)

    vgg11 = vgg.VGG11_no_bn(in_channels=3, num_classes=10)
    output_ml_setup.model_name = str(ModelType.vgg11_no_bn)
    output_ml_setup.model = vgg11
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup


def vgg11_bn_imagenet1k():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k(1)

    vgg11 = vgg11_bn(progress=False, weights=None, num_classes=1000)
    output_ml_setup.model_name = str(ModelType.vgg11_no_bn)
    output_ml_setup.model = vgg11
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = False
    return output_ml_setup