import torch
import torch.nn as nn
from py_src.ml_setup_base.model import ModelType
from py_src.ml_setup_base.dataset import DatasetType
from py_src.ml_setup_base.base import MlSetup

from py_src.ml_setup_base.mnist_models import lenet4_mnist, lenet5_mnist, lenet5_random_mnist, lenet5_large_fc_mnist
from py_src.ml_setup_base.regnet import regnet_y_400mf_imagenet1k, regnet_x_200mf_cifar10
from py_src.ml_setup_base.squeezenet import squeezenet1_1_imagenet1k
from py_src.ml_setup_base.vgg import vgg11_mnist, vgg11_cifar10, vgg11_bn_cifar10, vgg11_bn_imagenet1k
from py_src.ml_setup_base.resnet import resnet18_cifar10, resnet18_cifar100, resnet18_imagenet100, resnet18_imagenet1k, \
    resnet50_imagenet1k, resnet34_imagenet1k
from py_src.ml_setup_base.simplenet import simplenet_cifar10, simplenet_cifar100
from py_src.ml_setup_base.mobilenet import mobilenet_v2_cifar10, mobilenet_v3_large_imagenet1k
from py_src.ml_setup_base.cct import cct7_3x1_cifar10, cct7_3x1_cifar100, cct7_7x2_imagenet10, cct7_7x2_imagenet100, cct7_7x2_imagenet1k, cct14_7x2_imagenet1k
from py_src.ml_setup_base.shufflenet import shufflenet_v2_cifar10, shufflenet_v2_x2_0_imagenet1k
from py_src.ml_setup_base.efficientnet import efficientnet_v2_s_imagenet1k, efficientnet_b1_imagenet1k, efficientnet_b0_cifar10
from py_src.ml_setup_base.mnasnet import mnasnet1_0_imagenet1k, mnasnet0_5_imagenet1k
from py_src.ml_setup_base.densenet import densenet121_imagenet1k, densenet121_cifar10
from py_src.ml_setup_base.convnext import conveNeXt_tiny_imagenet1k
from py_src.ml_setup_base.alexnet import alexnet_imagenet1k
from py_src.ml_setup_base.resnext import resnext50_32x4d_imagenet1k
from py_src.ml_setup_base.vit import vit_b_32_imagenet1k
from py_src.ml_setup_base.wide_resnet50_2 import wide_resnet50_2_imagenet1k
from py_src.ml_setup_base.dla import dla_cifar10

__all__ = [ 'MlSetup',
            'lenet4_mnist', 'lenet5_mnist', 'lenet5_random_mnist', 'lenet5_large_fc_mnist',
            'regnet_y_400mf_imagenet1k', 'regnet_x_200mf_cifar10',
            'vgg11_mnist', 'vgg11_cifar10', 'vgg11_bn_cifar10', 'vgg11_bn_imagenet1k',
            'resnet18_cifar10', 'resnet18_cifar100', 'resnet18_imagenet100', 'resnet18_imagenet1k', 'resnet50_imagenet1k',
            'simplenet_cifar10', 'simplenet_cifar100',
            'mobilenet_v2_cifar10', 'mobilenet_v3_large_imagenet1k',
            'cct7_3x1_cifar10', 'cct7_3x1_cifar100', 'cct7_7x2_imagenet10', 'cct7_7x2_imagenet100', 'cct7_7x2_imagenet1k', 'cct14_7x2_imagenet1k',
            'shufflenet_v2_cifar10',
            'efficientnet_v2_s_imagenet1k', 'efficientnet_b1_imagenet1k', 'efficientnet_b0_cifar10',
            'mnasnet1_0_imagenet1k', 'mnasnet0_5_imagenet1k',
            'densenet121_imagenet1k', 'densenet121_cifar10',
            'conveNeXt_tiny_imagenet1k',
            'alexnet_imagenet1k',
            'vit_b_32_imagenet1k',
            'wide_resnet50_2_imagenet1k',
            'dla_cifar10',
           ]



""" Helper function """
def get_ml_setup_from_config(model_type: str, dataset_type: str = 'default', pytorch_preset_version=None):
    model_type = ModelType[model_type]
    dataset_type_enum = DatasetType[dataset_type]
    output_ml_setup = get_ml_setup_from_model_type(model_type, dataset_type=dataset_type_enum, pytorch_preset_version=pytorch_preset_version)
    return output_ml_setup

def get_ml_setup_from_model_type(model_name, dataset_type=DatasetType.default, pytorch_preset_version=None):
    if model_name == ModelType.lenet5:
        if dataset_type in [dataset_type.default, dataset_type.mnist]:
            output_ml_setup = lenet5_mnist()
        elif dataset_type in [dataset_type.random_mnist]:
            output_ml_setup = lenet5_random_mnist()
        else:
            raise NotImplementedError
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
            raise NotImplementedError
    elif model_name == ModelType.resnet34:
        if dataset_type in [dataset_type.default, dataset_type.imagenet1k]:
            output_ml_setup = resnet34_imagenet1k()
        else:
            raise NotImplementedError
    elif model_name == ModelType.resnet50:
        if dataset_type in [dataset_type.default, dataset_type.imagenet1k]:
            assert pytorch_preset_version is not None
            output_ml_setup = resnet50_imagenet1k(pytorch_preset_version)
        else:
            raise NotImplementedError
    elif model_name == ModelType.simplenet:
        assert dataset_type in [dataset_type.default, dataset_type.cifar10]
        output_ml_setup = simplenet_cifar10()
    elif model_name == ModelType.cct_7_3x1_32:
        if dataset_type in [dataset_type.default, dataset_type.cifar10]:
            output_ml_setup = cct7_3x1_cifar10()
        elif dataset_type in [dataset_type.cifar100]:
            output_ml_setup = cct7_3x1_cifar100()
        else:
            raise NotImplementedError
    elif model_name == ModelType.cct_7_7x2_224:
        if dataset_type in [dataset_type.default, dataset_type.imagenet1k]:
            output_ml_setup = cct7_7x2_imagenet1k()
        elif dataset_type in [dataset_type.imagenet100]:
            output_ml_setup = cct7_7x2_imagenet100()
        else:
            raise NotImplementedError
    elif model_name == ModelType.cct_14_7x2_224:
        if dataset_type in [dataset_type.default, dataset_type.imagenet1k]:
            output_ml_setup = cct14_7x2_imagenet1k()
        else:
            raise NotImplementedError
    elif model_name == ModelType.lenet5_large_fc:
        assert dataset_type in [dataset_type.default, dataset_type.mnist]
        output_ml_setup = lenet5_large_fc_mnist()
    elif model_name == ModelType.mobilenet_v3_small:
        raise NotImplementedError
    elif model_name == ModelType.mobilenet_v3_large:
        if dataset_type in [DatasetType.default, DatasetType.imagenet1k]:
            output_ml_setup = mobilenet_v3_large_imagenet1k()
        else:
            raise NotImplementedError
    elif model_name == ModelType.mobilenet_v2:
        if dataset_type in [DatasetType.default, DatasetType.cifar10]:
            output_ml_setup = mobilenet_v2_cifar10()
        else:
            raise NotImplementedError
    elif model_name == ModelType.vgg11_no_bn:
        if dataset_type in [DatasetType.default, DatasetType.mnist]:
            output_ml_setup = vgg11_mnist()
        elif dataset_type in [DatasetType.cifar10]:
            output_ml_setup = vgg11_cifar10()
        else:
            raise NotImplementedError
    elif model_name == ModelType.vgg11_bn:
        if dataset_type in [DatasetType.default, DatasetType.imagenet1k]:
            output_ml_setup = vgg11_bn_imagenet1k()
        elif dataset_type in [DatasetType.cifar10]:
            output_ml_setup = vgg11_bn_cifar10()
        else:
            raise NotImplementedError
    elif model_name == ModelType.vit_b_32:
        if dataset_type in [DatasetType.default, DatasetType.imagenet1k]:
            output_ml_setup = vit_b_32_imagenet1k()
        else:
            raise NotImplementedError
    elif model_name == ModelType.efficientnet_v2_s:
        if dataset_type in [DatasetType.default, DatasetType.imagenet1k]:
            output_ml_setup = efficientnet_v2_s_imagenet1k()
        else:
            raise NotImplementedError
    elif model_name == ModelType.efficientnet_b1:
        if dataset_type in [DatasetType.default, DatasetType.imagenet1k]:
            assert pytorch_preset_version is not None
            output_ml_setup = efficientnet_b1_imagenet1k(pytorch_preset_version)
        else:
            raise NotImplementedError
    elif model_name == ModelType.efficientnet_b0:
        if dataset_type in [DatasetType.default, DatasetType.cifar10]:
            output_ml_setup = efficientnet_b0_cifar10()
        else:
            raise NotImplementedError
    elif model_name == ModelType.shufflenet_v2:
        if dataset_type in [DatasetType.default, DatasetType.cifar10]:
            output_ml_setup = shufflenet_v2_cifar10()
        else:
            raise NotImplementedError
    elif model_name == ModelType.shufflenet_v2_x2_0:
        if dataset_type in [DatasetType.default, DatasetType.imagenet1k]:
            output_ml_setup = shufflenet_v2_x2_0_imagenet1k()
        else:
            raise NotImplementedError
    elif model_name == ModelType.squeezenet1_1:
        if dataset_type in [DatasetType.default, DatasetType.imagenet1k]:
            output_ml_setup = squeezenet1_1_imagenet1k()
        else:
            raise NotImplementedError
    elif model_name == ModelType.mnasnet1_0:
        if dataset_type in [DatasetType.default, DatasetType.imagenet1k]:
            output_ml_setup = mnasnet1_0_imagenet1k()
        else:
            raise NotImplementedError
    elif model_name == ModelType.mnasnet0_5:
        if dataset_type in [DatasetType.default, DatasetType.imagenet1k]:
            output_ml_setup = mnasnet0_5_imagenet1k()
        else:
            raise NotImplementedError
    elif model_name == ModelType.densenet121:
        if dataset_type in [DatasetType.default, DatasetType.imagenet1k]:
            output_ml_setup = densenet121_imagenet1k()
        elif dataset_type in [DatasetType.cifar10]:
            output_ml_setup = densenet121_cifar10()
        else:
            raise NotImplementedError
    elif model_name == ModelType.regnet_x_200mf:
        if dataset_type in [dataset_type.default, dataset_type.cifar10]:
            output_ml_setup = regnet_x_200mf_cifar10()
        else:
            raise NotImplementedError
    elif model_name == ModelType.regnet_y_400mf:
        if dataset_type in [dataset_type.default, dataset_type.imagenet1k]:
            assert pytorch_preset_version is not None
            output_ml_setup = regnet_y_400mf_imagenet1k(pytorch_preset_version)
        else:
            raise NotImplementedError
    elif model_name == ModelType.convnext_tiny:
        if dataset_type in [dataset_type.default, dataset_type.imagenet1k]:
            output_ml_setup = conveNeXt_tiny_imagenet1k()
        else:
            raise NotImplementedError
    elif model_name == ModelType.alexnet:
        if dataset_type in [dataset_type.default, dataset_type.imagenet1k]:
            output_ml_setup = alexnet_imagenet1k()
        else:
            raise NotImplementedError
    elif model_name == ModelType.resnext50_32x4d:
        if dataset_type in [dataset_type.default, dataset_type.imagenet1k]:
            assert pytorch_preset_version is not None
            output_ml_setup = resnext50_32x4d_imagenet1k(pytorch_preset_version)
        else:
            raise NotImplementedError
    elif model_name == ModelType.wide_resnet50_2:
        if dataset_type in [dataset_type.default, dataset_type.imagenet1k]:
            assert pytorch_preset_version is not None
            output_ml_setup = wide_resnet50_2_imagenet1k(pytorch_preset_version)
        else:
            raise NotImplementedError
    elif model_name == ModelType.dla:
        if dataset_type in [dataset_type.default, dataset_type.cifar10]:
            output_ml_setup = dla_cifar10()
        else:
            raise NotImplementedError
    else:
        raise ValueError(f'Invalid model type: {model_name}')
    return output_ml_setup

