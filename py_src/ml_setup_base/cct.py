import torch.nn as nn
import py_src.third_party.compact_transformers.src.cct as cct
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType
from py_src.ml_setup_base.other_setup import get_pytorch_training_imagenet


def cct7_3x1_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10()

    output_ml_setup.model = cct.cct_7_3x1_32()
    output_ml_setup.model_name = str(ModelType.cct_7_3x1_32.name)
    output_ml_setup.model_type = ModelType.cct_7_3x1_32
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct7_3x1_cifar100():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar100()

    output_ml_setup.model = cct.cct_7_3x1_32(num_classes=100)
    output_ml_setup.model_name = str(ModelType.cct_7_3x1_32.name)
    output_ml_setup.model_type = ModelType.cct_7_3x1_32
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct7_7x2_imagenet1k():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k(1)

    output_ml_setup.model = cct.cct_7_7x2_224()
    output_ml_setup.model_name = str(ModelType.cct_7_7x2_224.name)
    output_ml_setup.model_type = ModelType.cct_7_7x2_224
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct7_7x2_imagenet100():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet100(1)

    output_ml_setup.model = cct.cct_7_7x2_224()
    output_ml_setup.model_name = str(ModelType.cct_7_7x2_224.name)
    output_ml_setup.model_type = ModelType.cct_7_7x2_224
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct7_7x2_imagenet10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet10(1)

    output_ml_setup.model = cct.cct_7_7x2_224()
    output_ml_setup.model_name = str(ModelType.cct_7_7x2_224.name)
    output_ml_setup.model_type = ModelType.cct_7_7x2_224
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def cct14_7x2_imagenet1k():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k_custom(auto_augment_policy='imagenet',
                                                             val_crop_size=224, val_resize_size=256, train_crop_size=224)
    loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn = get_pytorch_training_imagenet(2, label_smoothing=0.1, mixup_alpha=0.8, cutmix_alpha=1.0)
    output_ml_setup.model = cct.cct_14_7x2_224()
    output_ml_setup.model_name = str(ModelType.cct_14_7x2_224.name)
    output_ml_setup.model_type = ModelType.cct_14_7x2_224
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True

    output_ml_setup.criterion = loss_fn
    output_ml_setup.collate_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)
    output_ml_setup.sampler_fn = sampler_fn

    return output_ml_setup