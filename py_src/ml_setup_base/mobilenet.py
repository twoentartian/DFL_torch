import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
from py_src.models import mobilenet
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType
from py_src.ml_setup_base.other_setup import get_pytorch_training_imagenet

def mobilenet_v2_cifar10():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_cifar10()

    output_ml_setup.model = mobilenet.MobileNetV2(10)
    output_ml_setup.model_name = str(ModelType.mobilenet_v2.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup

def mobilenet_v3_large_imagenet1k(pytorch_preset_version=2):
    output_ml_setup = MlSetup()
    if pytorch_preset_version == 1:
        dataset = ml_setup_dataset.dataset_imagenet1k_custom(auto_augment_policy='imagenet', random_erase_prob=0.2)
        loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn = get_pytorch_training_imagenet(1)
    elif pytorch_preset_version == 2:
        dataset = ml_setup_dataset.dataset_imagenet1k_custom(auto_augment_policy='ta_wide', random_erase_prob=0.1, val_resize_size=232)
        loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn = get_pytorch_training_imagenet(2, mixup_alpha=0.2, cutmix_alpha=1.0, label_smoothing=0.1)
    else:
        raise NotImplementedError
    output_ml_setup.model = models.mobilenet_v3_large(progress=False)
    output_ml_setup.model_name = str(ModelType.mobilenet_v3_large.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    output_ml_setup.criterion = loss_fn
    output_ml_setup.collate_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)
    output_ml_setup.sampler_fn = sampler_fn
    return output_ml_setup