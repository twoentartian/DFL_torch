import torch.nn as nn
import torchvision.models as models
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType
from py_src.ml_setup_base.other_setup import get_pytorch_training_imagenet

def mnasnet0_5_imagenet1k():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k(1, train_crop_size=224, val_resize_size=256, val_crop_size=224)
    loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn = get_pytorch_training_imagenet(1)
    output_ml_setup.model = models.mnasnet0_5(progress=False)
    output_ml_setup.model_name = str(ModelType.mnasnet0_5.name)
    output_ml_setup.model_type = ModelType.mnasnet0_5
    output_ml_setup.get_info_from_dataset(dataset)

    output_ml_setup.training_batch_size = 512
    output_ml_setup.has_normalization_layer = True
    output_ml_setup.criterion = loss_fn
    output_ml_setup.collate_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)
    output_ml_setup.sampler_fn = sampler_fn
    return output_ml_setup

def mnasnet1_0_imagenet1k():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k(1, train_crop_size=224, val_resize_size=256, val_crop_size=224)
    loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn = get_pytorch_training_imagenet(1)
    output_ml_setup.model = models.mnasnet1_0(progress=False)
    output_ml_setup.model_name = str(ModelType.mnasnet1_0.name)
    output_ml_setup.model_type = ModelType.mnasnet1_0
    output_ml_setup.get_info_from_dataset(dataset)

    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    output_ml_setup.criterion = loss_fn
    output_ml_setup.collate_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)
    output_ml_setup.sampler_fn = sampler_fn
    return output_ml_setup