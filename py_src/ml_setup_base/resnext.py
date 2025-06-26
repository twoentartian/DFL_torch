from torchvision import models
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.other_setup import get_pytorch_training_imagenet


def resnext50_32x4d_imagenet1k(pytorch_preset_version=2):
    output_ml_setup = MlSetup()
    if pytorch_preset_version == 1:
        dataset = ml_setup_dataset.dataset_imagenet1k_custom(val_crop_size=224, val_resize_size=256, train_crop_size=224)
    elif pytorch_preset_version == 2:
        dataset = ml_setup_dataset.dataset_imagenet1k_custom(auto_augment_policy='ta_wide', random_erase_prob=0.1,
                                                             val_crop_size=224, val_resize_size=232, train_crop_size=224)
    else:
        raise NotImplementedError
    output_ml_setup.model = models.resnext50_32x4d(progress=False, num_classes=1000)
    output_ml_setup.model_name = str(ModelType.resnext50_32x4d.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    if pytorch_preset_version == 1:
        loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn = get_pytorch_training_imagenet(1)
    elif pytorch_preset_version == 2:
        loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn = get_pytorch_training_imagenet(2, enable_collate_and_sampler=True)
    else:
        raise NotImplementedError

    output_ml_setup.criterion = loss_fn
    output_ml_setup.collate_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)
    output_ml_setup.sampler_fn = sampler_fn
    return output_ml_setup
