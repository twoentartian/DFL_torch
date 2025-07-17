from torchvision import models
from py_src.ml_setup_base.base import MlSetup
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.model import ModelType
from py_src.ml_setup_base.other_setup import get_pytorch_training_imagenet


def vit_b_32_imagenet1k():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k_custom(auto_augment_policy='imagenet',
                                                             val_crop_size=224, val_resize_size=256, train_crop_size=224)
    output_ml_setup.model = models.vit_b_32(progress=False, num_classes=1000)
    output_ml_setup.model_name = str(ModelType.vit_b_32.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn = get_pytorch_training_imagenet(2, label_smoothing=0.11, mixup_alpha=0.2, cutmix_alpha=1.0)
    output_ml_setup.criterion = loss_fn
    output_ml_setup.collate_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)
    output_ml_setup.sampler_fn = sampler_fn
    output_ml_setup.clip_grad_norm = 1
    return output_ml_setup