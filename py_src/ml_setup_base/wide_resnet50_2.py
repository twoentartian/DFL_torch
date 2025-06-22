from torchvision import models
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.other_setup import get_pytorch_training_imagenet

def wide_resnet50_2_imagenet1k(pytorch_preset_version=2):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k(pytorch_preset_version)

    output_ml_setup.model = models.wide_resnet50_2(progress=False, num_classes=1000)
    output_ml_setup.model_name = str(ModelType.wide_resnet50_2.name)
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn = get_pytorch_training_imagenet(pytorch_preset_version)
    output_ml_setup.criterion = loss_fn
    output_ml_setup.collate_fn = collate_fn
    output_ml_setup.model_ema = (model_ema_decay, model_ema_steps)
    output_ml_setup.sampler_fn = sampler_fn
    return output_ml_setup