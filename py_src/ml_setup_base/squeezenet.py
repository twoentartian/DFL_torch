import torch.nn as nn
from py_src.ml_setup_base.base import MlSetup
import py_src.ml_setup_base.dataset as ml_setup_dataset

from torchvision import models
from py_src.ml_setup_base.model import ModelType

def squeezenet1_1_imagenet1k():
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k(1)

    model_ft = models.squeezenet1_1(weights=None)

    output_ml_setup.model_name = str(ModelType.squeezenet1_1.name)
    output_ml_setup.model_type = ModelType.squeezenet1_1
    output_ml_setup.model = model_ft
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 256
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup