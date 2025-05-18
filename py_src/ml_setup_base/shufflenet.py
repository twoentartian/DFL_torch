import torch.nn as nn
from py_src.models import shufflenet
from py_src.ml_setup_base.base import MlSetup
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.model import ModelType

def shufflenet_v2_cifar10():
    output_ml_setup = MlSetup()

    dataset = ml_setup_dataset.dataset_cifar10()

    model_ft = shufflenet.ShuffleNet(10, g=1, scale_factor=1)
    output_ml_setup.model_name = str(ModelType.shufflenet_v2.name)
    output_ml_setup.model = model_ft
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup