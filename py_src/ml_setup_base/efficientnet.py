from torchvision import transforms
import torch.nn as nn
from py_src.ml_setup_base.base import MlSetup
import py_src.ml_setup_base.dataset as ml_setup_dataset

from torchvision import models
from py_src.ml_setup_base.model import ModelType

def efficientnet_v2_l_imagenet1k():
    output_ml_setup = MlSetup()
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img_size = 224
    crop_size = 224
    transform_train = transforms.Compose([
            transforms.Resize(img_size),  # , interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
        ])

    transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    dataset = ml_setup_dataset.dataset_cifar100(rescale_to_224=True, transforms_training=transform_train, transforms_testing=transform_test)

    model_ft = models.efficientnet_v2_l(weights=None)
    in_features = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = nn.Linear(in_features, 100)

    output_ml_setup.model_name = str(ModelType.efficientnet_v2)
    output_ml_setup.model = model_ft
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 32
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup