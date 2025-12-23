from typing import Literal

import torch.nn.functional as F
import torchvision

import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.base import MlSetup, CriterionType
from py_src.ml_setup_base.model import ModelType

from py_src.third_party.ddpm.ddpm.unet import UNet
from py_src.third_party.ddpm.ddpm.diffusion import GaussianDiffusion, generate_linear_schedule, generate_cosine_schedule

def get_ddpm_model_for_cifar10(use_labels=False, num_classes:int=10, img_channel=3,  base_channels=128, channel_mults=(1, 2, 2, 2), time_emb_dim=128 * 4,
                        norm='gn', dropout=0.1, activation="silu", attention_resolutions=(1,), schedule: Literal["cosine", "linear"]="linear",
                        num_timesteps=1000, schedule_low=1e-4, schedule_high=2e-2, ema_decay=0.9999, ema_update_rate=1, loss_type="l2"):
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }
    model = UNet(
        img_channels=img_channel,
        base_channels=base_channels,
        channel_mults=channel_mults,
        time_emb_dim=time_emb_dim,
        norm=norm,
        dropout=dropout,
        activation=activations[activation],
        attention_resolutions=attention_resolutions,
        num_classes=None if not use_labels else num_classes,
        initial_pad=0,
    )
    if schedule == "cosine":
        betas = generate_cosine_schedule(num_timesteps)
    else:
        betas = generate_linear_schedule(
            num_timesteps,
            schedule_low * 1000 / num_timesteps,
            schedule_high * 1000 / num_timesteps,
        )
    diffusion = GaussianDiffusion(
        model, (32, 32), 3, 10,
        betas,
        ema_decay=ema_decay,
        ema_update_rate=ema_update_rate,
        ema_start=2000,
        loss_type=loss_type,
    )
    return diffusion

class RescaleChannels(object):
    def __call__(self, sample):
        return 2 * sample - 1

def get_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


def ddpm_cifar10():
    output_ml_setup = MlSetup()

    dataset = ml_setup_dataset.dataset_cifar10(transforms_training=get_transform(), transforms_testing=get_transform())
    model = get_ddpm_model_for_cifar10()
    output_ml_setup.model = model
    output_ml_setup.model_name = str(ModelType.ddpm_cifar10.name)
    output_ml_setup.model_type = ModelType.ddpm_cifar10
    output_ml_setup.get_info_from_dataset(dataset)

    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True
    output_ml_setup.criterion = CriterionType.DiffusionModel

    def post_training(model):
        model.update_ema()

    output_ml_setup.func_handler_post_training.append(post_training)
    return output_ml_setup
