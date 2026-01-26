import yaml, os, sys
import torch.nn as nn
from timm.data import create_dataset, create_loader, Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

import py_src.third_party.compact_transformers.src.cct as cct
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.dataset import default_path_imagenet1k
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
    dataset = ml_setup_dataset.dataset_imagenet1k(pytorch_preset_version=1)

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
    dataset = ml_setup_dataset.dataset_imagenet100(pytorch_preset_version=1)

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
    dataset = ml_setup_dataset.dataset_imagenet10(pytorch_preset_version=1)

    output_ml_setup.model = cct.cct_7_7x2_224(num_classes=10)
    output_ml_setup.model_name = str(ModelType.cct_7_7x2_224.name)
    output_ml_setup.model_type = ModelType.cct_7_7x2_224
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.criterion = nn.CrossEntropyLoss()
    output_ml_setup.training_batch_size = 64
    output_ml_setup.has_normalization_layer = True
    return output_ml_setup



def timm_build_loaders(cfg: dict, data_root: str):
    img_size = int(cfg["img_size"])
    input_size = (3, img_size, img_size)

    # ImageNet expects folder structure like:
    #   <data_root>/train/<class>/*.JPEG
    #   <data_root>/val/<class>/*.JPEG   (or "validation" depending on your setup)
    train_ds = create_dataset(
        name=cfg["dataset"],
        root=data_root,
        split="train",
        is_training=True,
        batch_size=cfg["batch_size"],
    )
    val_ds = create_dataset(
        name=cfg["dataset"],
        root=data_root,
        split="val",  # sometimes "validation"
        is_training=False,
        batch_size=cfg["batch_size"],
    )

    common = dict(
        input_size=input_size,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        mean=tuple(cfg["mean"]),
        std=tuple(cfg["std"]),
        crop_pct=float(cfg["crop_pct"]),
        pin_memory=True,
        use_prefetcher=False,  # <- set True only if you want timm to move/normalize on-GPU
    )

    train_loader = create_loader(
        train_ds,
        is_training=True,
        # map your YAML keys to timm create_loader args
        scale=tuple(cfg["scale"]),
        interpolation=str(cfg["train_interpolation"]),  # "random"
        auto_augment=str(cfg["aa"]),                    # YAML "aa" -> timm "auto_augment"
        re_prob=float(cfg["reprob"]),
        re_mode=str(cfg["remode"]),
        **common,
    )

    val_loader = create_loader(
        val_ds,
        is_training=False,
        interpolation=str(cfg["interpolation"]),        # "bicubic"
        **common,
    )

    return train_loader, val_loader

def timm_build_mixup_and_loss(cfg: dict):
    use_mix = (float(cfg.get("mixup", 0)) > 0) or (float(cfg.get("cutmix", 0)) > 0)
    use_mix = False
    mixup_fn = None
    if use_mix:
        mixup_fn = Mixup(
            mixup_alpha=float(cfg["mixup"]),
            cutmix_alpha=float(cfg["cutmix"]),
            prob=float(cfg["mixup_prob"]),
            switch_prob=float(cfg["mixup_switch_prob"]),
            mode=str(cfg["mixup_mode"]),  # "batch"
            label_smoothing=float(cfg["smoothing"]),
            num_classes=int(cfg["num_classes"]),
        )
        criterion = SoftTargetCrossEntropy()
    else:
        s = float(cfg["smoothing"])
        criterion = LabelSmoothingCrossEntropy(smoothing=s) if s > 0 else nn.CrossEntropyLoss()

    return mixup_fn, criterion


def cct14_7x2_imagenet1k():

    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_imagenet1k_custom(auto_augment_policy='imagenet', val_crop_size=224, val_resize_size=256, train_crop_size=224)

    cfg = yaml.safe_load(open(f"{os.path.dirname(os.path.abspath(__file__))}/cct_imagenet_config.yaml", "r"))

    train_loader, val_loader = timm_build_loaders(cfg, default_path_imagenet1k)
    mixup_fn, criterion = timm_build_mixup_and_loss(cfg)

    output_ml_setup.model = cct.cct_14_7x2_224()
    output_ml_setup.model_name = str(ModelType.cct_14_7x2_224.name)
    output_ml_setup.model_type = ModelType.cct_14_7x2_224
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True

    output_ml_setup.criterion = criterion
    output_ml_setup.collate_fn = None
    output_ml_setup.model_ema = None
    output_ml_setup.mixup_fn = mixup_fn
    output_ml_setup.sampler_fn = None

    output_ml_setup.override_training_dataset_loader = train_loader
    output_ml_setup.override_testing_dataset_loader = val_loader

    return output_ml_setup