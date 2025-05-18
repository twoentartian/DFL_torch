import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.v2 import MixUp, RandomChoice, CutMix
from py_src.ml_setup_base.sampler import RASampler

def get_mixup_cutmix(*, mixup_alpha, cutmix_alpha, num_classes):
    mixup_cutmix = []
    if mixup_alpha > 0:
        mixup_cutmix.append(MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0:
        mixup_cutmix.append(CutMix(alpha=cutmix_alpha, num_classes=num_classes))
    if not mixup_cutmix:
        return None
    return RandomChoice(mixup_cutmix)

"""get pytorch training setup, version can be 1 or 2"""
class collate_fn_inst():
    def __init__(self, target):
        self.target = target

    def __call__(self, batch):
        return self.target(*default_collate(batch))


def get_pytorch_training_imagenet(version=2):
    if version == 1:
        collate_fn = None
        loss_fn = nn.CrossEntropyLoss()
        model_ema_decay = None
        model_ema_steps = None
        sampler_fn = None
        return loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn
    elif version == 2:
        mixup_cutmix = get_mixup_cutmix(mixup_alpha=0.2, cutmix_alpha=1.0, num_classes=1000)
        collate_fn = collate_fn_inst(mixup_cutmix)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        model_ema_decay = 0.99998
        model_ema_steps = 32
        def sampler_fn(dataset):
            return RASampler(dataset, shuffle=True, repetitions=4)
        return loss_fn, collate_fn, model_ema_decay, model_ema_steps, sampler_fn
    else:
        raise NotImplementedError