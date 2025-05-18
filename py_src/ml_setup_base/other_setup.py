import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.v2 import MixUp, RandomChoice, CutMix


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
def get_pytorch_training_imagenet(version=2):
    if version == 1:
        collect_fn = None
        loss_fn = nn.CrossEntropyLoss()
        model_ema_decay = None
        model_ema_steps = None
        return loss_fn, collect_fn, model_ema_decay, model_ema_steps
    elif version == 2:
        mixup_cutmix = get_mixup_cutmix(mixup_alpha=0.2, cutmix_alpha=1.0, num_classes=1000)
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        model_ema_decay = 0.99998
        model_ema_steps = 32
        return loss_fn, collate_fn, model_ema_decay, model_ema_steps
    else:
        raise NotImplementedError