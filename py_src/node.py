import torch

class Node:
    def __init__(self, model: torch.):
        self.model = torchvision.models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1,
                                                width_per_group=64, replace_stride_with_dilation=None)