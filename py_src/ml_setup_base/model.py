from enum import Enum, auto


class ModelType(Enum):
    lenet5 = auto()
    resnet18_bn = auto()
    resnet18_gn = auto()
    resnet34 = auto()
    resnet50 = auto()
    simplenet = auto()
    cct7 = auto()
    vit_b_16 = auto()
    lenet5_large_fc = auto()
    mobilenet_v3_small = auto()
    mobilenet_v3_large = auto()
    mobilenet_v2 = auto()
    lenet4 = auto()
    vgg11_no_bn = auto()
    efficientnet_b1 = auto()
    efficientnet_v2_s = auto()
    shufflenet_v2 = auto()
    shufflenet_v2_x2_0 = auto()
    squeezenet1_1 = auto()