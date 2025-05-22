import torch
import torchvision.models as models
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util
from py_src.ml_setup_base.model import ModelType

if __name__ == "__main__":
    output_path = "pytorch_pretrained"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "resnet18_imagenet1k.model.pt"), resnet18.state_dict(), model_name=str(ModelType.resnet18_bn.name))

    resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "resnet34_imagenet1k.model.pt"), resnet34.state_dict(), model_name=str(ModelType.resnet34.name))

    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    util.save_model_state(os.path.join(output_path, "resnet50_imagenet1k.model.pt"), resnet50.state_dict(), model_name=str(ModelType.resnet50.name))

    squeezenet1_1 = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "squeezenet1_1.model.pt"), squeezenet1_1.state_dict(), model_name=str(ModelType.squeezenet1_1.name))

    shufflenet_v2_x2_0 = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "shufflenet_v2_x2_0.model.pt"), shufflenet_v2_x2_0.state_dict(), model_name=str(ModelType.shufflenet_v2_x2_0.name))

