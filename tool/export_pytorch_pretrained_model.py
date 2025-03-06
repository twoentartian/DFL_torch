import torch
import torchvision.models as models
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util

if __name__ == "__main__":
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    output_path = "pytorch_pretrained"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    util.save_model_state(os.path.join(output_path, "resnet18_imagenet1k.model.pt"), resnet18.state_dict(), model_name="resnet18_bn")