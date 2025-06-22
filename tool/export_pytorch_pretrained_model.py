import torch
import torchvision.models as models
import os
import sys

from sympy.codegen.cxxnodes import using

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util
from py_src.ml_setup_base.model import ModelType

if __name__ == "__main__":
    output_path = "pytorch_pretrained"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "resnet18_imagenet1k.model.pt"), model.state_dict(), model_name=str(ModelType.resnet18_bn.name))

    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "resnet34_imagenet1k.model.pt"), model.state_dict(), model_name=str(ModelType.resnet34.name))

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "resnet50_imagenet1k_v1.model.pt"), model.state_dict(), model_name=str(ModelType.resnet50.name))

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    util.save_model_state(os.path.join(output_path, "resnet50_imagenet1k_v2.model.pt"), model.state_dict(), model_name=str(ModelType.resnet50.name))

    model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "vgg11_bn_imagenet1k_v1.model.pt"), model.state_dict(), model_name=str(ModelType.vgg11_bn.name))

    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "squeezenet1_1.model.pt"), model.state_dict(), model_name=str(ModelType.squeezenet1_1.name))

    model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "shufflenet_v2_x2_0.model.pt"), model.state_dict(), model_name=str(ModelType.shufflenet_v2_x2_0.name))

    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "mobilenet_v3_large_imagenet_v1.model.pt"), model.state_dict(), model_name=str(ModelType.mobilenet_v3_large.name))

    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    util.save_model_state(os.path.join(output_path, "mobilenet_v3_large_imagenet_v2.model.pt"), model.state_dict(), model_name=str(ModelType.mobilenet_v3_large.name))

    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "efficientnet_v2_s.model.pt"), model.state_dict(), model_name=str(ModelType.efficientnet_v2_s.name))

    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "efficientnet_b1_imagenet_v1.model.pt"), model.state_dict(), model_name=str(ModelType.efficientnet_b1.name))

    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2)
    util.save_model_state(os.path.join(output_path, "efficientnet_b1_imagenet_v2.model.pt"), model.state_dict(), model_name=str(ModelType.efficientnet_b1.name))

    model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "mnasnet1_0_imagenet_v1.model.pt"), model.state_dict(), model_name=str(ModelType.mnasnet1_0.name))

    model = models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "mnasnet0_5_imagenet_v1.model.pt"), model.state_dict(), model_name=str(ModelType.mnasnet0_5.name))

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "densenet121_imagenet_v1.model.pt"), model.state_dict(), model_name=str(ModelType.densenet121.name))

    model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "regnet_y_400_mf_imagenet_v1.model.pt"), model.state_dict(), model_name=str(ModelType.regnet_y_400mf.name))

    model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2)
    util.save_model_state(os.path.join(output_path, "regnet_y_400_mf_imagenet_v2.model.pt"), model.state_dict(), model_name=str(ModelType.regnet_y_400mf.name))

    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "convnext_tiny_imagenet_v1.model.pt"), model.state_dict(), model_name=str(ModelType.convnext_tiny.name))

    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "alexnet_imagenet.model.pt"), model.state_dict(), model_name=str(ModelType.alexnet.name))

    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "resnext50_32x4d_imagenet_v1.model.pt"), model.state_dict(), model_name=str(ModelType.resnext50_32x4d.name))

    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    util.save_model_state(os.path.join(output_path, "resnext50_32x4d_imagenet_v2.model.pt"), model.state_dict(), model_name=str(ModelType.resnext50_32x4d.name))

    model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "vit_b_32_imagenet_v1.model.pt"), model.state_dict(), model_name=str(ModelType.vit_b_32.name))

    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    util.save_model_state(os.path.join(output_path, "wide_resnet50_2_imagenet_v1.model.pt"), model.state_dict(), model_name=str(ModelType.wide_resnet50_2.name))

    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    util.save_model_state(os.path.join(output_path, "wide_resnet50_2_imagenet_v2.model.pt"), model.state_dict(), model_name=str(ModelType.wide_resnet50_2.name))
