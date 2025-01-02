import torch
import torch.nn as nn
import math

""" a vgg11 model without bn layers """
__layer_variance_mean = {}
# (variance, mean)
__layer_variance_mean['conv_layers.0.weight'] = (0.014527, 0.000820)
__layer_variance_mean['conv_layers.3.weight'] = (0.000235, 0.000254)
__layer_variance_mean['conv_layers.6.weight'] = (0.000117, -0.000014)
__layer_variance_mean['conv_layers.8.weight'] = (0.000059, 0.000017)
__layer_variance_mean['conv_layers.11.weight'] = (0.000060, -0.000055)
__layer_variance_mean['conv_layers.13.weight'] = (0.000031, -0.000224)
__layer_variance_mean['conv_layers.16.weight'] = (0.000033, -0.000260)
__layer_variance_mean['conv_layers.18.weight'] = (0.000038, -0.000767)
__layer_variance_mean['linear_layers.0.weight'] = (0.000006, -0.000063)
__layer_variance_mean['linear_layers.3.weight'] = (0.000035, -0.000042)
__layer_variance_mean['linear_layers.6.weight'] = (0.000480, -0.000004)

__layer_variance_mean['conv_layers.0.bias'] = (0.014471, -0.007999)
__layer_variance_mean['conv_layers.3.bias'] = (0.000314, 0.004178)
__layer_variance_mean['conv_layers.6.bias'] = (0.000195, 0.004282)
__layer_variance_mean['conv_layers.8.bias'] = (0.000129, 0.005683)
__layer_variance_mean['conv_layers.11.bias'] = (0.000148, 0.006729)
__layer_variance_mean['conv_layers.13.bias'] = (0.000136, 0.007548)
__layer_variance_mean['conv_layers.16.bias'] = (0.000270, 0.008515)
__layer_variance_mean['conv_layers.18.bias'] = (0.000148, 0.001585)
__layer_variance_mean['linear_layers.0.bias'] = (0.000033, 0.000631)
__layer_variance_mean['linear_layers.3.bias'] = (0.000063, 0.001991)
__layer_variance_mean['linear_layers.6.bias'] = (0.006112, 0.000515)


def weights_init_trained(module):
    layer_name = getattr(module, "_module_name", None)
    if layer_name is not None:
        for name, param in module.named_parameters():
            if name == 'weight':
                mean, var = __layer_variance_mean[f'{layer_name}.weight']
                std = math.sqrt(var) if var > 0 else 0.0
                param.data.normal_(mean=mean, std=std)  # Normal(0,1)
            elif name == 'bias':
                mean, var = __layer_variance_mean[f'{layer_name}.bias']
                std = math.sqrt(var) if var > 0 else 0.0
                param.data.normal_(mean=mean, std=std)


class VGG11_no_bn(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super(VGG11_no_bn, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
