import torch
from torchvision import transforms, models, datasets


class MlSetup:
    def __init__(self):
        self.model = None
        self.training_data = None
        self.testing_data = None
        self.criterion = None
        self.training_batch_size = None
        self.learning_rate = None
        self.dataset_label = None


""" CIFAR10 """
def dataset_cifar10():
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    transforms_cifar_train = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(32, padding=4, padding_mode='reflect'), transforms.RandomHorizontalFlip(p=0.5), transforms.Normalize(*stats)])
    transforms_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar_train)
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar_test)
    cifar10_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return cifar10_train, cifar10_test, cifar10_labels


""" CIFAR10 + ResNet18 """
def resnet18_cifar10():
    resnet18_cifar10 = MlSetup()
    resnet18_cifar10.model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
    resnet18_cifar10.training_data, resnet18_cifar10.testing_data, resnet18_cifar10.dataset_label = dataset_cifar10()
    resnet18_cifar10.criterion = torch.nn.CrossEntropyLoss()
    resnet18_cifar10.training_batch_size = 64
    resnet18_cifar10.learning_rate = 0.001
    return resnet18_cifar10
