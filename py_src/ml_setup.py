import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

    def self_validate(self):
        pass  # do nothing for now


""" CIFAR10 """
def dataset_cifar10():
    dataset_path = './data/cifar10'
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    transforms_cifar_train = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(32, padding=4, padding_mode='reflect'), transforms.RandomHorizontalFlip(p=0.5), transforms.Normalize(*stats)])
    transforms_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms_cifar_train)
    cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms_cifar_test)
    cifar10_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return cifar10_train, cifar10_test, cifar10_labels


""" MNIST """
def dataset_mnist():
    dataset_path = './data/mnist'
    mnist_train = datasets.MNIST(root=dataset_path, train=True, download=True)
    mean = mnist_train.data.float().mean() / 255
    std = mnist_train.data.float().std() / 255
    transforms_mnist_train = transforms.Compose([transforms.RandomRotation(5, fill=(0,)), transforms.RandomCrop(28, padding=2), transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])])
    transforms_mnist_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])])
    train_data = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transforms_mnist_train)
    test_data = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transforms_mnist_test)
    mnist_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return train_data, test_data, mnist_labels


""" LeNet5 """
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        return np.prod(size)

def lenet5():
    return LeNet5()


""" MNIST + LeNet5 """
def mnist_lenet5():
    mnist_lenet5 = MlSetup()
    mnist_lenet5.model = lenet5()
    mnist_lenet5.training_data, mnist_lenet5.testing_data, mnist_lenet5.dataset_label = dataset_mnist()
    mnist_lenet5.criterion = torch.nn.CrossEntropyLoss()
    mnist_lenet5.training_batch_size = 64
    mnist_lenet5.learning_rate = 0.001
    return mnist_lenet5


""" CIFAR10 + ResNet18 """
def resnet18_cifar10():
    resnet18_cifar10 = MlSetup()
    resnet18_cifar10.model = models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)
    resnet18_cifar10.training_data, resnet18_cifar10.testing_data, resnet18_cifar10.dataset_label = dataset_cifar10()
    resnet18_cifar10.criterion = torch.nn.CrossEntropyLoss()
    resnet18_cifar10.training_batch_size = 64
    resnet18_cifar10.learning_rate = 0.001
    return resnet18_cifar10
