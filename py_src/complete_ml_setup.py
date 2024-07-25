import torch

from py_src import ml_setup

class PredefinedCompleteMlSetup:
    def __init__(self, arg_ml_setup, optimizer, epochs):
        self.ml_setup = arg_ml_setup
        self.optimizer = optimizer
        self.epochs = epochs

    @staticmethod
    def get_lenet5():
        arg_ml_setup = ml_setup.mnist_lenet5()
        optimizer = torch.optim.SGD(arg_ml_setup.model.parameters(), lr=0.001)
        return PredefinedCompleteMlSetup(arg_ml_setup, optimizer, 20)

    @staticmethod
    def get_resnet18():
        arg_ml_setup = ml_setup.resnet18_cifar10()
        optimizer = torch.optim.SGD(arg_ml_setup.model.parameters(), lr=0.001)
        return PredefinedCompleteMlSetup(arg_ml_setup, optimizer, 100)

