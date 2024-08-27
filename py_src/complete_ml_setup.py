import torch

from py_src import ml_setup


class FastTrainingSetup(object):
    @staticmethod
    def get_optimizer_lr_scheduler_epoch(arg_ml_setup: ml_setup, model):
        if arg_ml_setup.model_name == 'lenet5':
            epochs = 20
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            return optimizer, None, epochs
        elif arg_ml_setup.model_name == 'resnet18':
            epochs = 30
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, steps_per_epoch=steps_per_epoch, epochs=epochs)
            return optimizer, lr_scheduler, epochs

        else:
            raise NotImplementedError

