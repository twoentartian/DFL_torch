import torch
import math

from py_src import ml_setup

class FastTrainingSetup(object):
    @staticmethod
    def get_optimizer_lr_scheduler_epoch(arg_ml_setup: ml_setup, model):
        if arg_ml_setup.model_name == 'lenet5':
            epochs = 20
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            return optimizer, None, epochs
        elif arg_ml_setup.model_name == 'resnet18_bn':
            lr = 0.1
            epochs = 30
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
            return optimizer, lr_scheduler, epochs
        elif arg_ml_setup.model_name == 'resnet18_gn':
            lr = 0.1
            epochs = 30
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
            return optimizer, lr_scheduler, epochs
        elif arg_ml_setup.model_name == 'simplenet':
            lr = 0.1
            epochs = 150
            optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-3, weight_decay=0.001)
            steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
            milestones_epoch = [100, 190, 306, 390, 440, 540]
            milestones = [steps_per_epoch * i for i in milestones_epoch]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
            return optimizer, lr_scheduler, epochs
        elif arg_ml_setup.model_name == 'cct7':
            lr = 6e-4
            weight_decay = 6e-2
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            min_lr = 1e-5
            warmup_lr = 1e-6
            warmup_epochs = 10
            decay_epochs = 280
            cooldown_epochs = 10
            total_epochs = warmup_epochs + decay_epochs + cooldown_epochs
            def lr_lambda(current_epoch):
                if current_epoch < warmup_epochs:
                    # Linear warmup from warmup_lr to lr
                    lr_current = warmup_lr + (lr - warmup_lr) * (current_epoch / warmup_epochs)
                    return lr_current / lr
                elif current_epoch < warmup_epochs + decay_epochs:
                    # Cosine decay from lr to min_lr
                    t = current_epoch - warmup_epochs
                    T = decay_epochs
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * t / T))
                    lr_current = min_lr + (lr - min_lr) * cosine_decay
                    return lr_current / lr
                else:
                    # Cooldown: learning rate held at min_lr
                    return min_lr / lr

            epochs = 300
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=6e-2)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return optimizer, lr_scheduler, epochs
        else:
            raise NotImplementedError
