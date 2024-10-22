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
            steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
            initial_lr = 55e-5
            weight_decay = 6e-2
            warmup_lr = 1e-5
            min_lr = 1e-5
            warmup_epochs = 10
            epochs = 300
            cooldown_epochs = 10
            warmup_steps = warmup_epochs * steps_per_epoch
            cosine_steps = (epochs - warmup_epochs) * steps_per_epoch
            cooldown_steps = cooldown_epochs * steps_per_epoch
            total_steps = warmup_steps + cosine_steps + cooldown_steps
            optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
            # lr_scheduler = CosineLRScheduler(
            #     optimizer,
            #     t_initial,
            #     lr_min=min_lr,
            #     warmup_lr_init=warmup_lr,
            #     warmup_t=warmup_epochs,
            #     initialize=True,
            #     cycle_limit=1,
            #     t_in_epochs=False
            # )

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear warmup
                    lr = warmup_lr + (initial_lr - warmup_lr) * (current_step / warmup_steps)
                elif current_step < warmup_steps + cosine_steps:
                    # Cosine annealing
                    t = current_step - warmup_steps
                    T = cosine_steps
                    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * t / T))
                else:
                    # Cooldown phase
                    lr = min_lr
                return lr / initial_lr
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return optimizer, lr_scheduler, epochs
        else:
            raise NotImplementedError
