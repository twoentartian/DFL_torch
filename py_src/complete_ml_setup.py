import torch
import math

from scipy.odr import Model

from py_src import ml_setup
from py_src.ml_setup_base.model import ModelType
from py_src.ml_setup_base.dataset import DatasetType

""" this class records the hyperparameters need for training a model in generate_high_accuracy_model.py """
class FastTrainingSetup(object):
    @staticmethod
    def get_optimizer_lr_scheduler_epoch(arg_ml_setup: ml_setup, model, preset=0):
        if arg_ml_setup.model_name == str(ModelType.lenet5.name) or arg_ml_setup.model_name == str(ModelType.lenet4.name):
            if arg_ml_setup.dataset_name == str(DatasetType.mnist.name):
                epochs = 20
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                return optimizer, None, epochs
            elif arg_ml_setup.dataset_name == str(DatasetType.random_mnist.name):
                lr = 0.01
                epochs = 100
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
                return optimizer, lr_scheduler, epochs
                # epochs = 50
                # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                # return optimizer, None, epochs
            else:
                raise NotImplementedError

        if arg_ml_setup.model_name == str(ModelType.lenet5_large_fc.name):
            epochs = 20
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            return optimizer, None, epochs
        elif arg_ml_setup.model_name == str(ModelType.resnet18_bn.name) or arg_ml_setup.model_name == str(ModelType.resnet18_gn.name):
            if "imagenet" in arg_ml_setup.dataset_name:
                lr = 0.1
                epochs = 60
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
            elif str(DatasetType.cifar10.name) == arg_ml_setup.dataset_name:
                lr = 0.1
                epochs = 30
                if preset == 0:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
            elif str(DatasetType.cifar100.name) == arg_ml_setup.dataset_name:
                lr = 0.1
                epochs = 50
                if preset == 0:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                elif preset == 1:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                else:
                    raise NotImplementedError
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
            else:
                raise NotImplementedError
            return optimizer, lr_scheduler, epochs
        elif arg_ml_setup.model_name == str(ModelType.simplenet.name):
            lr = 0.1
            epochs = 150
            optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-3, weight_decay=0.001)
            steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
            milestones_epoch = [100, 190, 306, 390, 440, 540]
            milestones = [steps_per_epoch * i for i in milestones_epoch]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
            return optimizer, lr_scheduler, epochs
        elif arg_ml_setup.model_name == str(ModelType.vgg11_bn.name):
            if arg_ml_setup.dataset_name == str(DatasetType.cifar10.name):
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                milestones_epoch = [30, 60, 90]
                milestones = [steps_per_epoch * i for i in milestones_epoch]
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
                return optimizer, lr_scheduler, epochs
            else:
                raise NotImplementedError
        elif arg_ml_setup.model_name == ModelType.cct_7_3x1_32.name:
            if arg_ml_setup.dataset_name == str(DatasetType.cifar10.name):
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                initial_lr = 55e-5
                if preset == 0:
                    weight_decay = 6e-2
                elif preset == 1:
                    weight_decay = 1e-2
                else:
                    raise NotImplementedError
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
            elif arg_ml_setup.dataset_name == str(DatasetType.cifar100.name):
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                initial_lr = 6e-4
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
        elif arg_ml_setup.model_name == ModelType.mobilenet_v2.name:
            if arg_ml_setup.dataset_name == str(DatasetType.cifar10.name):
                epochs = 200
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=4e-5, momentum=0.9)
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                milestones_epoch = [100]
                milestones = [steps_per_epoch * i for i in milestones_epoch]
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
                return optimizer, lr_scheduler, epochs
            else:
                raise NotImplementedError
        elif arg_ml_setup.model_name == ModelType.efficientnet_b0.name:
            if arg_ml_setup.dataset_name == DatasetType.cifar10.name:
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                milestones_epoch = [30, 60, 90]
                milestones = [steps_per_epoch * i for i in milestones_epoch]
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
            else:
                raise NotImplementedError
            return optimizer, lr_scheduler, epochs
        elif arg_ml_setup.model_name == ModelType.shufflenet_v2.name:
            if arg_ml_setup.dataset_name == DatasetType.cifar10.name:
                epochs = 300
                optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1, weight_decay = 4e-5, momentum = 0.9)
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                milestones_epoch = [150, 225]
                milestones = [steps_per_epoch * i for i in milestones_epoch]
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
            else:
                raise NotImplementedError
            return optimizer, lr_scheduler, epochs
        elif arg_ml_setup.model_name == ModelType.dla.name:
            if arg_ml_setup.dataset_name == DatasetType.cifar10.name:
                if preset == 0:
                    epochs = 120
                    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                    steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                    milestones_epoch = [30, 60, 90]
                    milestones = [steps_per_epoch * i for i in milestones_epoch]
                    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
                elif preset == 1:
                    epochs = 120
                    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=3e-4, momentum=0.9)
                    steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            return optimizer, lr_scheduler, epochs
        elif arg_ml_setup.model_name == ModelType.regnet_x_200mf.name:
            if arg_ml_setup.dataset_name == DatasetType.cifar10.name:
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                milestones_epoch = [30, 60, 90]
                milestones = [steps_per_epoch * i for i in milestones_epoch]
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*steps_per_epoch)
            else:
                raise NotImplementedError
            return optimizer, lr_scheduler, epochs
        elif arg_ml_setup.model_name == ModelType.densenet121.name:
            if arg_ml_setup.dataset_name == DatasetType.cifar10.name:
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
            else:
                raise NotImplementedError
            return optimizer, lr_scheduler, epochs
        elif arg_ml_setup.model_name == ModelType.densenet_cifar.name:
            if arg_ml_setup.dataset_name == DatasetType.cifar10.name:
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
            else:
                raise NotImplementedError
            return optimizer, lr_scheduler, epochs
        else:
            raise NotImplementedError
