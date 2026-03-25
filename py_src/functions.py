import math, sys
from typing import Optional
import torch
import lightning as L

from py_src import ml_setup, cuda, util


def batch_to_batch_size(batch):
    if isinstance(batch, torch.Tensor):
        batch_size = batch.size(0)
    elif isinstance(batch, tuple):
        batch_size = batch[0].size(0)
    elif isinstance(batch, list):
        batch_size = batch[0].size(0)
    else:
        raise NotImplementedError
    return batch_size


def train(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
          criterion,
          current_epoch: int,
          arg_ml_setup: ml_setup.MlSetup, device: torch.device,
          arg_amp: bool, scaler: Optional[torch.cuda.amp.GradScaler], 
          min_rounds=None, max_rounds=None, loss_threshold=None, backpropagation=True):
    # current_epoch, optimizer, lr_scheduler can be None when backpropagation is False

    model.train()
    model.to(device)

    train_correct = None
    train_loss = 0
    train_count = 0

    """ Training procedure """
    train_for_one_epoch_mode = False
    train_for_min_max_iteration_mode = False
    if min_rounds is not None and max_rounds is not None:
        moving_average = util.MovingAverage(min_rounds)
        train_for_min_max_iteration_mode = True
    else:
        moving_average = None
        train_for_one_epoch_mode = True
        max_rounds = int(sys.maxsize)

    training_iter_counter = 0
    exit_training = False

    def pre_train_check():
        nonlocal training_iter_counter
        if train_for_min_max_iteration_mode:
            training_iter_counter += 1

    def post_train_check(loss_value):
        nonlocal moving_average, exit_training, training_iter_counter
        if training_iter_counter >= max_rounds:
            exit_training = True
        if moving_average is not None:
            moving_average.add(loss_value)
            if moving_average.get_average() <= loss_threshold and training_iter_counter >= min_rounds:
                exit_training = True

    while training_iter_counter < max_rounds:
        # user defined step function
        if arg_ml_setup.override_train_step_function is not None:
            for batch_idx, batch in enumerate(dataloader):
                pre_train_check()
                assert backpropagation==True, "backpropagation has to be True for override_train_step_function"
                batch = cuda.to_device(batch, device)
                output = arg_ml_setup.override_train_step_function(batch_idx, batch, model, optimizer, lr_scheduler, arg_ml_setup)
                loss = output.loss_value
                train_loss += loss * output.sample_count
                train_count += output.sample_count
                train_correct = 0 if train_correct is None else train_correct
                train_correct += output.correct_count
                for func in arg_ml_setup.func_handler_post_training:
                    func(model=model)

                post_train_check(loss)
                if exit_training:
                    break

        # L.LightningModule
        elif isinstance(model, L.LightningModule):
            """ Lighting model """
            for batch_idx, batch in enumerate(dataloader):
                pre_train_check()

                batch = cuda.to_device(batch, device)
                if backpropagation:
                    optimizer.zero_grad(set_to_none=True)

                loss, batch_accuracy = model.training_step(batch, batch_idx)
                if backpropagation:
                    loss.backward()
                    model.optimizer_step(current_epoch, batch_idx, optimizer, optimizer_closure=None)

                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    for func in arg_ml_setup.func_handler_post_training:
                        func(model=model)
                batch_size = batch_to_batch_size(batch)
                train_loss += loss.item() * batch_size
                train_count += batch_size
                train_correct = 0 if train_correct is None else train_correct
                train_correct += batch_accuracy.item() * batch_size

                post_train_check(loss)
                if exit_training:
                    break
        else:
            """ Normal PyTorch model """
            for data, label in dataloader:
                pre_train_check()

                data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
                if backpropagation:
                    optimizer.zero_grad(set_to_none=True)
                if arg_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(data)
                        if criterion == ml_setup.CriterionType.Diffusion:
                            loss = outputs
                        elif isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss):
                            loss = criterion(outputs, label)
                        else:
                            raise NotImplementedError
                        
                        if backpropagation:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                else:
                    outputs = model(data)
                    if criterion == ml_setup.CriterionType.Diffusion:
                        loss = outputs
                    elif isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss):
                        loss = criterion(outputs, label)
                    else:
                        raise NotImplementedError

                    if backpropagation:
                        loss.backward()
                        optimizer.step()
                if backpropagation:
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    for func in arg_ml_setup.func_handler_post_training:
                        func(model=model)

                if isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss):
                    _, predicted = torch.max(outputs, 1)
                    train_correct = 0 if train_correct is None else train_correct
                    train_correct += (predicted == label).sum().item()
                train_loss += loss.item() * label.size(0)
                train_count += label.size(0)

                post_train_check(loss.item())
                if exit_training:
                    break

        if exit_training:
            break
        if train_for_one_epoch_mode:
            break
    output_loss = train_loss if moving_average is None else moving_average.get_average()

    return train_correct, output_loss, train_count, training_iter_counter


def val(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
          criterion,
          arg_ml_setup: ml_setup.MlSetup, device: torch.device,
          arg_amp: bool, scaler: Optional[torch.cuda.amp.GradScaler]):
    model.eval()
    model.to(device)

    total_loss, total_count, total_correct, total_variance = 0.0, 0, None, None

    if arg_ml_setup.override_evaluation_step_function is not None:
        """ user defined step function """
        for batch_idx, batch in enumerate(dataloader):
            batch = cuda.to_device(batch, device)
            output = arg_ml_setup.override_evaluation_step_function(batch_idx, batch, model, arg_ml_setup)
            loss = output.loss_value
            total_loss += loss * output.sample_count
            total_count += output.sample_count
            total_correct = 0 if total_correct is None else total_correct
            total_correct += output.correct_count

    elif isinstance(model, L.LightningModule):
        """ Lighting model """
        model.on_validation_epoch_start()
        for batch_idx, batch in enumerate(dataloader):
            batch_size = batch_to_batch_size(batch)
            batch = cuda.to_device(batch, device)
            model.validation_step(batch, batch_idx)
            total_count += batch_size
        model.on_validation_epoch_end()
        loss, correct_count = model.get_validation_result()
        total_loss += loss * total_count
        total_correct = 0 if total_correct is None else total_correct
        total_correct += correct_count
        total_variance = math.nan
    else:
        """ Normal PyTorch model """
        for d, l in dataloader:
            d = cuda.to_device(device, d)
            l = cuda.to_device(device, l)
            outputs = model(d)
            total_loss += criterion(outputs, l).item() * l.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct = 0 if total_correct is None else total_correct
            total_correct += (predicted == l).sum().item()
            total_count += l.size(0)

            total_variance = 0.0 if total_variance is None else total_variance
            total_variance += outputs.var(dim=0, unbiased=False).mean().item()

    return total_loss, total_count, total_correct, total_variance
