import os
import sys
import torch

from .runtime_parameters import RuntimeParameters

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import util, ml_setup, cuda

def rebuild_norm_layer_function(model: torch.nn.Module, initial_model_state, start_model_state, rebuild_norm_optimizer: torch.optim.Optimizer,
                                training_optimizer_state, norm_layers, ml_setup: ml_setup.MlSetup,
                                dataloader, parameter_rebuild_norm, runtime_parameter: RuntimeParameters, rebuild_on_device=None, logger=None):
    model_stat = model.state_dict()

    """reset the weights of norm layers"""
    assert sum([int(i) for i in [parameter_rebuild_norm.rebuild_norm_use_initial_norm_weights,
            parameter_rebuild_norm.rebuild_norm_use_start_model_norm_weights]]) <= 1, \
        "only rebuild_norm_use_start_model_norm_weights or rebuild_norm_use_initial_norm_weights can be set to True"
    rebuild_norm_layer_function.__reset_info_print = False
    for layer_name, layer_weights in model_stat.items():
        if layer_name in norm_layers:
            if parameter_rebuild_norm.rebuild_norm_use_initial_norm_weights:
                if not rebuild_norm_layer_function.__reset_info_print:
                    logger.info(f"reset norm weights to initial model weights")
                    rebuild_norm_layer_function.__reset_info_print = True
                model_stat[layer_name] = initial_model_state[layer_name].detach().clone()
            if parameter_rebuild_norm.rebuild_norm_use_start_model_norm_weights:
                if not rebuild_norm_layer_function.__reset_info_print:
                    logger.info(f"reset norm weights to starting model weights")
                    rebuild_norm_layer_function.__reset_info_print = True
                model_stat[layer_name] = start_model_state[layer_name].detach().clone()

    model.load_state_dict(model_stat)

    if rebuild_on_device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = rebuild_on_device

    model.train()
    model.to(device)
    criterion = ml_setup.criterion
    rebuild_norm_optimizer.load_state_dict(training_optimizer_state)
    cuda.CudaEnv.optimizer_to(rebuild_norm_optimizer, device)

    if runtime_parameter.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    training_iter_counter = 0
    moving_average = util.MovingAverage(parameter_rebuild_norm.rebuild_norm_for_min_rounds)
    while True:
        exit_training = False
        for data, label in dataloader:
            training_iter_counter += 1
            data, label = data.to(device), label.to(device)
            rebuild_norm_optimizer.zero_grad(set_to_none=True)
            if runtime_parameter.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    training_loss = criterion(outputs, label)
                    scaler.scale(training_loss).backward()
                    scaler.step(rebuild_norm_optimizer)
                    scaler.update()
            else:
                outputs = model(data)
                training_loss = criterion(outputs, label)
                training_loss.backward()
                rebuild_norm_optimizer.step()

            if runtime_parameter.verbose:
                if training_iter_counter % 10 == 0:
                    logger.info(f"current tick: {runtime_parameter.current_tick}, rebuilding norm for {training_iter_counter} rounds, loss = {moving_average.get_average():.3f}")

            training_loss_val = training_loss.item()
            moving_average.add(training_loss_val)
            if training_iter_counter == parameter_rebuild_norm.rebuild_norm_for_max_rounds:
                exit_training = True
                break
            if moving_average.get_average() <= parameter_rebuild_norm.rebuild_norm_until_loss and training_iter_counter >= parameter_rebuild_norm.rebuild_norm_for_min_rounds:
                exit_training = True
                break
        if exit_training:
            if logger is not None:
                logger.info(f"current tick: {runtime_parameter.current_tick}, rebuilding norm for {training_iter_counter} rounds(final), loss = {moving_average.get_average():.3f}")
            break