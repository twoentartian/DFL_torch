import os
import argparse
import logging
import shutil
import sys
import torch
import concurrent.futures
import copy
import time
from datetime import datetime
from torch.utils.data import DataLoader

from find_high_accuracy_path_v2.runtime_parameters import RuntimeParameters, WorkMode
from find_high_accuracy_path_v2.find_parameters import ParameterGeneral, ParameterMove, ParameterTrain, ParameterRebuildNorm
from find_high_accuracy_path import set_logging, get_files_to_process


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, model_average, model_variance_correct, special_torch_layers, cuda, util, configuration_file
from py_src.service import record_weights_difference, record_test_accuracy_loss, record_variance, record_model_stat, record_training_loss
from py_src.ml_setup import MlSetup

logger = logging.getLogger("find_high_accuracy_path_v2")

REPORT_FINISH_TIME_PER_TICK = 100

def load_existing_optimizer_stat(optimizer, optimizer_stat_dict_path, logger=None):
    assert os.path.exists(optimizer_stat_dict_path), f'starting optimizer {optimizer_stat_dict_path} is missing'
    start_model_optimizer_stat, _ = util.load_optimizer_state_file(optimizer_stat_dict_path)
    optimizer_test = copy.deepcopy(optimizer)
    try:
        optimizer_test.load_state_dict(start_model_optimizer_stat)
        optimizer.load_state_dict(start_model_optimizer_stat)
        if logger is not None:
            logger.info(f"successfully load optimizer state")
    except Exception as e:
        if logger is not None:
            logger.warning(f"fail to load optimizer state: {str(e)}, now skip")
        else:
            raise RuntimeError(f"fail to load optimizer state: {str(e)}, now skip")
    del optimizer_test

def pre_train(model, optimizer, criterion, dataloader, device, cpu_device, logger=None):
    if logger is not None:
        logger.info(f"pre training")
    temp_model_state = model.state_dict()
    cuda.CudaEnv.model_state_dict_to(temp_model_state, cpu_device)
    model.train()
    model.to(device)
    cuda.CudaEnv.optimizer_to(optimizer, device)
    for (training_index, (data, label)) in enumerate(dataloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        if training_index == 100:
            break
    model.load_state_dict(temp_model_state)
    model.to(device)
    if logger is not None:
        logger.info(f"pre training finished")

def find_layers_according_to_name_and_keyword(model_state_dict, layer_names, layer_name_keywords):
    found_layers = []
    ignored_layers = []
    if layer_names is None:
        _layer_names = []
    else:
        _layer_names = layer_names
    if layer_name_keywords is None:
        _layer_name_keywords = []
    else:
        _layer_name_keywords = layer_name_keywords
    for l in model_state_dict.keys():
        if l in _layer_names:
            found_layers.append(l)
        if special_torch_layers.is_keyword_in_layer_name(l, _layer_name_keywords):
            found_layers.append(l)
    for l in model_state_dict.keys():
        if l not in found_layers:
            ignored_layers.append(l)
    return found_layers, ignored_layers

def rebuild_norm_layer_function(model: torch.nn.Module, initial_model_state, rebuild_norm_optimizer: torch.optim.Optimizer,
                                training_optimizer_state, norm_layers, ml_setup: MlSetup,
                                dataloader, parameter_rebuild_norm, runtime_parameter: RuntimeParameters, rebuild_on_device=None, logger=None):
    model_stat = model.state_dict()

    """reset the weights of norm layers"""
    for layer_name, layer_weights in model_stat.items():
        if layer_name in norm_layers:
            model_stat[layer_name] = initial_model_state[layer_name]

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


def process_file_func(index, runtime_parameter: RuntimeParameters):
    config_file = configuration_file.load_configuration(runtime_parameter.config_file_path)
    start_point, end_point = runtime_parameter.start_and_end_point_for_paths[index]
    start_file_name = os.path.basename(start_point).replace('.model.pt', '')
    if end_point == "origin":
        assert runtime_parameter.work_mode == WorkMode.to_origin
        end_file_name = "origin"
    elif end_point == "inf":
        assert runtime_parameter.work_mode == WorkMode.to_inf
        end_file_name = "inf"
    else:
        end_file_name = os.path.basename(end_point).replace('.model.pt', '')

    """logger"""
    task_name = f"{start_file_name}-{end_file_name}"
    child_logger = logging.getLogger(f"find_high_accuracy_path.{task_name}")
    set_logging(child_logger, task_name, log_file_path=os.path.join(runtime_parameter.output_folder_path, "info.log"))
    child_logger.info("logging setup complete")

    if runtime_parameter.use_cpu:
        device = torch.device("cpu")
        gpu = None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu = cuda.CudaDevice(0)

    thread_per_process = runtime_parameter.total_cpu_count // runtime_parameter.worker_count
    torch.set_num_threads(thread_per_process)

    """output folders"""
    arg_output_folder_path = os.path.join(runtime_parameter.output_folder_path, f"{start_file_name}-{end_file_name}")
    if os.path.exists(arg_output_folder_path):
        child_logger.warning(f"{arg_output_folder_path} already exists")
    else:
        os.makedirs(arg_output_folder_path)

    """load models"""
    cpu_device = torch.device("cpu")
    start_model_stat_dict, start_model_name = util.load_model_state_file(start_point)
    child_logger.info(f"loading start model at {start_point}")
    if runtime_parameter.dataset_name is not None:
        current_ml_setup = ml_setup.get_ml_setup_from_config(start_model_name, dataset_type=runtime_parameter.dataset_name)
    else:
        current_ml_setup = ml_setup.get_ml_setup_from_config(start_model_name)
    child_logger.info(f"find model type is {start_model_name}")

    target_model: torch.nn.Module = copy.deepcopy(current_ml_setup.model)
    initial_model_stat = target_model.state_dict()
    target_model.load_state_dict(start_model_stat_dict)
    if runtime_parameter.work_mode == WorkMode.to_origin:
        end_model_stat_dict = {k: torch.zeros_like(v) for k, v in start_model_stat_dict.items()}
        child_logger.info(f"work mode: to_origin")
    elif runtime_parameter.work_mode == WorkMode.to_inf:
        end_model_stat_dict = {k: v * 2 for k, v in start_model_stat_dict.items()}
        child_logger.info(f"work mode: to_inf")
    elif runtime_parameter.work_mode == WorkMode.to_certain_model:
        end_model_stat_dict, end_model_name = util.load_model_state_file(end_point)
        child_logger.info(f"work mode: to_certain_model at {end_point}")
        assert end_model_name == start_model_name
    else:
        raise NotImplemented

    """assert start_model_state_dict != end_model_stat_dict"""
    for key in start_model_stat_dict.keys():
        assert not torch.equal(start_model_stat_dict[key], end_model_stat_dict[key]), f'starting model({start_point}) is same as ending model({end_point})'
        break

    """begin finding path"""
    general_parameter: ParameterGeneral = config_file.get_parameter_general(runtime_parameter, current_ml_setup)
    runtime_parameter.max_tick = general_parameter.max_tick
    runtime_parameter.current_tick = 0
    if general_parameter.test_dataset_use_whole is not None:
        child_logger.info(f"setting test_dataset_use_whole to {general_parameter.test_dataset_use_whole}")
        runtime_parameter.test_dataset_use_whole = general_parameter.test_dataset_use_whole
    else:
        child_logger.info(f"setting test_dataset_use_whole to default (False)")
        runtime_parameter.test_dataset_use_whole = False

    """load training data"""
    training_dataset = current_ml_setup.training_data
    if general_parameter.dataloader_worker is not None:
        dataloader = DataLoader(training_dataset, batch_size=current_ml_setup.training_batch_size, shuffle=True, num_workers=general_parameter.dataloader_worker, persistent_workers=True)
    else:
        dataloader = DataLoader(training_dataset, batch_size=current_ml_setup.training_batch_size, shuffle=True)
    criterion = current_ml_setup.criterion

    """get optimizer"""
    target_model.to(device)
    optimizer = config_file.get_optimizer_train(runtime_parameter, current_ml_setup, target_model.parameters())
    initial_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
    optimizer_for_rebuild_norm = None
    assert optimizer is not None

    """update parameters"""
    parameter_train: ParameterTrain = config_file.get_parameter_train(runtime_parameter, current_ml_setup)
    parameter_train.validate()
    parameter_move: ParameterMove = config_file.get_parameter_move(runtime_parameter, current_ml_setup)
    parameter_move.validate()
    parameter_rebuild_norm: ParameterRebuildNorm = config_file.get_parameter_rebuild_norm(runtime_parameter, current_ml_setup)
    parameter_rebuild_norm.validate()

    """services"""
    child_logger.info("setting services")
    all_model_stats = [target_model.state_dict(), end_model_stat_dict]

    weight_diff_service = record_weights_difference.ServiceWeightsDifferenceRecorder(1)
    weight_diff_service.initialize_without_runtime_parameters(all_model_stats, arg_output_folder_path)
    child_logger.info("setting service done: weight_diff_service")

    weight_change_service = record_weights_difference.ServiceWeightsDifferenceRecorder(1,
                                                                                       l1_save_file_name="weight_change_l1.csv",
                                                                                       l2_save_file_name="weight_change_l2.csv")
    model_state_of_last_tick = target_model.state_dict()
    weight_change_service.initialize_without_runtime_parameters(all_model_stats, arg_output_folder_path)
    child_logger.info("setting service done: weight_change_service")

    distance_to_origin_service = record_weights_difference.ServiceDistanceToOriginRecorder(1, [0])
    distance_to_origin_service.initialize_without_runtime_parameters({0: start_model_stat_dict}, arg_output_folder_path)
    child_logger.info("setting service done: distance_to_origin_service")

    variance_service = record_variance.ServiceVarianceRecorder(1)
    variance_service.initialize_without_runtime_parameters([0], [start_model_stat_dict], arg_output_folder_path)
    child_logger.info("setting service done: variance_service")

    if runtime_parameter.save_format != 'none':
        if not runtime_parameter.save_ticks:
            child_logger.info(f"record_model_service is ON at every {runtime_parameter.save_interval} tick")
            record_model_service = record_model_stat.ModelStatRecorder(sys.maxsize)  # the interval here has no effect
        else:
            child_logger.info("record_model_service is ON at certain ticks")
            record_model_service = record_model_stat.ModelStatRecorder(sys.maxsize)  # we don't set the interval here because only certain ticks should be recorded
        record_model_service.initialize_without_runtime_parameters([0], arg_output_folder_path, save_format=runtime_parameter.save_format)
    else:
        child_logger.info("record_model_service is OFF")
        record_model_service = None
    child_logger.info("setting service done: record_model_service")

    record_test_accuracy_loss_service = record_test_accuracy_loss.ServiceTestAccuracyLossRecorder(1, 100, use_fixed_testing_dataset=True, test_whole_dataset=runtime_parameter.test_dataset_use_whole)
    record_test_accuracy_loss_service.initialize_without_runtime_parameters(arg_output_folder_path, [0], target_model, criterion, current_ml_setup.testing_data,
                                                                            existing_model_for_testing=target_model, gpu=gpu, num_workers=general_parameter.dataloader_worker)
    child_logger.info("setting service done: record_test_accuracy_loss_service")

    record_training_loss_service = record_training_loss.ServiceTrainingLossRecorder(1)
    record_training_loss_service.initialize_without_runtime_parameters(arg_output_folder_path, [0])
    child_logger.info("setting service done: record_training_loss_service")

    """record variance"""
    variance_record = model_variance_correct.VarianceCorrector(model_variance_correct.VarianceCorrectionType.FollowOthers)
    variance_record.add_variance(target_model.state_dict())
    target_variance = variance_record.get_variance()

    """begin finding the path"""
    if runtime_parameter.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    ignore_move_layers = None
    norm_layers = None
    cuda.CudaEnv.model_state_dict_to(end_model_stat_dict, device)
    timer = time.time()
    while runtime_parameter.current_tick < runtime_parameter.max_tick:
        parameter_updated = False
        child_logger.info(f"tick: {runtime_parameter.current_tick}")

        """update end_model_stat_dict if work_mode = to_inf"""
        if runtime_parameter.work_mode == WorkMode.to_inf:
            target_model_stat = target_model.state_dict()
            end_model_stat_dict = {k: v.detach().clone() * 2 for k, v in target_model_stat.items()}
            cuda.CudaEnv.model_state_dict_to(end_model_stat_dict, device)

        if runtime_parameter.current_tick % REPORT_FINISH_TIME_PER_TICK == 0 and runtime_parameter.current_tick != 0:
            time_elapsed = time.time() - timer
            timer = time.time()
            remaining = (runtime_parameter.max_tick - runtime_parameter.current_tick) // REPORT_FINISH_TIME_PER_TICK
            time_to_finish = remaining * time_elapsed
            finish_time = timer + time_to_finish
            child_logger.info(f"time taken for {REPORT_FINISH_TIME_PER_TICK} ticks: {time_elapsed:.2f}s, expected to finish at {datetime.fromtimestamp(finish_time)}")

        """update parameter"""
        new_parameter_train: ParameterTrain = config_file.get_parameter_train(runtime_parameter, current_ml_setup)
        if new_parameter_train is not None:
            parameter_updated = True
            child_logger.info(f"update parameter (train) at tick {runtime_parameter.current_tick}")
            parameter_train = new_parameter_train
        new_parameter_move: ParameterMove = config_file.get_parameter_move(runtime_parameter, current_ml_setup)
        if new_parameter_move is not None:
            parameter_updated = True
            child_logger.info(f"update parameter (move) at tick {runtime_parameter.current_tick}")
            parameter_move = new_parameter_move
            # update layers to move
            ignore_move_layers, moved_layers = find_layers_according_to_name_and_keyword(start_model_stat_dict, parameter_move.layer_skip_move, parameter_move.layer_skip_move_keyword)
            child_logger.info(f"updating layers to move at tick {runtime_parameter.current_tick}")
            child_logger.info(f"ignore moving {len(ignore_move_layers)} layers: {ignore_move_layers}")
            child_logger.info(f"plan to move {len(moved_layers)} layers: {moved_layers}")
        new_parameter_rebuild_norm: ParameterRebuildNorm = config_file.get_parameter_rebuild_norm(runtime_parameter, current_ml_setup)
        if new_parameter_rebuild_norm is not None:
            parameter_updated = True
            child_logger.info(f"update parameter (rebuild_norm) at tick {runtime_parameter.current_tick}")
            parameter_rebuild_norm = new_parameter_rebuild_norm
            # update norm layer list
            norm_layers, non_norm_layers = find_layers_according_to_name_and_keyword(start_model_stat_dict, parameter_rebuild_norm.rebuild_norm_layer, parameter_rebuild_norm.rebuild_norm_layer_keyword)
            child_logger.info(f"updating norm layers at tick {runtime_parameter.current_tick}")
            child_logger.info(f"totally {len(norm_layers)} layers to rebuild: {norm_layers}")
            child_logger.info(f"totally {len(non_norm_layers)} non-rebuild layers: {non_norm_layers}")

        """if this is the first tick"""
        if runtime_parameter.current_tick == 0:
            if parameter_train.pretrain_optimizer and parameter_train.load_existing_optimizer:
                logger.critical("cannot enable both pretrain_optimizer and load_existing_optimizer")
            # pretrain optimizer
            if not runtime_parameter.debug_check_config_mode:
                if parameter_train.pretrain_optimizer:
                    pre_train(target_model, optimizer, criterion, dataloader, device, cpu_device, logger=child_logger)
            # load existing optimizer
            if not runtime_parameter.debug_check_config_mode:
                if parameter_train.load_existing_optimizer:
                    optimizer_state_dict_path = start_point.replace('model.pt', 'optimizer.pt')
                    load_existing_optimizer_stat(optimizer, optimizer_state_dict_path, logger=child_logger)

        """if parameter updated"""
        if parameter_updated:
            util.save_model_state(os.path.join(arg_output_folder_path, f"{runtime_parameter.current_tick}.model.pt"), target_model.state_dict(), current_ml_setup.model_name)
            util.save_optimizer_state(os.path.join(arg_output_folder_path, f"{runtime_parameter.current_tick}.optimizer.pt"), optimizer.state_dict(), current_ml_setup.model_name)

        """move model"""
        if not runtime_parameter.debug_check_config_mode:
            target_model_stat_dict = model_average.move_model_state_toward(target_model.state_dict(), end_model_stat_dict,
                                                                       parameter_move.step_size, parameter_move.adoptive_step_size,
                                                                       enable_merge_bias_with_weight=parameter_move.merge_bias_with_weights,
                                                                       ignore_layers=ignore_move_layers)
            target_model.load_state_dict(target_model_stat_dict)

        """variance correction"""
        if not runtime_parameter.debug_check_config_mode:
            if runtime_parameter.work_mode == WorkMode.to_certain_model:
                child_logger.info(f"current tick: rescale variance")
                target_model_stat_dict = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(target_model.state_dict(), target_variance)
                target_model.load_state_dict(target_model_stat_dict)

        """update learning rate"""
        if not runtime_parameter.debug_check_config_mode:
            if runtime_parameter.work_mode == WorkMode.to_inf or runtime_parameter.work_mode == WorkMode.to_origin:
                for param_group, initial_optimizer_state, (name, param) in zip(optimizer.param_groups, initial_optimizer_state_dict['param_groups'], target_model.named_parameters()):
                    if special_torch_layers.is_keyword_in_layer_name(name, parameter_rebuild_norm.rebuild_norm_layer_keyword):
                        continue  # skip norm layers
                    if 'weight' in name and param.requires_grad:  # Only adjust weights, not biases
                        current_layer_variance = torch.var(param.data).item()
                        new_lr = initial_optimizer_state['lr'] / target_variance[name] * current_layer_variance
                        child_logger.info(f"tick {runtime_parameter.current_tick}: update lr for layer {name} to {new_lr:.3E}")
                        param_group['lr'] = new_lr

        """training"""
        training_loss_val = 0
        if not runtime_parameter.debug_check_config_mode:
            target_model.train()
            target_model.to(device)
            cuda.CudaEnv.optimizer_to(optimizer, device)

            training_iter_counter = 0
            moving_average = util.MovingAverage(parameter_train.train_for_min_rounds)
            while True:
                exit_training = False
                for data, label in dataloader:
                    training_iter_counter += 1
                    data, label = data.to(device), label.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    if runtime_parameter.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = target_model(data)
                            training_loss = criterion(outputs, label)
                            scaler.scale(training_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        outputs = target_model(data)
                        training_loss = criterion(outputs, label)
                        training_loss.backward()
                        optimizer.step()
                    training_loss_val = training_loss.item()
                    moving_average.add(training_loss_val)

                    if runtime_parameter.verbose:
                        if training_iter_counter % 10 == 0:
                            child_logger.info(f"current tick: {runtime_parameter.current_tick}, training {training_iter_counter} rounds, loss = {moving_average.get_average():.3f}")

                    if training_iter_counter == parameter_train.train_for_max_rounds:
                        exit_training = True
                        break
                    if moving_average.get_average() <= parameter_train.train_until_loss and training_iter_counter >= parameter_train.train_for_min_rounds:
                        exit_training = True
                        break
                if exit_training:
                    child_logger.info(f"current tick: {runtime_parameter.current_tick}, training {training_iter_counter} rounds(final), loss = {moving_average.get_average():.3f}")
                    break

        """rebuilding normalization"""
        if not runtime_parameter.debug_check_config_mode:
            if parameter_rebuild_norm.rebuild_norm_for_max_rounds != 0:
                optimizer_for_rebuild_norm_new = config_file.get_optimizer_rebuild_norm(runtime_parameter, current_ml_setup, target_model.parameters())
                if optimizer_for_rebuild_norm_new is not None:
                    optimizer_for_rebuild_norm = optimizer_for_rebuild_norm_new
                training_optimizer_stat = optimizer.state_dict()
                rebuild_norm_layer_function(target_model, initial_model_stat, optimizer_for_rebuild_norm, training_optimizer_stat,
                                            norm_layers, current_ml_setup, dataloader, parameter_rebuild_norm, runtime_parameter, device, logger=child_logger)

        """variance correction"""
        if not runtime_parameter.debug_check_config_mode:
            if runtime_parameter.work_mode == WorkMode.to_certain_model:
                child_logger.info(f"current tick: rescale variance")
                target_model_stat_dict = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(target_model.state_dict(), target_variance)
                target_model.load_state_dict(target_model_stat_dict)

        """service"""
        run_service = True
        if runtime_parameter.debug_check_config_mode:
            run_service = runtime_parameter.current_tick % 1000 == 0

        if run_service:
            target_model_stat_dict = target_model.state_dict()
            all_model_stats = [target_model_stat_dict, end_model_stat_dict]
            for i in all_model_stats:
                cuda.CudaEnv.model_state_dict_to(i, device)
            weight_diff_service.trigger_without_runtime_parameters(runtime_parameter.current_tick, all_model_stats)
            weight_change_service.trigger_without_runtime_parameters(runtime_parameter.current_tick, [model_state_of_last_tick, target_model_stat_dict])
            model_state_of_last_tick = copy.deepcopy(target_model_stat_dict)
            distance_to_origin_service.trigger_without_runtime_parameters(runtime_parameter.current_tick, {0: target_model_stat_dict})
            variance_service.trigger_without_runtime_parameters(runtime_parameter.current_tick, [0], [target_model_stat_dict])
            if record_model_service is not None:
                record_flag = False
                if runtime_parameter.save_ticks is None:
                    record_flag = runtime_parameter.current_tick % runtime_parameter.save_interval == 0
                else:
                    if runtime_parameter.current_tick in runtime_parameter.save_ticks:
                        record_flag = True
                if record_flag:
                    record_model_service.trigger_without_runtime_parameters(runtime_parameter.current_tick, [0], [target_model_stat_dict])
            record_test_accuracy_loss_service.trigger_without_runtime_parameters(runtime_parameter.current_tick, {0: target_model_stat_dict})
            record_training_loss_service.trigger_without_runtime_parameters(runtime_parameter.current_tick, {0: training_loss_val})

        # update tick
        runtime_parameter.current_tick += 1

    # save final model and optimizer
    util.save_model_state(os.path.join(arg_output_folder_path, f"{runtime_parameter.current_tick}.model.pt"), target_model.state_dict(), current_ml_setup.model_name)
    util.save_optimizer_state(os.path.join(arg_output_folder_path, f"{runtime_parameter.current_tick}.optimizer.pt"), optimizer.state_dict(), current_ml_setup.model_name)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Move the model towards certain direction and keep its accuracy')
    parser.add_argument("start_folder", type=str, help="folder containing starting models")

    parser.add_argument("end_folder", type=str, help="folder containing destination models, or 'inf', 'origin' ")
    parser.add_argument("--config", type=str, help="the config file")
    parser.add_argument("--mapping_mode", type=str, default='auto', choices=['auto', 'all_to_all', 'each_to_each', 'one_to_all', 'all_to_one'])

    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("-w", "--worker", type=int, default=1, help='specify how many models to train in parallel')
    parser.add_argument("-d", "--dataset", type=str, default=None, help='specify the dataset name')

    parser.add_argument("--save_ticks", type=str, help='specify when to record the models (e.g. [1,2,3,5-10]), only works when --save_format is set to work.')
    parser.add_argument("--save_interval", type=int, default=1, help='specify the saving interval')
    parser.add_argument("--save_format", type=str, default='none', choices=['none', 'file', 'lmdb'])
    parser.add_argument("--cpu", action='store_true', help='force using CPU for training')
    parser.add_argument("--amp", action='store_true', help='enable auto mixed precision')
    parser.add_argument("--check_config", action='store_true', help='only check configuration')
    parser.add_argument("-v", "--verbose", action='store_true', help='verbose mode')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')

    args = parser.parse_args()

    if args.config is None:
        config_file_path = "find_high_accuracy_path_v2_config.py"
    else:
        config_file_path = args.config
    config_file = configuration_file.load_configuration(config_file_path)

    set_logging(logger, "main")
    logger.info("logging setup complete")

    runtime_parameter = RuntimeParameters()

    runtime_parameter.use_cpu = args.cpu
    runtime_parameter.use_amp = args.amp
    runtime_parameter.save_ticks = args.save_ticks
    runtime_parameter.save_interval = args.save_interval
    runtime_parameter.save_format = args.save_format
    runtime_parameter.config_file_path = config_file_path
    runtime_parameter.dataset_name = args.dataset
    runtime_parameter.debug_check_config_mode = args.check_config
    runtime_parameter.verbose = args.verbose

    # find all paths to process
    start_folder = args.start_folder
    if args.end_folder == "origin":
        runtime_parameter.work_mode = WorkMode.to_origin
        assert args.mapping_mode == "auto", "mapping mode has to be 'auto' for move to origin"
        files_in_start_folder = sorted(set(temp_file for temp_file in os.listdir(start_folder) if temp_file.endswith('model.pt')))
        paths_to_find = [(os.path.join(start_folder, i), "origin") for i in files_in_start_folder]
    elif args.end_folder == "inf":
        runtime_parameter.work_mode = WorkMode.to_inf
        assert args.mapping_mode == "auto", "mapping mode has to be 'auto' for move to inf"
        files_in_start_folder = sorted(set(temp_file for temp_file in os.listdir(start_folder) if temp_file.endswith('model.pt')))
        paths_to_find = [(os.path.join(start_folder, i), "inf") for i in files_in_start_folder]
    else:
        runtime_parameter.work_mode = WorkMode.to_certain_model
        paths_to_find = get_files_to_process(args.start_folder, args.end_folder, args.mapping_mode)
    paths_to_find_count = len(paths_to_find)
    runtime_parameter.start_and_end_point_for_paths = paths_to_find
    logger.info(f"totally {paths_to_find_count} paths to process: {paths_to_find}")

    # create output folder
    if args.output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.mkdir(output_folder_path)
    runtime_parameter.output_folder_path = output_folder_path
    info_file = open(os.path.join(output_folder_path, "arguments.txt"), 'x')
    info_file.write(f"{args}")
    info_file.flush()
    info_file.close()

    # backup config file
    shutil.copyfile(config_file_path, os.path.join(output_folder_path, os.path.basename(config_file_path)))
    shutil.copyfile(__file__, os.path.join(output_folder_path, os.path.basename(__file__)))

    # worker and cpu cores setting
    runtime_parameter.total_cpu_count = args.core
    runtime_parameter.worker_count = args.worker
    if runtime_parameter.worker_count > paths_to_find_count:
        runtime_parameter.worker_count = paths_to_find_count
    logger.info(f"worker: {runtime_parameter.worker_count}")

    logger.info(f"final runtime parameters: {runtime_parameter.print()}")

    # start process
    with concurrent.futures.ProcessPoolExecutor(max_workers=runtime_parameter.worker_count) as executor:
        futures = [executor.submit(process_file_func, index, runtime_parameter) for index, path in enumerate(paths_to_find)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()





