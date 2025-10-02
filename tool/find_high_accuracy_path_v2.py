import os
import argparse
import logging
import shutil
import sys
from typing import Optional

import numpy
import torch
import torch.nn as nn
import concurrent.futures
import copy
import time
from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from find_high_accuracy_path_v2.runtime_parameters import RuntimeParameters, WorkMode, Checkpoint
from find_high_accuracy_path_v2.find_parameters import ParameterGeneral, ParameterMove, ParameterTrain, ParameterRebuildNorm
from find_high_accuracy_path_v2.functions import rebuild_norm_layer_function
from py_src.simulation_runtime_parameters import SimulationPhase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, model_average, model_variance_correct, special_torch_layers, cuda, util, configuration_file
from py_src.service import record_weights_difference, record_test_accuracy_loss, record_variance, record_model_stat, record_training_loss_accuracy, record_consecutive_linear_interpolation

logger = logging.getLogger("find_high_accuracy_path_v2")

REPORT_FINISH_TIME_PER_TICK = 100
ENABLE_REBUILD_NORM = False

def set_logging(target_logger, task_name, log_file_path=None):
    class ExitOnExceptionHandler(logging.StreamHandler):
        def emit(self, record):
            if record.levelno == logging.CRITICAL:
                raise SystemExit(-1)

    formatter = logging.Formatter(f"[%(asctime)s] [%(levelname)8s] [{task_name}] --- %(message)s (%(filename)s:%(lineno)s)")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    target_logger.setLevel(logging.DEBUG)
    target_logger.addHandler(console)
    target_logger.addHandler(ExitOnExceptionHandler())

    if log_file_path is not None:
        file = logging.FileHandler(log_file_path)
        file.setLevel(logging.DEBUG)
        file.setFormatter(formatter)
        target_logger.addHandler(file)

    del console, formatter

def get_files_to_process(arg_start_folder, arg_end_folder, arg_mode):
    if not os.path.isdir(arg_start_folder):
        logger.critical(f"{arg_start_folder} does not exist")
    if not os.path.isdir(arg_end_folder):
        logger.critical(f"{arg_end_folder} does not exist")

    files_in_start_folder = sorted(set(temp_file for temp_file in os.listdir(arg_start_folder) if temp_file.endswith('model.pt')))
    files_in_end_folder = sorted(set(temp_file for temp_file in os.listdir(arg_end_folder) if temp_file.endswith('model.pt')))
    if arg_mode == "auto":
        if len(files_in_start_folder) == len(files_in_end_folder):
            arg_mode = "each_to_each"
        else:
            arg_mode = "all_to_all"

    output_paths = []
    if arg_mode == "all_to_all":
        for start_file in files_in_start_folder:
            for end_file in files_in_end_folder:
                output_paths.append((os.path.join(arg_start_folder, start_file), os.path.join(arg_end_folder, end_file)))

    elif arg_mode == "each_to_each":
        if len(files_in_start_folder) != len(files_in_end_folder):
            logger.critical(f"file counts mismatch: {len(files_in_start_folder)} != {len(files_in_end_folder)}")
        if files_in_start_folder != files_in_end_folder:
            logger.critical(f"file names mismatch: {files_in_start_folder} != {files_in_end_folder}")
        for file in files_in_start_folder:
            output_paths.append((os.path.join(arg_start_folder, file), os.path.join(arg_end_folder, file)))
    else:
        logger.critical(f"mode {arg_mode} not recognized")

    return sorted(output_paths)

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

# this function pretrain model weights / optimizer weights
def pre_train(model, optimizer, criterion, dataloader, device, train_iteration=0, train_model_weights=False, train_optimizer=False, logger=None):
    if logger is not None:
        if train_model_weights is False and train_optimizer is False:
            logger.info(f"skip pretraining")
            return
        logger.info(f"pre training, train_iteration:{train_iteration}, train_model_weights:{train_model_weights}, train_optimizer:{train_optimizer}")
    old_model_state = None if train_model_weights else copy.deepcopy(model.state_dict())
    old_optimizer_state = None if train_optimizer else copy.deepcopy(optimizer.state_dict())
    model.train()
    model.to(device)
    cuda.CudaEnv.optimizer_to(optimizer, device)
    training_index = 0
    while training_index < train_iteration:
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            training_index += 1
            if training_index % 100 == 0:
                logger.info(f"pre training iter {training_index}, loss: {loss_val}")
            if training_index >= train_iteration:
                break
    if old_model_state is not None:
        model.load_state_dict(old_model_state)
        if logger is not None:
            logger.info(f"pre training done, load old model state")
    if old_optimizer_state is not None:
        optimizer.load_state_dict(old_optimizer_state)
        if logger is not None:
            logger.info(f"pre training done, load old optimizer state")
    model.to(device)

def process_file_func(index, runtime_parameter: RuntimeParameters, checkpoint_file_path=None):
    config_file = configuration_file.load_configuration(runtime_parameter.config_file_path)
    checkpoint_content: Optional[Checkpoint] = None
    if checkpoint_file_path is None:
        # normal mode
        start_point, end_point = runtime_parameter.start_and_end_point_for_paths[index]
        start_file_name = os.path.basename(start_point).replace('.model.pt', '')
        if end_point == "origin":
            assert runtime_parameter.work_mode == WorkMode.to_origin
            end_file_name = "origin"
        elif end_point == "inf":
            assert runtime_parameter.work_mode == WorkMode.to_inf
            end_file_name = "inf"
        elif end_point == "mean":
            assert runtime_parameter.work_mode == WorkMode.to_mean
            end_file_name = "mean"
        else:
            end_file_name = os.path.basename(end_point).replace('.model.pt', '')
        """logger"""
        runtime_parameter.task_name = f"{start_file_name}-{end_file_name}"
    else:
        logger.info(f"loading checkpoint from {checkpoint_file_path}")
        checkpoint_content: Checkpoint = torch.load(checkpoint_file_path, weights_only=False)
        runtime_parameter.task_name = checkpoint_content.current_runtime_parameter.task_name

    child_logger = logging.getLogger(f"find_high_accuracy_path.{runtime_parameter.task_name}")
    set_logging(child_logger, runtime_parameter.task_name, log_file_path=os.path.join(runtime_parameter.output_folder_path, "info.log"))
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
    arg_output_folder_path = os.path.join(runtime_parameter.output_folder_path, f"{runtime_parameter.task_name}")
    if os.path.exists(arg_output_folder_path):
        child_logger.warning(f"{arg_output_folder_path} already exists")
    else:
        os.makedirs(arg_output_folder_path)

    """load models"""
    cpu_device = torch.device("cpu")
    if checkpoint_file_path is None:
        # normal mode
        start_model_stat_dict, start_model_name, dataset_name_in_model_state = util.load_model_state_file(start_point)
        if dataset_name_in_model_state is None:
            dataset_name_in_model_state = runtime_parameter.dataset_name
        child_logger.info(f"loading start model at {start_point}")
        if runtime_parameter.dataset_name is not None:
            assert dataset_name_in_model_state == runtime_parameter.dataset_name, "dataset name in cli mismatch the dataset name in model state file"
            current_ml_setup = ml_setup.get_ml_setup_from_config(start_model_name, dataset_type=runtime_parameter.dataset_name, pytorch_preset_version=runtime_parameter.pytorch_preset_version)
        else:
            dataset_name = 'default' if dataset_name_in_model_state is None else dataset_name_in_model_state
            current_ml_setup = ml_setup.get_ml_setup_from_config(start_model_name, dataset_type=dataset_name, pytorch_preset_version=runtime_parameter.pytorch_preset_version)
        runtime_parameter.model_name = current_ml_setup.model_name
        runtime_parameter.dataset_name = current_ml_setup.dataset_name
        child_logger.info(f"find model type is {start_model_name}")

        initial_model_stat = {k: v.detach().clone() for k, v in current_ml_setup.model.state_dict().items()}
        starting_point = {k: v.detach().clone() for k, v in start_model_stat_dict.items()}
        target_model: torch.nn.Module = copy.deepcopy(current_ml_setup.model)
        target_model.load_state_dict(start_model_stat_dict)
        if runtime_parameter.work_mode == WorkMode.to_origin:
            end_model_stat_dict = {k: torch.zeros_like(v) for k, v in start_model_stat_dict.items()}
            child_logger.info(f"work mode: to_origin")
        elif runtime_parameter.work_mode == WorkMode.to_inf:
            end_model_stat_dict = {k: v.detach() * 2 for k, v in start_model_stat_dict.items()}
            child_logger.info(f"work mode: to_inf")
        elif runtime_parameter.work_mode == WorkMode.to_mean:
            end_model_stat_dict = {k: torch.full_like(v, v.float().mean()) for k, v in start_model_stat_dict.items() }
            child_logger.info(f"work mode: to_mean")
        elif runtime_parameter.work_mode == WorkMode.to_certain_model:
            end_model_stat_dict, end_model_name, _ = util.load_model_state_file(end_point)
            child_logger.info(f"work mode: to_certain_model at {end_point}")
            assert end_model_name == start_model_name, f"start({start_model_name}) != end({end_model_name})"
        elif runtime_parameter.work_mode == WorkMode.to_vs:
            runtime_parameter.variance_sphere_model, temp_model_name = util.load_model_state_file(runtime_parameter.variance_sphere_file_path)
            assert temp_model_name == start_model_name, f"start({start_model_name}) != variance sphere({temp_model_name})"
            end_model_stat_dict = util.calculate_layer_wise_projection_to_variance_sphere(start_model_stat_dict, runtime_parameter.variance_sphere_model)
            child_logger.info(f"work mode: to_variance_sphere at {runtime_parameter.variance_sphere_file_path}")
        else:
            raise NotImplemented
        """assert start_model_state_dict != end_model_stat_dict"""
        for key in start_model_stat_dict.keys():
            assert not torch.equal(start_model_stat_dict[key], end_model_stat_dict[key]), f'starting model({start_point}) is same as ending model({end_point})'
            break
    else:
        # load from checkpoint
        assert checkpoint_content is not None
        start_model_stat_dict = checkpoint_content.current_model_stat
        initial_model_stat = checkpoint_content.init_model_stat
        starting_point = checkpoint_content.start_model_stat
        end_model_stat_dict = checkpoint_content.end_model_stat
        start_model_name = checkpoint_content.current_runtime_parameter.model_name
        dataset_name = checkpoint_content.current_runtime_parameter.dataset_name

        # update states
        runtime_parameter.work_mode = checkpoint_content.current_runtime_parameter.work_mode
        runtime_parameter.dataset_name = checkpoint_content.current_runtime_parameter.dataset_name
        runtime_parameter.model_name = checkpoint_content.current_runtime_parameter.model_name
        # update save states
        runtime_parameter.save_format = checkpoint_content.current_runtime_parameter.save_format
        runtime_parameter.save_interval = checkpoint_content.current_runtime_parameter.save_interval
        runtime_parameter.save_ticks = checkpoint_content.current_runtime_parameter.save_ticks

        current_ml_setup = ml_setup.get_ml_setup_from_config(start_model_name, dataset_type=dataset_name, pytorch_preset_version=runtime_parameter.pytorch_preset_version)
        target_model : torch.nn.Module = copy.deepcopy(current_ml_setup.model)
        target_model.load_state_dict(start_model_stat_dict)

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
    train_collate_fn = default_collate if current_ml_setup.collate_fn is None else current_ml_setup.collate_fn
    dataloader_worker = 0 if general_parameter.dataloader_worker is None else general_parameter.dataloader_worker
    dataloader_prefetch_factor = 4 if general_parameter.dataloader_prefetch_factor is None else general_parameter.dataloader_prefetch_factor
    persistent_workers = False if dataloader_worker == 0 else True
    sampler_fn = None if current_ml_setup.sampler_fn is None else current_ml_setup.sampler_fn(training_dataset)
    dataloader = DataLoader(training_dataset, batch_size=current_ml_setup.training_batch_size, shuffle=True if sampler_fn is None else None,
                            pin_memory=True, num_workers=dataloader_worker, persistent_workers=persistent_workers,
                            collate_fn=train_collate_fn, sampler=sampler_fn, prefetch_factor=dataloader_prefetch_factor)
    criterion = current_ml_setup.criterion

    """get optimizer"""
    target_model.to(device)
    optimizer = config_file.get_optimizer_train(runtime_parameter, current_ml_setup, target_model.parameters())
    initial_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
    optimizer_for_rebuild_norm = None
    assert optimizer is not None
    if checkpoint_file_path is not None:
        optimizer.load_state_dict(checkpoint_content.current_optimizer_stat)

    """update parameters"""
    parameter_train: ParameterTrain = config_file.get_parameter_train(runtime_parameter, current_ml_setup)
    parameter_train.fill_default()
    parameter_train.validate()
    parameter_move: ParameterMove = config_file.get_parameter_move(runtime_parameter, current_ml_setup)
    parameter_move.fill_default()
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
            record_model_service = record_model_stat.ModelStatRecorder(sys.maxsize, current_ml_setup.model_name, current_ml_setup.dataset_name)  # the interval here has no effect
        else:
            child_logger.info("record_model_service is ON at certain ticks")
            record_model_service = record_model_stat.ModelStatRecorder(sys.maxsize, current_ml_setup.model_name, current_ml_setup.dataset_name)  # we don't set the interval here because only certain ticks should be recorded
        record_model_service.initialize_without_runtime_parameters([0], arg_output_folder_path, save_format=runtime_parameter.save_format)
    else:
        child_logger.info("record_model_service is OFF")
        record_model_service = None
    child_logger.info("setting service done: record_model_service")

    record_test_accuracy_loss_service = record_test_accuracy_loss.ServiceTestAccuracyLossRecorder(runtime_parameter.service_test_accuracy_loss_interval,
                                                                                                  runtime_parameter.service_test_accuracy_loss_batch_size,
                                                                                                  current_ml_setup.model_name,
                                                                                                  current_ml_setup.dataset_name,
                                                                                                  store_top_accuracy_model_count=runtime_parameter.store_top_accuracy_model_count,
                                                                                                  use_fixed_testing_dataset=True,
                                                                                                  test_whole_dataset=runtime_parameter.test_dataset_use_whole)
    record_test_accuracy_loss_service.initialize_without_runtime_parameters(arg_output_folder_path, [0], target_model, criterion, current_ml_setup.testing_data,
                                                                            existing_model_for_testing=target_model, gpu=gpu, num_workers=general_parameter.dataloader_worker)
    child_logger.info("setting service done: record_test_accuracy_loss_service")

    record_training_loss_service = record_training_loss_accuracy.ServiceTrainingLossAccuracyRecorder(1)
    record_training_loss_service.initialize_without_runtime_parameters(arg_output_folder_path, [0])
    child_logger.info("setting service done: record_training_loss_service")

    record_consecutive_points_service = record_consecutive_linear_interpolation.ServiceConsecutiveLinearInterpolationRecorder(1, runtime_parameter.service_test_accuracy_loss_batch_size,
                                                                                                                              runtime_parameter.linear_interpolation_dataset_size,
                                                                                                                              runtime_parameter.linear_interpolation_points_size, 0)
    record_consecutive_points_service.initialize_without_runtime_parameters(arg_output_folder_path, target_model, criterion, current_ml_setup.training_data,
                                                                            existing_model_for_testing=None, gpu=gpu, num_workers=general_parameter.dataloader_worker)
    child_logger.info("setting service done: record_consecutive_points_service")

    """record variance"""
    variance_record = model_variance_correct.VarianceCorrector(model_variance_correct.VarianceCorrectionType.FollowOthers)
    variance_record.add_variance(starting_point)
    target_variance = variance_record.get_variance()

    """load checkpoint file"""
    re_init_norm_layer_list = False
    if checkpoint_file_path is not None:
        re_init_norm_layer_list = True
        checkpoint_folder_path = os.path.dirname(checkpoint_file_path)
        runtime_parameter.current_tick = checkpoint_content.current_runtime_parameter.current_tick

        # restore service
        if weight_diff_service is not None:
            weight_diff_service.continue_from_checkpoint(checkpoint_folder_path, runtime_parameter.current_tick)
        if weight_change_service is not None:
            weight_change_service.continue_from_checkpoint(checkpoint_folder_path, runtime_parameter.current_tick)
        if distance_to_origin_service is not None:
            distance_to_origin_service.continue_from_checkpoint(checkpoint_folder_path, runtime_parameter.current_tick)
        if variance_service is not None:
            variance_service.continue_from_checkpoint(checkpoint_folder_path, runtime_parameter.current_tick)
        if record_model_service is not None:
            record_model_service.continue_from_checkpoint(checkpoint_folder_path, runtime_parameter.current_tick)
        if record_test_accuracy_loss_service is not None:
            record_test_accuracy_loss_service.continue_from_checkpoint(checkpoint_folder_path, runtime_parameter.current_tick)
        if record_training_loss_service is not None:
            record_training_loss_service.continue_from_checkpoint(checkpoint_folder_path, runtime_parameter.current_tick)

    """begin finding the path"""
    if runtime_parameter.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    ignore_move_layers = None
    cuda.CudaEnv.model_state_dict_to(end_model_stat_dict, device)
    timer = time.time()

    latest_check_point_file_path = None
    norm_layer_names = None
    compensate_move_layer = None
    compensate_movex2_layer = None
    attention_layer = None
    ignore_move_layers = None
    ratio_step_size = None
    while runtime_parameter.current_tick < runtime_parameter.max_tick:
        parameter_updated = False
        child_logger.info(f"tick: {runtime_parameter.current_tick}")

        """save checkpoint file"""
        if runtime_parameter.current_tick % runtime_parameter.checkpoint_interval == 0:
            if not runtime_parameter.debug_check_config_mode:
                checkpoint_file_path = os.path.join(arg_output_folder_path, f"checkpoint_{runtime_parameter.current_tick}.checkpoint.pt")
                if latest_check_point_file_path is not None:
                    os.remove(latest_check_point_file_path)
                latest_check_point_file_path = checkpoint_file_path
                child_logger.info(f"save checkpoint file to {checkpoint_file_path}")
                check_point_file = Checkpoint()
                check_point_file.current_model_stat = target_model.state_dict()
                check_point_file.current_optimizer_stat = optimizer.state_dict()
                check_point_file.current_runtime_parameter = runtime_parameter
                check_point_file.current_general_parameter = general_parameter
                check_point_file.current_move_parameter = parameter_move
                check_point_file.current_train_parameter = parameter_train
                check_point_file.current_rebuild_norm_parameter = parameter_rebuild_norm
                check_point_file.start_model_stat = starting_point
                check_point_file.end_model_stat = end_model_stat_dict
                check_point_file.init_model_stat = initial_model_stat
                torch.save(check_point_file, checkpoint_file_path)

        """update end_model_stat_dict if work_mode = to_inf"""
        if runtime_parameter.work_mode == WorkMode.to_inf:
            target_model_stat = target_model.state_dict()
            end_model_stat_dict = {k: v.detach().clone() * 2 for k, v in target_model_stat.items()}
            cuda.CudaEnv.model_state_dict_to(end_model_stat_dict, device)
        if runtime_parameter.work_mode == WorkMode.to_vs:
            target_model_stat = target_model.state_dict()
            end_model_stat_dict = util.calculate_layer_wise_projection_to_variance_sphere(target_model_stat, runtime_parameter.variance_sphere_model)
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
            new_parameter_train.fill_default()
            new_parameter_train.validate()
            parameter_train = new_parameter_train
        new_parameter_move: ParameterMove = config_file.get_parameter_move(runtime_parameter, current_ml_setup)

        if new_parameter_move is not None:
            parameter_updated = True
            re_init_norm_layer_list = True
            new_parameter_move.fill_default()
            new_parameter_move.validate()
            parameter_move = new_parameter_move

            # calculate ratio step size
            if parameter_move.ratio_step_size is not None:
                ratio_step_size = {}
                current_model_stat = target_model.state_dict()
                for layer_name, current_layer_weights in current_model_stat.items():
                    distance = util.geodesic_distance(current_layer_weights, end_model_stat_dict[layer_name])
                    if distance is None:
                        continue
                    ratio_step_size[layer_name] = distance.item() * parameter_move.ratio_step_size
            else:
                ratio_step_size = None

        """ re init ignore moving layer list """
        if re_init_norm_layer_list:
            re_init_norm_layer_list = False
            norm_layer_names = []
            compensate_move_layer = []
            compensate_movex2_layer = []
            attention_layer = []
            ignore_move_layers = []
            norm_layers = special_torch_layers.find_normalization_layers(target_model)
            batch_norm_layer_names, _ = special_torch_layers.find_layers_according_to_name_and_keyword(start_model_stat_dict, [], norm_layers.batch_normalization)
            batch_norm_layer_names.sort()
            layer_norm_layer_names, _ = special_torch_layers.find_layers_according_to_name_and_keyword(start_model_stat_dict, [], norm_layers.layer_normalization)
            layer_norm_layer_names.sort()
            assert len(norm_layers.group_normalization) == 0, "group normalization layers are not supported yet."
            assert len(norm_layers.instance_normalization) == 0, "instance normalization layers are not supported yet."

            norm_layer_names.extend(batch_norm_layer_names)
            norm_layer_names.extend(layer_norm_layer_names)

            ignore_move_layers_from_config, _ = special_torch_layers.find_layers_according_to_name_and_keyword(start_model_stat_dict, parameter_move.layer_skip_move, parameter_move.layer_skip_move_keyword)
            ignore_move_layers.extend(ignore_move_layers_from_config)
            layer_compensate_x2, _ = special_torch_layers.find_layers_according_to_name_and_keyword(start_model_stat_dict, parameter_move.layer_compensate_x2, parameter_move.layer_compensate_x2_keyword)
            child_logger.info(f"updating layers to move at tick {runtime_parameter.current_tick}")

            # attention layers
            attention_layer, _ = special_torch_layers.find_layers_according_to_name_and_keyword(start_model_stat_dict, parameter_move.layer_attention, parameter_move.layer_attention_keyword)
            attention_layer.sort()
            if len(attention_layer) > 0:
                child_logger.info(f"found {len(attention_layer)} policy- attention layers (policy: {parameter_move.layer_attention_policy}): {attention_layer}")
            else:
                child_logger.info(f"found no policy attention layers (policy: {parameter_move.layer_attention_policy}))")

            if runtime_parameter.work_mode in [WorkMode.to_inf, WorkMode.to_mean, WorkMode.to_origin]:
                child_logger.info(f"layer norm layers added to compensate moving layer list (found by built-in norm layer detector)[{len(layer_norm_layer_names)} layers]: {layer_norm_layer_names}")
                for n in layer_compensate_x2:
                    assert n in layer_norm_layer_names, f"{n} is not a layer norm layer."
                compensate_move_layer.extend(layer_norm_layer_names) # we should move layer norm to compensate
                compensate_movex2_layer.extend(layer_compensate_x2)
                ignore_move_layers.extend(layer_norm_layer_names) # we do not move layer norm towards the destination direction

                child_logger.info(f"batch norm layers added to ignore moving layer list (found by built-in norm layer detector)[{len(batch_norm_layer_names)} layers]: {batch_norm_layer_names}")
                ignore_move_layers.extend(batch_norm_layer_names)

                compensate_move_layer = list(set(compensate_move_layer))
                compensate_move_layer.sort()
                compensate_movex2_layer = list(set(compensate_movex2_layer))
                compensate_movex2_layer.sort()
                ignore_move_layers = list(set(ignore_move_layers))
                ignore_move_layers.sort()

            # remove layers if mentioned in ignore_move_layers
            compensate_move_layer = list(set(compensate_move_layer) - set(ignore_move_layers_from_config))
            compensate_move_layer.sort()
            compensate_movex2_layer = list(set(compensate_movex2_layer) - set(ignore_move_layers_from_config))
            compensate_movex2_layer.sort()

            child_logger.info(f"ignore moving {len(ignore_move_layers)} layers: {ignore_move_layers}")
            child_logger.info(f"compensate moving {len(compensate_move_layer)} layers: {compensate_move_layer}")
            child_logger.info(f"compensate moving x2 {len(compensate_movex2_layer)} layers: {compensate_movex2_layer}")
            moved_layers = list(set(start_model_stat_dict.keys()) - set(ignore_move_layers) - set(compensate_move_layer) - set(compensate_movex2_layer))
            moved_layers.sort()
            child_logger.info(f"plan to move {len(moved_layers)} layers: {moved_layers}")
            if not runtime_parameter.silence_mode:
                input("Please check above information and press Enter to continue, or press Ctrl+C to quit")

        new_parameter_rebuild_norm: ParameterRebuildNorm = config_file.get_parameter_rebuild_norm(runtime_parameter, current_ml_setup)
        if (new_parameter_rebuild_norm is not None):
            parameter_updated = True
            child_logger.info(f"update parameter (rebuild_norm) at tick {runtime_parameter.current_tick}")
            parameter_rebuild_norm = new_parameter_rebuild_norm
            # update norm layer list
            child_logger.info(f"updating norm layers list at tick {runtime_parameter.current_tick}")
            if ENABLE_REBUILD_NORM and parameter_rebuild_norm.rebuild_norm_for_max_rounds != 0:
                extra_norm_layers, _ = special_torch_layers.find_layers_according_to_name_and_keyword(start_model_stat_dict, parameter_rebuild_norm.rebuild_norm_layer, parameter_rebuild_norm.rebuild_norm_layer_keyword)
                norm_layer_names.extend(extra_norm_layers)
                norm_layer_names = list(set(norm_layer_names))
                norm_layer_names.sort()
                child_logger.info(f"totally {len(norm_layer_names)} layers to rebuild: {norm_layer_names}")
                if not runtime_parameter.silence_mode:
                    input("Please check above information and press Enter to continue, or press Ctrl+C to quit")

        """if this is the first tick"""
        if runtime_parameter.current_tick == 0:
            if parameter_train.pretrain_optimizer and parameter_train.load_existing_optimizer:
                logger.critical("cannot enable both pretrain_optimizer and load_existing_optimizer")
            # pretrain optimizer
            if not runtime_parameter.debug_check_config_mode:
                if parameter_train.pretrain_optimizer or parameter_train.pretrain_optimizer:
                    pre_train(target_model, optimizer, criterion, dataloader, device, train_iteration=parameter_train.pretrain_iterations, logger=child_logger,
                              train_optimizer=parameter_train.pretrain_optimizer, train_model_weights=parameter_train.pretrain_model_weights)
            # load existing optimizer
            if not runtime_parameter.debug_check_config_mode:
                if parameter_train.load_existing_optimizer:
                    optimizer_state_dict_path = start_point.replace('model.pt', 'optimizer.pt')
                    load_existing_optimizer_stat(optimizer, optimizer_state_dict_path, logger=child_logger)

        """if parameter updated"""
        if parameter_updated:
            util.save_model_state(os.path.join(arg_output_folder_path, f"{runtime_parameter.current_tick}.model.pt"), target_model.state_dict(), current_ml_setup.model_name, current_ml_setup.dataset_name)
            util.save_optimizer_state(os.path.join(arg_output_folder_path, f"{runtime_parameter.current_tick}.optimizer.pt"), optimizer.state_dict(), current_ml_setup.model_name, current_ml_setup.dataset_name)

        """service"""
        run_service = True
        if runtime_parameter.debug_check_config_mode:
            run_service = runtime_parameter.current_tick % 1000 == 0
        if run_service:
            record_consecutive_points_service.trigger_without_runtime_parameters(runtime_parameter.current_tick, SimulationPhase.START_OF_TICK, target_model.state_dict())

        """move model"""
        if not runtime_parameter.debug_check_config_mode:
            # store attention layer weights
            old_attention_layer_weights = {}
            for layer_name, weights in target_model.state_dict().items():
                if layer_name in attention_layer:
                    old_attention_layer_weights[layer_name] = weights.detach().clone()

            target_model_stat_dict = model_average.move_model_state_toward(target_model.state_dict(), end_model_stat_dict,
                                                                       parameter_move.step_size, parameter_move.adoptive_step_size,
                                                                       enable_merge_bias_with_weight=parameter_move.merge_bias_with_weights,
                                                                       ignore_layers=ignore_move_layers, ratio_step_per_layer=ratio_step_size) # move towards destination
            compensate_end_model_stat_dict = {k: v.detach().clone() * 2 - end_model_stat_dict[k] for k, v in target_model.state_dict().items()}
            if len(compensate_move_layer) > 0:
                if ratio_step_size is not None:
                    child_logger.warning(f"using ratio step size in layer compensate might cause wrong results.")
                target_model_stat_dict = model_average.move_model_state_toward(target_model_stat_dict, compensate_end_model_stat_dict,
                                                                               parameter_move.step_size, parameter_move.adoptive_step_size,
                                                                               enable_merge_bias_with_weight=parameter_move.merge_bias_with_weights,
                                                                               move_layer=compensate_move_layer, ratio_step_per_layer=ratio_step_size)
            if len(compensate_movex2_layer) > 0:
                target_model_stat_dict = model_average.move_model_state_toward(target_model_stat_dict, compensate_end_model_stat_dict,
                                                                               parameter_move.step_size, parameter_move.adoptive_step_size,
                                                                               enable_merge_bias_with_weight=parameter_move.merge_bias_with_weights,
                                                                               move_layer=compensate_movex2_layer, ratio_step_per_layer=ratio_step_size)
            # attention layer: apply policy
            if len(attention_layer) > 0 and parameter_move.layer_attention_policy != 'none':
                if parameter_move.layer_attention_policy == 'ignore_kv':
                    def copy_qk_from_state_dict(source_sd: dict, target_sd: dict,
                            qkv_weight_key: str, qkv_bias_key: str = None):
                        """
                        Copy query and key weights (and optionally biases) from source_sd to target_sd.

                        Args:
                            source_sd (dict): state_dict of the source model/layer
                            target_sd (dict): state_dict of the target model/layer (to modify in place)
                            qkv_weight_key (str): key for qkv.weight (e.g., 'blocks.0.self_attn.qkv.weight')
                            qkv_bias_key (str): optional key for qkv.bias
                        """
                        # Get weights
                        W_src = source_sd[qkv_weight_key]  # shape: [3m, m]
                        W_tgt = target_sd[qkv_weight_key]

                        assert W_src.shape == W_tgt.shape, "Weight shape mismatch"

                        # Split Q, K, V
                        src_q, src_k, src_v = W_src.chunk(3, dim=0)
                        _, _, tgt_v = W_tgt.chunk(3, dim=0)

                        # Replace Q and K, keep V from target
                        new_weight = torch.cat([src_q, src_k, tgt_v], dim=0)
                        target_sd[qkv_weight_key] = new_weight

                        # Optionally handle bias
                        if qkv_bias_key and qkv_bias_key in source_sd and qkv_bias_key in target_sd:
                            B_src = source_sd[qkv_bias_key]
                            B_tgt = target_sd[qkv_bias_key]

                            assert B_src.shape == B_tgt.shape, "Bias shape mismatch"

                            src_qb, src_kb, src_vb = B_src.chunk(3, dim=0)
                            _, _, tgt_vb = B_tgt.chunk(3, dim=0)

                            new_bias = torch.cat([src_qb, src_kb, tgt_vb], dim=0)
                            target_sd[qkv_bias_key] = new_bias

                    for target_layer_name in attention_layer:
                        copy_qk_from_state_dict(old_attention_layer_weights, target_model_stat_dict, target_layer_name)
                else:
                    raise NotImplementedError

            target_model.load_state_dict(target_model_stat_dict)

        """variance correction"""
        if not runtime_parameter.debug_check_config_mode:
            if runtime_parameter.work_mode == WorkMode.to_certain_model:
                child_logger.info(f"current tick: {runtime_parameter.current_tick}, rescale variance")
                target_model_stat_dict = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(target_model.state_dict(), target_variance,
                                                                                                               ignore_layer_list=ignore_move_layers)
                target_model.load_state_dict(target_model_stat_dict)

        """update learning rate"""
        if not runtime_parameter.debug_check_config_mode:
            if runtime_parameter.work_mode in [WorkMode.to_inf, WorkMode.to_origin, WorkMode.to_mean]:
                for param_group, initial_optimizer_state, (name, param) in zip(optimizer.param_groups, initial_optimizer_state_dict['param_groups'], target_model.named_parameters()):
                    if name in norm_layer_names:
                        child_logger.info(f"tick {runtime_parameter.current_tick}: [keep] lr for layer {name} (norm layer): {initial_optimizer_state['lr']:.3E}")
                        continue  # skip norm layers
                    if 'weight' in name and param.requires_grad:  # Only adjust weights, not biases
                    # if param.requires_grad:
                        current_layer_variance = torch.var(param.data).item()
                        if runtime_parameter.across_vs_lr_policy == 'var':
                            new_lr = initial_optimizer_state['lr'] * current_layer_variance / target_variance[name]
                        elif runtime_parameter.across_vs_lr_policy == 'std':
                            new_lr = initial_optimizer_state['lr'] * numpy.sqrt(current_layer_variance / target_variance[name])
                        else:
                            raise NotImplementedError
                        child_logger.info(f"tick {runtime_parameter.current_tick}: update lr for layer {name} to {new_lr:.3E}")
                        param_group['lr'] = new_lr

        """training"""
        training_loss_val = 0
        training_accuracy_val = 0.0
        if not runtime_parameter.debug_check_config_mode:
            target_model.train()
            target_model.to(device)
            cuda.CudaEnv.optimizer_to(optimizer, device)

            training_iter_counter = 0
            moving_average = util.MovingAverage(parameter_train.train_for_min_rounds)
            while training_iter_counter < parameter_train.train_for_max_rounds:
                exit_training = False
                training_correct_val = 0
                training_total_val = 0
                for data, label in dataloader:
                    training_iter_counter += 1
                    data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    if runtime_parameter.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = target_model(data)
                            training_loss = criterion(outputs, label)
                            scaler.scale(training_loss).backward()
                            if current_ml_setup.clip_grad_norm is not None:
                                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                                scaler.unscale_(optimizer)
                                nn.utils.clip_grad_norm_(target_model.parameters(), current_ml_setup.clip_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        outputs = target_model(data)
                        training_loss = criterion(outputs, label)
                        training_loss.backward()
                        if current_ml_setup.clip_grad_norm is not None:
                            nn.utils.clip_grad_norm_(target_model.parameters(), current_ml_setup.clip_grad_norm)
                        optimizer.step()
                    _, predicted = torch.max(outputs, 1)
                    training_loss_val = training_loss.item()
                    moving_average.add(training_loss_val)
                    training_correct_val += (predicted == label).sum().item()
                    training_total_val += label.size(0)

                    if runtime_parameter.verbose:
                        if training_iter_counter % 10 == 0:
                            child_logger.info(f"current tick: {runtime_parameter.current_tick}, training {training_iter_counter} rounds, loss = {moving_average.get_average():.3f}")
                    if runtime_parameter.current_tick == 0 and training_loss_val > parameter_train.train_until_loss:
                        child_logger.warning(f"the loss for the first batch is larger than train_until_loss, ({training_loss_val}>{parameter_train.train_until_loss}). Check dataset selection !!!")
                    if training_iter_counter >= parameter_train.train_for_max_rounds:
                        exit_training = True
                        break
                    if moving_average.get_average() <= parameter_train.train_until_loss and training_iter_counter >= parameter_train.train_for_min_rounds:
                        exit_training = True
                        break
                training_accuracy_val = training_correct_val / training_total_val
                if exit_training:
                    child_logger.info(f"current tick: {runtime_parameter.current_tick}, training {training_iter_counter} rounds(final), loss = {moving_average.get_average():.3f}")
                    # training_loss_val = moving_average.get_average()
                    break

        """rebuilding normalization"""
        if not runtime_parameter.debug_check_config_mode:
            if ENABLE_REBUILD_NORM and (parameter_rebuild_norm.rebuild_norm_for_max_rounds != 0):
                optimizer_for_rebuild_norm_new = config_file.get_optimizer_rebuild_norm(runtime_parameter, current_ml_setup, target_model.parameters())
                if optimizer_for_rebuild_norm_new is not None:
                    optimizer_for_rebuild_norm = optimizer_for_rebuild_norm_new
                training_optimizer_stat = optimizer.state_dict()
                rebuild_norm_layer_function(target_model, initial_model_stat, start_model_stat_dict, optimizer_for_rebuild_norm, training_optimizer_stat,
                                            norm_layer_names, current_ml_setup, dataloader, parameter_rebuild_norm, runtime_parameter, device, logger=child_logger)

        """variance correction"""
        if not runtime_parameter.debug_check_config_mode:
            if runtime_parameter.work_mode == WorkMode.to_certain_model:
                child_logger.info(f"current tick: {runtime_parameter.current_tick}, rescale variance")
                target_model_stat_dict = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(target_model.state_dict(), target_variance, ignore_layer_list=ignore_move_layers)
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
            record_training_loss_service.trigger_without_runtime_parameters(runtime_parameter.current_tick, {0: training_loss_val}, {0: training_accuracy_val})
            record_consecutive_points_service.trigger_without_runtime_parameters(runtime_parameter.current_tick, SimulationPhase.END_OF_TICK, target_model.state_dict())
        # update tick
        runtime_parameter.current_tick += 1

    # save final model and optimizer
    util.save_model_state(os.path.join(arg_output_folder_path, f"{runtime_parameter.current_tick}.model.pt"), target_model.state_dict(), current_ml_setup.model_name, current_ml_setup.dataset_name)
    util.save_optimizer_state(os.path.join(arg_output_folder_path, f"{runtime_parameter.current_tick}.optimizer.pt"), optimizer.state_dict(), current_ml_setup.model_name, current_ml_setup.dataset_name)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Move the model towards certain direction and keep its accuracy')
    parser.add_argument("start_folder", nargs='?', type=str, help="folder containing starting models")

    parser.add_argument("end_folder", nargs='?', type=str, help="folder containing destination models, or 'inf', 'origin', 'mean', 'to_vs' ")

    parser.add_argument("--variance_sphere", type=str, help="specify the variance sphere by giving a model state, use when 'end_folder' is 'to_vs'")
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
    parser.add_argument("--store_top_accuracy_model_count", type=int, default=5, help='save n highest test accuracy models')
    parser.add_argument("--checkpoint_interval", type=int, default=100, help='save a checkpoint every n ticks')
    parser.add_argument("--continue_from_checkpoint", type=str, help='continue from a checkpoint file')
    parser.add_argument("-A", "--across_vs_lr_policy", type=str, choices=['std', 'var'], default='var', help='set the lr to follow variance or standard derivation of variance sphere')

    parser.add_argument( "--test_interval", type=int, default=1, help='specify the interval of measuring model on the test dataset.')
    parser.add_argument("--test_batch", type=int, default=100, help='specify the batch size of measuring model on the test dataset.')
    parser.add_argument("-P", "--torch_preset_version", type=int, default=None, help='specify the pytorch data training preset version')
    parser.add_argument("-S", "--silence", action='store_true', help='enable silence mode, do not interact with users, all checks will be bypassed')
    parser.add_argument("--linear_interpolation_points_size", type=int, default=0, help='specify the size of linear interpolation points between two consecutive points')
    parser.add_argument("--linear_interpolation_dataset_size", type=int, default=1000, help='specify the size of linear interpolation dataset size')

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
    runtime_parameter.service_test_accuracy_loss_interval = args.test_interval
    runtime_parameter.service_test_accuracy_loss_batch_size = args.test_batch
    runtime_parameter.store_top_accuracy_model_count = args.store_top_accuracy_model_count
    runtime_parameter.checkpoint_interval = args.checkpoint_interval
    runtime_parameter.pytorch_preset_version = args.torch_preset_version
    runtime_parameter.across_vs_lr_policy = args.across_vs_lr_policy
    runtime_parameter.silence_mode = args.silence
    runtime_parameter.linear_interpolation_points_size = args.linear_interpolation_points_size
    runtime_parameter.linear_interpolation_dataset_size = args.linear_interpolation_dataset_size

    # sanity check
    if runtime_parameter.across_vs_lr_policy == 'std':
        if args.end_folder not in ['origin', 'inf', 'mean']:
            raise RuntimeError('setting across_vs_lr_policy is not allowed for moving model towards origin/mean/inf')

    # find all paths to process
    if args.start_folder is not None and args.end_folder is not None:
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
        elif args.end_folder == "mean":
            runtime_parameter.work_mode = WorkMode.to_mean
            assert args.mapping_mode == "auto", "mapping mode has to be 'auto' for move to mean"
            files_in_start_folder = sorted(set(temp_file for temp_file in os.listdir(start_folder) if temp_file.endswith('model.pt')))
            paths_to_find = [(os.path.join(start_folder, i), "mean") for i in files_in_start_folder]
        elif args.end_folder == "to_vs":
            runtime_parameter.work_mode = WorkMode.to_vs
            assert args.variance_sphere is not None, "variance_sphere cannot be None for 'to_vs' mode"
            files_in_start_folder = sorted(set(temp_file for temp_file in os.listdir(start_folder) if temp_file.endswith('model.pt')))
            paths_to_find = [(os.path.join(start_folder, i), "to_vs") for i in files_in_start_folder]
            runtime_parameter.variance_sphere_file_path = args.variance_sphere
        else:
            runtime_parameter.work_mode = WorkMode.to_certain_model
            paths_to_find = get_files_to_process(args.start_folder, args.end_folder, args.mapping_mode)
        paths_to_find_count = len(paths_to_find)
        runtime_parameter.start_and_end_point_for_paths = paths_to_find
        logger.info(f"totally {paths_to_find_count} paths to process: {paths_to_find}")
    elif args.continue_from_checkpoint is not None:
        logger.info(f"continue algorithm based on checkpoint file {args.continue_from_checkpoint}.")
    else:
        logger.critical(f"this script can only be used with providing start/end folders or providing a checkpoint file.")

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

    logger.info(f"final runtime parameters: {runtime_parameter.print()}")

    if not runtime_parameter.use_cpu:
        assert torch.cuda.is_available()
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # worker and cpu cores setting
    runtime_parameter.total_cpu_count = args.core
    runtime_parameter.worker_count = args.worker
    if args.continue_from_checkpoint is not None:
        for index, path in enumerate(paths_to_find):
            process_file_func(index, runtime_parameter, args.continue_from_checkpoint)
    else:
        # normal mode
        if runtime_parameter.worker_count > paths_to_find_count:
            runtime_parameter.worker_count = paths_to_find_count
        logger.info(f"worker: {runtime_parameter.worker_count}")
        if runtime_parameter.worker_count == 1:
            for index, path in enumerate(paths_to_find):
                process_file_func(index, runtime_parameter, None)
        else:
            assert runtime_parameter.silence_mode, "silence_mode must be set for multiple workers"
            with concurrent.futures.ProcessPoolExecutor(max_workers=runtime_parameter.worker_count) as executor:
                futures = [executor.submit(process_file_func, index, runtime_parameter, None) for index, path in enumerate(paths_to_find)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()





