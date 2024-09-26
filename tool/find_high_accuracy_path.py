import os
import argparse
import logging
import shutil
import sys
import json
from enum import Enum
from typing import Final
import torch
import re
import torch.optim as optim
import concurrent.futures
import copy
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, model_average, model_variance_correct, special_torch_layers, cuda, util
from py_src.service import record_weights_difference, record_test_accuracy_loss, record_variance, record_model_stat, record_training_loss

logger = logging.getLogger("find_high_accuracy_path")

INFO_FILE_NAME = 'info.json'

MAX_CPU_COUNT: Final[int] = 32

ENABLE_DEDICATED_TRAINING_DATASET_FOR_REBUILDING_NORM: Final[bool] = False
ENABLE_NAN_CHECKING: Final[bool] = False


# the optimizers to find the pathway points
def get_optimizer_to_find_pathway_point(model_name, model_parameter, dataset, batch_size):
    if model_name == "resnet18_bn":
        epochs = 10
        optimizer = torch.optim.SGD(model_parameter, lr=0.005, momentum=0.9, weight_decay=5e-4)
        steps_per_epoch = len(dataset) // batch_size + 1
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.005, steps_per_epoch=steps_per_epoch, epochs=epochs)
    elif model_name == "simplenet":
        epochs = 40
        optimizer = torch.optim.Adadelta(model_parameter, lr=0.1, rho=0.9, eps=1e-3, weight_decay=0.001)
        steps_per_epoch = len(dataset) // batch_size + 1
        milestones_epoch = [10]
        milestones = [steps_per_epoch * i for i in milestones_epoch]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    else:
        raise NotImplementedError(f"{model_name} not implemented for finding pathway points")
    return epochs, optimizer, lr_scheduler

def get_optimizer_to_find_high_accuracy_path(model_name, model_parameter, train_lr, dataset, batch_size):
    if model_name == "lenet":
        optimizer = torch.optim.SGD(model_parameter, lr=train_lr)
    elif model_name == "resnet18_bn":
        optimizer = torch.optim.SGD(model_parameter, lr=train_lr)
    elif model_name == "simplenet":
        optimizer = torch.optim.Adadelta(model_parameter, lr=train_lr, rho=0.9, eps=1e-3, weight_decay=0.001)
    else:
        raise NotImplementedError(f"{model_name} not implemented for finding high accuracy path")
    return optimizer

def get_optimizer_to_rebuild_norm(model_name, model_parameter, lr, dataset, batch_size):
    if model_name == "resnet18_bn":
        optimizer = torch.optim.SGD(model_parameter, lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise NotImplementedError(f"{model_name} not implemented for rebuilding norm")
    return optimizer


class TrainMode(Enum):
    default_x_rounds = 0
    Adam_until_loss = 1
    default_until_loss = 2


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


# rebuild_norm_layers=None indicates rebuild all norm layers, when specified, then only rebuild norm for these layers

def rebuild_norm_layers(model, model_state, arg_ml_setup, epoch_of_rebuild, dataloader, rebuild_norm_round,
                        rebuild_lr=0.001, existing_optimizer=None, rebuild_on_device=None,
                        reset_norm_to_initial=False, initial_model_stat=None, display=False, rebuild_norm_layers_override=None):
    global __print_norm_to_rebuild_layers
    if rebuild_norm_layers_override is None:
        rebuild_norm_layers_override = []

    def is_processing_this_layer(model_name, layer_name, rebuild_norm_layers):
        if len(rebuild_norm_layers) == 0:
            if special_torch_layers.is_normalization_layer(model_name, layer_name):
                return True
            else:
                return False
        else:
            process_this_layer = False
            for single_layer in rebuild_norm_layers:
                if layer_name in single_layer:
                    process_this_layer = True
                    break
            return process_this_layer


    # get the list of layers to rebuild norm
    rebuild_norm_layers_to_process = []
    for layer_name, layer_weights in model_state.items():
        if is_processing_this_layer(arg_ml_setup.model_name, layer_name, rebuild_norm_layers_override):
            rebuild_norm_layers_to_process.append(layer_name)

    # reset target layers to initial state?
    if reset_norm_to_initial:
        assert initial_model_stat is not None
        # reset normalization layers
        for layer_name, layer_weights in model_state.items():
            if layer_name in rebuild_norm_layers_to_process:
                model_state[layer_name] = initial_model_stat[layer_name]

    model.load_state_dict(model_state)

    if rebuild_on_device is None:
        rebuild_on_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    model.train()
    model.to(rebuild_on_device)

    criterion = arg_ml_setup.criterion
    if existing_optimizer is None:
        optimizer_rebuild_norm = torch.optim.Adam(model.parameters(), lr=rebuild_lr)
    else:
        optimizer_rebuild_norm = existing_optimizer
    cuda.CudaEnv.optimizer_to(optimizer_rebuild_norm, rebuild_on_device)
    rebuild_states = []
    rebuilding_normalization_count = 0
    for epoch in range(epoch_of_rebuild):
        exit_flag = False
        for (rebuilding_normalization_index, (data, label)) in enumerate(dataloader):
            data, label = data.to(rebuild_on_device), label.to(rebuild_on_device)
            optimizer_rebuild_norm.zero_grad(set_to_none=True)
            output = model(data)
            rebuilding_loss = criterion(output, label)
            rebuilding_loss.backward()
            rebuilding_loss_val = rebuilding_loss.item()
            optimizer_rebuild_norm.step()
            # reset all layers except normalization
            current_model_stat = model.state_dict()
            for layer_name, layer_weights in current_model_stat.items():
                if layer_name not in rebuild_norm_layers_to_process:
                    current_model_stat[layer_name] = model_state[layer_name]
            model.load_state_dict(current_model_stat)
            rebuilding_normalization_count += 1
            rebuild_states.append((rebuilding_normalization_count, rebuilding_loss_val))
            if display:
                print(f"tick: {rebuilding_normalization_count}  loss: {rebuilding_loss_val}")
            if rebuilding_normalization_count >= rebuild_norm_round:
                exit_flag = True
                break
        if exit_flag:
            break
    output_model_state = model.state_dict()
    cuda.CudaEnv.model_state_dict_to(output_model_state, cpu_device)
    return output_model_state, rebuild_states, rebuild_norm_layers_to_process


__print_norm_to_rebuild_layers = True
def process_file_func(arg_env, arg_training_parameters, arg_average, arg_rebuild_norm, arg_pathway, arg_compute):
    global __print_norm_to_rebuild_layers

    arg_output_folder_path, start_model_path, end_model_path, arg_ml_setup, arg_max_tick = arg_env
    arg_rebuild_norm_lr, arg_rebuild_norm_round, arg_rebuild_norm_specified_layers = arg_rebuild_norm
    arg_step_size, arg_adoptive_step_size, arg_layer_skip_average, arg_layer_skip_average_keyword = arg_average
    arg_path_way_depth, arg_existing_pathway = arg_pathway
    arg_worker_count, arg_total_cpu_count, arg_save_format, arg_save_ticks, arg_use_cpu = arg_compute

    training_mode, training_parameter = arg_training_parameters
    if training_mode == TrainMode.default_x_rounds:
        train_lr, train_round = training_parameter
    elif training_mode == TrainMode.Adam_until_loss or training_mode == TrainMode.default_until_loss:
        target_train_loss = training_parameter
    else:
        raise NotImplementedError
    del training_parameter

    start_file_name = os.path.basename(start_model_path).replace('.model.pt', '')
    end_file_name = os.path.basename(end_model_path).replace('.model.pt', '')

    # logger
    task_name = f"{start_file_name}-{end_file_name}"
    child_logger = logging.getLogger(f"find_high_accuracy_path.{task_name}")
    set_logging(child_logger, task_name, log_file_path=os.path.join(arg_output_folder_path, "info.log"))
    child_logger.info("logging setup complete")

    if arg_use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    thread_per_process = arg_total_cpu_count // arg_worker_count
    torch.set_num_threads(thread_per_process)

    # check optimizer
    start_optimizer_path = start_model_path.replace('model.pt', 'optimizer.pt')
    assert os.path.exists(start_optimizer_path), f'starting optimizer {start_optimizer_path} is missing'

    # output folders
    arg_output_folder_path = os.path.join(arg_output_folder_path, f"{start_file_name}-{end_file_name}")
    if os.path.exists(arg_output_folder_path):
        child_logger.warning(f"{arg_output_folder_path} already exists")
    else:
        os.makedirs(arg_output_folder_path)

    # load models
    cpu_device = torch.device("cpu")
    start_model = copy.deepcopy(arg_ml_setup.model)
    initial_model_stat = start_model.state_dict()
    start_model_stat_dict, start_model_name = util.load_model_state_file(start_model_path)
    end_model_stat_dict, end_model_name = util.load_model_state_file(end_model_path)
    util.assert_if_both_not_none(start_model_name, end_model_name)

    start_model.load_state_dict(start_model_stat_dict)
    start_model_optimizer_stat, _ = util.load_optimizer_state_file(start_optimizer_path)

    # assert start_model_state_dict != end_model_stat_dict
    for key in start_model_stat_dict.keys():
        assert not torch.equal(start_model_stat_dict[key], end_model_stat_dict[key]), f'starting model({start_model_path}) is same as ending model({end_model_path})'
        break

    # load training data
    training_dataset = arg_ml_setup.training_data
    dataloader = DataLoader(training_dataset, batch_size=arg_ml_setup.training_batch_size, shuffle=True)
    criterion = arg_ml_setup.criterion
    if training_mode == TrainMode.default_x_rounds:
        optimizer = get_optimizer_to_find_high_accuracy_path(arg_ml_setup.model_name, start_model.parameters(), train_lr, training_dataset, arg_ml_setup.training_batch_size)
    elif training_mode == TrainMode.Adam_until_loss:
        optimizer = torch.optim.Adam(start_model.parameters(), lr=0.001)
    elif training_mode == TrainMode.default_until_loss:
        optimizer = get_optimizer_to_find_high_accuracy_path(arg_ml_setup.model_name, start_model.parameters(), 0.001, training_dataset, arg_ml_setup.training_batch_size)
    else:
        raise NotImplementedError

    # try loading optimizer state to optimizer
    optimizer_test = copy.deepcopy(optimizer)
    load_optimizer_success = False
    try:
        optimizer_test.load_state_dict(start_model_optimizer_stat)
        optimizer.load_state_dict(start_model_optimizer_stat)
        child_logger.info(f"successfully load optimizer state")
        load_optimizer_success = True
    except Exception as e:
        child_logger.info(f"fail to load optimizer state: {str(e)}, now skip")
    del optimizer_test

    # set dataloader for rebuilding norm
    if arg_rebuild_norm_round != 0:
        if ENABLE_DEDICATED_TRAINING_DATASET_FOR_REBUILDING_NORM:
            epoch_for_rebuilding_norm = 1
            dataset_for_rebuilding_norm = arg_ml_setup.training_data_for_rebuilding_normalization
            dataset_rebuild_norm_size = arg_rebuild_norm_round * arg_ml_setup.training_batch_size
            if dataset_rebuild_norm_size > len(dataset_for_rebuilding_norm):
                ratio = int(dataset_rebuild_norm_size // len(dataset_for_rebuilding_norm) + 1)
                assert dataset_rebuild_norm_size % ratio == 0
                dataset_rebuild_norm_size = dataset_rebuild_norm_size // ratio
                epoch_for_rebuilding_norm = epoch_for_rebuilding_norm * ratio
            indices = torch.randperm(len(dataset_for_rebuilding_norm))[:dataset_rebuild_norm_size]
            sub_dataset = torch.utils.data.Subset(dataset_for_rebuilding_norm, indices.tolist())
            dataloader_for_rebuilding_norm = DataLoader(sub_dataset, batch_size=arg_ml_setup.training_batch_size)
        else:
            dataloader_for_rebuilding_norm = dataloader  # use the training dataloader
            epoch_for_rebuilding_norm = 1
    else:
        dataloader_for_rebuilding_norm = None
        epoch_for_rebuilding_norm = None

    # ignored layers
    ignore_layers = []
    averaged_layers = []
    for layer_name in start_model_stat_dict.keys():
        if layer_name in arg_layer_skip_average:
            ignore_layers.append(layer_name)
        if special_torch_layers.is_keyword_in_layer_name(layer_name, arg_layer_skip_average_keyword):
            ignore_layers.append(layer_name)
    for layer_name in start_model_stat_dict.keys():
        if layer_name not in ignore_layers:
            averaged_layers.append(layer_name)
    child_logger.info(f"ignore moving {len(ignore_layers)} layers: {ignore_layers}")
    child_logger.info(f"moving {len(averaged_layers)} layers: {averaged_layers}")

    # services
    all_node_names = [0, 1]
    all_model_stats = [start_model_stat_dict, end_model_stat_dict]

    weight_diff_service = record_weights_difference.ServiceWeightsDifferenceRecorder(1)
    weight_diff_service.initialize_without_runtime_parameters(all_model_stats, arg_output_folder_path)
    distance_to_origin_service = record_weights_difference.ServiceDistanceToOriginRecorder(1, [0])
    distance_to_origin_service.initialize_without_runtime_parameters({0: start_model_stat_dict}, arg_output_folder_path)
    variance_service = record_variance.ServiceVarianceRecorder(1)
    variance_service.initialize_without_runtime_parameters([0], [start_model_stat_dict], arg_output_folder_path)
    if arg_save_format != 'none':
        if not arg_save_ticks:
            record_model_service = record_model_stat.ModelStatRecorder(1)
        else:
            record_model_service = record_model_stat.ModelStatRecorder(arg_max_tick)
        record_model_service.initialize_without_runtime_parameters([0], arg_output_folder_path, save_format=arg_save_format)
    else:
        record_model_service = None
    record_test_accuracy_loss_service = record_test_accuracy_loss.ServiceTestAccuracyLossRecorder(1, 100, use_fixed_testing_dataset=True)
    record_test_accuracy_loss_service.initialize_without_runtime_parameters(arg_output_folder_path, [0], start_model, criterion, training_dataset)
    record_training_loss_service = record_training_loss.ServiceTrainingLossRecorder(1)
    record_training_loss_service.initialize_without_runtime_parameters(arg_output_folder_path, [0])

    """find pathway points"""
    optimizer_paths_for_pathway_points = None
    if arg_path_way_depth != 0:
        if arg_existing_pathway is None:
            child_logger.info(f"finding pathway points with depth {arg_path_way_depth}")

            def find_pathway_point_between_two_model(m0, m1, target_model, model_name, fine_tune_dataloader, depth, index):
                # average models
                averaged_model_state = {}
                for layer_name in m0.keys():
                    layer_of_first_model = m0[layer_name]
                    if isinstance(layer_of_first_model, torch.Tensor) and layer_of_first_model.dtype in (torch.float32, torch.float64):
                        averaged_model_state[layer_name] = torch.mean(torch.stack([model[layer_name] for model in [m0, m1]]), dim=0)
                    elif "num_batches_tracked" in layer_name:
                        averaged_model_state[layer_name] = m0[layer_name]
                    else:
                        raise NotImplementedError

                # record variance
                ft_variance_record = model_variance_correct.VarianceCorrector(model_variance_correct.VarianceCorrectionType.FollowOthers)
                ft_variance_record.add_variance(m0)
                ft_variance_record.add_variance(m1)
                ft_target_variance = ft_variance_record.get_variance()

                # fine-tune the model with
                target_model.load_state_dict(averaged_model_state)
                target_model.to(device)
                target_model.train()
                ft_epochs, ft_optimizer, ft_lr_scheduler = get_optimizer_to_find_pathway_point(model_name, target_model.parameters(), fine_tune_dataloader.dataset, arg_ml_setup.training_batch_size)

                for epoch in range(ft_epochs):
                    # scale variance
                    current_model_state = target_model.state_dict()
                    cuda.CudaEnv.model_state_dict_to(current_model_state, cpu_device)
                    current_model_state = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(current_model_state, ft_target_variance)
                    cuda.CudaEnv.model_state_dict_to(current_model_state, device)
                    target_model.load_state_dict(current_model_state)

                    # training
                    train_loss = 0
                    count = 0
                    for data, label in dataloader:
                        data, label = data.to(device), label.to(device)
                        ft_optimizer.zero_grad()
                        outputs = target_model(data)
                        loss = criterion(outputs, label)
                        loss.backward()
                        ft_optimizer.step()
                        if ft_lr_scheduler is not None:
                            ft_lr_scheduler.step()
                        train_loss += loss.item()
                        count += 1
                    lrs = []
                    for param_group in ft_optimizer.param_groups:
                        lrs.append(param_group['lr'])
                    child_logger.info(f"find pathway points depth {depth} index {index}: epoch[{epoch}] loss={train_loss / count} lrs={lrs}")
                output_model_state = target_model.state_dict()
                cuda.CudaEnv.model_state_dict_to(output_model_state, torch.device("cpu"))
                return output_model_state

            def find_pathway_points(m0, m1, target_model, model_name, fine_tune_dataloader, depth):
                current_models = [m0, m1]
                for current_depth in range(1, depth+1):
                    new_models = []
                    for model_index in range(len(current_models) - 1):
                        pathway_model_state = find_pathway_point_between_two_model(current_models[model_index], current_models[model_index+1], target_model, model_name, fine_tune_dataloader, current_depth, model_index)
                        new_models.append(pathway_model_state)
                    new_model_list = [None] * (len(current_models) + len(new_models))
                    new_model_list[::2] = current_models
                    new_model_list[1::2] = new_models
                    current_models = new_model_list
                return current_models

            pathway_points = find_pathway_points(start_model_stat_dict, end_model_stat_dict, start_model, arg_ml_setup.model_name, dataloader, arg_path_way_depth)
            child_logger.info(f"find {len(pathway_points)} pathway points with depth {arg_path_way_depth}")
        else:
            child_logger.info(f"loading existing pathway points from {arg_existing_pathway}")

            def extract_x_y_from_filename(filename):
                match = re.match(r"(\d+)_over_(\d+)\.model\.pt", filename)
                if match:
                    x = int(match.group(1))
                    y = int(match.group(2))
                    return x, y
                return None, None

            model_list = []
            found_files = {}
            for file_name in os.listdir(arg_existing_pathway):
                if file_name.endswith(".model.pt"):
                    x, y = extract_x_y_from_filename(file_name)
                    if x is not None and y is not None:
                        if y not in found_files:
                            found_files[y] = []
                        found_files[y].append(x)
            if found_files:
                max_y = max(found_files.keys())
                expected_files = set(range(max_y + 1))

                if set(found_files[max_y]) == expected_files:
                    child_logger.info(f"all files from 0_over_{max_y}.model.pt to {max_y}_over_{max_y}.model.pt are present.")
                    # Load models into the list
                    for x in range(max_y + 1):
                        file_name = f"{x}_over_{max_y}.model.pt"
                        model_path = os.path.join(arg_existing_pathway, file_name)
                        model_state_dict, _ = util.load_model_state_file(model_path)
                        model_list.append(model_state_dict)
                else:
                    child_logger.critical(f"missing files for {max_y}. Expected {expected_files}, but found {found_files[max_y]}")
            else:
                child_logger.critical("no valid model files found in the folder.")
            for key in start_model_stat_dict.keys():
                assert torch.equal(start_model_stat_dict[key], model_list[0][key])
                assert torch.equal(model_list[-1][key], end_model_stat_dict[key])

            pathway_points = model_list
            assert 2 ** arg_path_way_depth + 1 == len(pathway_points)
            child_logger.info(f"loading and checking pathway points pass")

        # create folders and save pathway points
        output_path_pathway_points = os.path.join(arg_output_folder_path, "pathway_points")
        os.makedirs(output_path_pathway_points, exist_ok=True)
        optimizer_paths_for_pathway_points = []
        for index, pathway_model in enumerate(pathway_points):
            util.save_model_state(os.path.join(output_path_pathway_points, f"{index}_over_{len(pathway_points) - 1}.model.pt"), pathway_model, arg_ml_setup.model_name)
            optimizer_paths_for_pathway_points.append(os.path.join(output_path_pathway_points, f"{index}_over_{len(pathway_points) - 1}.optimizer.pt"))
    else:
        pathway_points = [start_model_stat_dict, end_model_stat_dict]

    start_model_stat = start_model_stat_dict
    target_direction_points = pathway_points[1:]
    current_tick = 0

    """record variance"""
    variance_record = model_variance_correct.VarianceCorrector(model_variance_correct.VarianceCorrectionType.FollowOthers)
    variance_record.add_variance(start_model_stat)
    target_variance = variance_record.get_variance()

    previous_pathway_index = -1
    while current_tick < arg_max_tick:
        """set end point"""
        path_len = arg_max_tick // len(target_direction_points)
        current_path_index = current_tick // path_len
        current_direction_point = target_direction_points[current_path_index]
        if previous_pathway_index != current_path_index:
            # entering a new pathway point region
            previous_pathway_index = current_path_index
            if optimizer_paths_for_pathway_points is not None:
                util.save_optimizer_state(optimizer_paths_for_pathway_points[current_path_index], optimizer.state_dict(), arg_ml_setup.model_name)

        """move tensor"""
        start_model_stat = model_average.move_model_state_toward(start_model_stat, current_direction_point, arg_step_size, arg_adoptive_step_size, False, ignore_layers=ignore_layers, random_scale=1.0)
        if ENABLE_NAN_CHECKING:
            util.check_for_nans_in_state_dict(start_model_stat)
        """rescale variance"""
        start_model_stat = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(start_model_stat, target_variance)
        if ENABLE_NAN_CHECKING:
            util.check_for_nans_in_state_dict(start_model_stat)
        """training"""
        start_model.load_state_dict(start_model_stat)
        training_loss_val = None
        start_model.train()
        start_model.to(device)
        cuda.CudaEnv.optimizer_to(optimizer, device)

        if training_mode == TrainMode.default_x_rounds:
            for (training_index, (data, label)) in enumerate(dataloader):
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = start_model(data)
                training_loss = criterion(output, label)
                training_loss.backward()
                optimizer.step()
                training_loss_val = training_loss.item()
                if training_index == train_round:
                    break
                assert training_index < train_round
        elif training_mode == TrainMode.Adam_until_loss or training_mode == TrainMode.default_until_loss:
            moving_max_size = 2
            moving_max = util.MovingMax(moving_max_size)
            while True:
                exit_training = False
                for (training_index, (data, label)) in enumerate(dataloader):
                    data, label = data.to(device), label.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    output = start_model(data)
                    training_loss = criterion(output, label)
                    training_loss.backward()
                    optimizer.step()
                    training_loss_val = training_loss.item()
                    max_loss = moving_max.add(training_loss_val)
                    child_logger.info(f"current tick: {current_tick}, training loss = {training_loss_val:.3f}, max loss = {max_loss:.3f}")
                    if max_loss < target_train_loss and training_index+1 >= moving_max_size:
                        exit_training = True
                        break
                if exit_training:
                    break
        else:
            raise NotImplementedError

        child_logger.info(f"current tick: {current_tick}, training loss = {training_loss_val:.3f}")

        start_model_stat = start_model.state_dict()

        """rebuilding normalization"""
        if arg_rebuild_norm_round != 0:
            optimizer_rebuild = optimizer
            start_model_stat, rebuild_states, layers_rebuild = rebuild_norm_layers(start_model, start_model_stat, arg_ml_setup, epoch_for_rebuilding_norm,
                                                                   dataloader_for_rebuilding_norm, arg_rebuild_norm_round, rebuild_lr=arg_rebuild_norm_lr,
                                                                   rebuild_on_device=device, existing_optimizer=optimizer_rebuild,
                                                                   initial_model_stat=initial_model_stat,
                                                                   reset_norm_to_initial=True, rebuild_norm_layers_override=arg_rebuild_norm_specified_layers)
            if __print_norm_to_rebuild_layers:
                __print_norm_to_rebuild_layers = False
                child_logger.info(f"layers to rebuild: {layers_rebuild}")
            rebuild_iter, rebuilding_loss_val = rebuild_states[-1]
            child_logger.info(f"current tick: {current_tick}, rebuilding finished at {rebuild_iter} rounds, rebuilding loss = {rebuilding_loss_val:.3f}")
            # remove norm layer variance
            target_variance = {k: v for k, v in target_variance.items() if not special_torch_layers.is_normalization_layer(arg_ml_setup.model_name, k)}

        cuda.CudaEnv.model_state_dict_to(start_model_stat, cpu_device)
        if ENABLE_NAN_CHECKING:
            util.check_for_nans_in_state_dict(start_model_stat)
        """scale variance back, due to SGD variance drift"""
        start_model_stat = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(start_model_stat, target_variance)
        if ENABLE_NAN_CHECKING:
            util.check_for_nans_in_state_dict(start_model_stat)

        # service
        all_model_stats = [start_model_stat, end_model_stat_dict]
        weight_diff_service.trigger_without_runtime_parameters(current_tick, all_model_stats)
        distance_to_origin_service.trigger_without_runtime_parameters(current_tick, {0: start_model_stat})
        variance_service.trigger_without_runtime_parameters(current_tick, [0], [start_model_stat])
        if record_model_service is not None:
            record_flag = False
            if arg_save_ticks is None:
                record_flag = True
            else:
                if current_tick in arg_save_ticks:
                    record_flag = True
            if record_flag:
                record_model_service.trigger_without_runtime_parameters(current_tick, [0], [start_model_stat])
        record_test_accuracy_loss_service.trigger_without_runtime_parameters(current_tick, {0: start_model_stat})
        record_training_loss_service.trigger_without_runtime_parameters(current_tick, {0: training_loss_val})

        current_tick += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Generate some high accuracy models')
    parser.add_argument("start_folder", type=str, help="folder containing starting models")
    parser.add_argument("end_folder", type=str, help="folder containing destination models")
    parser.add_argument("--mapping_mode", type=str, default='auto', choices=['auto', 'all_to_all', 'each_to_each', 'one_to_all', 'all_to_one'])
    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("-t", "--thread", type=int, default=1, help='specify how many models to train in parallel')
    parser.add_argument("-m", "--model_type", type=str, default='auto', choices=['auto', 'lenet5', 'resnet18_bn', 'resnet18_gn'])
    parser.add_argument("-T", "--max_tick", type=int, default=10000)

    # average parameters
    parser.add_argument("-s", "--step_size", type=float, default=0.001)
    parser.add_argument("-a", "--adoptive_step_size", type=float, default=0)
    parser.add_argument("--layer_skip_average", type=str, nargs='+')
    parser.add_argument("--layer_skip_average_keyword", type=str, nargs='+')

    # train parameters
    parser.add_argument("--lr", type=float, default=0.001, help='train the model with this learning rate, joint with "--training_round"')
    parser.add_argument("--training_round", type=int, default=1, help='train the model for x rounds, joint with "--lr"')
    parser.add_argument("--loss", type=float, default=-1, help='train the model until loss is smaller than, cannot use with "--training_round" or "--lr"')

    # rebuild norm parameter
    parser.add_argument("-r", "--rebuild_norm_round", type=int, default=0, help='train for x rounds to rebuild the norm layers')
    parser.add_argument("--rebuild_norm_lr", type=float, default=0.001)
    parser.add_argument("--rebuild_norm_layers", type=str, nargs="+", default=[], help='specify which layers to rebuild, default means all norm layers')

    # find pathway points parameters
    parser.add_argument("-p", "--pathway_depth", type=int, default=0, help='the depth of find pathway points, 1->find a mid point, 2->find 3 points(25% each), 3->find 7 points(12.5% each)')
    parser.add_argument("--existing_pathway", type=str, help='specify the folder containing existing pathway points')

    # compute parameters
    parser.add_argument("--save_ticks", type=str, help='specify when to record the models (e.g. [1,2,3,5-10]), only works when --save_format is set to work.')
    parser.add_argument("--save_format", type=str, default='none', choices=['none', 'file', 'lmdb'])
    parser.add_argument("--cpu", action='store_true', help='force using CPU for training')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    parser.add_argument("--use_predefined_optimal", action='store_true', help='use predefined optimal parameters')

    args = parser.parse_args()

    set_logging(logger, "main")
    logger.info("logging setup complete")

    mode = TrainMode.default_x_rounds
    learning_rate = args.lr
    training_round = args.training_round
    loss = args.loss
    if loss != -1:
        assert learning_rate == 0.001 and training_round == 1
        mode = TrainMode.default_until_loss

    start_folder = args.start_folder
    end_folder = args.end_folder
    mapping_mode = args.mapping_mode

    max_tick = args.max_tick
    step_size = args.step_size
    adoptive_step_size = args.adoptive_step_size

    rebuild_normalization_round = args.rebuild_norm_round

    pathway_depth = args.pathway_depth
    existing_pathway = args.existing_pathway

    rebuild_norm_lr = args.rebuild_norm_lr
    rebuild_norm_specified_layers = args.rebuild_norm_layers

    use_cpu = args.cpu
    paths_to_find = get_files_to_process(args.start_folder, args.end_folder, mapping_mode)
    if args.save_ticks is not None:
        save_ticks = util.expand_int_args(args.save_ticks)
    else:
        save_ticks = None
    save_format = args.save_format
    layer_skip_average = args.layer_skip_average if args.layer_skip_average is not None else []
    layer_skip_average_keyword = args.layer_skip_average_keyword if args.layer_skip_average_keyword is not None else []
    paths_to_find_count = len(paths_to_find)
    logger.info(f"totally {paths_to_find_count} paths to process: {paths_to_find}")

    worker_count = args.thread
    total_cpu_count = args.core
    if total_cpu_count > MAX_CPU_COUNT:
        total_cpu_count = MAX_CPU_COUNT
    model_type = args.model_type

    # load info.json
    with open(os.path.join(start_folder, INFO_FILE_NAME)) as f:
        start_folder_info = json.load(f)
    with open(os.path.join(end_folder, INFO_FILE_NAME)) as f:
        end_folder_info = json.load(f)
    assert start_folder_info['model_type'] == end_folder_info['model_type']

    if model_type == 'auto':
        model_type = start_folder_info['model_type']
    else:
        assert model_type == start_folder_info['model_type']

    # predefine optimal parameters
    if args.use_predefined_optimal:
        if model_type == 'lenet5':
            learning_rate = 0.01
            max_tick = 30000
            step_size = 0.001
            adoptive_step_size = 0
            training_round = 2
            rebuild_normalization_round = 0
        elif model_type == 'resnet18_bn':
            raise NotImplementedError
        elif model_type == 'resnet18_gn':
            raise NotImplementedError
        else:
            raise NotImplementedError

    # prepare model and dataset
    current_ml_setup = ml_setup.get_ml_setup_from_model_type(model_type)

    # create output folder
    if args.output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.mkdir(output_folder_path)
    shutil.copyfile(__file__, os.path.join(output_folder_path, os.path.basename(__file__)))
    info_file = open(os.path.join(output_folder_path, "arguments.txt"), 'x')
    info_file.write(f"{args}")
    info_file.flush()
    info_file.close()

    # finding path
    if worker_count > paths_to_find_count:
        worker_count = paths_to_find_count
    logger.info(f"worker: {worker_count}")

    if mode == TrainMode.default_x_rounds:
        args = [( (output_folder_path, start_file, end_file, current_ml_setup, max_tick),
                  (mode, (learning_rate, training_round)),
                  (step_size, adoptive_step_size, layer_skip_average, layer_skip_average_keyword),
                  (rebuild_norm_lr, rebuild_normalization_round, rebuild_norm_specified_layers),
                  (pathway_depth, existing_pathway),
                  (worker_count, total_cpu_count, save_format, save_ticks, use_cpu) ) for (start_file, end_file) in paths_to_find]
    elif mode == TrainMode.Adam_until_loss or mode == TrainMode.default_until_loss:
        args = [( (output_folder_path, start_file, end_file, current_ml_setup, max_tick),
                  (mode, loss),
                  (step_size, adoptive_step_size, layer_skip_average, layer_skip_average_keyword),
                  (rebuild_norm_lr, rebuild_normalization_round, rebuild_norm_specified_layers),
                  (pathway_depth, existing_pathway),
                  (worker_count, total_cpu_count, save_format, save_ticks, use_cpu) ) for (start_file, end_file) in paths_to_find]
    else:
        raise NotImplementedError
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_file_func, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
