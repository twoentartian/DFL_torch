import os
import argparse
import logging
import shutil
import sys
import json
from enum import Enum
from typing import Final, Optional
import torch
import re
import torch.optim as optim
import concurrent.futures
import copy
from dataclasses import dataclass, field
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, model_average, model_variance_correct, special_torch_layers, cuda, util
from py_src.service import record_weights_difference, record_test_accuracy_loss, record_variance, record_model_stat, record_training_loss

logger = logging.getLogger("find_high_accuracy_path")

INFO_FILE_NAME = 'info.json'

MAX_CPU_COUNT: Final[int] = 32

ENABLE_NAN_CHECKING: Final[bool] = False


class TrainMode(Enum):
    default_x_rounds = 0
    default_until_loss = 1

class FindPathArgs:
    # general options
    start_model_path = None
    end_model_path = None
    output_folder_path = None
    ml_setup = None
    tick = None

    # training_options
    training_options = None
    pretrain_optimizer = None
    lr = None
    training_round = None
    loss = None

    # average options
    step_size = None
    adoptive_step_size = None
    layer_skip_average = None
    layer_skip_average_keyword = None

    # rebuild norm options
    rebuild_norm_round = None
    rebuild_norm_loss = None
    rebuild_norm_layers = None
    rebuild_norm_with_dedicated_dataloader = None
    rebuild_norm_with_training_optimizer = None

    # pathway depth
    pathway_depth = None
    existing_pathway = None

    # other options
    worker_count = None
    total_cpu_count = None
    save_format = None
    save_ticks = None
    use_cpu = None
    use_amp = None

    def set_default(self):
        self.worker_count = 1
        self.tick = 10000
        self.step_size = 0.001
        self.adoptive_step_size = 0
        self.lr = 0.001
        self.training_round = 1
        self.loss = -1
        self.rebuild_norm_round = 0
        self.pathway_depth = 0
        self.save_format = "none"

    def set_training_option_x_rounds(self, lr, rounds):
        self.training_options = TrainMode.default_x_rounds, (lr, rounds)

    def set_training_option_until_loss(self, loss):
        self.training_options = TrainMode.default_until_loss, loss

    def setup_training_option(self):
        if self.loss == -1:
            assert self.lr is not None and self.training_round is not None
            self.set_training_option_x_rounds(self.lr, self.training_round)
        else:
            assert self.loss > 0
            self.set_training_option_until_loss(self.loss)

    def update_object_from_json(self, json_target):
        train_mode = None
        for key, value in json_target.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"reading json file error: key {key} does not exists")
        self.setup_training_option()

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

def get_optimizer_to_find_high_accuracy_path(model_name, model_parameter):
    if model_name == "lenet5":
        optimizer = torch.optim.SGD(model_parameter, lr=0.01)
    elif model_name == "resnet18_bn":
        optimizer = torch.optim.SGD(model_parameter, lr=0.001)
    elif model_name == "simplenet":
        optimizer = torch.optim.Adadelta(model_parameter, lr=0.01, rho=0.9, eps=1e-3, weight_decay=0.001)
    elif model_name == "cct7":
        optimizer = torch.optim.SGD(model_parameter, lr=0.001)
    else:
        raise NotImplementedError(f"{model_name} not implemented for finding high accuracy path")
    return optimizer

def get_optimizer_to_rebuild_norm(model_name, model_parameter):
    if model_name == "resnet18_bn":
        optimizer = torch.optim.SGD(model_parameter, lr=0.001, momentum=0.9, weight_decay=5e-4)
    else:
        raise NotImplementedError(f"{model_name} not implemented for rebuilding norm")
    return optimizer

def get_enable_merge_bias_weight_during_moving(model_name):
    if model_name == "lenet5":
        return True
    elif model_name == "resnet18_bn":
        return False
    elif model_name == "simplenet":
        return False
    elif model_name == "cct7":
        return True
    else:
        raise NotImplementedError(f"{model_name} not implemented for get_enable_merge_bias_weight_during_moving")


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

def rebuild_norm_layers(model, model_state, arg_ml_setup, dataloader, rebuild_norm_round, rebuild_norm_loss, existing_optimizer_state_or_optimizer,
                        rebuild_on_device=None, reset_norm_to_initial=False, initial_model_stat=None, display=False, rebuild_norm_layers_override=None,
                        use_amp=False):
    global __print_norm_to_rebuild_layers
    if rebuild_norm_layers_override is None:
        rebuild_norm_layers_override = []

    def is_processing_this_layer(model_name, layer_name, rebuild_norm_layers_override):
        if len(rebuild_norm_layers_override) == 0:
            if special_torch_layers.is_normalization_layer(model_name, layer_name):
                return True
            else:
                return False
        else:
            process_this_layer = False
            for single_layer in rebuild_norm_layers_override:
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

    if isinstance(existing_optimizer_state_or_optimizer, dict):
        # this is an optimizer state
        optimizer_rebuild_norm = get_optimizer_to_rebuild_norm(arg_ml_setup.model_name, model.parameters())
        optimizer_rebuild_norm.load_state_dict(existing_optimizer_state_or_optimizer)
    elif isinstance(existing_optimizer_state_or_optimizer, optim.Optimizer):
        # this is an optimizer
        optimizer_rebuild_norm = existing_optimizer_state_or_optimizer
    else:
        raise ValueError(f"existing_optimizer_state_or_optimizer should be an optimizer or an optimizer state dict.")
    cuda.CudaEnv.optimizer_to(optimizer_rebuild_norm, rebuild_on_device)

    rebuild_states = []
    rebuilding_normalization_count = 0

    # # freeze unprocessed layers
    # for layer_name, param in model.named_parameters():
    #     if layer_name not in rebuild_norm_layers_to_process:
    #         param.requires_grad = False
    start_model_stat = model.state_dict()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    moving_max_size = 2
    moving_max = util.MovingMax(moving_max_size)
    while True:
        exit_flag = False
        for (rebuilding_normalization_index, (data, label)) in enumerate(dataloader):
            data, label = data.to(rebuild_on_device), label.to(rebuild_on_device)
            optimizer_rebuild_norm.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    rebuilding_loss = criterion(outputs, label)
                    scaler.scale(rebuilding_loss).backward()
                    scaler.step(optimizer_rebuild_norm)
                    scaler.update()
            else:
                outputs = model(data)
                rebuilding_loss = criterion(outputs, label)
                rebuilding_loss.backward()
                optimizer_rebuild_norm.step()
            rebuilding_loss_val = rebuilding_loss.item()
            rebuilding_normalization_count += 1
            rebuild_states.append((rebuilding_normalization_count, rebuilding_loss_val))
            current_model_stat = model.state_dict()
            for layer_name, layer_weights in current_model_stat.items():
                if layer_name not in rebuild_norm_layers_to_process:
                    current_model_stat[layer_name] = start_model_stat[layer_name]
            model.load_state_dict(current_model_stat)
            if display:
                print(f"tick: {rebuilding_normalization_count}  loss: {rebuilding_loss_val}")
            if rebuilding_normalization_count >= rebuild_norm_round:
                exit_flag = True
                break
            max_loss = moving_max.add(rebuilding_loss_val)
            if rebuild_norm_loss is not None:
                if max_loss < rebuild_norm_loss and rebuilding_normalization_index + 1 >= moving_max_size:
                    exit_flag = True
                    break
        if exit_flag:
            break
    output_model_state = model.state_dict()
    return output_model_state, rebuild_states, rebuild_norm_layers_to_process


__print_norm_to_rebuild_layers = True
def process_file_func(args: [FindPathArgs]):
    global __print_norm_to_rebuild_layers

    assert len(args) >= 1
    arg0 = args[0]

    start_file_name = os.path.basename(arg0.start_model_path).replace('.model.pt', '')
    end_file_name = os.path.basename(arg0.end_model_path).replace('.model.pt', '')

    # logger
    task_name = f"{start_file_name}-{end_file_name}"
    child_logger = logging.getLogger(f"find_high_accuracy_path.{task_name}")
    set_logging(child_logger, task_name, log_file_path=os.path.join(arg0.output_folder_path, "info.log"))
    child_logger.info("logging setup complete")

    if arg0.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    thread_per_process = arg0.total_cpu_count // arg0.worker_count
    torch.set_num_threads(thread_per_process)

    # output folders
    arg_output_folder_path = os.path.join(arg0.output_folder_path, f"{start_file_name}-{end_file_name}")
    if os.path.exists(arg_output_folder_path):
        child_logger.warning(f"{arg_output_folder_path} already exists")
    else:
        os.makedirs(arg_output_folder_path)

    """load models"""
    cpu_device = torch.device("cpu")
    start_model = copy.deepcopy(arg0.ml_setup.model)
    initial_model_stat = start_model.state_dict()
    start_model_stat_dict, start_model_name = util.load_model_state_file(arg0.start_model_path)
    end_model_stat_dict, end_model_name = util.load_model_state_file(arg0.end_model_path)
    util.assert_if_both_not_none(start_model_name, end_model_name)
    start_model.load_state_dict(start_model_stat_dict)

    # assert start_model_state_dict != end_model_stat_dict
    for key in start_model_stat_dict.keys():
        assert not torch.equal(start_model_stat_dict[key], end_model_stat_dict[key]), f'starting model({arg0.start_model_path}) is same as ending model({arg0.end_model_path})'
        break

    """load training data"""
    training_dataset = arg0.ml_setup.training_data
    dataloader = DataLoader(training_dataset, batch_size=arg0.ml_setup.training_batch_size, shuffle=True)
    criterion = arg0.ml_setup.criterion
    optimizer = get_optimizer_to_find_high_accuracy_path(arg0.ml_setup.model_name, start_model.parameters())

    """Optimizer related config"""
    if arg0.pretrain_optimizer:
        """pre-train an optimizer"""
        child_logger.info(f"pre training")
        temp_model_state = start_model.state_dict()
        cuda.CudaEnv.model_state_dict_to(temp_model_state, cpu_device)
        start_model.train()
        start_model.to(device)
        cuda.CudaEnv.optimizer_to(optimizer, device)
        for (training_index, (data, label)) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = start_model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            if training_index == 100:
                break
        start_model.load_state_dict(temp_model_state)
    else:
        """use existing optimizer"""
        # check optimizer
        start_optimizer_path = arg0.start_model_path.replace('model.pt', 'optimizer.pt')
        assert os.path.exists(start_optimizer_path), f'starting optimizer {start_optimizer_path} is missing'
        start_model_optimizer_stat, _ = util.load_optimizer_state_file(start_optimizer_path)

        # try loading optimizer state to optimizer
        optimizer_test = copy.deepcopy(optimizer)
        try:
            optimizer_test.load_state_dict(start_model_optimizer_stat)
            optimizer.load_state_dict(start_model_optimizer_stat)
            child_logger.info(f"successfully load optimizer state")
        except Exception as e:
            child_logger.info(f"fail to load optimizer state: {str(e)}, now skip")
        del optimizer_test

    # services
    all_node_names = [0, 1]
    all_model_stats = [start_model_stat_dict, end_model_stat_dict]

    weight_diff_service = record_weights_difference.ServiceWeightsDifferenceRecorder(1)
    weight_diff_service.initialize_without_runtime_parameters(all_model_stats, arg_output_folder_path)
    weight_change_service = record_weights_difference.ServiceWeightsDifferenceRecorder(1,
                                                                                       l1_save_file_name="weight_change_l1.csv",
                                                                                       l2_save_file_name="weight_change_l2.csv")
    weight_change_service.initialize_without_runtime_parameters(all_model_stats, arg_output_folder_path)
    distance_to_origin_service = record_weights_difference.ServiceDistanceToOriginRecorder(1, [0])
    distance_to_origin_service.initialize_without_runtime_parameters({0: start_model_stat_dict}, arg_output_folder_path)
    variance_service = record_variance.ServiceVarianceRecorder(1)
    variance_service.initialize_without_runtime_parameters([0], [start_model_stat_dict], arg_output_folder_path)
    if arg0.save_format != 'none':
        if not arg0.save_ticks:
            record_model_service = record_model_stat.ModelStatRecorder(1)
        else:
            record_model_service = record_model_stat.ModelStatRecorder(sys.maxsize) # we don't set the interval here because only certain ticks should be recorded
        record_model_service.initialize_without_runtime_parameters([0], arg_output_folder_path, save_format=arg0.save_format)
    else:
        record_model_service = None
    record_test_accuracy_loss_service = record_test_accuracy_loss.ServiceTestAccuracyLossRecorder(1, 100, use_fixed_testing_dataset=True)
    record_test_accuracy_loss_service.initialize_without_runtime_parameters(arg_output_folder_path, [0], start_model, criterion, training_dataset, use_cuda=True)
    record_training_loss_service = record_training_loss.ServiceTrainingLossRecorder(1)
    record_training_loss_service.initialize_without_runtime_parameters(arg_output_folder_path, [0])

    """find pathway points"""
    optimizer_paths_for_pathway_points = None
    if arg0.pathway_depth != 0:
        assert len(args) == 1, "find_pathway_points can only be used for 1 stage finding"
        if arg0.existing_pathway is None:
            child_logger.info(f"finding pathway points with depth {arg0.pathway_depth}")

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
                ft_epochs, ft_optimizer, ft_lr_scheduler = get_optimizer_to_find_pathway_point(model_name, target_model.parameters(), fine_tune_dataloader.dataset, arg.ml_setup.training_batch_size)

                for epoch in range(ft_epochs):
                    # scale variance
                    current_model_state = target_model.state_dict()
                    current_model_state = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(current_model_state, ft_target_variance)
                    target_model.load_state_dict(current_model_state)

                    # training
                    train_loss = 0
                    count = 0
                    for data, label in dataloader:
                        data, label = data.to(device), label.to(device)
                        ft_optimizer.zero_grad(set_to_none=True)
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

            pathway_points = find_pathway_points(start_model_stat_dict, end_model_stat_dict, start_model, arg0.ml_setup.model_name, dataloader, arg0.pathway_depth)
            child_logger.info(f"find {len(pathway_points)} pathway points with depth {arg0.pathway_depth}")
        else:
            child_logger.info(f"loading existing pathway points from {arg0.existing_pathway}")

            def extract_x_y_from_filename(filename):
                match = re.match(r"(\d+)_over_(\d+)\.model\.pt", filename)
                if match:
                    x = int(match.group(1))
                    y = int(match.group(2))
                    return x, y
                return None, None

            model_list = []
            found_files = {}
            for file_name in os.listdir(arg0.existing_pathway):
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
                        model_path = os.path.join(arg0.existing_pathway, file_name)
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
            assert 2 ** arg0.pathway_depth + 1 == len(pathway_points)
            child_logger.info(f"loading and checking pathway points pass")

        # create folders and save pathway points
        output_path_pathway_points = os.path.join(arg_output_folder_path, "pathway_points")
        os.makedirs(output_path_pathway_points, exist_ok=True)
        optimizer_paths_for_pathway_points = []
        for index, pathway_model in enumerate(pathway_points):
            util.save_model_state(os.path.join(output_path_pathway_points, f"{index}_over_{len(pathway_points) - 1}.model.pt"), pathway_model, arg0.ml_setup.model_name)
            optimizer_paths_for_pathway_points.append(os.path.join(output_path_pathway_points, f"{index}_over_{len(pathway_points) - 1}.optimizer.pt"))
    else:
        pathway_points = [start_model_stat_dict, end_model_stat_dict]

    start_model_stat = start_model_stat_dict
    target_direction_points = pathway_points[1:]

    """record variance"""
    variance_record = model_variance_correct.VarianceCorrector(model_variance_correct.VarianceCorrectionType.FollowOthers)
    variance_record.add_variance(start_model_stat)
    target_variance = variance_record.get_variance()

    current_tick = 0
    modeL_state_of_last_tick = copy.deepcopy(start_model_stat)
    cuda.CudaEnv.model_state_dict_to(modeL_state_of_last_tick, device)

    for stage_index, arg in enumerate(args):
        start_tick_of_this_stage = current_tick

        # set training parameters
        training_mode, training_parameter = arg.training_options
        if training_mode == TrainMode.default_x_rounds:
            train_lr, train_round = training_parameter
        elif training_mode == TrainMode.default_until_loss:
            target_train_loss = training_parameter
        else:
            raise NotImplementedError
        del training_parameter

        # set dataloader for rebuilding norm
        if arg0.rebuild_norm_round != 0:
            child_logger.info(f"rebuild norm layers for {arg0.rebuild_norm_round} rounds")
            if arg0.rebuild_norm_with_dedicated_dataloader:
                dataset_for_rebuilding_norm = arg0.ml_setup.training_data_for_rebuilding_normalization
                dataloader_for_rebuilding_norm = DataLoader(dataset_for_rebuilding_norm, batch_size=arg0.ml_setup.training_batch_size)
                child_logger.info(f"use dedicated dataloader for rebuilding norm")
            else:
                dataloader_for_rebuilding_norm = dataloader  # use the training dataloader
                child_logger.info(f"use training dataloader for rebuilding norm")
        else:
            dataloader_for_rebuilding_norm = None
            epoch_for_rebuilding_norm = None

        # ignored layers
        ignore_layers = []
        averaged_layers = []
        if arg.layer_skip_average is None:
            arg.layer_skip_average = []
        if arg.layer_skip_average_keyword is None:
            arg.layer_skip_average_keyword = []
        for layer_name in start_model_stat_dict.keys():
            if layer_name in arg.layer_skip_average:
                ignore_layers.append(layer_name)
            if special_torch_layers.is_keyword_in_layer_name(layer_name, arg.layer_skip_average_keyword):
                ignore_layers.append(layer_name)
            if special_torch_layers.is_ignored_layer_averaging(layer_name):
                ignore_layers.append(layer_name)
        for layer_name in start_model_stat_dict.keys():
            if layer_name not in ignore_layers:
                averaged_layers.append(layer_name)
        child_logger.info(f"ignore moving {len(ignore_layers)} layers: {ignore_layers}")
        child_logger.info(f"plan to move {len(averaged_layers)} layers: {averaged_layers}")

        """begin find path for this stage"""
        previous_pathway_index = -1
        if arg.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        while current_tick < arg.tick + start_tick_of_this_stage:
            cuda.CudaEnv.model_state_dict_to(start_model_stat, device)

            if stage_index == 0:
                """set end point for first stage"""
                path_len = arg.tick // len(target_direction_points)
                current_path_index = current_tick // path_len
                current_direction_point = target_direction_points[current_path_index]
                if previous_pathway_index != current_path_index:
                    # move current pathway points to computing device and move else back to cpu
                    for i in range(len(target_direction_points)):
                        if i == current_path_index:
                            cuda.CudaEnv.model_state_dict_to(target_direction_points[i], device)
                        else:
                            cuda.CudaEnv.model_state_dict_to(target_direction_points[i], cpu_device)
                    # entering a new pathway point region
                    previous_pathway_index = current_path_index
                    if optimizer_paths_for_pathway_points is not None:
                        util.save_optimizer_state(optimizer_paths_for_pathway_points[current_path_index], optimizer.state_dict(), arg.ml_setup.model_name)
            else:
                """for other stages, just set the end model as direction"""
                current_direction_point = end_model_stat_dict

            """move tensor"""
            merge_bias_weight = get_enable_merge_bias_weight_during_moving(arg.ml_setup.model_name)
            start_model_stat = model_average.move_model_state_toward(start_model_stat, current_direction_point, arg.step_size, arg.adoptive_step_size,
                                                                     enable_merge_bias_with_weight=merge_bias_weight, ignore_layers=ignore_layers)

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

            training_round_counter = 0
            if training_mode == TrainMode.default_x_rounds:
                for (training_index, (data, label)) in enumerate(dataloader):
                    training_round_counter += 1
                    data, label = data.to(device), label.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    if arg.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = start_model(data)
                            training_loss = criterion(outputs, label)
                            scaler.scale(training_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        outputs = start_model(data)
                        training_loss = criterion(outputs, label)
                        training_loss.backward()
                        optimizer.step()
                    training_loss_val = training_loss.item()
                    if training_round_counter == train_round:
                        break
                child_logger.info(f"current tick: {current_tick}, training {training_round_counter} rounds, loss = {training_loss_val:.3f}")
            elif training_mode == TrainMode.default_until_loss:
                moving_max_size = 2
                moving_max = util.MovingMax(moving_max_size)
                while True:
                    training_round_counter += 1
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

            start_model_stat = start_model.state_dict()

            """rebuilding normalization"""
            if arg.rebuild_norm_round != 0:
                if arg.rebuild_norm_with_training_optimizer:
                    optimizer_info = "optimizer"
                    i = optimizer
                else:
                    optimizer_info = "optimizer state"
                    i = optimizer.state_dict()

                start_model_stat, rebuild_states, layers_rebuild = rebuild_norm_layers(start_model, start_model_stat,
                                                                                       arg.ml_setup,
                                                                                       dataloader_for_rebuilding_norm,
                                                                                       arg.rebuild_norm_round,
                                                                                       arg.rebuild_norm_loss,
                                                                                       i, rebuild_on_device=device,
                                                                                       initial_model_stat=initial_model_stat,
                                                                                       reset_norm_to_initial=True,
                                                                                       rebuild_norm_layers_override=arg.rebuild_norm_layers,
                                                                                       use_amp=arg.use_amp)

                if __print_norm_to_rebuild_layers:
                    __print_norm_to_rebuild_layers = False
                    child_logger.info(f"layers to rebuild: {layers_rebuild}")
                rebuild_iter, rebuilding_loss_val = rebuild_states[-1]
                child_logger.info(f"current tick: {current_tick}, rebuilding finished at {rebuild_iter} rounds with {optimizer_info}, rebuilding loss = {rebuilding_loss_val:.3f}")
                # remove norm layer variance
                # target_variance = {k: v for k, v in target_variance.items() if k not in layers_rebuild}

            if ENABLE_NAN_CHECKING:
                util.check_for_nans_in_state_dict(start_model_stat)
            """scale variance back, due to SGD variance drift"""
            start_model_stat = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(start_model_stat, target_variance)
            if ENABLE_NAN_CHECKING:
                util.check_for_nans_in_state_dict(start_model_stat)

            """service"""
            all_model_stats = [start_model_stat, end_model_stat_dict]
            for i in all_model_stats:
                cuda.CudaEnv.model_state_dict_to(i, device)
            weight_diff_service.trigger_without_runtime_parameters(current_tick, all_model_stats)
            weight_change_service.trigger_without_runtime_parameters(current_tick, [modeL_state_of_last_tick, start_model_stat])
            modeL_state_of_last_tick = copy.deepcopy(start_model_stat)
            distance_to_origin_service.trigger_without_runtime_parameters(current_tick, {0: start_model_stat})
            variance_service.trigger_without_runtime_parameters(current_tick, [0], [start_model_stat])
            if record_model_service is not None:
                record_flag = False
                if arg.save_ticks is None:
                    record_flag = True
                else:
                    current_save_tick_for_this_stage = [i+start_tick_of_this_stage for i in arg.save_ticks]
                    if current_tick in current_save_tick_for_this_stage:
                        record_flag = True
                if record_flag:
                    record_model_service.trigger_without_runtime_parameters(current_tick, [0], [start_model_stat])
            record_test_accuracy_loss_service.trigger_without_runtime_parameters(current_tick, {0: start_model_stat})
            record_training_loss_service.trigger_without_runtime_parameters(current_tick, {0: training_loss_val})

            current_tick += 1

        # save final model and optimizer
        cuda.CudaEnv.optimizer_to(optimizer, cpu_device)
        util.save_model_state(os.path.join(arg_output_folder_path, f"{current_tick}.model.pt"), start_model_stat, arg.ml_setup.model_name)
        util.save_optimizer_state(os.path.join(arg_output_folder_path, f"{current_tick}.optimizer.pt"), optimizer.state_dict(), arg.ml_setup.model_name)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Generate some high accuracy models')
    parser.add_argument("start_folder", type=str, help="folder containing starting models")
    parser.add_argument("end_folder", type=str, help="folder containing destination models")
    parser.add_argument("--mapping_mode", type=str, default='auto', choices=['auto', 'all_to_all', 'each_to_each', 'one_to_all', 'all_to_one'])
    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("-t", "--thread", type=int, default=1, help='specify how many models to train in parallel')
    parser.add_argument("-m", "--model_type", type=str, default='auto', choices=['auto', 'lenet5', 'resnet18_bn', 'resnet18_gn'])
    parser.add_argument("-T", "--tick", type=int, default=10000)

    # average parameters
    parser.add_argument("-s", "--step_size", type=float, default=0.001)
    parser.add_argument("-a", "--adoptive_step_size", type=float, default=0)
    parser.add_argument("--layer_skip_average", type=str, nargs='+')
    parser.add_argument("--layer_skip_average_keyword", type=str, nargs='+')

    # train parameters
    parser.add_argument("--lr", type=float, default=0.001, help='train the model with this learning rate, joint with "--training_round"')
    parser.add_argument("--training_round", type=int, default=1, help='train the model for x rounds, joint with "--lr"')
    parser.add_argument("--loss", type=float, default=-1, help='train the model until loss is smaller than, cannot use with "--training_round" or "--lr"')
    parser.add_argument("--pretrain_optimizer", action='store_true', help='pretrain an optimizer rather than using existing optimizer state')

    # rebuild norm parameter
    parser.add_argument("-r", "--rebuild_norm_round", type=int, default=0, help='rebuild norm for x rounds to rebuild the norm layers')
    parser.add_argument("--rebuild_norm_loss", type=float, default=0, help='rebuild norm until loss is smaller than this threshold')
    parser.add_argument("--rebuild_norm_layers", type=str, nargs="+", default=[], help='specify which layers to rebuild, default means all norm layers')
    parser.add_argument("--dedicated_rebuild_norm_dataloader", action='store_true', help='use a dedicated, sample order preserved dataloader')
    parser.add_argument("--rebuild_norm_reuse_train_optimizer", action='store_true', help='reuse the training optimizer for building norm layers')

    # find pathway points parameters
    parser.add_argument("-p", "--pathway_depth", type=int, default=0, help='the depth of find pathway points, 1->find a mid point, 2->find 3 points(25% each), 3->find 7 points(12.5% each)')
    parser.add_argument("--existing_pathway", type=str, help='specify the folder containing existing pathway points')

    # compute parameters
    parser.add_argument("--save_ticks", type=str, help='specify when to record the models (e.g. [1,2,3,5-10]), only works when --save_format is set to work.')
    parser.add_argument("--save_format", type=str, default='none', choices=['none', 'file', 'lmdb'])
    parser.add_argument("--cpu", action='store_true', help='force using CPU for training')
    parser.add_argument("--amp", action='store_true', help='enable auto mixed precision')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    # parser.add_argument("--use_predefined_optimal", action='store_true', help='use predefined optimal parameters')

    # load json file for multiple configs?
    parser.add_argument("--json", type=str, default=None, help='specify the json file for multiple stages')

    args = parser.parse_args()

    set_logging(logger, "main")
    logger.info("logging setup complete")

    find_path_arg_template = []
    if args.json is not None:
        """for multiple args"""
        with open(args.json, 'r') as file:
            all_stages = json.load(file)
        for single_stage_json in all_stages:
            stage = FindPathArgs()
            stage.set_default()
            stage.update_object_from_json(single_stage_json)
            find_path_arg_template.append(stage)
    else:
        """for single arg"""
        assert args.json is None
        find_path_arg = FindPathArgs()

        # train parameters
        find_path_arg.pretrain_optimizer = args.pretrain_optimizer
        loss = args.loss
        if loss != -1:
            assert args.lr == 0.001 and args.training_round == 1, "you can only specify loss or lr+training_round"
            find_path_arg.set_training_option_until_loss(loss)
            # mode = TrainMode.default_until_loss
        else:
            find_path_arg.set_training_option_x_rounds(args.lr, args.training_round)

        # average parameters
        find_path_arg.step_size = args.step_size
        find_path_arg.adoptive_step_size = args.adoptive_step_size
        find_path_arg.layer_skip_average = args.layer_skip_average if args.layer_skip_average is not None else []
        find_path_arg.layer_skip_average_keyword = args.layer_skip_average_keyword if args.layer_skip_average_keyword is not None else []

        # rebuild norm
        find_path_arg.rebuild_norm_round = args.rebuild_norm_round
        find_path_arg.rebuild_norm_loss = args.rebuild_norm_loss
        find_path_arg.rebuild_norm_layers = args.rebuild_norm_layers
        find_path_arg.rebuild_norm_with_dedicated_dataloader = args.dedicated_rebuild_norm_dataloader
        find_path_arg.rebuild_norm_with_training_optimizer = args.rebuild_norm_reuse_train_optimizer

        # pathway depth
        find_path_arg.pathway_depth = args.pathway_depth
        find_path_arg.existing_pathway = args.existing_pathway

        # general_info
        find_path_arg.tick = args.tick
        find_path_arg.use_cpu = args.cpu
        find_path_arg.use_amp = args.amp
        find_path_arg.save_format = args.save_format
        if args.save_ticks is not None:
            find_path_arg.save_ticks = util.expand_int_args(args.save_ticks)
        else:
            find_path_arg.save_ticks = None

        """add to list"""
        find_path_arg_template.append(find_path_arg)

    start_folder = args.start_folder
    end_folder = args.end_folder
    mapping_mode = args.mapping_mode
    paths_to_find = get_files_to_process(start_folder, end_folder, mapping_mode)
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
    if args.json is not None:
        shutil.copyfile(args.json, os.path.join(output_folder_path, os.path.basename(args.json)))
    info_file = open(os.path.join(output_folder_path, "arguments.txt"), 'x')
    info_file.write(f"{args}")
    info_file.flush()
    info_file.close()

    # finding path
    if worker_count > paths_to_find_count:
        worker_count = paths_to_find_count
    logger.info(f"worker: {worker_count}")

    args = []
    for (start_file, end_file) in paths_to_find:
        find_path_arg = copy.deepcopy(find_path_arg_template)
        for each_stage in find_path_arg:
            each_stage.start_model_path = start_file
            each_stage.end_model_path = end_file
            each_stage.output_folder_path = output_folder_path
            each_stage.worker_count = worker_count
            each_stage.total_cpu_count = total_cpu_count
            each_stage.ml_setup = current_ml_setup
        args.append(find_path_arg)

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_file_func, arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
