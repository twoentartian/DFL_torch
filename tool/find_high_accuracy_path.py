import os
import argparse
import logging
import shutil
import sys
import json
from typing import Final
import torch
import torch.optim as optim
import concurrent.futures
import copy
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, model_average, model_variance_correct, special_torch_layers, cuda, util
from py_src.service import record_weights_difference, record_test_accuracy_loss, record_variance, record_model_stat

logger = logging.getLogger("find_high_accuracy_path")

INFO_FILE_NAME = 'info.json'

MAX_CPU_COUNT: Final[int] = 32

ENABLE_DEDICATED_TRAINING_DATASET_FOR_REBUILDING_NORM: Final[bool] = True
ENABLE_REBUILD_NORM_FOR_STARTING_ENDING_MODEL: Final[bool] = True
ENABLE_NAN_CHECKING: Final[bool] = False
ENABLE_PRE_TRAINING: Final[bool] = False


def set_logging(base_logger, task_name, log_file_path=None):
    class ExitOnExceptionHandler(logging.StreamHandler):
        def emit(self, record):
            if record.levelno == logging.CRITICAL:
                raise SystemExit(-1)

    formatter = logging.Formatter(f"[%(asctime)s] [%(levelname)8s] [{task_name}] --- %(message)s (%(filename)s:%(lineno)s)")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    base_logger.setLevel(logging.DEBUG)
    base_logger.addHandler(console)
    base_logger.addHandler(ExitOnExceptionHandler())

    if log_file_path is not None:
        file = logging.FileHandler(log_file_path)
        file.setLevel(logging.DEBUG)
        file.setFormatter(formatter)
        logger.addHandler(file)

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


def rebuild_norm_layers(model, model_state, arg_ml_setup, epoch_of_rebuild, dataloader, rebuild_lr, existing_optimizer=None, rebuild_on_device=None, reset_norm_to_initial=False, initial_model_stat=None, display=False):
    if reset_norm_to_initial:
        assert initial_model_stat is not None
        # reset normalization layers
        for layer_name, layer_weights in model_state.items():
            if special_torch_layers.is_normalization_layer(arg_ml_setup.model_name, layer_name):
                model_state[layer_name] = initial_model_stat[layer_name]
    model.load_state_dict(model_state)

    if rebuild_on_device is None:
        rebuild_on_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    model.train()
    model.to(rebuild_on_device)
    rebuilding_normalization_count = 0
    criterion = arg_ml_setup.criterion

    if existing_optimizer is None:
        optimizer_rebuild_norm = torch.optim.SGD(model.parameters(), lr=rebuild_lr)
    else:
        optimizer_rebuild_norm = existing_optimizer
    cuda.CudaEnv.optimizer_to(optimizer_rebuild_norm, rebuild_on_device)
    rebuild_states = []
    for epoch in range(epoch_of_rebuild):
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
                if not special_torch_layers.is_normalization_layer(arg_ml_setup.model_name, layer_name):
                    current_model_stat[layer_name] = model_state[layer_name]
            model.load_state_dict(current_model_stat)
            rebuilding_normalization_count += 1
            rebuild_states.append((rebuilding_normalization_count, rebuilding_loss_val))
            if display:
                print(f"tick: {rebuilding_normalization_count}  loss: {rebuilding_loss_val}")
    output_model_state = model.state_dict()
    cuda.CudaEnv.model_state_dict_to(output_model_state, cpu_device)
    return output_model_state, rebuild_states


def process_file_func(arg_output_folder_path, start_model_path, end_model_path, arg_ml_setup, arg_lr, arg_lr_rebuild_norm, arg_max_tick, arg_training_round, arg_rebuild_normalization_round, arg_step_size, arg_adoptive_step_size, layer_skip_average, arg_worker_count, arg_total_cpu_count, arg_save_format, arg_use_cpu):
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

    arg_output_folder_path = os.path.join(arg_output_folder_path, f"{start_file_name}-{end_file_name}")
    if os.path.exists(arg_output_folder_path):
        child_logger.warning(f"{arg_output_folder_path} already exists")
    else:
        os.makedirs(arg_output_folder_path)

    # load models
    cpu_device = torch.device("cpu")
    start_model = copy.deepcopy(arg_ml_setup.model)
    initial_model_stat = start_model.state_dict()
    start_model_stat_dict = torch.load(start_model_path, map_location=cpu_device)
    end_model_stat_dict = torch.load(end_model_path, map_location=cpu_device)
    start_model.load_state_dict(start_model_stat_dict)
    start_model_optimizer_stat = torch.load(start_optimizer_path, map_location=cpu_device)  # load optimizer
    # assert start_model_state_dict != end_model_stat_dict
    for key in start_model_stat_dict.keys():
        assert not torch.equal(start_model_stat_dict[key], end_model_stat_dict[key]), f'starting model({start_model_path}) is same as ending model({end_model_path})'
        break

    # load training data
    training_dataset = arg_ml_setup.training_data
    dataloader = DataLoader(training_dataset, batch_size=arg_ml_setup.training_batch_size, shuffle=True)
    criterion = arg_ml_setup.criterion
    optimizer = torch.optim.SGD(start_model.parameters(), lr=arg_lr)
    optimizer.load_state_dict(start_model_optimizer_stat)
    if arg_rebuild_normalization_round != 0:
        if ENABLE_DEDICATED_TRAINING_DATASET_FOR_REBUILDING_NORM:
            epoch_for_rebuilding_norm = 1
            dataset_for_rebuilding_norm = arg_ml_setup.training_data_for_rebuilding_normalization
            dataset_rebuild_norm_size = arg_rebuild_normalization_round * arg_ml_setup.training_batch_size
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

    # services
    all_node_names = [0, 1]
    all_model_stats = [start_model_stat_dict, end_model_stat_dict]

    weight_diff_service = record_weights_difference.ServiceWeightsDifferenceRecorder(1)
    weight_diff_service.initialize_without_runtime_parameters(all_model_stats, arg_output_folder_path)
    variance_service = record_variance.ServiceVarianceRecorder(1)
    variance_service.initialize_without_runtime_parameters(all_node_names, all_model_stats, arg_output_folder_path)
    if arg_save_format != 'none':
        record_model_service = record_model_stat.ModelStatRecorder(1)
        record_model_service.initialize_without_runtime_parameters([0], arg_output_folder_path, save_format=arg_save_format)
    else:
        record_model_service = None
    record_test_accuracy_loss_service = record_test_accuracy_loss.ServiceTestAccuracyLossRecorder(1, 100, use_fixed_testing_dataset=True)
    record_test_accuracy_loss_service.initialize_without_runtime_parameters(arg_output_folder_path, [0], start_model, criterion, training_dataset)

    # begin finding path
    """rebuilding normalization for start and end points"""
    if ENABLE_REBUILD_NORM_FOR_STARTING_ENDING_MODEL:
        if arg_rebuild_normalization_round != 0:
            start_model_stat_dict, _ = rebuild_norm_layers(start_model, start_model_stat_dict, arg_ml_setup, epoch_for_rebuilding_norm,
                                                           dataloader_for_rebuilding_norm, arg_lr_rebuild_norm, optimizer,
                                                           rebuild_on_device=device, initial_model_stat=initial_model_stat, reset_norm_to_initial=True)
            end_model_stat_dict, _ = rebuild_norm_layers(start_model, end_model_stat_dict, arg_ml_setup, epoch_for_rebuilding_norm,
                                                         dataloader_for_rebuilding_norm, arg_lr_rebuild_norm, optimizer,
                                                         rebuild_on_device=device, initial_model_stat=initial_model_stat, reset_norm_to_initial=True)

    start_model_stat = start_model_stat_dict

    """pre training"""
    if ENABLE_PRE_TRAINING:
        child_logger.info(f"[{start_file_name}--{end_file_name}] pre training")
        start_model.load_state_dict(start_model_stat)
        cuda.CudaEnv.model_state_dict_to(start_model_stat, cpu_device)
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
            if training_index == 100:
                break
        start_model.load_state_dict(start_model_stat)

    current_tick = 0

    while current_tick < arg_max_tick:
        """record variance"""
        variance_record = model_variance_correct.VarianceCorrector(model_variance_correct.VarianceCorrectionType.FollowOthers)
        variance_record.add_variance(start_model_stat)
        """move tensor"""
        start_model_stat = model_average.move_model_state_toward(start_model_stat, end_model_stat_dict, arg_step_size, arg_adoptive_step_size, True, ignore_layer_keywords=layer_skip_average)
        if ENABLE_NAN_CHECKING:
            util.check_for_nans_in_state_dict(start_model_stat)
        """rescale variance"""
        target_variance = variance_record.get_variance()
        for layer_name, single_layer_variance in target_variance.items():
            if special_torch_layers.is_ignored_layer_averaging(layer_name):
                continue
            start_model_stat[layer_name] = model_variance_correct.VarianceCorrector.scale_tensor_to_variance(start_model_stat[layer_name], single_layer_variance)
        if ENABLE_NAN_CHECKING:
            util.check_for_nans_in_state_dict(start_model_stat)
        """training"""
        start_model.load_state_dict(start_model_stat)
        training_loss_val = None
        start_model.train()
        start_model.to(device)
        cuda.CudaEnv.optimizer_to(optimizer, device)
        for (training_index, (data, label)) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = start_model(data)
            training_loss = criterion(output, label)
            training_loss.backward()
            optimizer.step()
            training_loss_val = training_loss.item()
            if training_index == arg_training_round:
                break
            assert training_index < arg_training_round
        child_logger.info(f"[{start_file_name}--{end_file_name}] current tick: {current_tick}, training loss = {training_loss_val}")
        """rebuilding normalization"""
        if arg_rebuild_normalization_round != 0:
            start_model_stat = start_model.state_dict()
            start_model_stat, (rebuilding_iter, rebuilding_loss) = rebuild_norm_layers(start_model, start_model_stat, arg_ml_setup, epoch_for_rebuilding_norm,
                                                                                       dataloader_for_rebuilding_norm, arg_lr_rebuild_norm, optimizer,
                                                                                       rebuild_on_device=device, initial_model_stat=initial_model_stat, reset_norm_to_initial=True)
            child_logger.info(f"[{start_file_name}--{end_file_name}] current tick: {current_tick}, rebuilding finished at {rebuilding_iter} rounds, rebuilding loss = {rebuilding_loss}")

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
        variance_service.trigger_without_runtime_parameters(current_tick, all_node_names, all_model_stats)
        if record_model_service is not None:
            record_model_service.trigger_without_runtime_parameters(current_tick, [0], [start_model_stat])
        record_test_accuracy_loss_service.trigger_without_runtime_parameters(current_tick, {0: start_model_stat})

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
    parser.add_argument("-s", "--step_size", type=float, default=0.001)
    parser.add_argument("-a", "--adoptive_step_size", type=float, default=0)
    parser.add_argument("--training_round", type=int, default=1)
    parser.add_argument("-r", "--rebuild_norm_round", type=int, default=0, help='train for x rounds to rebuild the norm layers')
    parser.add_argument("--layer_skip_average", type=str, nargs='+')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_rebuild_norm", type=float, default=0.001)
    parser.add_argument("--save_format", type=str, default='none', choices=['none', 'file', 'lmdb'])
    parser.add_argument("--cpu", action='store_true', help='force using CPU for training')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    parser.add_argument("--use_predefined_optimal", action='store_true', help='use predefined optimal parameters')

    args = parser.parse_args()

    set_logging(logger, "main")
    logger.info("logging setup complete")

    start_folder = args.start_folder
    end_folder = args.end_folder
    mode = args.mapping_mode
    max_tick = args.max_tick
    step_size = args.step_size
    adoptive_step_size = args.adoptive_step_size
    training_round = args.training_round
    rebuild_normalization_round = args.rebuild_norm_round
    learning_rate = args.lr
    learning_rate_rebuild_norm = args.lr_rebuild_norm
    use_cpu = args.cpu
    paths_to_find = get_files_to_process(args.start_folder, args.end_folder, mode)
    save_format = args.save_format
    layer_skip_average = args.layer_skip_average
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
    args = [(output_folder_path, start_file, end_file, current_ml_setup, learning_rate, learning_rate_rebuild_norm, max_tick, training_round, rebuild_normalization_round, step_size, adoptive_step_size, layer_skip_average, worker_count, total_cpu_count, save_format, use_cpu) for (start_file, end_file) in paths_to_find]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_file_func, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
