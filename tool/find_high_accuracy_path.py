import os
import argparse
import logging
import sys
import torch
import torch.optim as optim
import concurrent.futures
import copy
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, model_average, model_variance_correct, special_torch_layers, cuda
from py_src.service import record_weights_difference, record_test_accuracy_loss, record_variance, record_model_stat

logger = logging.getLogger("find_high_accuracy_path")

def set_logging(base_logger, task_name):
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

    del console, formatter

def get_files_to_process(start_folder, end_folder):
    if not os.path.isdir(start_folder):
        logger.critical(f"{start_folder} does not exist")
    if not os.path.isdir(end_folder):
        logger.critical(f"{end_folder} does not exist")

    files_in_folder1 = set(os.listdir(start_folder))
    files_in_folder2 = set(os.listdir(end_folder))

    if len(files_in_folder1) != len(files_in_folder2):
        logger.critical(f"file counts mismatch: {len(files_in_folder1)} != {len(files_in_folder2)}")

    if files_in_folder1 != files_in_folder2:
        logger.critical(f"file names mismatch: {files_in_folder1} != {files_in_folder2}")

    file_names = []
    for file in files_in_folder1:
        file_names.append(file)

    return sorted(file_names)

def get_file_name_without_extension(file_path):
    base_name = os.path.basename(file_path)  # Get the file name with extension
    file_name, _ = os.path.splitext(base_name)  # Split the name and extension
    return file_name

class InverseLRScheduler(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(InverseLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [base_lr / (1 + self.gamma * self.last_epoch) for base_lr in self.base_lrs]


def process_file_func(output_folder_path, start_model_path, end_model_path, arg_ml_setup, arg_lr, arg_max_tick, arg_training_round, arg_step_size, arg_adoptive_step_size, arg_worker_count):
    thread_per_process = os.cpu_count() // worker_count
    torch.set_num_threads(thread_per_process)

    start_file_name = get_file_name_without_extension(start_model_path)
    end_file_name = get_file_name_without_extension(end_model_path)
    assert start_file_name == end_file_name
    output_folder_path = os.path.join(output_folder_path, f"{start_file_name}-{end_file_name}")
    if os.path.exists(output_folder_path):
        logger.critical(f"{output_folder_path} already exists")
    else:
        os.makedirs(output_folder_path)

    # load models
    cpu_device = torch.device("cpu")
    start_model = arg_ml_setup.model
    start_model_stat_dict = torch.load(start_model_path, map_location=cpu_device)
    end_model_state_dict = torch.load(end_model_path, map_location=cpu_device)
    # cuda.CudaEnv.model_state_dict_to(start_model_stat_dict, cpu_device)
    # cuda.CudaEnv.model_state_dict_to(end_model_state_dict, cpu_device)
    start_model.load_state_dict(start_model_stat_dict)

    # load training data
    training_dataset = arg_ml_setup.training_data
    dataloader = DataLoader(training_dataset, batch_size=arg_ml_setup.training_batch_size, shuffle=True)
    criterion = arg_ml_setup.criterion
    optimizer = torch.optim.SGD(start_model.parameters(), lr=arg_lr)

    # services
    all_node_names = [0, 1]
    all_model_stats = [start_model_stat_dict, end_model_state_dict]

    weight_diff_service = record_weights_difference.ServiceWeightsDifferenceRecorder(1)
    weight_diff_service.initialize_without_runtime_parameters(all_model_stats, output_folder_path)
    variance_service = record_variance.ServiceVarianceRecorder(1)
    variance_service.initialize_without_runtime_parameters(all_node_names, all_model_stats, output_folder_path)
    record_model_service = record_model_stat.ModelStatRecorder(1)
    record_model_service.initialize_without_runtime_parameters([0], output_folder_path)
    record_test_accuracy_loss_service = record_test_accuracy_loss.ServiceTestAccuracyLossRecorder(1, 100, use_fixed_testing_dataset=True)
    record_test_accuracy_loss_service.initialize_without_runtime_parameters(output_folder_path, [0], start_model, criterion, training_dataset)

    # begin finding path
    """pre training"""
    start_model_stat = start_model.state_dict()
    for (training_index, (data, label)) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)
        output = start_model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        if training_index == 100:
            break
    start_model.load_state_dict(start_model_stat)

    current_tick = 0
    while current_tick < arg_max_tick:
        """record variance"""
        variance_record = model_variance_correct.VarianceCorrector(model_variance_correct.VarianceCorrectionType.FollowOthers)
        variance_record.add_variance(start_model_stat)
        """move tensor"""
        for layer_name in start_model_stat.keys():
            dst_tensor = end_model_state_dict[layer_name]
            start_model_stat[layer_name] = model_average.move_tensor_toward(start_model_stat[layer_name], dst_tensor, arg_step_size, arg_adoptive_step_size)
        """rescale variance"""
        target_variance = variance_record.get_variance()
        for layer_name, single_layer_variance in target_variance.items():
            if special_torch_layers.is_ignored_layer(layer_name):
                continue
            start_model_stat[layer_name] = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(start_model_stat[layer_name], single_layer_variance)
        """training"""
        start_model.load_state_dict(start_model_stat)
        loss_val = None
        for (training_index, (data, label)) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            output = start_model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            if training_index == arg_training_round:
                break
        start_model_stat = start_model.state_dict()
        """scale variance back, due to SGD variance drift"""
        for layer_name, single_layer_variance in target_variance.items():
            if special_torch_layers.is_ignored_layer(layer_name):
                continue
            start_model_stat[layer_name] = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(start_model_stat[layer_name], single_layer_variance)

        # service
        all_model_stats = [start_model_stat, end_model_state_dict]
        weight_diff_service.trigger_without_runtime_parameters(current_tick, all_model_stats)
        variance_service.trigger_without_runtime_parameters(current_tick, all_node_names, all_model_stats)
        record_model_service.trigger_without_runtime_parameters(current_tick, [0], [start_model_stat])
        record_test_accuracy_loss_service.trigger_without_runtime_parameters(current_tick, {0: start_model_stat})

        current_tick += 1
        print(f"[{start_file_name}] current tick: {current_tick}, training loss = {loss_val}")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Generate some high accuracy models')
    parser.add_argument("start_folder", type=str, help="folder containing starting models")
    parser.add_argument("end_folder", type=str, help="folder containing destination models")
    parser.add_argument("-c", '--cores', type=int, default=os.cpu_count(), help='specify how many models to train in parallel')
    parser.add_argument("-m", "--model_type", type=str, default='lenet5', choices=['lenet5', 'resnet18'])
    parser.add_argument("-t", "--max_tick", type=int, default=10000)
    parser.add_argument("-s", "--step_size", type=float, default=0.002)
    parser.add_argument("-a", "--adoptive_step_size", type=float, default=0.001)
    parser.add_argument("--training_round", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)

    args = parser.parse_args()

    set_logging(logger, "main")
    logger.info("logging setup complete")

    start_folder = args.start_folder
    end_folder = args.end_folder
    max_tick = args.max_tick
    step_size = args.step_size
    adoptive_step_size = args.adoptive_step_size
    training_round = args.training_round
    learning_rate = args.lr
    files = get_files_to_process(args.start_folder, args.end_folder)
    file_count = len(files)
    logger.info(f"totally {len(files)} files to process: {files}")

    worker_count = args.cores
    model_type = args.model_type

    # prepare model and dataset
    current_ml_setup = None
    if model_type == 'lenet5':
        current_ml_setup = ml_setup.mnist_lenet5()
    elif model_type == 'resnet18':
        current_ml_setup = ml_setup.resnet18_cifar10()
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    # create output folder
    output_folder_path = os.path.join(os.curdir, f"{__file__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")}")
    os.mkdir(output_folder_path)

    # finding path
    if worker_count > file_count:
        worker_count = file_count
    args = [(output_folder_path, os.path.join(start_folder, f), os.path.join(end_folder, f), current_ml_setup, learning_rate, max_tick, training_round, step_size, adoptive_step_size, worker_count) for f in files]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_file_func, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()

