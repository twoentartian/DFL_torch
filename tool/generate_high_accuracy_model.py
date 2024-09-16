import argparse
import torch
import os
import sys
import random
import copy
import json
import numpy as np
from datetime import datetime
import concurrent.futures
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, complete_ml_setup
from py_src.service import record_model_stat

def re_initialize_model(model, arg_ml_setup):
    random_data = os.urandom(4)
    seed = int.from_bytes(random_data, byteorder="big")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if arg_ml_setup.weights_init_func is None:
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    else:
        model.apply(arg_ml_setup.weights_init_func)


def training_model(output_folder, index, arg_number_of_models, arg_ml_setup: ml_setup, arg_use_cpu: bool, arg_worker_count, arg_total_cpu_count, arg_save_format):
    thread_per_process = arg_total_cpu_count // arg_worker_count
    torch.set_num_threads(thread_per_process)
    if arg_use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digit_number_of_models = len(str(arg_number_of_models))
    model = copy.deepcopy(arg_ml_setup.model)
    model.to(device)
    dataset = copy.deepcopy(arg_ml_setup.training_data)
    batch_size = arg_ml_setup.training_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = arg_ml_setup.criterion
    optimizer, lr_scheduler, epochs = complete_ml_setup.FastTrainingSetup.get_optimizer_lr_scheduler_epoch(arg_ml_setup, model)

    # services
    if arg_save_format != 'none':
        record_model_service = record_model_stat.ModelStatRecorder(1)
        record_model_service.initialize_without_runtime_parameters([0], output_folder, save_format=arg_save_format, lmdb_db_name=f"{str(index).zfill(digit_number_of_models)}")
    else:
        record_model_service = None

    # reset random weights
    re_initialize_model(model, arg_ml_setup)

    model.train()
    print(f"INDEX[{index}] begin training")
    for epoch in range(epochs):
        train_loss = 0
        count = 0
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            train_loss += loss.item()
            count += 1
        lrs = []
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
        print(f"INDEX[{index}] epoch[{epoch}] loss={train_loss/count} lrs={lrs}")

        # services
        if record_model_service is not None:
            model_stat = model.state_dict()
            record_model_service.trigger_without_runtime_parameters(epoch, [0], [model_stat])
    print(f"INDEX[{index}] finish training")

    model_info = {}
    model_info["state_dict"] = model.state_dict()
    model_info["model_name"] = arg_ml_setup.model_name
    torch.save(model_info, os.path.join(output_folder, f"{str(index).zfill(digit_number_of_models)}.model.pt"))
    optimizer_info = {}
    optimizer_info["state_dict"] = optimizer.state_dict()
    optimizer_info["model_name"] = arg_ml_setup.model_name
    torch.save(optimizer_info, os.path.join(output_folder, f"{str(index).zfill(digit_number_of_models)}.optimizer.pt"))
    del model, dataset, dataloader, criterion, optimizer, model_info, optimizer_info
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Generate some high accuracy models')
    parser.add_argument("-n", "--number_of_models", type=int, default=1)
    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("-t", "--thread", type=int, default=1, help='specify how many models to train in parallel')
    parser.add_argument("-m", "--model_type", type=str, default='lenet5', choices=['lenet5', 'resnet18'])
    parser.add_argument("--norm_method", type=str, default='auto', choices=['auto', 'bn', 'ln', 'gn'])
    parser.add_argument("--cpu", action='store_true', help='force using CPU for training')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    parser.add_argument("--save_format", type=str, default='none', choices=['none', 'file', 'lmdb'], help='which format to save the training states')

    args = parser.parse_args()

    number_of_models = args.number_of_models
    worker_count = args.thread
    total_cpu_cores = args.core
    model_type = args.model_type
    use_cpu = args.cpu
    output_folder_name = args.output_folder_name
    save_format = args.save_format
    norm_method = args.norm_method

    # prepare model and dataset
    current_ml_setup = ml_setup.get_ml_setup_from_config(model_type, norm_method)

    output_model_name = current_ml_setup.model_name

    # create output folder
    if output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, output_folder_name)
    os.mkdir(output_folder_path)

    # write info file
    info_content = {}
    info_content['model_type'] = current_ml_setup.model_name
    info_content['model_count'] = number_of_models
    info_content['generated_by_cpu'] = use_cpu
    if current_ml_setup.has_normalization_layer:
        info_content['norm_method'] = norm_method
    json_data = json.dumps(info_content)
    with open(os.path.join(output_folder_path, 'info.json'), 'w') as f:
        f.write(json_data)

    # training
    if worker_count > number_of_models:
        worker_count = number_of_models
    args = [(output_folder_path, i, number_of_models, current_ml_setup, use_cpu, worker_count, total_cpu_cores, save_format) for i in range(number_of_models)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(training_model, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
