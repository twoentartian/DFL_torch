import argparse
import torch
import os
import sys
import random
import copy
import numpy as np
from datetime import datetime
import concurrent.futures
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, complete_ml_setup

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


def training_model(output_folder, index, arg_number_of_models, arg_ml_setup: ml_setup, arg_use_cpu: bool, arg_worker_count):
    thread_per_process = os.cpu_count() // arg_worker_count
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
    print(f"INDEX[{index}] finish training")

    torch.save(model.state_dict(), os.path.join(output_folder, f"{str(index).zfill(digit_number_of_models)}.pt"))
    del model, dataset, dataloader, criterion, optimizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Generate some high accuracy models')
    parser.add_argument("-n", "--number_of_models", type=int, default=1)
    parser.add_argument("-c", '--parallel', type=int, default=os.cpu_count(), help='specify how many models to train in parallel')
    parser.add_argument("-m", "--model_type", type=str, default='lenet5', choices=['lenet5', 'resnet18'])
    parser.add_argument("--cpu", action='store_true', help='force using CPU for training')

    args = parser.parse_args()

    number_of_models = args.number_of_models
    worker_count = args.cores
    model_type = args.model_type
    use_cpu = args.cpu

    # prepare model and dataset
    current_ml_setup = None
    if model_type == 'lenet5':
        current_ml_setup = ml_setup.lenet5_mnist()
    elif model_type == 'resnet18':
        current_ml_setup = ml_setup.resnet18_cifar10()
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    # create output folder
    output_folder_path = os.path.join(os.curdir, f"{__file__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")}")
    os.mkdir(output_folder_path)

    # training
    if worker_count > number_of_models:
        worker_count = number_of_models
    args = [(output_folder_path, i, number_of_models, current_ml_setup, use_cpu, worker_count) for i in range(number_of_models)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(training_model, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
