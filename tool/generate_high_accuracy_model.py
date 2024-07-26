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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def re_initialize_model(model, ml_setup):
    random_data = os.urandom(4)
    seed = int.from_bytes(random_data, byteorder="big")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if ml_setup.weights_init_func is None:
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    else:
        model.apply(ml_setup.weights_init_func)


def training_model(output_folder, index, number_of_models, complete_ml_setup: complete_ml_setup.PredefinedCompleteMlSetup):
    digit_number_of_models = len(str(number_of_models))
    model = copy.deepcopy(complete_ml_setup.ml_setup.model)
    model.to(device)
    dataset = copy.deepcopy(complete_ml_setup.ml_setup.training_data)
    batch_size = complete_ml_setup.ml_setup.training_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = complete_ml_setup.ml_setup.criterion
    optimizer = copy.deepcopy(complete_ml_setup.optimizer)
    epochs = complete_ml_setup.epochs

    # reset random weights
    re_initialize_model(model, complete_ml_setup.ml_setup)

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
            train_loss += loss.item()
            count += 1
        print(f"INDEX[{index}] epoch[{epoch}] loss={train_loss/count}")
    print(f"INDEX[{index}] finish training")

    torch.save(model.state_dict(), os.path.join(output_folder, f"{str(index).zfill(digit_number_of_models)}.pt"))
    del model, dataset, dataloader, criterion, optimizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Generate some high accuracy models')
    parser.add_argument("-n", "--number_of_models", type=int, default=1)
    parser.add_argument("-c", '--cores', type=int, default=os.cpu_count(), help='specify how many models to train in parallel')
    parser.add_argument("-m", "--model_type", type=str, default='lenet5', choices=['lenet5', 'resnet18'])

    args = parser.parse_args()

    number_of_models = args.number_of_models
    worker_count = args.cores
    model_type = args.model_type

    # prepare model and dataset
    current_complete_ml_setup = None
    if model_type == 'lenet5':
        current_complete_ml_setup = complete_ml_setup.PredefinedCompleteMlSetup.get_lenet5()
    elif model_type == 'resnet18':
        current_complete_ml_setup = complete_ml_setup.PredefinedCompleteMlSetup.get_resnet18()
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    # create output folder
    output_folder_path = os.path.join(os.curdir, f"{__file__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")}")
    os.mkdir(output_folder_path)

    # training
    if worker_count > number_of_models:
        worker_count = number_of_models
    args = [(output_folder_path, i, number_of_models, current_complete_ml_setup) for i in range(number_of_models)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(training_model, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
