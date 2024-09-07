import os
import torch
from torch.utils.data import DataLoader
import argparse
import json
import sys
import copy
import csv
import hashlib
import numpy as np
import random
from datetime import datetime
from typing import Final

import find_high_accuracy_path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, special_torch_layers, cuda


MAX_CPU_COUNT: Final[int] = 32

def save_first_norm_layer_to_file(file_path, model_state):
    bn_layer_name = None
    for key in model_state.keys():
        if "bn" in key and "weight" in key:
            bn_layer_name = key
            break
    if bn_layer_name:
        # Extract the weights of the first bn layer
        bn_weights = model_state[bn_layer_name].cpu().numpy()

        # Save the weights to a CSV file
        with open(file_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Weights'])  # Header
            writer.writerows([[weight] for weight in bn_weights])


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Get the data and label from the original dataset
        data, label = self.original_dataset[idx]
        # Return the data, label, and index
        return data, label, idx


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Rebuild norm layer for a model')
    parser.add_argument("model_path", type=str, help="model path")

    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("-r", "--rebuild_norm_round", type=int, default=50)
    parser.add_argument("-t", "--rebuild_count", type=int, default=1)
    parser.add_argument("-m", "--model_type", type=str, default='auto', choices=['auto', 'lenet5', 'resnet18_bn', 'resnet18_gn'])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')

    args = parser.parse_args()

    # args
    total_cpu_count = args.core
    if total_cpu_count > MAX_CPU_COUNT:
        total_cpu_count = MAX_CPU_COUNT
    model_path = args.model_path
    model_type = args.model_type
    rebuild_round = args.rebuild_norm_round
    rebuild_count = args.rebuild_count
    lr = args.lr

    # envs
    torch.set_num_threads(args.core)

    assert os.path.exists(model_path), f"model file {model_path} does not exist"
    model_file_name = os.path.basename(model_path)
    assert '.model.pt' in model_file_name, f"model file {model_file_name} does not have .model.pt extension"
    if model_type == 'auto':
        folder_path = os.path.dirname(model_path)
        model_info_file = os.path.join(folder_path, 'info.json')
        assert os.path.exists(model_info_file), f"model info file {model_info_file} does not exist, please specify model type with -m"
        with open(model_info_file) as f:
            model_info = json.load(f)
        model_type = model_info['model_type']

    current_ml_setup = ml_setup.get_ml_setup_from_model_type(model_type)

    print(f'Current ML setup: {current_ml_setup.model_name}')

    # create output folder
    if args.output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.mkdir(output_folder_path)

    """ start rebuilding norm """
    cpu_device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_model = copy.deepcopy(current_ml_setup.model)
    initial_model_stat = target_model.state_dict()

    optimizer_stat_path = model_path.replace('model.pt', 'optimizer.pt')
    assert os.path.exists(optimizer_stat_path), f'starting optimizer {optimizer_stat_path} is missing'

    dataset = current_ml_setup.training_data_for_rebuilding_normalization
    dataset_rebuild_norm_size = rebuild_round * current_ml_setup.training_batch_size
    epoch_for_rebuilding_norm = 1
    if dataset_rebuild_norm_size > len(dataset):
        ratio = int(dataset_rebuild_norm_size // len(dataset) + 1)
        assert dataset_rebuild_norm_size % ratio == 0, f"dataset_rebuild_norm_size={dataset_rebuild_norm_size}, ratio={ratio}"
        dataset_rebuild_norm_size = dataset_rebuild_norm_size // ratio
        epoch_for_rebuilding_norm = epoch_for_rebuilding_norm * ratio

    indices = torch.randperm(len(current_ml_setup.training_data_for_rebuilding_normalization))[:dataset_rebuild_norm_size]
    # index_dataset = IndexedDataset(current_ml_setup.training_data_for_rebuilding_normalization)
    sub_dataset = torch.utils.data.Subset(current_ml_setup.training_data_for_rebuilding_normalization, indices.tolist())
    dataloader_for_rebuilding_norm = DataLoader(sub_dataset, batch_size=current_ml_setup.training_batch_size)
    criterion = current_ml_setup.criterion

    for rebuild_count_index in range(rebuild_count):
        starting_model_stat = torch.load(model_path, map_location=cpu_device)

        # optimizer = torch.optim.SGD(target_model.parameters(), lr=lr)
        # optimizer_stat = torch.load(optimizer_stat_path, map_location=cpu_device)
        # optimizer.load_state_dict(optimizer_stat)

        optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)
        final_state, rebuild_states = find_high_accuracy_path.rebuild_norm_layers(target_model, starting_model_stat, current_ml_setup,
                                                                                  epoch_for_rebuilding_norm, dataloader_for_rebuilding_norm, lr,
                                                                                  existing_optimizer=optimizer, rebuild_on_device=device,
                                                                                  initial_model_stat=initial_model_stat, reset_norm_to_initial=True, display=True)
        rebuild_iter, rebuilding_loss_val = rebuild_states[-1]
        print(f"rebuild norm layer finished at {rebuild_iter} rounds, rebuilding loss = {rebuilding_loss_val}")

        # save model state
        save_csv_file_name = f'{output_folder_path}/{rebuild_count_index}.csv'
        state_dict = target_model.state_dict()
        save_first_norm_layer_to_file(save_csv_file_name, state_dict)




