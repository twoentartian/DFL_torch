import os
import torch
from torch.utils.data import DataLoader
import argparse
import json
import sys
import copy
import csv
import hashlib
from datetime import datetime
from typing import Final

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

    dataset_rebuild_norm_size = rebuild_round * current_ml_setup.training_batch_size
    indices = torch.randperm(len(current_ml_setup.training_data_for_rebuilding_normalization))[:dataset_rebuild_norm_size]
    index_dataset = IndexedDataset(current_ml_setup.training_data_for_rebuilding_normalization)
    sub_dataset = torch.utils.data.Subset(index_dataset, indices.tolist())
    dataloader_for_rebuilding_norm = DataLoader(sub_dataset, batch_size=current_ml_setup.training_batch_size)
    criterion = current_ml_setup.criterion

    for rebuild_count_index in range(rebuild_count):
        optimizer = torch.optim.SGD(target_model.parameters(), lr=lr)
        cuda.CudaEnv.optimizer_to(optimizer, device)

        starting_model_stat = torch.load(model_path, map_location=cpu_device)

        # reset normalization layers
        for layer_name, layer_weights in starting_model_stat.items():
            if special_torch_layers.is_normalization_layer(current_ml_setup.model_name, layer_name):
                starting_model_stat[layer_name] = initial_model_stat[layer_name]

        # prepare for rebuild
        target_model.load_state_dict(starting_model_stat)
        target_model.to(device)
        rebuilding_normalization_index = None
        rebuilding_loss_val = None

        # rebuild norm
        for (rebuilding_normalization_index, (data, label, indices)) in enumerate(dataloader_for_rebuilding_norm):
            if rebuilding_normalization_index == 0:
                data_numpy = data.numpy()
                tensor_bytes = data_numpy.tobytes()
                hash_object = hashlib.sha256(tensor_bytes)
                tensor_hash = hash_object.hexdigest()
                print(f"{rebuilding_normalization_index} round: hash of training data batch is {tensor_hash}")

            data, label = data.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = target_model(data)
            rebuilding_loss = criterion(output, label)
            rebuilding_loss.backward()
            rebuilding_loss_val = rebuilding_loss.item()
            optimizer.step()
            # reset all layers except normalization
            target_model_stat = target_model.state_dict()
            for layer_name, layer_weights in target_model_stat.items():
                if not special_torch_layers.is_normalization_layer(current_ml_setup.model_name, layer_name):
                    target_model_stat[layer_name] = starting_model_stat[layer_name]
            target_model.load_state_dict(target_model_stat)

        print(f"rebuild norm layer finished at {rebuild_round} rounds, rebuilding loss = {rebuilding_loss_val}")

        # save model state
        save_csv_file_name = f'{output_folder_path}/{rebuild_count_index}.csv'
        state_dict = target_model.state_dict()
        save_first_norm_layer_to_file(save_csv_file_name, state_dict)





