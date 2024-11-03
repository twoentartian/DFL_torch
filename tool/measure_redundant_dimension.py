import argparse
import os
import sys
import torch
import copy
import logging
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util, ml_setup, cuda, model_average

logger = logging.getLogger("find_high_accuracy_path")

def extract_models_in_folder(folder):
    if not os.path.isdir(folder):
        print(f"{folder} does not exist")
    files_in_start_folder = sorted(set(temp_file for temp_file in os.listdir(folder) if temp_file.endswith('model.pt')))
    return files_in_start_folder

def get_precise_training_setup(model, model_name, current_ml_setup):
    if model_name == 'lenet5' or model_name == 'resnet18_large_fc':
        training_dataset = current_ml_setup.training_data
        dataloader = DataLoader(training_dataset, batch_size=1000, shuffle=True, num_workers=8)
        criterion = current_ml_setup.criterion
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        lr_scheduler = None
    else:
        raise NotImplemented
    return dataloader, criterion, optimizer, lr_scheduler

def train_model(model, optimizer, dataloader, criterion, training_round, model_name=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    round = 0
    total_loss = 0
    print_loss_interval = 100
    model_name_info = f"[{model_name}]" if model_name != '' else ''
    while round < training_round:
        for (data, label) in dataloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            total_loss += loss_val
            round += 1
            if round % print_loss_interval == 0:
                print(f"{model_name_info}round: {round} training loss={total_loss/print_loss_interval:.4f}")
                total_loss = 0
            if round == training_round:
                break
    print(f"{model_name_info}finish training, round={round}, loss={total_loss/training_round:.4f}")
    return model

def train_all_models(model_start, models_end, step_size, adoptive_step_size):
    """train the starting model"""
    model_state_dict, model_name = util.load_model_state_file(model_start)
    print(f"starting model type: {model_name}")
    current_ml_setup = ml_setup.get_ml_setup_from_model_type(model_name)
    model = copy.deepcopy(current_ml_setup.model)
    print(f"train the starting model")

    """load starting model"""
    cpu_device = torch.device('cpu')
    dataloader, criterion, optimizer, lr_scheduler = get_precise_training_setup(model, model_name, current_ml_setup)
    model.load_state_dict(model_state_dict)
    train_model(model, optimizer, dataloader, criterion, training_round, model_name='start')
    start_model_state = model.state_dict()
    cuda.CudaEnv.model_state_dict_to(start_model_state, cpu_device)
    util.save_model_state(os.path.join(output_folder_path, "start.model.pt"), start_model_state)

    """load destination models"""
    end_model_states = []
    for end_model_path in models_end:
        end_model_state_dict, end_model_name = util.load_model_state_file(end_model_path)
        current_model_state = model_average.move_model_state_toward(start_model_state, end_model_state_dict, step_size, adoptive_step_size)
        assert end_model_name == model_name, f"model name mismatch {model_name} != {end_model_name}"
        dataloader, criterion, optimizer, lr_scheduler = get_precise_training_setup(model, model_name, current_ml_setup)
        model.load_state_dict(current_model_state)
        end_file_name = os.path.basename(end_model_path)
        end_file_name = end_file_name.split(".")[0]
        train_model(model, optimizer, dataloader, criterion, training_round, model_name=f"{end_file_name}")
        end_model_state = model.state_dict()
        cuda.CudaEnv.model_state_dict_to(end_model_state, cpu_device)
        util.save_model_state(os.path.join(output_folder_path, f"end_{end_file_name}.model.pt"), end_model_state)
        end_model_states.append(end_model_state)

    return start_model_state, end_model_states

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Try to calculate the dimension of redundant high accuracy space')

    parser.add_argument("start_folder", type=str, help="folder containing starting models")
    parser.add_argument("end_folder", type=str, help="folder containing destination models")

    parser.add_argument("-s", "--step_size", type=float, default=0.001)
    parser.add_argument("-a", "--adoptive_step_size", type=float, default=0)
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    parser.add_argument("-r", "--training_round", type=int, default=10, help='specify the round of training')

    args = parser.parse_args()

    training_round = args.training_round

    """output folder"""
    if args.output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.mkdir(output_folder_path)

    model_start = extract_models_in_folder(args.start_folder)[0]
    print(f"starting model: {model_start}")
    model_start = os.path.join(args.start_folder, model_start)
    models_end = extract_models_in_folder(args.end_folder)
    print(f"destination {len(models_end)} models: {models_end}")
    models_end = [os.path.join(args.end_folder, model) for model in models_end]

    start_model_state, end_model_states = train_all_models(model_start, models_end, args.step_size, args.adoptive_step_size)

    """calculate the dimension of high accuracy space"""
    total_layers = start_model_state.keys()
    for layer_name in total_layers:
        layers_of_end_models = []
        for end_model in end_model_states:
            diff = end_model[layer_name] - start_model_state[layer_name]
            layers_of_end_models.append(diff.flatten())
        layers_of_end_models = torch.stack(layers_of_end_models)
        u, s, vh = torch.linalg.svd(layers_of_end_models)
        threshold_list = []
        effective_rank_list = []
        for threshold in np.logspace(-6, 2, num=1000):
            effective_rank = torch.sum(s > float(threshold)).item()
            threshold_list.append(threshold)
            effective_rank_list.append(effective_rank)

        print(f"plotting for {layer_name}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(threshold_list, effective_rank_list, marker='o', linestyle='-', label='dimension')
        ax.set_xlabel('threshold')
        ax.set_ylabel('dimension')
        ax.set_xscale('log')
        ax.set_title('Dimension of high accuracy space')
        ax.grid(True)
        ax.legend()
        fig.savefig(os.path.join(output_folder_path, f"{layer_name}.pdf"))
        plt.close(fig)