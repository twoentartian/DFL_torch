import argparse
import torch
import random
import os
import sys
import numpy as np
import copy
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, util, special_torch_layers


def get_optimizer_lr_scheduler_epoch(arg_ml_setup: ml_setup, model):
    if arg_ml_setup.model_name == 'lenet5':
        raise NotImplementedError
    elif arg_ml_setup.model_name == 'resnet18_bn':
        lr = 0.1
        epochs = 30
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.training_batch_size + 1
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
        return optimizer, lr_scheduler, epochs
    elif arg_ml_setup.model_name == 'resnet18_gn':
        raise NotImplementedError
    else:
        raise NotImplementedError

def get_layers_follow_initial_model(arg_ml_setup: ml_setup):
    if arg_ml_setup.model_name == 'lenet5':
        raise NotImplementedError
    elif arg_ml_setup.model_name == 'resnet18_bn':
        return ["num_batches_tracked", "running_mean", "running_var"]
    elif arg_ml_setup.model_name == 'resnet18_gn':
        raise NotImplementedError
    else:
        raise NotImplementedError


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser(description='Rebuild a new model based on the existing model weights, retrain from existing variance')
    parser.add_argument("model_path", type=str, help="model path")
    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')

    args = parser.parse_args()

    # args
    total_cpu_count = args.core
    model_path = args.model_path

    torch.set_num_threads(args.core)

    assert os.path.exists(model_path), f"model file {model_path} does not exist"
    model_file_name = os.path.basename(model_path)
    assert '.model.pt' in model_file_name, f"model file {model_file_name} does not have .model.pt extension"

    cpu_device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    starting_model_info = torch.load(model_path, map_location=cpu_device)
    starting_model_stat = starting_model_info["state_dict"]
    model_type = starting_model_info["model_name"]

    current_ml_setup = ml_setup.get_ml_setup_from_config(model_type)

    print(f'Current ML setup: {current_ml_setup.model_name}')

    # create output folder
    if args.output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.mkdir(output_folder_path)

    """ start retraining """
    target_model = copy.deepcopy(current_ml_setup.model)
    optimizer, lr_scheduler, epochs = get_optimizer_lr_scheduler_epoch(current_ml_setup, target_model)
    keywords_of_layers_follow_initial = get_layers_follow_initial_model(current_ml_setup)

    initial_model_state = target_model.state_dict()
    new_state_dict = {}
    # re-generate the model weights
    for layer_name, weights in starting_model_stat.items():
        if special_torch_layers.is_keyword_in_layer_name(layer_name, keywords_of_layers_follow_initial):
            new_state_dict[layer_name] = initial_model_state[layer_name]
            continue
        variance = torch.var(weights)
        new_weights = torch.randn_like(weights) * torch.sqrt(variance)
        new_state_dict[layer_name] = new_weights

    target_model.load_state_dict(new_state_dict)

    training_dataset = current_ml_setup.training_data
    dataloader = DataLoader(training_dataset, batch_size=current_ml_setup.training_batch_size, shuffle=True)
    criterion = current_ml_setup.criterion

    log_file = open(os.path.join(output_folder_path, f"retrained.log"), "w")
    log_file.write("epoch,loss,lrs" + "\n")
    log_file.flush()

    target_model.train()
    target_model.to(device)
    for epoch in range(epochs):
        train_loss = 0
        count = 0
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = target_model(data)
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
        print(f"epoch[{epoch}] loss={train_loss/count} lrs={lrs}")
        log_file.write(f"{epoch},{train_loss/count},{lrs}" + "\n")
        log_file.flush()
    log_file.flush()
    log_file.close()

    util.save_model_state(os.path.join(output_folder_path, f"retrained.model.pt"),
                          target_model.state_dict(), current_ml_setup.model_name)

