import torch
from torch.utils.data import DataLoader
import os
import argparse
from datetime import datetime
import sys
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="path to the starting model state")
    parser.add_argument('-o', '--output', type=str, help="output path")

    args = parser.parse_args()

    if args.output is None:
        now_str = datetime.now().strftime("trained_model_%Y-%m-%d_%H-%M-%S_%f")
        args.output = f"{now_str}.model.pt"

    # args
    model_path = args.model
    output_folder_name = args.output

    # torch devices
    cpu_device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_info = torch.load(model_path, map_location=cpu_device)
    model_type = model_info["model_name"]
    model_state_dict = model_info["state_dict"]

    current_ml_setup = ml_setup.get_ml_setup_from_model_type(model_type)

    print(f'Current ML setup: {current_ml_setup.model_name}')

    # create output folder
    if output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, output_folder_name)
    os.mkdir(output_folder_path)

    target_model = copy.deepcopy(current_ml_setup.model)
    target_model.load_state_dict(model_state_dict)
    target_model.to(device)

    dataset = copy.deepcopy(current_ml_setup.training_data)
    batch_size = current_ml_setup.training_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = current_ml_setup.criterion

    if current_ml_setup.model_name == "resnet18_bn":
        epochs = 30
        optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        steps_per_epoch = len(dataset) // current_ml_setup.training_batch_size + 1
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, steps_per_epoch=steps_per_epoch, epochs=epochs)
    else:
        raise NotImplementedError()

    target_model.train()
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
    print(f"finish training")

    model_info = {}
    model_info["state_dict"] = target_model.state_dict()
    model_info["model_name"] = current_ml_setup.model_name
    torch.save(model_info, os.path.join(output_folder_path, f"trained.model.pt"))

    optimizer_info = {}
    optimizer_info["state_dict"] = optimizer.state_dict()
    optimizer_info["model_name"] = current_ml_setup.model_name
    torch.save(optimizer_info, os.path.join(output_folder_path, f"trained.optimizer.pt"))


