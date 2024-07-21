import argparse
import torch
import os
import sys
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup


def testing_model(model, current_ml_setup):
    testing_dataset = current_ml_setup.testing_data
    criterion = current_ml_setup.criterion
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = DataLoader(testing_dataset, batch_size=100, shuffle=True)

    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            loss = criterion(outputs, label)

            test_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
    return test_loss / total, correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure model test accuracy and loss.')
    parser.add_argument("model_file", type=str)
    parser.add_argument("-m", "--model_type", type=str, default='lenet5', choices=['lenet5', 'resnet18'])

    args = parser.parse_args()

    model_file_path = args.model_file
    model_type = args.model_type

    current_ml_setup = None
    if model_type == 'lenet5':
        current_ml_setup = ml_setup.mnist_lenet5()
    elif model_type == 'resnet18':
        current_ml_setup = ml_setup.resnet18_cifar10()
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    if not os.path.exists(model_file_path):
        print(f"file not found. {model_file_path}")
    model = current_ml_setup.model
    model.load_state_dict(torch.load(model_file_path))

    loss, acc = testing_model(model, current_ml_setup)
    print(f"loss={loss}, acc={acc}")
