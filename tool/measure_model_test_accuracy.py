import argparse
import torch
import os
import sys
import json
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup


def testing_model(model, current_ml_setup):
    testing_dataset = current_ml_setup.testing_data
    training_dataset = current_ml_setup.training_data
    criterion = current_ml_setup.criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader_test = DataLoader(testing_dataset, batch_size=100, shuffle=True, num_workers=4, persistent_workers=True)
    dataloader_train = DataLoader(training_dataset, batch_size=100, shuffle=True, num_workers=4, persistent_workers=True)

    model.eval()
    model.to(device)
    test_loss = 0
    train_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader_test):
            print(f"test batch_idx: {batch_idx}")
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            loss = criterion(outputs, label)
            test_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        test_loss = test_loss / total
        test_accuracy = correct / total

        for batch_idx, (data, label) in enumerate(dataloader_train):
            print(f"train batch_idx: {batch_idx}")
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            loss = criterion(outputs, label)
            train_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        train_loss = train_loss / total
        train_accuracy = correct / total

    return test_loss, test_accuracy, train_loss, train_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure model test accuracy and loss.')
    parser.add_argument("model_file", type=str)
    parser.add_argument("-m", "--model_type", type=str, default="auto")
    parser.add_argument("-d", "--dataset_type", type=str, default="default")

    args = parser.parse_args()

    model_file_path = args.model_file
    model_type_from_cli = args.model_type

    model_info = torch.load(model_file_path)
    if model_type_from_cli == "auto":
        model_type = model_info["model_name"]
    else:
        model_type = model_type_from_cli
    assert model_type is not None, "model_type is None"
    model_state = model_info["state_dict"]

    current_ml_setup = ml_setup.get_ml_setup_from_config(model_type, dataset_type=args.dataset_type)

    if not os.path.exists(model_file_path):
        print(f"file not found. {model_file_path}")
    model = current_ml_setup.model
    model.load_state_dict(model_state)

    test_loss, test_accuracy, train_loss, train_accuracy = testing_model(model, current_ml_setup)
    print(f"test loss={test_loss}, test acc={test_accuracy}")
    print(f"train loss={train_loss}, train acc={train_accuracy}")
