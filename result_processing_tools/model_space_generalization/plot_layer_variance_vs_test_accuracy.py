import argparse
import os
import matplotlib.pyplot as plt
import sys
import torch
import re
import lmdb
import io
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import ml_setup, util

def measure_accuracy(current_ml_setup, model_stat_dict, args) -> float:
    model = current_ml_setup.model
    model.load_state_dict(model_stat_dict)
    testing_dataset = current_ml_setup.testing_data
    criterion = current_ml_setup.criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader_test = DataLoader(testing_dataset, batch_size=args.test_batch, shuffle=False, num_workers=8, persistent_workers=True)
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader_test):
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            loss = criterion(outputs, label)
            test_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        test_loss = test_loss / total
        test_accuracy = correct / total

    return test_loss, test_accuracy


def find_subfolders(folders: List[str]) -> List[str]:
    """
    Find all subfolders within the given folders.

    Args:
        folders (List[str]): List of folder paths (set A)

    Returns:
        List[str]: List of all subfolders (set B)
    """
    subfolders = []

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: Folder '{folder}' does not exist, skipping...")
            continue

        if not folder_path.is_dir():
            print(f"Warning: '{folder}' is not a directory, skipping...")
            continue

        # Find all subdirectories
        for item in folder_path.iterdir():
            if item.is_dir():
                subfolders.append(str(item))

    return subfolders


def process_folders(args):
    """
    Process all folders and measure accuracy for models found in subfolders.

    Args:
        folders (List[str]): List of input folders (set A)
        output_csv (str): Output CSV file path
    """
    # Find all subfolders (set B)
    subfolders = find_subfolders(args.folders)
    print(f"Found {len(subfolders)} subfolders to process")

    processed_count = 0

    current_ml_setup = None
    selected_layer_as_variance = None
    variance_and_accuracy_groups = []
    subfolder_index = 0
    for subfolder in subfolders:
        lmdb_path = Path(subfolder) / "model_stat.lmdb"
        lmdb_accuracy_path = Path(subfolder) / "accuracy.csv"
        if lmdb_path.exists() and lmdb_accuracy_path.exists():
        #     this is a lmdb database
            env = lmdb.open(str(lmdb_path), readonly=True)
            accuracy_df = pd.read_csv(str(lmdb_accuracy_path))
            with env.begin() as txn:
                keys = sorted(txn.cursor().iternext(values=False))
                all_ticks = []
                record_node_name = None
                for key in keys:
                    items = key.decode("utf-8").split("/")
                    node_name = items[0]
                    tick = items[1].replace(".model.pt", "")
                    if record_node_name is None:
                        record_node_name = node_name
                    else:
                        assert record_node_name == node_name, f"node name changes: {record_node_name} -> {node_name}"
                    all_ticks.append(int(tick))
                all_ticks = sorted(all_ticks)
                cursor = txn.cursor()
                for tick in all_ticks:
                    key = f"{record_node_name}/{tick}.model.pt"
                    key_b = key.encode("utf-8")
                    print(f"loading lmdb: {key}")
                    value = cursor.get(key_b)
                    buffer = io.BytesIO(value)
                    state_dict = torch.load(buffer, map_location=torch.device("cpu"))
                    layer_info = util.get_layer_info(state_dict)
                    accuracy = accuracy_df.loc[accuracy_df["tick"] == tick, "0"].values[0]
                    if selected_layer_as_variance is None:
                        selected_layer_as_variance = util.prompt_selection(list(layer_info.keys()))
                    variance_and_accuracy_groups.append({'accuracy': accuracy, 'variance': layer_info[selected_layer_as_variance]["variance"], "type": "series", "group": subfolder_index})
                subfolder_index += 1
        else:
        #     this is a single model
            model_file = Path(subfolder) / "0.model.pt"
            output_csv_path = Path(subfolder) / 'accuracy.csv'
            model_stat_dict, model_name = util.load_model_state_file(str(model_file))
            if not os.path.exists(output_csv_path):
                if current_ml_setup is None:
                    if args.dataset is not None:
                        current_ml_setup = ml_setup.get_ml_setup_from_config(model_name, dataset_type=args.dataset)
                    else:
                        current_ml_setup = ml_setup.get_ml_setup_from_config(model_name)
                if model_file.exists():
                    loss, accuracy = measure_accuracy(current_ml_setup, model_stat_dict, args)
                    result = {'tick': 0,'phase': None,'0': accuracy}
                    processed_count += 1
                    print(f"Processed {processed_count}: {subfolder} -> Accuracy: {accuracy}, Loss: {loss}")
                    df = pd.DataFrame(result, index=[0])
                    df.to_csv(output_csv_path, index=False)
                else:
                    print(f"No 0.model.pt found in: {subfolder}")
            accuracy_df = pd.read_csv(output_csv_path)
            accuracy = accuracy_df.at[0, '0']
            layer_info = util.get_layer_info(model_stat_dict)
            if selected_layer_as_variance is None:
                selected_layer_as_variance = util.prompt_selection(list(layer_info.keys()))
            wd = float(re.search(r'_wd([0-9.eE+-]+)', subfolder).group(1))
            variance_and_accuracy_groups.append({'accuracy': accuracy, 'variance': layer_info[selected_layer_as_variance]["variance"], "type": "single", "group": subfolder_index, "wd":wd})
            subfolder_index += 1
    return variance_and_accuracy_groups

def main():
    parser = argparse.ArgumentParser(description='Measure accuracy of models in subfolders and save results to CSV')
    parser.add_argument('folders',nargs='+',help='One or more folders to search for subfolders containing model files')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    parser.add_argument("-d", "--dataset", type=str, default=None, help='specify the dataset name')
    parser.add_argument("--test_batch", type=int, default=100, help='specify the batch size of measuring model on the test dataset.')
    args = parser.parse_args()

    if args.output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        args.output_folder_name = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    os.mkdir(args.output_folder_name)

    print(f"Input folders: {args.folders}")
    print(f"Output file: {args.output_folder_name}")
    print("-" * 50)

    variance_and_accuracy_groups = process_folders(args)
    print(variance_and_accuracy_groups)
    df = pd.DataFrame(variance_and_accuracy_groups)

    plt.figure(figsize=(24, 16))

    scatter = plt.scatter(df['variance'], df['accuracy'],
                          c=df['group'],
                          cmap='viridis',  # You can change this colormap
                          s=100,  # Size of points
                          alpha=0.7,  # Transparency
                          edgecolors='black',  # Border around points
                          linewidth=0.5)
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Group', rotation=270, labelpad=20, fontsize=12)
    plt.xlabel('Variance', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Variance by Group', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder_name, "layer_variance_and_accuracy.pdf"))

if __name__ == "__main__":
    main()