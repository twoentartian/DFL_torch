import os
import torch
import pandas as pd
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util

def calculate_l1_l2_distance_per_layer(model_state_dict):
    layer_distances = {}

    for layer_name, param_tensor in model_state_dict.items():
        l1_distance = torch.sum(torch.abs(param_tensor)).item()
        l2_distance = torch.sqrt(torch.sum(param_tensor ** 2)).item()
        layer_distances[layer_name] = {'L1': l1_distance, 'L2': l2_distance}

    return layer_distances

def process_models(folder_path):
    l1_distances = {}
    l2_distances = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".model.pt"):
            # Extract the time tick (integer part) from the file name
            time_tick = int(file_name.split('.')[0])

            # Load the model state dictionary
            model_path = os.path.join(folder_path, file_name)
            model_state_dict, model_type = util.load_model_state_file(model_path)

            print(f"processing {file_name}")
            # Calculate L1 and L2 distances per layer
            distances = calculate_l1_l2_distance_per_layer(model_state_dict)

            # Store distances by time tick
            l1_distances[time_tick] = {layer: dist['L1'] for layer, dist in distances.items()}
            l2_distances[time_tick] = {layer: dist['L2'] for layer, dist in distances.items()}

    # Convert the dictionaries to DataFrames
    l1_df = pd.DataFrame(l1_distances).T
    l2_df = pd.DataFrame(l2_distances).T

    l1_df = l1_df.sort_index()
    l2_df = l2_df.sort_index()

    return l1_df, l2_df

def main():
    parser = argparse.ArgumentParser(description="Process *.model.pt files in a folder and calculate L1/L2 distances per layer.")
    parser.add_argument('folder_path', type=str, help='Path to the folder containing the *.model.pt files')
    args = parser.parse_args()

    folder_path = args.folder_path

    # Process the models and calculate distances
    l1_df, l2_df = process_models(folder_path)

    # Save the results to CSV files
    l1_df.to_csv(os.path.join(folder_path, "l1_distances_to_origin.csv"))
    l2_df.to_csv(os.path.join(folder_path, "l2_distances_to_origin.csv"))

    print("L1 and L2 distances saved to l1_distances.csv and l2_distances.csv.")


if __name__ == "__main__":
    main()
