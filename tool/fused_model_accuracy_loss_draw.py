import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

fusion_accuracy_mapping_file_path = "fusion_model_accuracy_on_test.csv"


def get_propagating_src(fusion_accuracy_mapping):
    propagating_src = [col for col in fusion_accuracy_mapping.columns if 'accuracy' not in col and 'loss' not in col]
    return propagating_src


def draw_fusion_accuracy_loss_graph(fusion_df, command, save_name):
    propagating_src = get_propagating_src(fusion_df)

    value_to_plot = command
    if len(propagating_src) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(fusion_df[propagating_src[0]], fusion_df[propagating_src[1]], fusion_df[propagating_src[2]], s=5, c=fusion_df[value_to_plot], vmax=3 if value_to_plot=="loss" else None, cmap='hot')
        plt.colorbar(sc)
        ax.set_xlabel('Model 0')
        ax.set_ylabel('Model 1')
        ax.set_zlabel('Model 2')
        ax.set_title(f'3D Scatter Plot with {value_to_plot} as Temperature')
        plt.show()
    elif len(propagating_src) == 2:
        plt.figure()
        sc = plt.scatter(fusion_df[propagating_src[0]], fusion_df[propagating_src[1]], c=fusion_df[value_to_plot], s=5, vmax=3 if value_to_plot=="loss" else None, cmap='viridis')
        plt.colorbar(sc, label=command)
        plt.title(f'Heatmap of {value_to_plot.capitalize()}')
        plt.xlabel('Model 1')
        plt.ylabel('Model 0')
        plt.savefig(save_name)
    elif len(propagating_src) == 1:
        plt.figure()
        plt.plot(fusion_df[propagating_src[0]], fusion_df[value_to_plot], marker='o', linestyle='-', color='royalblue')
        plt.title(f'Line Graph of {value_to_plot.capitalize()} Over Dimension "0"')
        plt.xlabel('Model "0"')
        plt.ylabel(value_to_plot.capitalize())
        plt.grid(True)
        plt.savefig(save_name)
    else:
        print(f'Invalid propagating source count {len(propagating_src)}')
        exit(-1)


def get_file_name_without_extension(file_path):
    # Split the file path to get the file name and extension
    file_name_with_extension = os.path.basename(file_path)

    # Split the file name and extension
    file_name, file_extension = os.path.splitext(file_name_with_extension)

    return file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process loss or accuracy data.')

    # Add the command argument
    parser.add_argument('command', choices=['loss', 'accuracy'], help='Specify the command (loss or accuracy)')

    # Add the file path argument
    parser.add_argument('file_path', nargs='?', type=str, default=fusion_accuracy_mapping_file_path, help='File path to process')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Retrieve the values
    command = args.command
    file_path = args.file_path

    fusion_df = pd.read_csv(file_path)
    file_name = get_file_name_without_extension(file_path)
    folder_path = os.path.dirname(file_path)
    output_path = os.path.join(folder_path, f"{file_name}__{command}.pdf")
    draw_fusion_accuracy_loss_graph(fusion_df, command, output_path)