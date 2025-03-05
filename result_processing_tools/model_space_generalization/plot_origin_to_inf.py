import argparse
import pandas as pd
from matplotlib import pyplot as plt
import os

target_node = 0
target_layer_name = "conv1.weight"

def load_and_merge_data(folder_path):
    # Define file paths
    files = {
        'variance': os.path.join(folder_path, f"variance/{target_node}.csv"),
        'loss': os.path.join(folder_path, "loss.csv"),
        'accuracy': os.path.join(folder_path, "accuracy.csv")
    }

    # Load data
    dfs = {}
    for key, path in files.items():
        df = pd.read_csv(path)
        df.drop("phase", axis=1, inplace=True)
        if key == "loss":
            df.rename(columns={"0": "loss"}, inplace=True)
        if key == "accuracy":
            df.rename(columns={"0": "accuracy"}, inplace=True)
        dfs[key] = df

    # Merge data on 'tick'
    merged_df = dfs['variance']
    for key in ['loss', 'accuracy']:
        merged_df = merged_df.merge(dfs[key], on='tick', how='outer')

    return merged_df

def plot(whole_df, save_name, loss_limit=None, accuracy_limit=None, vertical=False, scatter=False):
    if vertical:
        fig, axes = plt.subplots(2, 1, figsize=(5, 12))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss vs. conv1.weight
    axs = axes[0]
    if scatter:
        axs.scatter(whole_df[target_layer_name], whole_df["loss"], alpha=0.3, s=1)
    else:
        axs.plot(whole_df[target_layer_name], whole_df["loss"])
    axs.set_xlabel(target_layer_name)
    axs.set_ylabel("Loss")
    axs.set_title(f"Loss vs. Variance of {target_layer_name}")
    if loss_limit is not None:
        axs.set_ylim(loss_limit)

    # Plot Accuracy vs. conv1.weight
    axs = axes[1]
    if scatter:
        axs.scatter(whole_df[target_layer_name], whole_df["accuracy"], alpha=0.3, s=1)
    else:
        axs.plot(whole_df[target_layer_name], whole_df["accuracy"])
    axs.set_xlabel(target_layer_name)
    axs.set_ylabel("Accuracy")
    axs.set_title(f"Accuracy vs. Variance of {target_layer_name}")
    if accuracy_limit is not None:
        axs.set_ylim(accuracy_limit)

    fig.tight_layout()
    fig.savefig(save_name)


def main():
    parser = argparse.ArgumentParser(description="Merge and concatenate CSV data from two folders.")
    parser.add_argument("to_origin", type=str, help="Low loss space folder from starting point to origin")
    parser.add_argument("to_inf", type=str, help="Low loss space folder from starting point to infinite")
    args = parser.parse_args()

    # Load and merge data from both folders
    df1 = load_and_merge_data(args.to_origin)
    df2 = load_and_merge_data(args.to_inf)

    # Concatenate the two dataframes
    final_df = pd.concat([df1, df2], ignore_index=True)

    sorted_final_df = final_df.sort_values(by=[target_layer_name], ascending=True)
    plot(sorted_final_df, "model_space_generalization_variance.pdf", scatter=True)
    sorted_df1 = df1.sort_values(by=[target_layer_name], ascending=True)
    plot(sorted_df1, "model_space_generalization_variance_origin_part.pdf",
         loss_limit=[0.8, 1.2], accuracy_limit=[0.72, 0.77], vertical=True, scatter=True)


if __name__ == "__main__":
    main()
