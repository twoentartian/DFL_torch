import json
import os
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

plot_alpha = 0.5
plot_size = 1

ignore_layers_keyword = ["bias"]

def deduplicate_weights_dbscan(weights_trajectory, shrink_ratio: float | None = None):
    if shrink_ratio is not None:
        assert 0 < shrink_ratio < 1
    scaler = StandardScaler()
    weights_trajectory_scaled = scaler.fit_transform(weights_trajectory)
    dimension = weights_trajectory_scaled.shape[1]
    if dimension == 2:
        dbscan = DBSCAN(eps=0.05, min_samples=100)
    elif dimension == 3:
        dbscan = DBSCAN(eps=0.05, min_samples=100)
    else:
        raise ValueError(f"Unexpected dimension: {dimension}")
    labels = dbscan.fit_predict(weights_trajectory_scaled)
    processed_labels = set()
    weights_trajectory_reduced = []
    index_reduced = []
    for index, label in enumerate(labels):
        if label == -1:
            weights_trajectory_reduced.append(weights_trajectory[index])
            index_reduced.append(index)
        else:
            if label not in processed_labels:
                processed_labels.add(label)
                weights_trajectory_reduced.append(weights_trajectory[index])
                index_reduced.append(index)
    weights_trajectory_reduced = np.array(weights_trajectory_reduced)
    index_reduced = np.array(index_reduced)
    print(f"De-duplicate trajectory points: {weights_trajectory.shape[0]} -> {weights_trajectory_reduced.shape[0]}")
    if shrink_ratio is None:
        return weights_trajectory_reduced, index_reduced
    else:
        target_len = weights_trajectory.shape[0] * shrink_ratio
        current_len = weights_trajectory_reduced.shape[0]
        sample_rate = round(current_len / target_len)
        if sample_rate < 1:
            sample_rate = 1
        print(f"De-duplicate extra sampling rate: {sample_rate}")
        return weights_trajectory_reduced[::sample_rate], index_reduced[::sample_rate]

def plot_pca_all_path(info_file_path, data_path, shrink_ratio=None):
    with open(info_file_path) as f:
        info_target = json.load(f)
    all_targets = []
    for k, v in info_target["targets"].items():
        all_targets.extend(v)
    all_layers = info_target["layer_names"]
    all_layers = sorted(all_layers)
    filtered_layers = []
    for layer in all_layers:
        for keyword in ignore_layers_keyword:
            if keyword not in layer:
                filtered_layers.append(layer)
    all_layers = filtered_layers
    all_dimensions = info_target["dimension"]

    for d in all_dimensions:
        if d == 2:
            fig, axs = plt.subplots(1, len(all_layers), figsize=(len(all_layers) * 15, 15), squeeze=False)
            file_name = "pca_2d"
            for layer_index, layer in enumerate(all_layers):
                print(f"processing layer {layer}")
                ax = axs[0, layer_index]
                ax.set_title(f'PCA Projection of layer {layer}')
                for target_index, target in enumerate(all_targets):
                    csv_file_path = os.path.join(data_path, f"{target}_{layer}_{d}d.csv")
                    df = pd.read_csv(csv_file_path)
                    pattern = r'^PCA Dimension \d+$'
                    pca_dimensions = [col for col in df.columns if re.match(pattern, col)]
                    assert len(pca_dimensions) == d

                    projected_2d = df[pca_dimensions].to_numpy()

                    if shrink_ratio is not None:
                        projected_2d_final, index_final = deduplicate_weights_dbscan(projected_2d, shrink_ratio=shrink_ratio)
                    else:
                        projected_2d_final = projected_2d
                        index_final = range(projected_2d.shape[0])
                    sc = ax.scatter(projected_2d_final[:, 0], projected_2d_final[:, 1], s=plot_size, alpha=plot_alpha,
                                    c=df["tick"][index_final], cmap='viridis')
                    if target_index == 0:
                        plt.colorbar(sc, label='Model Index')
            fig.tight_layout()
            fig.savefig(f"{data_path}/{file_name}.pdf")
            fig.savefig(f"{data_path}/{file_name}.jpg", dpi=400)
            plt.close(fig)
        if d == 3:
            pass
            # not implemented yet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot high accuracy paths')
    parser.add_argument("path", type=str, help="the folder containing info.json and PCA results")
    parser.add_argument("-i","--info", type=str, help="info file path, default: {PCA results path}/info.json")
    parser.add_argument("-s", "--shrink", type=float, default=0.1, help="shrink ratio, a value between 0 and 1 to reduce output image size")

    args = parser.parse_args()

    path = Path(args.path)
    shrink_ratio = args.shrink
    if args.info is None:
        info_path = os.path.join(path, "info.json")
    else:
        info_path = Path(args.info)
    plot_pca_all_path(info_path, path, shrink_ratio=shrink_ratio)
