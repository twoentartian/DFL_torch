import os
import argparse
import io
import sys
import lmdb
import torch
import umap
import pickle
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plot_alpha = 0.5
plot_size = 1

def load_models_from_lmdb(lmdb_path, arg_node_name):
    lmdb_env = lmdb.open(lmdb_path, readonly=True)
    tick_and_models = {}
    with lmdb_env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key = key.decode("utf-8")
            key = key.replace('.pt', '')
            items = key.split('/')
            current_node_name = int(items[0])
            if current_node_name != arg_node_name:
                continue
            tick = int(items[1])
            buffer = io.BytesIO(value)
            state_dict = torch.load(buffer)
            tick_and_models[tick] = state_dict
    return tick_and_models

def visualize_single_path_all_weights(arg_path_folder, arg_output_folder, arg_node_name: int):
    assert os.path.exists(arg_path_folder)
    lmdb_path = os.path.join(arg_path_folder, "model_stat.lmdb")
    assert os.path.exists(lmdb_path)
    tick_and_models = load_models_from_lmdb(lmdb_path, arg_node_name)
    max_tick = max(tick_and_models.keys())
    sample_model = tick_and_models[next(iter(tick_and_models))]
    print("model digest:")
    for layer_name, weights in sample_model.items():
        print(f"layer: {layer_name}, weights: {weights.shape}")

    # plot
    for layer_name, weights in sample_model.items():
        output_folder = os.path.join(arg_output_folder, layer_name)
        os.mkdir(output_folder)
        weights_count = len(weights.view(-1))
        for weight_index in range(weights_count):
            print(f"processing: {layer_name}:{weight_index}")
            ticks = []
            values = []
            for tick in range(max_tick+1):
                model = tick_and_models[tick]
                layer_weights = model[layer_name]
                layer_weights = layer_weights.view(-1)
                values.append(layer_weights[weight_index].item())
                ticks.append(tick)

            fig, ax = plt.subplots()
            ax.scatter(ticks, values, s=plot_size, alpha=plot_alpha, c='b')
            ax.set_title(f"Weight value change at {layer_name}:{weight_index}")
            ax.set_xlabel("ticks")
            ax.set_ylabel("weight value")
            ax.grid(True)
            plt.tight_layout()
            file_name = os.path.join(output_folder, f"{layer_name}_{weight_index}")
            plt.savefig(f"{file_name}.pdf")
            # plt.savefig(f"{file_name}.png")
            plt.close(fig)

def extract_weights(model_stat, layer_name):
    weights = model_stat[layer_name].numpy()
    return weights.flatten()

def visualize_single_path(arg_path_folder, arg_output_folder, arg_node_name: int, methods, dimension=None):
    if dimension is None:
        dimension = [2, 3]
    assert os.path.exists(arg_path_folder)

    lmdb_path = os.path.join(arg_path_folder, "model_stat.lmdb")
    assert os.path.exists(lmdb_path)
    tick_and_models = load_models_from_lmdb(lmdb_path, arg_node_name)
    sample_model = tick_and_models[next(iter(tick_and_models))]
    ticks_ordered = sorted(tick_and_models.keys())
    print("model digest:")
    for layer_name, weights in sample_model.items():
        print(f"layer: {layer_name}, weights: {weights.shape}")

    for method in methods:
        assert method in ['umap', 'pca', 'tsne']

        for layer_name in sample_model.keys():
            print(f"processing {method}: {layer_name}")
            weights_list = [extract_weights(tick_and_models[tick], layer_name) for tick in ticks_ordered]
            weights_array = np.array(weights_list)

            if 2 in dimension:
                if method == 'umap':
                    umap_2d = umap.UMAP(n_components=2)
                    projected_2d = umap_2d.fit_transform(weights_array)
                elif method == 'pca':
                    pca_2d = PCA(n_components=2)
                    projected_2d = pca_2d.fit_transform(weights_array)
                elif method == 'tsne':
                    tsne_2d = TSNE(n_components=2, perplexity=30)
                    projected_2d = tsne_2d.fit_transform(weights_array)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                sc = ax.scatter(projected_2d[:, 0], projected_2d[:, 1], s=plot_size, alpha=plot_alpha, c=range(len(projected_2d)), cmap='viridis')
                plt.colorbar(sc, label='Model Index')
                ax.set_xlabel('UMAP Dimension 1')
                ax.set_ylabel('UMAP Dimension 2')
                ax.set_title(f'2D UMAP Projection of layer {layer_name}')
                plt.tight_layout()
                file_name = os.path.join(arg_output_folder, f"{method}_2d_{layer_name}")
                plt.savefig(f"{file_name}.pdf")
                plt.savefig(f"{file_name}.png")
                plt.close(fig)

            if 3 in dimension:
                if method == 'umap':
                    umap_3d = umap.UMAP(n_components=3)
                    projected_3d = umap_3d.fit_transform(weights_array)
                elif method == 'pca':
                    pca_3d = PCA(n_components=3)
                    projected_3d = pca_3d.fit_transform(weights_array)
                elif method == 'tsne':
                    tsne_3d = TSNE(n_components=3, perplexity=30)
                    projected_3d = tsne_3d.fit_transform(weights_array)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                sc = ax.scatter(projected_3d[:, 0], projected_3d[:, 1], projected_3d[:, 2], s=plot_size, alpha=plot_alpha, c=range(len(projected_3d)), cmap='viridis')
                plt.colorbar(sc, label='Model Index')
                ax.set_xlabel('UMAP Dimension 1')
                ax.set_ylabel('UMAP Dimension 2')
                ax.set_zlabel('UMAP Dimension 3')
                ax.set_title(f'3D UMAP Projection of layer {layer_name}')
                file_name = os.path.join(arg_output_folder, f"{method}_3d_{layer_name}")
                pickle.dump(fig, open(f'{file_name}.plt3d', 'wb'))
                plt.close(fig)

def visualize_all_path(arg_path_folder, arg_output_folder, arg_node_name: int, methods, only_layer=None, dimension=None):
    if dimension is None:
        dimension = [2, 3]
    assert os.path.exists(arg_path_folder)
    all_sub_folders = [f.path for f in os.scandir(arg_path_folder) if f.is_dir()]
    assert len(all_sub_folders) > 0

    layers_and_trajectory = {}
    layers_and_trajectory_length = {}
    all_layer_names = None
    for single_sub_folder in all_sub_folders:
        lmdb_path = os.path.join(single_sub_folder, "model_stat.lmdb")
        print(f"loading lmdb: {lmdb_path}")
        tick_and_models = load_models_from_lmdb(lmdb_path, arg_node_name)
        ticks_ordered = sorted(tick_and_models.keys())
        sample_model = tick_and_models[next(iter(tick_and_models))]
        if all_layer_names is None:
            all_layer_names = sample_model.keys()
        for layer_name in sample_model.keys():
            if only_layer is not None and (only_layer not in layer_name):
                continue
            weights_list = [extract_weights(tick_and_models[tick], layer_name) for tick in ticks_ordered]
            if layer_name not in layers_and_trajectory.keys():
                layers_and_trajectory[layer_name] = []
                layers_and_trajectory_length[layer_name] = []
            layers_and_trajectory[layer_name].extend(weights_list)
            layers_and_trajectory_length[layer_name].append(len(weights_list))

    for layer_name in all_layer_names:
        for method in methods:
            print(f"processing {method}: {layer_name}")
            if 2 in dimension:
                if method == 'umap':
                    umap_2d = umap.UMAP(n_components=2)
                    projected_2d = umap_2d.fit_transform(layers_and_trajectory[layer_name])
                elif method == 'pca':
                    pca_2d = PCA(n_components=2)
                    projected_2d = pca_2d.fit_transform(layers_and_trajectory[layer_name])
                elif method == 'tsne':
                    tsne_2d = TSNE(n_components=2, perplexity=30)
                    projected_2d = tsne_2d.fit_transform(layers_and_trajectory[layer_name])
                else:
                    raise ValueError(f"Unsupported method: {method}")

                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                count = 0
                for index, trajectory_length in enumerate(layers_and_trajectory_length[layer_name]):
                    sc = ax.scatter(projected_2d[count:count+trajectory_length, 0], projected_2d[count:count+trajectory_length, 1], s=plot_size, alpha=plot_alpha, c=range(trajectory_length), cmap='viridis')
                    count += trajectory_length
                    if index == 0:
                        plt.colorbar(sc, label='Model Index')
                ax.set_xlabel('UMAP Dimension 1')
                ax.set_ylabel('UMAP Dimension 2')
                ax.set_title(f'2D UMAP Projection of layer {layer_name}')
                plt.tight_layout()
                file_name = os.path.join(arg_output_folder, f"{method}_2d_{layer_name}")
                plt.savefig(f"{file_name}.pdf")
                plt.savefig(f"{file_name}.png")
                plt.close(fig)

            if 3 in dimension:
                if method == 'umap':
                    umap_3d = umap.UMAP(n_components=3)
                    projected_3d = umap_3d.fit_transform(layers_and_trajectory[layer_name])
                elif method == 'pca':
                    pca_3d = PCA(n_components=3)
                    projected_3d = pca_3d.fit_transform(layers_and_trajectory[layer_name])
                elif method == 'tsne':
                    tsne_3d = TSNE(n_components=3, perplexity=30)
                    projected_3d = tsne_3d.fit_transform(layers_and_trajectory[layer_name])
                else:
                    raise ValueError(f"Unsupported method: {method}")

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                count = 0
                for index, trajectory_length in enumerate(layers_and_trajectory_length[layer_name]):
                    sc = ax.scatter(projected_3d[count:count + trajectory_length, 0], projected_3d[count:count + trajectory_length, 1], projected_3d[count:count + trajectory_length, 2], s=plot_size, alpha=plot_alpha, c=range(trajectory_length), cmap='viridis')
                    count += trajectory_length
                    if index == 0:
                        plt.colorbar(sc, label='Model Index')
                ax.set_xlabel('UMAP Dimension 1')
                ax.set_ylabel('UMAP Dimension 2')
                ax.set_zlabel('UMAP Dimension 3')
                ax.set_title(f'3D UMAP Projection of layer {layer_name}')
                file_name = os.path.join(arg_output_folder, f"{method}_3d_{layer_name}")
                pickle.dump(fig, open(f'{file_name}.plt3d', 'wb'))
                plt.close(fig)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Visualize high accuracy paths')
    parser.add_argument("mode", type=str, choices=['single_path_all_weights', 'all_path', 'single_path'])
    parser.add_argument("path_folder", type=str)
    parser.add_argument("-m", "--dimension_reduce", type=str, nargs='+', choices=['umap', 'tsne', 'pca'])
    parser.add_argument("--node_name", type=int, default=0)
    parser.add_argument("--disable_3d", action='store_true')
    parser.add_argument("--disable_2d", action='store_true')
    parser.add_argument("--layer", type=str)

    args = parser.parse_args()
    mode = args.mode
    path_folder = args.path_folder
    node_name = args.node_name
    plot_dimensions = [2, 3]
    if args.disable_3d:
        plot_dimensions.remove(3)
    if args.disable_2d:
        plot_dimensions.remove(2)
    assert len(plot_dimensions) > 0
    dimension_reduction_methods = args.dimension_reduce
    if 'all_path' in mode or 'single_path' in mode:
        assert dimension_reduction_methods is not None
    only_layer = args.layer

    # create output folder
    output_folder_path = os.path.join(os.curdir, f"{__file__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")}")
    os.mkdir(output_folder_path)

    if mode == 'single_path_all_weights':
        visualize_single_path_all_weights(path_folder, output_folder_path, node_name)
    elif mode == 'all_path':
        visualize_all_path(path_folder, output_folder_path, node_name, dimension_reduction_methods, only_layer=only_layer, dimension=plot_dimensions)
    elif mode == "single_path":
        visualize_single_path(path_folder, output_folder_path, node_name, dimension_reduction_methods, dimension=plot_dimensions)
