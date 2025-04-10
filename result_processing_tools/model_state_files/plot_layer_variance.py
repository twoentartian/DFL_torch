import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

ignore_layers_with_keywords = ["running_mean", "running_var", "num_batches_tracked"]

def load_models(directory):
    model_files = [f for f in os.listdir(directory) if f.endswith(".model.pt")]
    layer_variances = {}
    for file in model_files:
        path = os.path.join(directory, file)
        print(f"Loading model from {path}")
        info = torch.load(path, map_location="cpu")
        model = info["state_dict"]

        for layer_name, weights in model.items():
            ignore = False
            for k in ignore_layers_with_keywords:
                if k in layer_name:
                    ignore = True
                    break
            if ignore:
                continue

            if weights.ndim > 0:  # Ignore scalars
                flattened = weights.view(-1).numpy()
                if layer_name not in layer_variances:
                    layer_variances[layer_name] = []
                layer_variances[layer_name].append(np.var(flattened))

    return layer_variances


def plot_variance(layer_variances, output_path):

    model_indices = range(len(next(iter(layer_variances.values()))))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(layer_variances)))

    fig, axes = plt.subplots(1, 1, figsize=(20, 20), squeeze=False)

    print(f"Plotting")
    axs = axes[0, 0]
    for (layer, variances), color in zip(layer_variances.items(), colors):
        axs.plot(model_indices, variances, label=layer, color=color)

    axs.set_xlabel("Model Index")
    axs.set_ylabel("Layer Variance")
    axs.set_yscale("log")
    axs.set_title("Layer Variance Across Models")
    axs.legend()

    fig.tight_layout()
    fig.savefig(f"{output_path}/layer_variances.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=".", help="Directory containing model files")
    args = parser.parse_args()

    layer_variances = load_models(args.directory)
    plot_variance(layer_variances, args.directory)


if __name__ == "__main__":
    main()
