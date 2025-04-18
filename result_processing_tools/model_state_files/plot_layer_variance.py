import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

ignore_layers_with_keywords = ["running_mean", "running_var", "num_batches_tracked"]
ignore_batch_norm_layers = True
plot_mean = False
fig_size = (10,6)

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
                layer_variances[layer_name].append( (np.var(flattened), np.mean(flattened)) )

    return layer_variances


def plot_variance(layer_variances, output_path):

    model_indices = range(len(next(iter(layer_variances.values()))))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(layer_variances)))

    fig_var, axes_var = plt.subplots(1, 1, figsize=fig_size, squeeze=False)

    print(f"Plotting")
    axs_var = axes_var[0, 0]
    for (layer, variances_mean), color in zip(layer_variances.items(), colors):
        variances, mean = zip(*variances_mean)
        if ignore_batch_norm_layers:
            if "bn" in layer:
                continue
        axs_var.plot(model_indices, variances, label=layer, color=color, linewidth=1)

    axs_var.set_xlabel("Model Index")
    axs_var.set_ylabel("Layer Variance")
    axs_var.set_yscale("log")
    # axs.set_title("Layer Variance Across Models")
    axs_var.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    fig_var.tight_layout()
    fig_var.savefig(f"{output_path}/layer_variances.pdf")

    if plot_mean:
        fig_mean, axes_mean = plt.subplots(1, 1, figsize=fig_size, squeeze=False)
        axs_mean = axes_mean[0, 0]
        for (layer, variances_mean), color in zip(layer_variances.items(), colors):
            variances, mean = zip(*variances_mean)
            if ignore_batch_norm_layers:
                if "bn" in layer:
                    continue
            axs_mean.plot(model_indices, mean, label=layer, color=color, linewidth=1)

        axs_mean.set_xlabel("Model Index")
        axs_mean.set_ylabel("Layer Mean")
        axs_mean.set_yscale("log")
        # axs.set_title("Layer Variance Across Models")
        axs_mean.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        fig_mean.tight_layout()
        fig_mean.savefig(f"{output_path}/layer_mean.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=".", help="Directory containing model files")
    args = parser.parse_args()

    layer_variances = load_models(args.directory)
    plot_variance(layer_variances, args.directory)


if __name__ == "__main__":
    main()
