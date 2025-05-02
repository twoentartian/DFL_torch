import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import util

def load_model_from_state_dict(model_path):

    model_state, model_name = util.load_model_state_file(model_path)
    return model_state, model_name


def plot_weights_and_calculate_stats(model, output_folder, model_file_name):
    """
    Iterate through each layer of the model, plot the weight distribution,
    and print variance & mean of weights and (if present) bias.
    """
    # Collect (layer_name, weight_tensor) pairs for plotting
    layer_weights = []
    layer_biases = []

    for name, param in model.items():
        if 'weight' in name:
            layer_weights.append((name, param))
        elif 'bias' in name:
            layer_biases.append((name, param))
        else:
            raise NotImplementedError

    # Plot weight distributions
    fig, axes = plt.subplots(nrows=len(layer_weights), figsize=(8, 3 * len(layer_weights)))
    if len(layer_weights) == 1:
        # In case there's only 1 layer with weights
        axes = [axes]

    for ax, (name, param) in zip(axes, layer_weights):
        data = param.detach().cpu().numpy().flatten()
        ax.hist(data, bins=200, alpha=0.7)
        ax.set_title(f"Weight distribution - {name}")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, f"{model_file_name}_layer_variance.pdf"))

    # Print stats: variance & mean of weights
    print("Layer-wise weight statistics:")
    for name, param in layer_weights:
        data = param.detach().cpu().numpy()
        variance = data.var()
        mean = data.mean()
        print(f"  {name}: count={data.size} var={variance:.6f}, mean={mean:.6f}")

    # Print stats: variance & mean of biases (if they exist)
    if len(layer_biases) > 0:
        print("\nLayer-wise bias statistics:")
        for name, param in layer_biases:
            data = param.detach().cpu().numpy()
            variance = data.var()
            mean = data.mean()
            print(f"  {name}: count={data.size} var={variance:.6f}, mean={mean:.6f}")



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Plot weights distribution of a PyTorch model.")
    parser.add_argument("model_path",type=str,help="Path to the saved model file (e.g. my_model.pth).")
    args = parser.parse_args()

    model_path = args.model_path
    model_file_name = os.path.basename(model_path)
    output_folder = os.path.dirname(model_path)
    model, model_name = load_model_from_state_dict(model_path)
    print(f"model name: {model_name}")

    plot_weights_and_calculate_stats(model, output_folder, model_file_name)


if __name__ == "__main__":
    main()
