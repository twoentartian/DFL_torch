import torch
import argparse
import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util


SKIP_LAYERS = ["num_batches_tracked", "running_mean", "running_var"]

def is_skip_layer(layer_name):
    skip = False
    for n in SKIP_LAYERS:
        if n in layer_name:
            skip = True
            break
    return skip

def calculate_l1_l2_weight_difference(state_dict_a, state_dict_b):
    output = {}
    for (layer_name_a, param_a), (layer_name_b, param_b) in zip(state_dict_a.items(), state_dict_b.items()):
        param_a.double()
        param_b.double()
        if is_skip_layer(layer_name_a):
            continue
        if layer_name_a == layer_name_b:
            l1_diff = torch.sum(torch.abs(param_a - param_b)).item()
            l2_diff = torch.sqrt(torch.sum((param_a - param_b) ** 2)).item()
            output[layer_name_a] = (l1_diff, l2_diff)
        else:
            print(f"Warning: Layer names don't match: {layer_name_a} vs {layer_name_b}")
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', type=str, help="path to model states")
    parser.add_argument('-o', '--output', type=str, help="output folder path")

    args = parser.parse_args()
    output_path = args.output
    models_path = args.models

    cpu_device = torch.device("cpu")
    models = []

    model_name = None
    output_path_when_not_specified = None
    for model_path in models_path:
        model, current_model_name = util.load_model_state_file(model_path)

        if model_name is None:
            model_name = current_model_name
        else:
            if current_model_name is not None:
                assert current_model_name == model_name, "Model name mismatch"
        model_folder = os.path.dirname(model_path)
        if output_path_when_not_specified is None:
            output_path_when_not_specified = model_folder
        else:
            if output_path_when_not_specified != model_folder:
                output_path_when_not_specified = ''
        models.append(model)

    if output_path is None:
        if output_path_when_not_specified == '':
            output_path = os.getcwd()
        else:
            output_path = output_path_when_not_specified
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for layer_name in models[0]:
        if is_skip_layer(layer_name):
            continue
        print(f"layer: {layer_name}")
        for model_index, model in enumerate(models):
            layer_weights = model[layer_name]
            formatted_weights = [ '%.4f' % v for v in layer_weights.flatten().numpy()[0:10].tolist() ]
            print(f"model {model_index} (len={layer_weights.nelement()}): {formatted_weights}")
    # if not is_skip_layer(layer)
    df_l1 = pd.DataFrame(index=[layer for layer in models[0]])
    df_l2 = pd.DataFrame(index=[layer for layer in models[0]])
    for index_a, model_a in enumerate(models):
        for index_b, model_b in enumerate(models):
            if index_a >= index_b:
                continue
            result = calculate_l1_l2_weight_difference(model_a, model_b)
            new_l1_column = {layer_name: l1_dis for layer_name, (l1_dis, _) in result.items()}
            df_l1[f"{index_a}-{index_b}"] = df_l1.index.map(new_l1_column)
            new_l2_column = {layer_name: l2_dis for layer_name, (_, l2_dis) in result.items()}
            df_l2[f"{index_a}-{index_b}"] = df_l2.index.map(new_l2_column)

    def calculate_difference(col_name):
        a, b = map(int, col_name.split('-'))  # Split by '-' and convert to integers
        return abs(a - b)  # Return the absolute difference
    sorted_columns = sorted(df_l1.columns, key=calculate_difference)
    df_l1 = df_l1[sorted_columns]
    df_l2 = df_l2[sorted_columns]
    df_l1.to_csv(os.path.join(output_path, "l1_distances.csv"))
    df_l2.to_csv(os.path.join(output_path, "l2_distances.csv"))
