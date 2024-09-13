import torch
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', type=str, help="path to model states")
    parser.add_argument('-o', '--output', type=str, help="output path")

    args = parser.parse_args()

    models_path = args.models
    output_path = args.output

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

    assert len(models) >= 2

    output_model = {}
    for layer_name in models[0].keys():
        layer_of_first_model = models[0][layer_name]
        if isinstance(layer_of_first_model, torch.Tensor) and layer_of_first_model.dtype in (torch.float32, torch.float64):
            output_model[layer_name] = torch.mean(torch.stack([model[layer_name] for model in models]), dim=0)
        elif "num_batches_tracked" in layer_name:
            output_model[layer_name] = models[0][layer_name]
        else:
            raise NotImplementedError

    util.save_model_state(args.output, output_model, model_name)
