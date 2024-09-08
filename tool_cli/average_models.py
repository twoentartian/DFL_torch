import torch
import argparse
from datetime import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', type=str, help="path to model states")
    parser.add_argument('-o', '--output', type=str, help="output path")

    args = parser.parse_args()

    if args.output is None:
        now_str = datetime.now().strftime("averaged_model_%Y-%m-%d_%H-%M-%S_%f")
        args.output = f"{now_str}.model.pt"

    models_path = args.models

    cpu_device = torch.device("cpu")
    models = []
    for model_path in models_path:
        model_info = torch.load(model_path, map_location=cpu_device)
        model = model_info["state_dict"]
        models.append(model)

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

    torch.save(output_model, args.output)

