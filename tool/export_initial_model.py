import argparse
import copy
import logging
import os
import sys
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, util


logger = logging.getLogger("export_initial_model")


def resolve_output_path(output_arg: str | None, model_name: str, dataset_name: str) -> str:
    if output_arg is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        return os.path.join(os.curdir, f"initial_{model_name}_{dataset_name}_{timestamp}.model.pt")
    if output_arg.endswith(".model.pt"):
        return output_arg
    return f"{output_arg}.model.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export freshly initialized model weights to a .model.pt file.")
    parser.add_argument("-m", "--model", required=True, type=str, help="model type")
    parser.add_argument("-d", "--dataset", required=True, type=str, help="dataset type")
    parser.add_argument("-P", "--torch_preset_version", type=int, default=None, help="PyTorch preset version for ml_setup")
    parser.add_argument("--cpu", action="store_true", help="force CPU when resolving the ML setup")
    parser.add_argument("-o", "--output", default=None, help="output .model.pt path; '.model.pt' is appended if missing")
    parser.add_argument("--seed", type=int, default=None, help="optional random seed used before re-initialization")

    args = parser.parse_args()

    util.set_logging(logger, "export_init")

    if args.seed is not None:
        util.set_seed(args.seed, logger)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    arg_ml_setup = ml_setup.get_ml_setup_from_config(
        args.model,
        dataset_type=args.dataset,
        pytorch_preset_version=args.torch_preset_version,
        device=device,
    )

    model = copy.deepcopy(arg_ml_setup.model)
    logger.info(f"re-initialize model {arg_ml_setup.model_name} on dataset {arg_ml_setup.dataset_name}")
    arg_ml_setup.re_initialize_model(model)

    output_path = resolve_output_path(args.output, arg_ml_setup.model_name, arg_ml_setup.dataset_name)
    output_folder = os.path.dirname(output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    util.save_model_state(output_path, model.state_dict(), arg_ml_setup.model_name, arg_ml_setup.dataset_name)
    logger.info(f"saved initialized model weights to {output_path}")
