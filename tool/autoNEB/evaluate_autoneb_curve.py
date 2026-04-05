import argparse
import logging
import os
import sys
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import util
from tool.autoNEB.autoneb_utils import (
    build_loader,
    build_model_tensor_map,
    evaluate_state_on_loader,
    get_trainable_parameter_names,
    interpolate_state_dict,
    load_point_paths,
    path_distance,
    resolve_model_and_setup,
    set_reproducibility,
)


def resolve_reference_endpoints(
    point_paths: list[str],
    args,
    device: torch.device,
):
    if len(point_paths) < 2:
        raise RuntimeError("Need at least two ordered path points to evaluate a curve")
    start_state, end_state, model_name, dataset_name, current_ml_setup = resolve_model_and_setup(
        point_paths[0],
        point_paths[-1],
        args.model,
        args.dataset,
        args.torch_preset_version,
        device=device,
    )
    del start_state, end_state
    return model_name, dataset_name, current_ml_setup


def load_ordered_states(point_paths: list[str]) -> tuple[list[dict[str, torch.Tensor]], list[str], list[str]]:
    states = []
    model_names = []
    dataset_names = []
    for point_path in point_paths:
        state, model_name, dataset_name = util.load_model_state_file(point_path)
        states.append(state)
        model_names.append(model_name)
        dataset_names.append(dataset_name)
    return states, model_names, dataset_names


def progress_for_segments(
    states: list[dict[str, torch.Tensor]],
    parameter_names: tuple[str, ...],
) -> list[float]:
    cumulative = [0.0]
    for index in range(1, len(states)):
        cumulative.append(cumulative[-1] + path_distance(states[index - 1], states[index], parameter_names))
    total_distance = cumulative[-1]
    if total_distance <= 0.0:
        if len(states) == 1:
            return [0.0]
        return [index / (len(states) - 1) for index in range(len(states))]
    return [distance / total_distance for distance in cumulative]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the piecewise-linear curve through AutoNEB pivots.")
    parser.add_argument(
        "points",
        nargs="+",
        help="either a folder produced by find_autoneb_path.py or an ordered list of .model.pt pivot files",
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="override the model name stored in the checkpoints")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="override the dataset name stored in the checkpoints")
    parser.add_argument("-P", "--torch_preset_version", type=int, default=None, help="PyTorch preset version for ml_setup")
    parser.add_argument(
        "-ip_points","--interpolation_points",
        type=int,
        default=9,
        help="number of sampled interpolation points between each neighboring pair, excluding endpoints",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="evaluation batch size")
    parser.add_argument(
        "-k",
        "--batch_count",
        type=int,
        default=100,
        help="number of batches to sample when --whole_dataset is not set",
    )
    parser.add_argument("--whole_dataset", action="store_true", help="evaluate on the whole chosen split")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="which dataset split to evaluate",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="number of dataloader workers")
    parser.add_argument("-c", "--core", type=int, default=os.cpu_count(), help="number of CPU threads to use")
    parser.add_argument("--cpu", action="store_true", help="force CPU evaluation")
    parser.add_argument("--amp", action="store_true", help="enable mixed precision on CUDA")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("-o", "--output", default=None, help="output CSV path")

    args = parser.parse_args()
    if args.interpolation_points < 0:
        raise RuntimeError("--interpolation_points must be non-negative")
    if args.batch_count is not None and args.batch_count <= 0:
        raise RuntimeError("--batch_count must be positive when provided")

    logger = logging.getLogger("evaluate_autoneb_curve")
    util.set_logging(logger, "curve_eval")

    set_reproducibility(args.seed)
    torch.set_num_threads(args.core)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    point_paths = load_point_paths(args.points)
    if len(point_paths) < 2:
        raise RuntimeError("Need at least two path points to evaluate a curve")
    for point_path in point_paths:
        if not os.path.exists(point_path):
            raise FileNotFoundError(f"path point {point_path} does not exist")

    logger.info(f"evaluating {len(point_paths)} path points on device {device}")
    model_name, dataset_name, current_ml_setup = resolve_reference_endpoints(point_paths, args, device)
    states, model_names, dataset_names = load_ordered_states(point_paths)

    for model_name_from_file in model_names:
        util.assert_if_both_not_none(model_name, model_name_from_file)
    if args.dataset is None:
        for dataset_name_from_file in dataset_names:
            util.assert_if_both_not_none(dataset_name, dataset_name_from_file)

    shared_model = current_ml_setup.model.to(device)
    model_tensors = build_model_tensor_map(shared_model)
    parameter_names = get_trainable_parameter_names(shared_model)
    if not parameter_names:
        raise RuntimeError("No trainable parameters found in the resolved model")

    split_names = ["train", "test"] if args.split == "both" else [args.split]
    loaders = {
        split_name: build_loader(
            current_ml_setup,
            split=split_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            batch_count=None if args.whole_dataset else args.batch_count,
            use_whole_dataset=args.whole_dataset,
        )
        for split_name in split_names
    }

    if args.output is None:
        if len(args.points) == 1 and os.path.isdir(args.points[0]):
            output_path = os.path.join(args.points[0], "autoneb_curve.csv")
        else:
            output_path = os.path.join(
                os.path.dirname(point_paths[0]) or os.curdir,
                f"autoneb_curve_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}.csv",
            )
    else:
        output_path = args.output

    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    pivot_progress = progress_for_segments(states, parameter_names)
    total_segments = len(states) - 1

    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "sample_index",
            "sample_kind",
            "split",
            "left_pivot_index",
            "right_pivot_index",
            "alpha_local",
            "path_progress",
            "loss",
            "accuracy",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        sample_index = 0
        for segment_index in range(total_segments):
            local_alphas = [0.0]
            for interp_index in range(1, args.interpolation_points + 1):
                local_alphas.append(interp_index / (args.interpolation_points + 1))
            local_alphas.append(1.0)

            for alpha_local in local_alphas:
                if segment_index > 0 and alpha_local == 0.0:
                    continue

                if alpha_local == 0.0:
                    sample_kind = "pivot"
                    sampled_state = states[segment_index]
                elif alpha_local == 1.0:
                    sample_kind = "pivot"
                    sampled_state = states[segment_index + 1]
                else:
                    sample_kind = "interpolation"
                    sampled_state = interpolate_state_dict(
                        states[segment_index],
                        states[segment_index + 1],
                        alpha_local,
                    )

                path_progress = pivot_progress[segment_index] * (1.0 - alpha_local) + pivot_progress[segment_index + 1] * alpha_local

                for split_name in split_names:
                    metrics = evaluate_state_on_loader(
                        shared_model,
                        model_tensors,
                        sampled_state,
                        loaders[split_name],
                        current_ml_setup,
                        device,
                        args.amp,
                        batch_limit=None if args.whole_dataset else args.batch_count,
                    )
                    writer.writerow(
                        {
                            "sample_index": sample_index,
                            "sample_kind": sample_kind,
                            "split": split_name,
                            "left_pivot_index": segment_index,
                            "right_pivot_index": segment_index + 1,
                            "alpha_local": alpha_local,
                            "path_progress": path_progress,
                            "loss": metrics["loss"],
                            "accuracy": metrics["accuracy"],
                        }
                    )
                sample_index += 1

    logger.info(f"curve evaluation saved to {output_path}")


