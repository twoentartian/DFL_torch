import argparse
import csv
import logging
import os
import shutil
import sys
from dataclasses import dataclass

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import util
from tool.autoNEB.autoneb_utils import (
    InfiniteLoader,
    apply_projected_adam_update,
    build_loader,
    clone_state_dict,
    compute_loss_and_gradients,
    compute_loss_only,
    create_output_folder,
    get_trainable_parameter_names,
    interpolate_state_dict,
    path_progress_positions,
    project_gradients,
    redistribute_path,
    resolve_model_and_setup,
    save_json,
    set_reproducibility,
    tangent_for_pivot,
)


DEFAULT_SCHEDULE = "0.01x1000x4,0.001x1000x4,0.001x2000x2,0.0001x1000x4"


@dataclass
class InsertionCandidate:
    segment_index: int
    alpha: float
    true_loss: float
    guessed_loss: float
    residual: float
    normalized_residual: float


def parse_schedule(schedule_text: str) -> list[tuple[float, int]]:
    schedule: list[tuple[float, int]] = []
    for chunk in schedule_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split("x")
        if len(parts) != 3:
            raise ValueError(f"Unable to parse schedule chunk {chunk!r}; expected lr x steps x repeats")
        learning_rate = float(parts[0])
        steps = int(parts[1])
        repeats = int(parts[2])
        if steps <= 0 or repeats <= 0:
            raise ValueError(f"Schedule chunk {chunk!r} must use positive steps and repeats")
        for _ in range(repeats):
            schedule.append((learning_rate, steps))
    if not schedule:
        raise ValueError("Schedule cannot be empty")
    return schedule


def ensure_valid_paths(start_point: str, end_point: str) -> None:
    if not os.path.exists(start_point):
        raise FileNotFoundError(f"starting model {start_point} does not exist")
    if not os.path.exists(end_point):
        raise FileNotFoundError(f"ending model {end_point} does not exist")
    if os.path.isdir(start_point) or os.path.isdir(end_point):
        raise RuntimeError("AutoNEB expects model file paths, not folders")


def initialize_path(
    start_state: dict[str, torch.Tensor],
    end_state: dict[str, torch.Tensor],
    intermediate_points: int,
) -> list[dict[str, torch.Tensor]]:
    path_states = [clone_state_dict(start_state), clone_state_dict(end_state)]
    if intermediate_points > 0:
        path_states.insert(1, interpolate_state_dict(start_state, end_state, 0.5))
    return path_states


def run_neb_cycle(
    path_states: list[dict[str, torch.Tensor]],
    shared_model: torch.nn.Module,
    batch_provider: InfiniteLoader,
    current_ml_setup,
    parameter_names: tuple[str, ...],
    device: torch.device,
    learning_rate: float,
    steps: int,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    weight_decay: float,
    enable_amp: bool,
    logger,
    log_interval: int,
) -> list[dict[str, torch.Tensor]]:
    if len(path_states) <= 2:
        return [clone_state_dict(state) for state in path_states]

    optimizer_states: list[dict[str, dict[str, object]] | None] = [None for _ in path_states]
    updated_path = [clone_state_dict(state) for state in path_states]

    for step_index in range(steps):
        updated_path = redistribute_path(updated_path, parameter_names)
        batch = batch_provider.next()

        losses: list[float] = []
        gradients_by_index: dict[int, dict[str, torch.Tensor]] = {}
        for pivot_index, state in enumerate(updated_path):
            if 0 < pivot_index < len(updated_path) - 1:
                loss_value, gradients = compute_loss_and_gradients(
                    shared_model,
                    state,
                    batch,
                    current_ml_setup,
                    device,
                    enable_amp,
                )
                gradients_by_index[pivot_index] = gradients
            else:
                loss_value = compute_loss_only(
                    shared_model,
                    state,
                    batch,
                    current_ml_setup,
                    device,
                    enable_amp,
                )
            losses.append(loss_value)

        next_path = [clone_state_dict(updated_path[0])]
        next_optimizer_states: list[dict[str, dict[str, object]] | None] = [None]
        for pivot_index in range(1, len(updated_path) - 1):
            tangent = tangent_for_pivot(
                updated_path[pivot_index - 1],
                updated_path[pivot_index],
                updated_path[pivot_index + 1],
                losses[pivot_index - 1],
                losses[pivot_index + 1],
                parameter_names,
            )
            projected_gradients = project_gradients(
                gradients_by_index[pivot_index],
                tangent,
                parameter_names,
            )
            updated_state, next_state = apply_projected_adam_update(
                updated_path[pivot_index],
                projected_gradients,
                optimizer_states[pivot_index],
                parameter_names,
                learning_rate,
                adam_beta1,
                adam_beta2,
                adam_eps,
                weight_decay,
            )
            next_path.append(updated_state)
            next_optimizer_states.append(next_state)

        next_path.append(clone_state_dict(updated_path[-1]))
        next_optimizer_states.append(None)
        updated_path = next_path
        optimizer_states = next_optimizer_states

        if log_interval > 0 and ((step_index + 1) % log_interval == 0 or step_index == steps - 1):
            logger.info(
                f"cycle step {step_index + 1}/{steps}, max pivot loss = {max(losses):.6f}, min pivot loss = {min(losses):.6f}"
            )

    return updated_path


def evaluate_insertions(
    path_states: list[dict[str, torch.Tensor]],
    shared_model: torch.nn.Module,
    batch,
    current_ml_setup,
    device: torch.device,
    enable_amp: bool,
    insertion_eval_points: int,
    insertion_threshold: float,
    remaining_points: int,
    max_insertions_per_cycle: int | None,
) -> tuple[list[float], list[InsertionCandidate], list[InsertionCandidate]]:
    pivot_losses = [
        compute_loss_only(shared_model, state, batch, current_ml_setup, device, enable_amp)
        for state in path_states
    ]
    loss_range = max(pivot_losses) - min(pivot_losses)
    normalization = loss_range if loss_range > 1e-12 else 1.0

    candidates: list[InsertionCandidate] = []
    for segment_index in range(len(path_states) - 1):
        best_candidate: InsertionCandidate | None = None
        for point_index in range(1, insertion_eval_points + 1):
            alpha = point_index / (insertion_eval_points + 1)
            sampled_state = interpolate_state_dict(
                path_states[segment_index],
                path_states[segment_index + 1],
                alpha,
            )
            true_loss = compute_loss_only(
                shared_model,
                sampled_state,
                batch,
                current_ml_setup,
                device,
                enable_amp,
            )
            guessed_loss = pivot_losses[segment_index] * (1.0 - alpha) + pivot_losses[segment_index + 1] * alpha
            residual = true_loss - guessed_loss
            normalized_residual = residual / normalization
            current_candidate = InsertionCandidate(
                segment_index=segment_index,
                alpha=alpha,
                true_loss=true_loss,
                guessed_loss=guessed_loss,
                residual=residual,
                normalized_residual=normalized_residual,
            )
            if best_candidate is None or current_candidate.residual > best_candidate.residual:
                best_candidate = current_candidate
        if best_candidate is not None:
            candidates.append(best_candidate)

    candidates.sort(key=lambda item: item.normalized_residual, reverse=True)
    insert_limit = remaining_points if max_insertions_per_cycle is None else min(remaining_points, max_insertions_per_cycle)

    selected = [candidate for candidate in candidates if candidate.normalized_residual >= insertion_threshold][:insert_limit]
    if len(selected) < insert_limit:
        already_selected = {(candidate.segment_index, candidate.alpha) for candidate in selected}
        for candidate in candidates:
            candidate_key = (candidate.segment_index, candidate.alpha)
            if candidate_key in already_selected:
                continue
            selected.append(candidate)
            already_selected.add(candidate_key)
            if len(selected) >= insert_limit:
                break

    return pivot_losses, candidates, selected


def apply_insertions(
    path_states: list[dict[str, torch.Tensor]],
    selected_candidates: list[InsertionCandidate],
) -> list[dict[str, torch.Tensor]]:
    output_states = [clone_state_dict(state) for state in path_states]
    for candidate in sorted(selected_candidates, key=lambda item: item.segment_index, reverse=True):
        inserted_state = interpolate_state_dict(
            path_states[candidate.segment_index],
            path_states[candidate.segment_index + 1],
            candidate.alpha,
        )
        output_states.insert(candidate.segment_index + 1, inserted_state)
    return output_states


def save_final_path(
    output_folder_path: str,
    path_states: list[dict[str, torch.Tensor]],
    model_name: str,
    dataset_name: str,
    parameter_names: tuple[str, ...],
    start_point: str,
    end_point: str,
    schedule: list[tuple[float, int]],
    args_dict: dict[str, object],
    cycle_rows: list[dict[str, object]],
) -> None:
    pivot_files = []
    for pivot_index, state in enumerate(path_states):
        pivot_file_name = f"pivot_{pivot_index:03d}.model.pt"
        util.save_model_state(
            os.path.join(output_folder_path, pivot_file_name),
            state,
            model_name,
            dataset_name,
        )
        pivot_files.append(pivot_file_name)

    manifest = {
        "script": os.path.basename(__file__),
        "start_point": start_point,
        "end_point": end_point,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "pivot_files": pivot_files,
        "pivot_progress": path_progress_positions(path_states, parameter_names),
        "schedule": [{"lr": lr, "steps": steps} for lr, steps in schedule],
        "arguments": args_dict,
    }
    save_json(os.path.join(output_folder_path, "path_manifest.json"), manifest)

    with open(os.path.join(output_folder_path, "cycle_metrics.csv"), "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "cycle_index",
            "learning_rate",
            "steps",
            "pivot_count_before",
            "pivot_count_after",
            "max_pivot_loss",
            "min_pivot_loss",
            "best_segment_residual",
            "best_segment_normalized_residual",
            "inserted_points",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in cycle_rows:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find an AutoNEB path between two trained models.")
    parser.add_argument("start_point", type=str, help="path to the starting .model.pt checkpoint")
    parser.add_argument("end_point", type=str, help="path to the ending .model.pt checkpoint")
    parser.add_argument("intermediate_points", type=int, help="number of intermediate pivots to find between the two endpoints")
    parser.add_argument("-m", "--model", type=str, default=None, help="override the model name stored in the checkpoints")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="override the dataset name stored in the checkpoints")
    parser.add_argument("-P", "--torch_preset_version", type=int, default=None, help="PyTorch preset version for ml_setup")
    parser.add_argument("-b", "--neb_batch_size", type=int, default=None, help="training batch size used for each NEB step")
    parser.add_argument("--num_workers", type=int, default=0, help="number of dataloader workers")
    parser.add_argument("-c", "--core", type=int, default=os.cpu_count(), help="number of CPU threads to use")
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE, help="comma-separated Adam learning-rate x steps x repeats schedule")
    parser.add_argument("--adam_beta1", "--momentum", dest="adam_beta1", type=float, default=0.9, help="Adam beta1 used inside each NEB cycle")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 used inside each NEB cycle")
    parser.add_argument("--adam_eps", type=float, default=1.0e-8, help="Adam epsilon used inside each NEB cycle")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1.0e-4, help="weight decay used inside each Adam pivot update")
    parser.add_argument("--insert_threshold", type=float, default=0.2, help="normalized residual threshold for pivot insertion")
    parser.add_argument("--insert_eval_points", type=int, default=9, help="number of sampled interpolation points per segment")
    parser.add_argument("--max_insertions_per_cycle", type=int, default=None, help="optional cap on pivots inserted per cycle")
    parser.add_argument("--cpu", action="store_true", help="force CPU execution")
    parser.add_argument("--amp", action="store_true", help="enable mixed precision on CUDA")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--log_interval", type=int, default=100, help="cycle-step logging interval")
    parser.add_argument("-o", "--output_folder_name", default=None, help="output folder name")

    args = parser.parse_args()

    if args.intermediate_points < 0:
        raise RuntimeError("--intermediate_points must be non-negative")
    if args.insert_eval_points <= 0:
        raise RuntimeError("--insert_eval_points must be positive")
    if not 0.0 <= args.adam_beta1 < 1.0:
        raise RuntimeError("--adam_beta1 must be in [0, 1)")
    if not 0.0 <= args.adam_beta2 < 1.0:
        raise RuntimeError("--adam_beta2 must be in [0, 1)")
    if args.adam_eps <= 0.0:
        raise RuntimeError("--adam_eps must be positive")

    ensure_valid_paths(args.start_point, args.end_point)
    schedule = parse_schedule(args.schedule)

    logger = logging.getLogger("find_autoneb_path")
    util.set_logging(logger, "autoneb")

    set_reproducibility(args.seed)
    torch.set_num_threads(args.core)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    start_state, end_state, model_name, dataset_name, current_ml_setup = resolve_model_and_setup(
        args.start_point,
        args.end_point,
        args.model,
        args.dataset,
        args.torch_preset_version,
        device=device,
    )
    logger.info(f"resolved model = {model_name}, dataset = {dataset_name}, device = {device}")

    shared_model = current_ml_setup.model
    parameter_names = get_trainable_parameter_names(shared_model)
    if not parameter_names:
        raise RuntimeError("No trainable parameters found in the resolved model")

    neb_batch_size = current_ml_setup.training_batch_size if args.neb_batch_size is None else args.neb_batch_size
    if neb_batch_size is None or neb_batch_size <= 0:
        raise RuntimeError("Unable to infer a valid NEB batch size; please pass --neb_batch_size")

    train_loader = InfiniteLoader(
        build_loader(
            current_ml_setup,
            split="train",
            batch_size=neb_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            batch_count=None,
            use_whole_dataset=True,
        )
    )

    output_folder_path = create_output_folder(__file__, args.output_folder_name)
    with open(os.path.join(output_folder_path, "arguments.txt"), "w", encoding="utf-8") as file:
        file.write(str(args))
    shutil.copyfile(__file__, os.path.join(output_folder_path, os.path.basename(__file__)))
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "autoneb_utils.py"),
        os.path.join(output_folder_path, "autoneb_utils.py"),
    )

    path_states = initialize_path(start_state, end_state, args.intermediate_points)
    cycle_rows: list[dict[str, object]] = []

    if args.intermediate_points == 0:
        final_path = [clone_state_dict(start_state), clone_state_dict(end_state)]
    else:
        final_path = path_states
        cycle_index = 0
        schedule_index = 0
        while True:
            remaining_points = args.intermediate_points - (len(final_path) - 2)
            if remaining_points < 0:
                raise RuntimeError("path has grown beyond the requested number of intermediate points")

            learning_rate, steps = schedule[min(schedule_index, len(schedule) - 1)]
            cycle_index += 1
            schedule_index += 1
            pivot_count_before = len(final_path)
            logger.info(f"cycle {cycle_index}: pivots = {pivot_count_before}, lr = {learning_rate}, steps = {steps}")

            final_path = run_neb_cycle(
                final_path,
                shared_model,
                train_loader,
                current_ml_setup,
                parameter_names,
                device,
                learning_rate,
                steps,
                args.adam_beta1,
                args.adam_beta2,
                args.adam_eps,
                args.weight_decay,
                args.amp,
                logger,
                args.log_interval,
            )

            insertion_batch = train_loader.next()
            pivot_losses, candidates, selected_candidates = evaluate_insertions(
                final_path,
                shared_model,
                insertion_batch,
                current_ml_setup,
                device,
                args.amp,
                args.insert_eval_points,
                args.insert_threshold,
                remaining_points,
                args.max_insertions_per_cycle,
            )

            if remaining_points > 0 and selected_candidates:
                final_path = apply_insertions(final_path, selected_candidates)

            best_candidate = candidates[0] if candidates else None
            cycle_rows.append(
                {
                    "cycle_index": cycle_index,
                    "learning_rate": learning_rate,
                    "steps": steps,
                    "pivot_count_before": pivot_count_before,
                    "pivot_count_after": len(final_path),
                    "max_pivot_loss": max(pivot_losses),
                    "min_pivot_loss": min(pivot_losses),
                    "best_segment_residual": None if best_candidate is None else best_candidate.residual,
                    "best_segment_normalized_residual": None if best_candidate is None else best_candidate.normalized_residual,
                    "inserted_points": len(selected_candidates),
                }
            )

            current_intermediate_points = len(final_path) - 2
            logger.info(
                f"cycle {cycle_index} finished: pivot_count = {len(final_path)}, inserted_points = {len(selected_candidates)}, remaining_intermediate_points = {args.intermediate_points - current_intermediate_points}"
            )

            if current_intermediate_points >= args.intermediate_points and schedule_index >= len(schedule):
                break

    final_path = redistribute_path(final_path, parameter_names)
    save_final_path(
        output_folder_path,
        final_path,
        model_name,
        dataset_name,
        parameter_names,
        args.start_point,
        args.end_point,
        schedule,
        vars(args),
        cycle_rows,
    )
    logger.info(f"saved final AutoNEB path with {len(final_path) - 2} intermediate pivots to {output_folder_path}")
