import contextlib
import copy
import json
import math
import os
import random
import sys
from typing import Any, Iterable, Sequence

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import cuda, ml_setup, util
from py_src.ml_setup_base.base import CriterionType, MlSetup


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            output[key] = value.detach().cpu().clone()
        else:
            output[key] = copy.deepcopy(value)
    return output


def interpolate_state_dict(
    left_state: dict[str, Any],
    right_state: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    if alpha <= 0.0:
        return clone_state_dict(left_state)
    if alpha >= 1.0:
        return clone_state_dict(right_state)

    output: dict[str, Any] = {}
    for key in left_state.keys():
        left_value = left_state[key]
        right_value = right_state[key]
        if torch.is_tensor(left_value) and torch.is_tensor(right_value):
            if left_value.dtype.is_floating_point and right_value.dtype.is_floating_point:
                output[key] = ((1.0 - alpha) * left_value + alpha * right_value).detach().cpu()
            else:
                output[key] = left_value.detach().cpu().clone() if alpha < 0.5 else right_value.detach().cpu().clone()
        else:
            output[key] = copy.deepcopy(left_value if alpha < 0.5 else right_value)
    return output


def get_trainable_parameter_names(model: torch.nn.Module) -> tuple[str, ...]:
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def path_distance(
    left_state: dict[str, Any],
    right_state: dict[str, Any],
    parameter_names: Sequence[str],
) -> float:
    total = 0.0
    for name in parameter_names:
        difference = left_state[name].detach().float() - right_state[name].detach().float()
        total += float(torch.sum(difference * difference).item())
    return math.sqrt(max(total, 0.0))


def redistribute_path(
    states: list[dict[str, Any]],
    parameter_names: Sequence[str],
) -> list[dict[str, Any]]:
    if len(states) <= 2:
        return [clone_state_dict(state) for state in states]

    cumulative = [0.0]
    for index in range(1, len(states)):
        cumulative.append(cumulative[-1] + path_distance(states[index - 1], states[index], parameter_names))

    total_distance = cumulative[-1]
    if total_distance <= 0.0:
        return [clone_state_dict(state) for state in states]

    redistributed = [clone_state_dict(states[0])]
    segment_index = 0
    for pivot_index in range(1, len(states) - 1):
        target_distance = total_distance * pivot_index / (len(states) - 1)
        while segment_index < len(cumulative) - 2 and cumulative[segment_index + 1] < target_distance:
            segment_index += 1
        left_distance = cumulative[segment_index]
        right_distance = cumulative[segment_index + 1]
        alpha = 0.0 if right_distance <= left_distance else (target_distance - left_distance) / (right_distance - left_distance)
        redistributed.append(interpolate_state_dict(states[segment_index], states[segment_index + 1], alpha))
    redistributed.append(clone_state_dict(states[-1]))
    return redistributed


def path_progress_positions(
    states: list[dict[str, Any]],
    parameter_names: Sequence[str],
) -> list[float]:
    if not states:
        return []

    cumulative = [0.0]
    for index in range(1, len(states)):
        cumulative.append(cumulative[-1] + path_distance(states[index - 1], states[index], parameter_names))

    total_distance = cumulative[-1]
    if total_distance <= 0.0:
        if len(states) == 1:
            return [0.0]
        return [index / (len(states) - 1) for index in range(len(states))]
    return [distance / total_distance for distance in cumulative]


def tangent_for_pivot(
    previous_state: dict[str, Any],
    current_state: dict[str, Any],
    next_state: dict[str, Any],
    previous_loss: float,
    next_loss: float,
    parameter_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    if next_loss > previous_loss:
        tangent = {name: (next_state[name] - current_state[name]).detach().cpu().float() for name in parameter_names}
    else:
        tangent = {name: (current_state[name] - previous_state[name]).detach().cpu().float() for name in parameter_names}

    squared_norm = 0.0
    for value in tangent.values():
        squared_norm += float(torch.sum(value * value).item())
    if squared_norm <= 0.0:
        return {name: torch.zeros_like(current_state[name], dtype=torch.float32) for name in parameter_names}

    norm = math.sqrt(squared_norm)
    return {name: value / norm for name, value in tangent.items()}


def project_gradients(
    gradients: dict[str, torch.Tensor],
    tangent: dict[str, torch.Tensor],
    parameter_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    numerator = None
    denominator = None
    for name in parameter_names:
        gradient = gradients.get(name)
        if gradient is None:
            continue
        tangent_value = tangent[name].to(gradient.device, dtype=gradient.dtype)
        dot_product = torch.sum(gradient * tangent_value)
        tangent_norm = torch.sum(tangent_value * tangent_value)
        numerator = dot_product if numerator is None else numerator + dot_product
        denominator = tangent_norm if denominator is None else denominator + tangent_norm

    if numerator is None or denominator is None or float(denominator.item()) <= 0.0:
        return {name: value.detach().cpu().clone() for name, value in gradients.items()}

    coefficient = numerator / denominator
    projected = {}
    for name, gradient in gradients.items():
        tangent_value = tangent[name].to(gradient.device, dtype=gradient.dtype)
        projected[name] = (gradient - coefficient * tangent_value).detach().cpu().clone()
    return projected


def apply_projected_adam_update(
    state: dict[str, Any],
    projected_gradients: dict[str, torch.Tensor],
    adam_state: dict[str, dict[str, Any]] | None,
    parameter_names: Sequence[str],
    learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    weight_decay: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    updated_state = clone_state_dict(state)
    next_state: dict[str, dict[str, Any]] = {}
    previous_state = {} if adam_state is None else adam_state

    for name in parameter_names:
        if name not in projected_gradients:
            if name in previous_state:
                old_state = previous_state[name]
                next_state[name] = {
                    "exp_avg": old_state["exp_avg"].detach().cpu().clone(),
                    "exp_avg_sq": old_state["exp_avg_sq"].detach().cpu().clone(),
                    "step": int(old_state["step"]),
                }
            continue

        parameter_value = state[name].detach().cpu()
        gradient = projected_gradients[name].detach().cpu().to(dtype=parameter_value.dtype)
        if weight_decay != 0.0:
            gradient = gradient + weight_decay * parameter_value

        if name in previous_state:
            old_state = previous_state[name]
            exp_avg = old_state["exp_avg"].detach().cpu().to(dtype=gradient.dtype)
            exp_avg_sq = old_state["exp_avg_sq"].detach().cpu().to(dtype=gradient.dtype)
            step = int(old_state["step"])
        else:
            exp_avg = torch.zeros_like(gradient)
            exp_avg_sq = torch.zeros_like(gradient)
            step = 0

        step += 1
        exp_avg = adam_beta1 * exp_avg + (1.0 - adam_beta1) * gradient
        exp_avg_sq = adam_beta2 * exp_avg_sq + (1.0 - adam_beta2) * (gradient * gradient)

        bias_correction1 = 1.0 - adam_beta1 ** step
        bias_correction2 = 1.0 - adam_beta2 ** step
        denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + adam_eps
        step_size = learning_rate / bias_correction1

        updated_state[name] = (parameter_value - step_size * (exp_avg / denom)).detach().cpu()
        next_state[name] = {
            "exp_avg": exp_avg.detach().cpu().clone(),
            "exp_avg_sq": exp_avg_sq.detach().cpu().clone(),
            "step": step,
        }
    return updated_state, next_state


class InfiniteLoader:
    def __init__(self, loader: Iterable[Any]):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self) -> Any:
        while True:
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)


def build_loader(
    current_ml_setup: MlSetup,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    batch_count: int | None = None,
    use_whole_dataset: bool = False,
) -> Iterable[Any]:
    if split == "train":
        override_loader = current_ml_setup.override_training_dataset_loader
        dataset = current_ml_setup.training_data
        collate_fn = current_ml_setup.collate_fn
        sampler_fn = current_ml_setup.sampler_fn
    elif split == "test":
        override_loader = current_ml_setup.override_testing_dataset_loader
        dataset = current_ml_setup.testing_data
        collate_fn = current_ml_setup.collate_fn_val
        sampler_fn = None
    else:
        raise ValueError(f"unsupported split {split}")

    if override_loader is not None:
        return override_loader

    if collate_fn is None:
        collate_fn = current_ml_setup.collate_fn if split == "test" and current_ml_setup.collate_fn is not None else default_collate

    target_dataset = dataset
    if not use_whole_dataset and batch_count is not None and dataset is not None:
        subset_size = min(len(dataset), batch_size * batch_count)
        if subset_size < len(dataset):
            indices = torch.randperm(len(dataset))[:subset_size].tolist()
            target_dataset = Subset(dataset, indices)

    sampler = None
    if sampler_fn is not None and split == "train" and target_dataset is not None:
        sampler = sampler_fn(target_dataset)

    return DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def batch_size_of(batch: Any) -> int:
    if torch.is_tensor(batch):
        return int(batch.size(0))
    if isinstance(batch, dict):
        for value in batch.values():
            if torch.is_tensor(value):
                return int(value.size(0))
    if isinstance(batch, (tuple, list)) and batch:
        return batch_size_of(batch[0])
    raise NotImplementedError("Unable to infer batch size from batch")


def extract_inputs_and_targets(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise NotImplementedError("Expected a standard (inputs, targets) batch")


def autocast_context(device: torch.device, enable_amp: bool):
    if not enable_amp or device.type != "cuda":
        return contextlib.nullcontext()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def compute_standard_loss(outputs: Any, targets: Any, criterion: Any) -> torch.Tensor:
    if criterion == CriterionType.Diffusion:
        if not torch.is_tensor(outputs):
            raise RuntimeError("Diffusion models must return a scalar loss tensor")
        return outputs
    if callable(criterion):
        return criterion(outputs, targets)
    raise NotImplementedError(f"Unsupported criterion type: {type(criterion)}")


def compute_accuracy(outputs: Any, targets: Any) -> int | None:
    if not torch.is_tensor(outputs) or not torch.is_tensor(targets):
        return None
    if outputs.ndim < 2 or targets.dtype.is_floating_point:
        return None
    predictions = outputs.argmax(dim=1)
    if predictions.shape[0] != targets.shape[0]:
        return None
    return int((predictions == targets).sum().item())


def compute_loss_and_gradients(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    batch: Any,
    current_ml_setup: MlSetup,
    device: torch.device,
    enable_amp: bool,
) -> tuple[float, dict[str, torch.Tensor]]:
    if current_ml_setup.override_train_step_function is not None:
        raise NotImplementedError("AutoNEB path finding currently supports only standard train steps")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)

    batch_on_device = cuda.to_device(batch, device)
    if isinstance(model, L.LightningModule):
        with autocast_context(device, enable_amp):
            loss_tensor, _ = model.training_step(batch_on_device, 0)
    else:
        inputs, targets = extract_inputs_and_targets(batch_on_device)
        with autocast_context(device, enable_amp):
            outputs = model(inputs)
            loss_tensor = compute_standard_loss(outputs, targets, current_ml_setup.criterion)

    loss_tensor.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            gradients[name] = parameter.grad.detach().cpu().clone()
    return float(loss_tensor.detach().item()), gradients


def compute_loss_only(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    batch: Any,
    current_ml_setup: MlSetup,
    device: torch.device,
    enable_amp: bool,
) -> float:
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    batch_on_device = cuda.to_device(batch, device)

    if current_ml_setup.override_evaluation_step_function is not None:
        with torch.no_grad():
            output = current_ml_setup.override_evaluation_step_function(0, batch_on_device, model, current_ml_setup)
        return float(output.loss_value)

    if isinstance(model, L.LightningModule):
        with torch.no_grad():
            with autocast_context(device, enable_amp):
                loss_tensor, _ = model.training_step(batch_on_device, 0)
        return float(loss_tensor.detach().item())

    inputs, targets = extract_inputs_and_targets(batch_on_device)
    with torch.no_grad():
        with autocast_context(device, enable_amp):
            outputs = model(inputs)
            loss_tensor = compute_standard_loss(outputs, targets, current_ml_setup.criterion)
    return float(loss_tensor.detach().item())


def evaluate_state_on_loader(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    loader: Iterable[Any],
    current_ml_setup: MlSetup,
    device: torch.device,
    enable_amp: bool,
    batch_limit: int | None = None,
) -> dict[str, float | int | None]:
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_count = 0
    total_correct: int | None = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_limit is not None and batch_index >= batch_limit:
                break

            batch_on_device = cuda.to_device(batch, device)
            if current_ml_setup.override_evaluation_step_function is not None:
                output = current_ml_setup.override_evaluation_step_function(batch_index, batch_on_device, model, current_ml_setup)
                batch_loss = float(output.loss_value)
                batch_count_value = int(output.sample_count)
                batch_correct = None if output.correct_count is None else int(output.correct_count)
            elif isinstance(model, L.LightningModule):
                with autocast_context(device, enable_amp):
                    loss_tensor, batch_accuracy = model.training_step(batch_on_device, batch_index)
                batch_count_value = batch_size_of(batch_on_device)
                batch_loss = float(loss_tensor.detach().item())
                batch_correct = int(float(batch_accuracy) * batch_count_value) if batch_accuracy is not None else None
            else:
                inputs, targets = extract_inputs_and_targets(batch_on_device)
                with autocast_context(device, enable_amp):
                    outputs = model(inputs)
                    loss_tensor = compute_standard_loss(outputs, targets, current_ml_setup.criterion)
                batch_count_value = batch_size_of(batch_on_device)
                batch_loss = float(loss_tensor.detach().item())
                batch_correct = compute_accuracy(outputs, targets)

            total_loss += batch_loss * batch_count_value
            total_count += batch_count_value
            if batch_correct is None:
                total_correct = None
            elif total_correct is not None:
                total_correct += batch_correct

    return {
        "loss": total_loss / total_count if total_count > 0 else math.nan,
        "accuracy": None if total_correct is None or total_count == 0 else total_correct / total_count,
        "sample_count": total_count,
    }


def resolve_model_and_setup(
    start_path: str,
    end_path: str,
    model_override: str | None,
    dataset_override: str | None,
    torch_preset_version: int | None,
    device: torch.device | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str, str, MlSetup]:
    start_state, start_model_name, start_dataset_name = util.load_model_state_file(start_path)
    end_state, end_model_name, end_dataset_name = util.load_model_state_file(end_path)

    if model_override is not None:
        model_name = model_override
    else:
        util.assert_if_both_not_none(start_model_name, end_model_name)
        model_name = start_model_name if start_model_name is not None else end_model_name
    if model_name is None:
        raise RuntimeError("Unable to resolve model name; please provide --model")

    if dataset_override is not None:
        dataset_name = dataset_override
    else:
        util.assert_if_both_not_none(start_dataset_name, end_dataset_name)
        dataset_name = start_dataset_name if start_dataset_name is not None else end_dataset_name
        dataset_name = "default" if dataset_name is None else dataset_name

    current_ml_setup = ml_setup.get_ml_setup_from_config(
        model_name,
        dataset_type=dataset_name,
        pytorch_preset_version=torch_preset_version,
        device=device,
    )
    return start_state, end_state, model_name, dataset_name, current_ml_setup


def create_output_folder(script_path: str, output_folder_name: str | None) -> str:
    if output_folder_name is None:
        from datetime import datetime

        output_folder_path = os.path.join(os.curdir, f"{os.path.basename(script_path)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}")
    else:
        output_folder_path = os.path.join(os.curdir, output_folder_name)
    os.makedirs(output_folder_path, exist_ok=False)
    return output_folder_path


def save_json(path: str, content: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=2, sort_keys=True)


def load_point_paths(path_args: Sequence[str]) -> list[str]:
    if len(path_args) == 1 and os.path.isdir(path_args[0]):
        folder_path = path_args[0]
        manifest_path = os.path.join(folder_path, "path_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
            pivot_files = manifest.get("pivot_files", [])
            if pivot_files:
                return [os.path.join(folder_path, file_name) for file_name in pivot_files]

        pivot_candidates = sorted(
            file_name
            for file_name in os.listdir(folder_path)
            if file_name.endswith(".model.pt") and file_name.startswith("pivot_")
        )
        if not pivot_candidates:
            pivot_candidates = sorted(file_name for file_name in os.listdir(folder_path) if file_name.endswith(".model.pt"))
        return [os.path.join(folder_path, file_name) for file_name in pivot_candidates]

    return list(path_args)
