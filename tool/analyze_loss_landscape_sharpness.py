import os
import argparse
import sys
import re
import io
from typing import List, Iterable, Optional, Union, Callable, Tuple
from pathlib import Path
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import lmdb
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util, ml_setup


@torch.no_grad()
def compute_loss_of_point(
    model: torch.nn.Module, dataloader: Iterable, criterion: Callable, device: Optional[torch.device] = None,
) -> float:
    """Average loss over the entire dataloader (no grad)."""
    model.eval()
    if device is not None:
        model.to(device)
    total_loss, total_n = 0.0, 0
    for batch in dataloader:
        # Supports (inputs, targets) or (inputs, targets, *extras)
        inputs, targets = batch[:2] if isinstance(batch, (tuple, list)) else batch
        if device is not None:
            inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        bs = inputs.shape[0] if hasattr(inputs, "shape") else 1
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / max(total_n, 1)


def _flatten_params(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params if p.requires_grad])


def _param_views_like(flat: torch.Tensor, params: Iterable[torch.nn.Parameter]) -> List[Tuple[torch.nn.Parameter, slice]]:
    views, offset = [], 0
    for p in params:
        if not p.requires_grad:
            continue
        n = p.numel()
        views.append((p, slice(offset, offset + n)))
        offset += n
    assert offset == flat.numel()
    return views


def _sample_unit_direction_like(model: torch.nn.Module, device: Optional[torch.device]) -> List[torch.Tensor]:
    """Sample a single random unit direction with the same shapes as trainable params (global L2-norm = 1)."""
    vecs = []
    sqsum = 0.0
    for p in model.parameters():
        if not p.requires_grad:
            vecs.append(None)
            continue
        d = torch.randn_like(p, device=device)
        vecs.append(d)
        sqsum += (d**2).sum().item()
    scale = (sqsum ** 0.5) or 1.0
    for i, p in enumerate(model.parameters()):
        if not p.requires_grad:
            continue
        vecs[i] = vecs[i] / scale
    return vecs


def _per_param_unit_dirs(model: torch.nn.Module) -> List[Optional[torch.Tensor]]:
    """Random direction normalized to unit L2 *per parameter tensor*."""
    dirs: List[Optional[torch.Tensor]] = []
    for p in model.parameters():
        if not p.requires_grad:
            dirs.append(None)
            continue
        d = torch.randn_like(p)
        d = d / torch.linalg.vector_norm(d).clamp_min(1e-12)
        dirs.append(d)
    return dirs


def _per_param_l2_norms(model: torch.nn.Module) -> List[Optional[float]]:
    norms: List[Optional[float]] = []
    for p in model.parameters():
        if not p.requires_grad:
            norms.append(None)
        else:
            n = torch.linalg.vector_norm(p.detach()).item()
            norms.append(max(n, 1e-12))  # avoid zero step
    return norms

@torch.no_grad()
def loss_landscape_sharpness(
        model: torch.nn.Module, dataloader: Iterable, criterion: Callable,
        change_ratio=None, sample_count: int = 100, device: Optional[Union[str, torch.device]] = None,
) -> dict[float, List[float]]:
    """
    Estimate sharpness by averaging loss increase when stepping the model weights along random directions.

    Args:
        model: nn.Module whose *current weights* will be probed (restored after each probe).
        dataloader: Iterable yielding (inputs, targets[, ...]) used to evaluate loss.
        criterion: Loss function mapping (outputs, targets) -> scalar loss.
        change_ratio: List of step scales. Each step is r * ||θ|| * u, where u is a unit random direction and ||θ|| is the global L2 norm of trainable params.
        sample_count: Number of random directions to sample.
        device: Optional device for evaluation (e.g., "cuda", torch.device("cuda"), or None for current device).

    Returns:
        A dict with key = change_ratio, value = averaged loss across the sample_count.
    """
    if change_ratio is None:
        change_ratio = [0.001, 0.002, 0.003]
    device = torch.device(device) if isinstance(device, str) else (device or next(model.parameters()).device)
    baseline_loss = compute_loss_of_point(model, dataloader, criterion, device=device)
    # Cache original weights
    orig_state = deepcopy(model.state_dict())
    model.eval()  # keep eval for stable measurement
    norms = _per_param_l2_norms(model)
    output = {}
    for r in change_ratio:
        output[r] = []
        for i in range(sample_count):
            model.load_state_dict(deepcopy(orig_state), strict=True)
            dirs = _per_param_unit_dirs(model)
            # Apply perturbation: θ' = θ + (r * ||θ||) * u
            for (p, d, n) in zip(model.parameters(), dirs, norms):
                if p.requires_grad and d is not None and n is not None:
                    p.add_(d, alpha=r * n)
            # Measure loss at θ'
            perturbed_loss = compute_loss_of_point(model, dataloader, criterion, device=device)
            delta_loss = abs(perturbed_loss - baseline_loss)
            print(f"[info] finish ratio {r}, sample {i}.")
            output[r].append(delta_loss)
    return output


def get_model_weights_from_file(path, tick=None):
    model_weights = None
    model_name = None
    dataset_name = None
    if model_weight_file_path.is_file():
        print(f"[info] '{model_weight_file_path}' is a file.")
        model_weights, model_name, dataset_name = util.load_model_state_file(path)
    elif model_weight_file_path.is_dir():
        print(f"[info] '{model_weight_file_path}' is a folder.")
        lmdb_data_path = model_weight_file_path / "data.mdb"
        lmdb_lock_path = model_weight_file_path / "lock.mdb"
        if lmdb_data_path.exists() and lmdb_lock_path.exists():
            print(f"[info] '{model_weight_file_path}' is a valid LMDB folder.")
        else:
            print(f"[error] LMDB files are missing", file=sys.stderr)
            sys.exit(1)
        assert args.tick is not None, "tick has to be provided in LMDB mode"
        lmdb_index = int(args.tick)
        env = lmdb.open(str(model_weight_file_path), readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            all_keys = set()
            for key, value in cursor:
                tick = int(re.search(r'/(\d+)\.model\.pt$', key.decode()).group(1))
                all_keys.add(tick)
                if tick == lmdb_index:
                    buffer = io.BytesIO(value)
                    model_weights = torch.load(buffer, map_location=torch.device("cpu"))
                    break
            if model_weights is None:
                print(f"[error] tick is not in the lmdb, all ticks: {all_keys}", file=sys.stderr)
                sys.exit(1)
    else:
        print(f"[error] Path does not exist: {model_weight_file_path}", file=sys.stderr)
        sys.exit(1)
    return model_weights, model_name, dataset_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the sharpness of loss landscape by sampling several points. ')
    parser.add_argument("model_weights_path", type=str, help="file containing the model weights, can be a .model.pt file or a lmdb directory.")
    parser.add_argument("-t", "--tick", type=int, help="specify the model weights tick index for a lmdb file.")
    parser.add_argument("-m", "--model", type=str, default=None, help='specify the model name')
    parser.add_argument("-d", "--dataset", type=str, default=None, help='specify the dataset name')
    parser.add_argument("--cpu", action="store_true", help="force using CPU for training")
    parser.add_argument("-r", "--change_ratio", type=float, default=[0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128], nargs="+", help="specify the list of change ratio")
    parser.add_argument("-s", "--sample_count", type=int, default=100, help="specify the number of samples")

    args = parser.parse_args()
    model_name_arg = args.model
    dataset_name_arg = args.dataset
    change_ratio = args.change_ratio
    sample_count = args.sample_count

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_weight_file_path = Path(args.model_weights_path).expanduser().resolve()
    tick = None if args.tick is None else int(args.tick)
    model_weights, model_name_from_model, dataset_name_from_model = get_model_weights_from_file(model_weight_file_path, tick)

    if model_name_from_model is not None and model_name_arg is not None:
        assert model_name_from_model == model_name_arg, f"model name mismatch {model_name_from_model} != {model_name_arg}"
    model_name = model_name_from_model if model_name_from_model is not None else model_name_arg
    if dataset_name_from_model is not None and dataset_name_arg is not None:
        assert dataset_name_from_model == dataset_name_arg, f"dataset name mismatch {dataset_name_from_model} != {dataset_name_arg}"
    dataset_name = dataset_name_from_model if dataset_name_from_model is not None else dataset_name_arg

    current_ml_setup = ml_setup.get_ml_setup_from_config(model_name, dataset_type=dataset_name)
    dataloader_worker = 2
    dataloader_prefetch_factor = 4
    dataloader = DataLoader(current_ml_setup.training_data, batch_size=current_ml_setup.training_batch_size, shuffle=False,
                            pin_memory=True, num_workers=dataloader_worker, persistent_workers=True,
                            prefetch_factor=dataloader_prefetch_factor)
    criterion = current_ml_setup.criterion
    target_model: torch.nn.Module = deepcopy(current_ml_setup.model)
    target_model.load_state_dict(model_weights)
    target_model.to(device)

    result = loss_landscape_sharpness(target_model, dataloader, criterion, change_ratio=change_ratio, sample_count=sample_count, device=device)
    print(f"[info] final result is \n{result}.")

    data = {k: [float(x) for x in v] for k, v in result.items()}
    result_df = pd.DataFrame(data)
    result_df.to_csv(f"{model_weight_file_path}.loss_sharpness.csv")