#!/usr/bin/env python3
"""
Relative flatness κ^φ_Tr(w) for ResNet-18 on CIFAR-10.

- Assumes a torchvision.models.resnet18(num_classes=10) model.
- Uses the penultimate features (before model.fc) as φ(x).
- Computes the Hessian of the empirical loss w.r.t. the final fc weight matrix W,
  then contracts it with the Gram matrix of W's rows, as in Def. 3.

This is *expensive* (full Hessian of last layer). Use a small subset of data
via max_batches if needed.
"""

import argparse
import os
import sys
import re
import io
from copy import deepcopy
from pathlib import Path
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd.functional import hessian
from torchvision import datasets, transforms, models
from typing import Iterable, Callable, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util, ml_setup


# ------------------------ ResNet18 feature extraction ------------------------ #

@torch.no_grad()
def resnet18_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute φ(x) = penultimate features of ResNet-18 (before the final fc layer).

    Works for standard torchvision.models.resnet18.
    """
    # This follows torchvision's ResNet.forward, but stops before fc.
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)         # [B, C, 1, 1]
    x = torch.flatten(x, 1)      # [B, m]
    return x


def compute_empirical_loss_with_W(
    W: torch.Tensor,
    b: torch.Tensor,
    model: nn.Module,
    dataloader: Iterable,
    criterion: Callable,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    """
    Empirical loss E_emp(w, φ(S)) with weight matrix W (shape [d, m]) and bias b.
    Uses ResNet-18 features φ(x) from resnet18_features(model, x).
    """
    model.eval()
    total_loss = 0.0
    total_n = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        # If max_batches is None or <= 0, use all batches
        if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            phi = resnet18_features(model, x)  # [B, m]

        logits = F.linear(phi, W, b)          # [B, num_classes]
        loss = criterion(logits, y)           # mean loss

        bs = x.size(0)
        total_loss = total_loss + loss * bs
        total_n += bs

    return total_loss / total_n


def relative_flatness_kappa_tr_resnet18(
    model: nn.Module,
    dataloader: Iterable,
    criterion: Callable,
    device: torch.device,
    max_batches: int,
) -> float:
    """
    Compute κ^φ_Tr(w) for the final fc layer of a ResNet-18 model on CIFAR-10.

    Args:
        model: ResNet-18 model with a final nn.Linear as `model.fc`.
        dataloader: Iterable of (x, y) batches (e.g., CIFAR-10 train or val).
        criterion: loss function, e.g. nn.CrossEntropyLoss().
        device: torch.device, defaults to model's device.
        max_batches: if given, only use the first `max_batches` batches to
                     approximate the empirical loss (for speed).

    Returns:
        Scalar κ^φ_Tr(w) as Python float.
    """
    model.to(device)

    if not isinstance(model.fc, nn.Linear):
        raise TypeError("Expected model.fc to be nn.Linear for ResNet-18.")

    # W0 is the weight matrix (rows = classes, columns = feature dims)
    W0 = model.fc.weight        # [d, m]
    b0 = model.fc.bias          # [d]

    # Work with a clone of W0 as the variable for Hessian computation
    W0 = W0.detach().clone().requires_grad_(True)

    def loss_closure(W: torch.Tensor) -> torch.Tensor:
        return compute_empirical_loss_with_W(
            W=W,
            b=b0,
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            max_batches=max_batches,
        )

    # H has shape [d, m, d, m]
    H = hessian(loss_closure, W0)

    # Extract block traces Tr(H_{s,s'}) by summing the diagonal over the input dim
    # H[s, t, s', t'] -> we need sum_t H[s, t, s', t]
    H_diag = H.diagonal(dim1=1, dim2=3)  # [d, d, m]
    block_traces = H_diag.sum(-1)        # [d, d], Tr(H_{s,s'})

    # Gram matrix of rows: G_{s,s'} = <w_s, w_{s'}>
    G = W0 @ W0.t()                      # [d, d]

    # κ = sum_{s,s'} <w_s, w_{s'}> * Tr(H_{s,s'})
    kappa = (G * block_traces).sum()

    return float(kappa.item())


def get_model_weights_from_file(path, tick=None):
    model_weights = None
    model_name = None
    dataset_name = None
    output_name = None
    if path.is_file():
        print(f"[info] '{path}' is a file.")
        model_weights, model_name, dataset_name = util.load_model_state_file(path)
        output_name = path
    elif path.is_dir():
        print(f"[info] '{path}' is a folder.")
        lmdb_data_path = path / "data.mdb"
        lmdb_lock_path = path / "lock.mdb"
        if lmdb_data_path.exists() and lmdb_lock_path.exists():
            print(f"[info] '{path}' is a valid LMDB folder.")
        else:
            print(f"[error] LMDB files are missing", file=sys.stderr)
            sys.exit(1)
        assert tick is not None, "tick has to be provided in LMDB mode"
        lmdb_index = int(tick)
        env = lmdb.open(str(path), readonly=True)
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
        output_name = f"{path}_tick{tick}"
    else:
        print(f"[error] Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    return model_weights, model_name, dataset_name, output_name


def main():
    parser = argparse.ArgumentParser(description='Relative flatness κ^φ_Tr(w) for a model. This is expensive (full Hessian of last layer). Use a small subset of data via max_batches if needed.')
    parser.add_argument("model_weights_path", type=str, nargs="?", default=None, help="file containing the model weights, can be a .model.pt file or a lmdb directory.")
    parser.add_argument("-t", "--tick", type=int, help="specify the model weights tick index for a lmdb file.")
    parser.add_argument("-m", "--model", type=str, default=None, help='specify the model name')
    parser.add_argument("-d", "--dataset", type=str, default=None, help='specify the dataset name')
    parser.add_argument("--cpu", action="store_true", help="force using CPU for training")
    parser.add_argument("--max-batches", type=int, default=100, help="Use only first N batches for Hessian loss (for speed). If <= 0, use all batches.")
    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')

    args = parser.parse_args()
    model_file_path = args.model_weights_path
    model_name_arg = args.model
    dataset_name_arg = args.dataset
    number_batch = args.max_batches
    number_of_core = args.core

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_weight_file_path = Path(model_file_path).expanduser().resolve()
    tick = None if args.tick is None else int(args.tick)
    model_weights, model_name_from_model, dataset_name_from_model, output_name = get_model_weights_from_file(model_weight_file_path, tick)

    if model_name_from_model is not None and model_name_arg is not None:
        assert model_name_from_model == model_name_arg, f"model name mismatch {model_name_from_model} != {model_name_arg}"
    model_name = model_name_from_model if model_name_from_model is not None else model_name_arg
    if dataset_name_from_model is not None and dataset_name_arg is not None:
        assert dataset_name_from_model == dataset_name_arg, f"dataset name mismatch {dataset_name_from_model} != {dataset_name_arg}"
    dataset_name = dataset_name_from_model if dataset_name_from_model is not None else dataset_name_arg

    current_ml_setup = ml_setup.get_ml_setup_from_config(model_name, dataset_type=dataset_name)
    dataloader_worker = 8 if number_of_core > 8 else number_of_core
    dataloader_prefetch_factor = 4
    dataloader = DataLoader(current_ml_setup.training_data, batch_size=current_ml_setup.training_batch_size, shuffle=False,
                            pin_memory=True, num_workers=dataloader_worker, persistent_workers=True,
                            prefetch_factor=dataloader_prefetch_factor)
    criterion = current_ml_setup.criterion
    target_model: torch.nn.Module = deepcopy(current_ml_setup.model)
    target_model.load_state_dict(model_weights)
    target_model.to(device)

    kappa = relative_flatness_kappa_tr_resnet18(
        model=target_model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        max_batches=number_batch,
    )

    log_info = (f"Relative flatness κ^φ_Tr(w): {kappa:.6e} (with batch_size={current_ml_setup.training_batch_size}, number_of_batch={number_batch}), "
          f"model_name={current_ml_setup.model_name}, dataset_name:{current_ml_setup.dataset_name}")
    print(log_info)
    with open(f"{model_file_path}.relative_flatness.log", "a") as output_file:
        output_file.write(log_info)

if __name__ == "__main__":
    main()
