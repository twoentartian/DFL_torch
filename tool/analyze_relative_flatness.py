#!/usr/bin/env python3
"""
Relative flatness κ^φ_Tr(w) for torchvision CNN classifiers with a final nn.Linear head:
  - ResNet / ResNeXt (model.fc)
  - RegNet (model.fc)
  - DenseNet (model.classifier)
  - VGG (last Linear in model.classifier)

Efficient analytic computation for CrossEntropyLoss(mean):

  κ = (1/N) Σ_i ||φ_i||^2 * Tr( G * H_z(p_i) )
where:
  φ_i: input to the final linear layer
  W: final linear weight, shape [d, m]
  G = W W^T
  p = softmax(logits)
  Tr(G * (diag(p) - p p^T)) = (diag(G)·p) - (p^T G p)

Per-sample contribution:
  ||φ||^2 * ( (diag(G)·p) - (p^T G p) )
"""

import argparse
import os
import sys
import re
import io
import logging
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional, Tuple

import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util, ml_setup


# ------------------------ Logging ------------------------ #

def setup_logger(verbosity: int) -> logging.Logger:
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("relative_flatness")


# ------------------------ Helpers: final linear + hook ------------------------ #

def find_last_linear(model: nn.Module) -> nn.Linear:
    """
    Return the last nn.Linear encountered in model.modules() order.
    This matches common torchvision classifiers:
      - ResNet/ResNeXt/RegNet: fc is last Linear
      - DenseNet: classifier is last Linear
      - VGG: classifier contains multiple Linear, last is the output head
    """
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise TypeError("No nn.Linear found in model; cannot compute κ^φ_Tr(w).")
    return last


class _CaptureInputHook:
    """Captures the first positional input to a module during forward."""
    def __init__(self):
        self.x = None  # captured tensor

    def __call__(self, module, inputs, output):
        # inputs is a tuple; for Linear it's (phi,)
        self.x = inputs[0]


# ------------------------ κ computation (analytic, CrossEntropy) ------------------------ #

def _is_cross_entropy_mean(criterion: nn.Module) -> bool:
    if not isinstance(criterion, nn.CrossEntropyLoss):
        return False
    return getattr(criterion, "reduction", "mean") == "mean"


@torch.no_grad()
def relative_flatness_kappa_tr_cross_entropy(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
    log_every: int = 10,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Efficient κ^φ_Tr(w) for *any* torchvision-style classifier whose final head is nn.Linear,
    under CrossEntropyLoss(mean).

    This avoids explicit Hessian construction and works for:
      resnext50_32x4d, regnet_y_400mf, densenet121, vgg11, resnet*, etc.
    """
    if logger is None:
        logger = logging.getLogger("relative_flatness")

    model.eval()
    model.to(device)

    if not _is_cross_entropy_mean(criterion):
        raise TypeError(
            "This fast κ implementation supports nn.CrossEntropyLoss(reduction='mean') only."
        )

    head = find_last_linear(model)

    # Grab W,b from the head
    W = head.weight.detach().to(device)                    # [d, m]
    b = head.bias.detach().to(device) if head.bias is not None else None

    logger.info(f"Using final head: {head.__class__.__name__} "
                f"(out_features={head.out_features}, in_features={head.in_features})")

    # Gram matrix of rows: G = W W^T  [d, d]
    logger.info("Building Gram matrix G = W W^T ...")
    G = W @ W.t()
    diagG = torch.diag(G)                                  # [d]

    # Hook to capture φ(x) = input to final linear
    cap = _CaptureInputHook()
    handle = head.register_forward_hook(cap)

    total = 0.0
    total_n = 0
    used_batches = 0

    logger.info("Starting κ accumulation over batches...")
    try:
        for batch_idx, (x, y) in enumerate(dataloader):
            if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            cap.x = None
            logits = model(x)                               # [B, d]
            phi = cap.x                                     # [B, m] (captured)

            if phi is None:
                raise RuntimeError(
                    "Failed to capture penultimate features via hook. "
                    "The selected last nn.Linear might not be used in forward."
                )

            p = torch.softmax(logits, dim=1)                 # [B, d]

            # per-sample: ||φ||^2
            phi_norm2 = (phi * phi).sum(dim=1)               # [B]

            # Tr(G * (diag(p) - p p^T)) = (diag(G)·p) - (p^T G p)
            pG = p @ G                                       # [B, d]
            pGp = (pG * p).sum(dim=1)                        # [B]
            diag_term = (p * diagG).sum(dim=1)               # [B]
            trace_term = diag_term - pGp                     # [B]

            contrib = (phi_norm2 * trace_term).sum()         # scalar
            bs = x.size(0)
            total += float(contrib.item())
            total_n += bs
            used_batches += 1

            if log_every > 0 and ((batch_idx + 1) % log_every == 0):
                logger.info(
                    f"Progress: batch {batch_idx + 1}"
                    + (f"/{max_batches}" if (max_batches is not None and max_batches > 0) else "")
                    + f", samples={total_n}, running κ={(total / max(total_n, 1)):.6e}"
                )
    finally:
        handle.remove()

    kappa = total / max(total_n, 1)
    logger.info(f"Done. Used batches={used_batches}, samples={total_n}, κ={kappa:.6e}")
    return float(kappa)


# ------------------------ Weight loading (file / LMDB) ------------------------ #

def get_model_weights_from_file(path: Path, tick: Optional[int] = None) -> Tuple[dict, Optional[str], Optional[str], str]:
    model_weights = None
    model_name = None
    dataset_name = None
    output_name = None

    if path.is_file():
        model_weights, model_name, dataset_name = util.load_model_state_file(path)
        output_name = str(path)

    elif path.is_dir():
        lmdb_data_path = path / "data.mdb"
        lmdb_lock_path = path / "lock.mdb"
        if not (lmdb_data_path.exists() and lmdb_lock_path.exists()):
            print(f"[error] LMDB files are missing", file=sys.stderr)
            sys.exit(1)

        if tick is None:
            raise AssertionError("tick has to be provided in LMDB mode")

        lmdb_index = int(tick)
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            all_ticks = set()
            for key, value in cursor:
                m = re.search(r"/(\d+)\.model\.pt$", key.decode())
                if m is None:
                    continue
                t = int(m.group(1))
                all_ticks.add(t)
                if t == lmdb_index:
                    buffer = io.BytesIO(value)
                    model_weights = torch.load(buffer, map_location=torch.device("cpu"))
                    break
            if model_weights is None:
                print(f"[error] tick {lmdb_index} not in lmdb, all ticks: {sorted(all_ticks)[:20]}...", file=sys.stderr)
                sys.exit(1)

        output_name = f"{path}_tick{lmdb_index}"

    else:
        print(f"[error] Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    return model_weights, model_name, dataset_name, output_name


# ------------------------ Main ------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Relative flatness κ^φ_Tr(w) for torchvision CNNs with final Linear head. "
                    "Efficient analytic computation for CrossEntropyLoss."
    )
    parser.add_argument("model_weights_path", type=str, nargs="?", default=None,
                        help="file containing model weights, can be a .model.pt file or an lmdb directory.")
    parser.add_argument("-t", "--tick", type=int, help="model weights tick index for lmdb mode.")
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="model name (e.g., resnext50_32x4d, regnet_y_400mf, densenet121, vgg11)")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="dataset name (e.g., imagenet1k)")
    parser.add_argument("--cpu", action="store_true", help="force using CPU")
    parser.add_argument("--max-batches", type=int, default=100,
                        help="use only first N batches (<=0 means all)")
    parser.add_argument("--log-every", type=int, default=10,
                        help="log progress every N batches (<=0 disables)")
    parser.add_argument("-c", "--core", type=int, default=os.cpu_count(),
                        help="number of CPU cores to use for dataloader")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity")
    parser.add_argument("-P", "--torch_preset_version", type=int, default=None,
                        help="specify the pytorch data training preset version")

    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    if args.model_weights_path is None:
        raise SystemExit("model_weights_path is required")

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    model_weight_file_path = Path(args.model_weights_path).expanduser().resolve()
    tick = None if args.tick is None else int(args.tick)
    model_weights, model_name_from_file, dataset_name_from_file, output_name = get_model_weights_from_file(
        model_weight_file_path, tick
    )

    # Resolve model/dataset names
    model_name = model_name_from_file if model_name_from_file is not None else args.model
    dataset_name = dataset_name_from_file if dataset_name_from_file is not None else args.dataset
    if model_name is None or dataset_name is None:
        raise SystemExit("model name and dataset name must be provided (either in weight file or via --model/--dataset)")

    if model_name_from_file is not None and args.model is not None:
        assert model_name_from_file == args.model, f"model name mismatch {model_name_from_file} != {args.model}"
    if dataset_name_from_file is not None and args.dataset is not None:
        assert dataset_name_from_file == args.dataset, f"dataset name mismatch {dataset_name_from_file} != {args.dataset}"

    logger.info(f"Model: {model_name}, Dataset: {dataset_name}")

    current_ml_setup = ml_setup.get_ml_setup_from_config(
        model_name, dataset_type=dataset_name, pytorch_preset_version=args.torch_preset_version
    )

    number_batch = args.max_batches
    max_batches = None if (number_batch is None or number_batch <= 0) else int(number_batch)

    number_of_core = int(args.core)
    num_workers = min(8, max(0, number_of_core))
    use_workers = num_workers > 0

    dl_kwargs = dict(
        batch_size=current_ml_setup.training_batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
    )
    if use_workers:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

    training_dataset_func = ml_setup.dataset_type_to_setup[current_ml_setup.dataset_type]
    dataset_setup_wo_augmentation = training_dataset_func(
        pytorch_preset_version=args.torch_preset_version, augmentation=False
    )
    dataloader = DataLoader(dataset_setup_wo_augmentation.training_data, **dl_kwargs)

    criterion = current_ml_setup.criterion
    target_model: nn.Module = deepcopy(current_ml_setup.model)
    target_model.load_state_dict(model_weights)
    target_model.to(device)

    logger.info("Computing κ^φ_Tr(w) ...")
    kappa = relative_flatness_kappa_tr_cross_entropy(
        model=target_model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        max_batches=max_batches,
        log_every=int(args.log_every),
        logger=logger,
    )

    log_info = (
        f"Relative flatness κ^φ_Tr(w): {kappa:.6e} "
        f"(batch_size={current_ml_setup.training_batch_size}, max_batches={max_batches}) "
        f"model_name={current_ml_setup.model_name}, dataset_name={current_ml_setup.dataset_name}\n"
    )
    print(log_info, end="")

    out_path = f"{args.model_weights_path}.relative_flatness.log"
    with open(out_path, "a") as f:
        f.write(log_info)
    logger.info(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
