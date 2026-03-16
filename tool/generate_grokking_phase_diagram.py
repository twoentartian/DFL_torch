"""
Reproduce Figure 6 from "Towards Understanding Grokking" (Liu et al., 2022).

Figure 6 is a 2-D phase diagram over (weight_decay, learning_rate).
Each cell is classified into one of four phases:
    - Comprehension : both train & val accuracy are high, val generalizes early
    - Grokking      : both train & val accuracy are high, but val lags behind train
    - Memorization  : train accuracy is high, val accuracy stays low
    - Confusion     : both train & val accuracy are low

Usage example
-------------
# Run the full sweep:
python generate_grokking_phase_diagram.py \
    -dexp "x+y" --modulus 97 -tp 50 \
    -epoch 100000 \
    -o phase_diagram_output

# Customise the grid (10x10, narrower ranges):
python generate_grokking_phase_diagram.py \
    -dexp "x+y" --modulus 97 \
    --lr_max 1e-2 --n_lr 10 \
    --wd_max 10 --n_wd 10 \
    -o phase_diagram_output

# Plot results using the separate plotting script:
python plot_grokking_phase_diagram.py -o phase_diagram_output
"""

import os
import sys
import copy
import argparse
import logging
import json
from datetime import datetime
from itertools import product

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import project modules (same pattern as generate_grokking.py)
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util, ml_setup
from py_src.ml_setup_base import transformer_for_grokking
from py_src.ml_setup_base.dataset_modular import ArithmeticDataset, ArithmeticIterator

from generate_grokking import (
    GrokkingParameters,
    train_grokking,
    generate_dataset,
    loading_dataset_from,
)

logger = logging.getLogger("generate_grokking_phase_diagram")

# ---------------------------------------------------------------------------
# Grid defaults
# ---------------------------------------------------------------------------
DEFAULT_LR_MIN = 1e-5
DEFAULT_LR_MAX = 1e-2
DEFAULT_N_LR   = 20

DEFAULT_WD_MIN = 0.0
DEFAULT_WD_MAX = 20.0
DEFAULT_N_WD   = 20


def make_lr_grid(lr_min: float, lr_max: float, n: int) -> list:
    """Log-spaced learning rates from lr_min to lr_max."""
    return list(np.logspace(np.log10(lr_min), np.log10(lr_max), n))


def make_wd_grid(wd_min: float, wd_max: float, n: int) -> list:
    """Linearly-spaced weight decay values from wd_min to wd_max."""
    return list(np.linspace(wd_min, wd_max, n))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def cell_output_dir(base: str, lr: float, wd: float) -> str:
    return os.path.join(base, f"lr{lr:.4e}_wd{wd:.4e}")


def cell_log_csv(base: str, lr: float, wd: float) -> str:
    return os.path.join(cell_output_dir(base, lr, wd), "00.log.csv")


def save_cell_metadata(cell_dir: str, lr: float, wd: float, extra: dict = None):
    """Persist (lr, wd) and any extra info so the plotting script needs no CLI args."""
    meta = {"learning_rate": lr, "weight_decay": wd}
    if extra:
        meta.update(extra)
    with open(os.path.join(cell_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Training one cell
# ---------------------------------------------------------------------------

def train_cell(args, lr: float, wd: float,
               train_ds: ArithmeticDataset,
               val_ds:   ArithmeticDataset,
               device:   torch.device,
               current_ml_setup):
    """Train a single (lr, wd) cell and save results under its own sub-folder."""
    cell_dir = cell_output_dir(args.output_folder_path, lr, wd)
    os.makedirs(cell_dir, exist_ok=True)

    save_cell_metadata(cell_dir, lr, wd, extra={
        "modulus":    args.modulus,
        "train_pct":  args.train_pct,
        "epoch":      args.epoch,
        "model_type": args.model_type,
    })

    # Fresh model for every cell
    model = copy.deepcopy(current_ml_setup.model)
    if any(x is not None for x in [args.m_nlayer, args.m_n_heads,
                                    args.m_d_model, args.m_context_len]):
        m_nlayer      = args.m_nlayer      or 2
        m_n_heads     = args.m_n_heads     or 4
        m_d_model     = args.m_d_model     or 128
        m_context_len = args.m_context_len or 50
        model = transformer_for_grokking.Transformer(
            n_layers=m_nlayer, n_heads=m_n_heads,
            d_model=m_d_model, max_context_len=m_context_len)
    current_ml_setup.re_initialize_model(model)
    model.to(device)

    batch_size = current_ml_setup.training_batch_size if args.batchsize is None else args.batchsize
    train_dl = ArithmeticIterator(train_ds, device, batchsize_hint=batch_size)
    val_dl   = ArithmeticIterator(val_ds,   device, batchsize_hint=batch_size)

    params = GrokkingParameters()
    params.set_env(cell_dir, True, logger=logger)
    params.set_ml_env(model, current_ml_setup.model_name,
                      current_ml_setup.dataset_name, train_ds.tokenizer)
    params.set_ml_hyperparameter(
        learning_rate=lr,
        weight_decay=wd,
        min_lr=lr,          # constant learning rate
        warmup_epoch=10,
        total_epoch=args.epoch,
    )
    params.set_dataloader(train_dl, val_dl)
    params.set_model_save("00", save_format="none")

    logger.info(f"  → training cell lr={lr:.4e}  wd={wd:.4e}")
    train_grokking(params)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep LR × WD to reproduce the grokking phase diagram (Figure 6)")

    # ---- output ----
    parser.add_argument("-o", "--output_folder_name", default=None)

    # ---- grid: learning rate (log-spaced) ----
    parser.add_argument("--lr_min", type=float, default=DEFAULT_LR_MIN,
                        help=f"Minimum learning rate (default: {DEFAULT_LR_MIN})")
    parser.add_argument("--lr_max", type=float, default=DEFAULT_LR_MAX,
                        help=f"Maximum learning rate (default: {DEFAULT_LR_MAX})")
    parser.add_argument("--n_lr",   type=int,   default=DEFAULT_N_LR,
                        help=f"Number of log-spaced LR points (default: {DEFAULT_N_LR})")

    # ---- grid: weight decay (linear-spaced) ----
    parser.add_argument("--wd_max", type=float, default=DEFAULT_WD_MAX,
                        help=f"Maximum weight decay; always starts from {DEFAULT_WD_MIN} "
                             f"(default: {DEFAULT_WD_MAX})")
    parser.add_argument("--n_wd",   type=int,   default=DEFAULT_N_WD,
                        help=f"Number of linear-spaced WD points (default: {DEFAULT_N_WD})")

    # ---- dataset ----
    parser.add_argument("-dpath", "--dataset_path", type=str, default=None)
    parser.add_argument("-dexp",  "--dataset_exp",  type=str, default=None)
    parser.add_argument("--modulus",   type=int,   default=97)
    parser.add_argument("-tp", "--train_pct", type=float, default=50)
    parser.add_argument("-st", "--split_type", type=str, default="random",
                        choices=["random", "chessboard", "updown", "leftright",
                                 "tl_to_br", "tr_to_bl", "interlace_row",
                                 "interlace_col", "chessboard_random"])
    parser.add_argument("-ol", "--operand_length", type=int, default=None)

    # ---- training ----
    parser.add_argument("-epoch", "--epoch",    type=int, default=100000)
    parser.add_argument("-bs",    "--batchsize", type=int, default=None)
    parser.add_argument("-m", "--model_type",   type=str,
                        default="transformer_for_grokking")

    # ---- optional model overrides ----
    parser.add_argument("--m_nlayer",      default=None, type=int)
    parser.add_argument("--m_n_heads",     default=None, type=int)
    parser.add_argument("--m_d_model",     default=None, type=int)
    parser.add_argument("--m_context_len", default=None, type=int)

    # ---- misc ----
    parser.add_argument("-s", "--random_seed", type=int, default=None)

    args = parser.parse_args()

    # ---- logging ----
    util.set_logging(logger, "main")
    logger.info("phase diagram sweep starting")

    if args.random_seed is not None:
        util.set_seed(args.random_seed, logger)

    # ---- build grids ----
    learning_rates = make_lr_grid(args.lr_min, args.lr_max, args.n_lr)
    weight_decays  = make_wd_grid(DEFAULT_WD_MIN, args.wd_max, args.n_wd)
    logger.info(f"LR grid ({args.n_lr} pts, log-spaced):    {args.lr_min:.2e} … {args.lr_max:.2e}")
    logger.info(f"WD grid ({args.n_wd} pts, linear-spaced): {DEFAULT_WD_MIN} … {args.wd_max}")

    # ---- output folder ----
    if args.output_folder_name is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_folder_path = os.path.join(
            os.curdir, f"generate_grokking_phase_diagram_{ts}")
    else:
        args.output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.makedirs(args.output_folder_path, exist_ok=True)

    with open(os.path.join(args.output_folder_path, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv))

    grid_spec = {
        "learning_rates": learning_rates,
        "weight_decays":  weight_decays,
    }
    with open(os.path.join(args.output_folder_path, "grid_spec.json"), "w") as f:
        json.dump(grid_spec, f, indent=2)

    total_cells = len(learning_rates) * len(weight_decays)
    logger.info(f"Grid: {args.n_lr} LRs × {args.n_wd} WDs = {total_cells} cells")

    # ---- device & model setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_ml_setup = ml_setup.get_ml_setup_from_config(
        args.model_type, dataset_type="arithmetic_exp_unknown", device=device)

    # ---- dataset (shared across all cells) ----
    if args.dataset_path is not None:
        train_ds, val_ds = loading_dataset_from(args.dataset_path)
    else:
        train_ds, val_ds = generate_dataset(
            args.output_folder_path,
            args.train_pct,
            args.dataset_exp,
            args.modulus,
            args.split_type,
            args.operand_length,
        )

    done = 0
    for lr, wd in product(learning_rates, weight_decays):
        done += 1
        logger.info(f"[{done}/{total_cells}] lr={lr:.4e}  wd={wd:.4e}")
        if os.path.exists(cell_log_csv(args.output_folder_path, lr, wd)):
            logger.info("  → already done, skipping")
            continue
        train_cell(args, lr, wd, train_ds, val_ds, device, current_ml_setup)

    logger.info("All cells complete. Run plot_grokking_phase_diagram.py to visualise.")
