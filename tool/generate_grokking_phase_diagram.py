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
# Run the full sweep (one process per cell, sequentially):
python generate_grokking_phase_diagram.py \
    -dexp "x+y" --modulus 97 -tp 50 \
    -epoch 5000 \
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
from pathlib import Path
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

# Re-use the helpers from the original script
from generate_grokking import (
    GrokkingParameters,
    train_grokking,
    generate_dataset,
    loading_dataset_from,
)

logger = logging.getLogger("generate_grokking_phase_diagram")

# ---------------------------------------------------------------------------
# Grid definition — 20 log-spaced values each
# ---------------------------------------------------------------------------
DEFAULT_LEARNING_RATES = list(np.logspace(-4, -1, 20))        # 1e-4 … 1e-1
DEFAULT_WEIGHT_DECAYS  = [0.0] + list(np.logspace(-4, 1, 19)) # 0 + 1e-4 … 10  → 20 total


# ---------------------------------------------------------------------------
# Helpers
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

    # Persist hyperparameters so the plotting script can read them independently
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
    params.set_env(cell_dir, logger=logger)
    params.set_ml_env(model, current_ml_setup.model_name,
                      current_ml_setup.dataset_name, train_ds.tokenizer)
    params.set_ml_hyperparameter(
        learning_rate=lr,
        weight_decay=wd,
        min_lr=args.min_lr if args.min_lr is not None else 1e-4,
        warmup_epoch=10,
        total_epoch=args.epoch if args.epoch is not None else 1e5,
    )
    params.set_dataloader(train_dl, val_dl)
    params.set_model_save("00", save_format="none")   # no intermediate snapshots

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

    # ---- grid ----
    parser.add_argument("-lrs", "--learning_rates", nargs="+", type=float,
                        default=DEFAULT_LEARNING_RATES,
                        help="List of learning rates to sweep "
                             "(default: 20 log-spaced from 1e-4 to 1e-1)")
    parser.add_argument("-wds", "--weight_decays", nargs="+", type=float,
                        default=DEFAULT_WEIGHT_DECAYS,
                        help="List of weight decay values to sweep "
                             "(default: 0 + 19 log-spaced from 1e-4 to 10)")

    # ---- dataset ----
    parser.add_argument("-dpath", "--dataset_path", type=str, default=None)
    parser.add_argument("-dexp",  "--dataset_exp",  type=str, default=None)
    parser.add_argument("--modulus",    type=int,   default=97)
    parser.add_argument("-tp", "--train_pct", type=float, default=50)
    parser.add_argument("-st", "--split_type", type=str, default="random",
                        choices=["random", "chessboard", "updown", "leftright",
                                 "tl_to_br", "tr_to_bl", "interlace_row",
                                 "interlace_col", "chessboard_random"])
    parser.add_argument("-ol", "--operand_length", type=int, default=None)

    # ---- training ----
    parser.add_argument("-epoch", "--epoch",    type=int,   default=1e5)
    parser.add_argument("-minlr", "--min_lr",   type=float, default=None)
    parser.add_argument("-bs",    "--batchsize", type=int,   default=None)
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

    # ---- random seed ----
    if args.random_seed is not None:
        util.set_seed(args.random_seed, logger)

    # ---- output folder ----
    if args.output_folder_name is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_folder_path = os.path.join(
            os.curdir, f"generate_grokking_phase_diagram_{ts}")
    else:
        args.output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.makedirs(args.output_folder_path, exist_ok=True)

    # Save the invocation command
    with open(os.path.join(args.output_folder_path, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv))

    # Save the full grid spec so the plotting script can reconstruct axes without
    # needing to re-parse CLI args or rely on folder-name parsing.
    grid_spec = {
        "learning_rates": args.learning_rates,
        "weight_decays":  args.weight_decays,
    }
    with open(os.path.join(args.output_folder_path, "grid_spec.json"), "w") as f:
        json.dump(grid_spec, f, indent=2)

    learning_rates = args.learning_rates
    weight_decays  = args.weight_decays
    logger.info(f"Grid: {len(learning_rates)} LRs × {len(weight_decays)} WDs "
                f"= {len(learning_rates) * len(weight_decays)} cells")

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

    total_cells = len(learning_rates) * len(weight_decays)
    done = 0
    for lr, wd in product(learning_rates, weight_decays):
        done += 1
        logger.info(f"[{done}/{total_cells}] lr={lr:.4e}  wd={wd:.4e}")
        # Skip cells that already have a completed log (allows safe resume)
        if os.path.exists(cell_log_csv(args.output_folder_path, lr, wd)):
            logger.info("  → already done, skipping")
            continue
        train_cell(args, lr, wd, train_ds, val_ds, device, current_ml_setup)

    logger.info("All cells complete. Run plot_grokking_phase_diagram.py to visualise.")
