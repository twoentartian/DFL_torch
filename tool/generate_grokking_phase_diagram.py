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
# Run the full sweep with 2 cells in parallel:
python generate_grokking_phase_diagram.py \
    -dexp "x+y" --modulus 97 -tp 50 \
    -epoch 100000 \
    -w 2 \
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
import multiprocessing
from pathlib import Path
from datetime import datetime
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

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
# Grid defaults
# ---------------------------------------------------------------------------
DEFAULT_LR_MIN    = 1e-5
DEFAULT_LR_MAX    = 1e-2
DEFAULT_WD_MIN    = 0.0
DEFAULT_WD_MAX    = 20.0
DEFAULT_N_LR      = 20
DEFAULT_N_WD      = 20


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
# Worker function — must be a top-level function for pickle (spawn)
# ---------------------------------------------------------------------------

def _worker(
    lr: float,
    wd: float,
    output_folder_path: str,
    # dataset args (re-generate inside worker to avoid cross-process sharing)
    dataset_path: str,
    train_pct: float,
    dataset_exp: str,
    modulus: int,
    split_type: str,
    operand_length,
    # training args
    epoch: int,
    batchsize,
    model_type: str,
    # model override args
    m_nlayer, m_n_heads, m_d_model, m_context_len,
    # misc
    random_seed,
):
    """
    Train a single (lr, wd) cell.  Runs in its own spawned process so that
    CUDA contexts are fully isolated between parallel jobs.
    """
    worker_logger = logging.getLogger(f"worker_lr{lr:.2e}_wd{wd:.2e}")
    util.set_logging(worker_logger, f"lr{lr:.2e}_wd{wd:.2e}")

    if random_seed is not None:
        util.set_seed(random_seed, worker_logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_ml_setup = ml_setup.get_ml_setup_from_config(
        model_type, dataset_type="arithmetic_exp_unknown", device=device)

    cell_dir = cell_output_dir(output_folder_path, lr, wd)
    os.makedirs(cell_dir, exist_ok=True)

    if dataset_path is not None:
        train_ds, val_ds = loading_dataset_from(dataset_path)
    else:
        train_ds, val_ds = generate_dataset(
            cell_dir, train_pct, dataset_exp, modulus, split_type, operand_length)

    save_cell_metadata(cell_dir, lr, wd, extra={
        "modulus":    modulus,
        "train_pct":  train_pct,
        "epoch":      epoch,
        "model_type": model_type,
    })

    model = copy.deepcopy(current_ml_setup.model)
    if any(x is not None for x in [m_nlayer, m_n_heads, m_d_model, m_context_len]):
        m_nlayer      = m_nlayer      or 2
        m_n_heads     = m_n_heads     or 4
        m_d_model     = m_d_model     or 128
        m_context_len = m_context_len or 50
        model = transformer_for_grokking.Transformer(
            n_layers=m_nlayer, n_heads=m_n_heads,
            d_model=m_d_model, max_context_len=m_context_len)
    current_ml_setup.re_initialize_model(model)
    model.to(device)

    batch_size = current_ml_setup.training_batch_size if batchsize is None else batchsize
    train_dl = ArithmeticIterator(train_ds, device, batchsize_hint=batch_size)
    val_dl   = ArithmeticIterator(val_ds,   device, batchsize_hint=batch_size)

    params = GrokkingParameters()
    params.set_env(cell_dir, True, logger=worker_logger)
    params.set_ml_env(model, current_ml_setup.model_name,
                      current_ml_setup.dataset_name, train_ds.tokenizer)
    params.set_ml_hyperparameter(
        learning_rate=lr,
        weight_decay=wd,
        min_lr=lr,          # constant LR (cosine bottoms out at lr itself)
        warmup_epoch=10,
        total_epoch=epoch,
    )
    params.set_dataloader(train_dl, val_dl)
    params.set_model_save("00", save_format="none")

    worker_logger.info(f"training cell lr={lr:.4e}  wd={wd:.4e}")
    train_grokking(params)
    return lr, wd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 'spawn' is required for CUDA safety; also works correctly on CPU-only.
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Sweep LR × WD to reproduce the grokking phase diagram (Figure 6)")

    # ---- output ----
    parser.add_argument("-o", "--output_folder_name", default=None)

    # ---- parallelism ----
    parser.add_argument("-w", "--workers", type=int, default=2,
                        help="Number of cells to train in parallel (default: 2). "
                             "With CUDA, keep this ≤ number of available GPUs "
                             "unless you intend multiple jobs per GPU.")

    # ---- grid: learning rate ----
    parser.add_argument("--lr_min", type=float, default=DEFAULT_LR_MIN,
                        help=f"Minimum learning rate, log-spaced grid "
                             f"(default: {DEFAULT_LR_MIN})")
    parser.add_argument("--lr_max", type=float, default=DEFAULT_LR_MAX,
                        help=f"Maximum learning rate, log-spaced grid "
                             f"(default: {DEFAULT_LR_MAX})")
    parser.add_argument("--n_lr", type=int, default=DEFAULT_N_LR,
                        help=f"Number of learning rate grid points (default: {DEFAULT_N_LR})")

    # ---- grid: weight decay ----
    parser.add_argument("--wd_min", type=float, default=DEFAULT_WD_MIN,
                        help=f"Minimum weight decay, linear-spaced grid "
                             f"(default: {DEFAULT_WD_MIN})")
    parser.add_argument("--wd_max", type=float, default=DEFAULT_WD_MAX,
                        help=f"Maximum weight decay, linear-spaced grid "
                             f"(default: {DEFAULT_WD_MAX})")
    parser.add_argument("--n_wd", type=int, default=DEFAULT_N_WD,
                        help=f"Number of weight decay grid points (default: {DEFAULT_N_WD})")

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
    weight_decays  = make_wd_grid(args.wd_min, args.wd_max, args.n_wd)
    logger.info(f"LR  grid ({args.n_lr} pts, log-spaced):    {args.lr_min:.2e} … {args.lr_max:.2e}")
    logger.info(f"WD  grid ({args.n_wd} pts, linear-spaced): {args.wd_min:.2g} … {args.wd_max:.2g}")

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
    logger.info(f"Grid: {args.n_lr} LRs × {args.n_wd} WDs = {total_cells} cells,  workers={args.workers}")

    # ---- pre-generate the shared dataset in the main process ----
    dataset_path = args.dataset_path
    if dataset_path is None and args.dataset_exp is not None:
        train_ds, _ = generate_dataset(
            args.output_folder_path,
            args.train_pct,
            args.dataset_exp,
            args.modulus,
            args.split_type,
            args.operand_length,
        )
        dataset_path = os.path.join(args.output_folder_path, train_ds.name)
        logger.info(f"Dataset pre-generated at {dataset_path}")

    # ---- build pending cell list (skip already-done cells) ----
    pending = [
        (lr, wd)
        for lr, wd in product(learning_rates, weight_decays)
        if not os.path.exists(cell_log_csv(args.output_folder_path, lr, wd))
    ]
    skipped = total_cells - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} already-completed cells")
    logger.info(f"Submitting {len(pending)} cells to pool (workers={args.workers})")

    # ---- shared kwargs for every worker ----
    worker_kwargs = dict(
        output_folder_path=args.output_folder_path,
        dataset_path=dataset_path,
        train_pct=args.train_pct,
        dataset_exp=args.dataset_exp,
        modulus=args.modulus,
        split_type=args.split_type,
        operand_length=args.operand_length,
        epoch=args.epoch,
        batchsize=args.batchsize,
        model_type=args.model_type,
        m_nlayer=args.m_nlayer,
        m_n_heads=args.m_n_heads,
        m_d_model=args.m_d_model,
        m_context_len=args.m_context_len,
        random_seed=args.random_seed,
    )

    # ---- dispatch ----
    completed = 0
    failed    = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_worker, lr, wd, **worker_kwargs): (lr, wd)
            for lr, wd in pending
        }
        for future in as_completed(futures):
            lr, wd = futures[future]
            try:
                future.result()
                completed += 1
                logger.info(
                    f"[{completed + skipped}/{total_cells}] done "
                    f"lr={lr:.4e} wd={wd:.4e}")
            except Exception as exc:
                failed += 1
                logger.error(
                    f"Cell lr={lr:.4e} wd={wd:.4e} raised an exception: {exc}",
                    exc_info=True)

    logger.info(
        f"Sweep complete. "
        f"Completed={completed}  Skipped={skipped}  Failed={failed}  "
        f"Total={total_cells}")
    if failed:
        logger.warning(
            f"{failed} cells failed. Re-run the script to retry them "
            f"(completed cells are skipped automatically).")
    logger.info("Run plot_grokking_phase_diagram.py to visualise.")
