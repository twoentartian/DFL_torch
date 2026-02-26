"""
visualizing_arithemetic_dataset.py
--------------
Visualise which (a, b) operand pairs belong to the training set vs the
validation set.

Usage
-----
    python visualizing_arithemetic_dataset.py <folder>

<folder> must contain exactly three files:
    train.txt      – training equations
    val.txt        – validation equations
    tokenizer.txt  – one token per line (the vocabulary)

The script reads the operator tokens directly from tokenizer.txt so it
works with any operator, including ones not seen at development time.
"""

import argparse
import re
import sys
import os
from itertools import permutations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

# ---------------------------------------------------------------------------
# Tokenizer / vocabulary helpers
# ---------------------------------------------------------------------------

def load_tokens(tokenizer_path: Path) -> list[str]:
    """Return the list of tokens in vocabulary order."""
    return tokenizer_path.read_text().strip().split("\n")


def extract_operators(tokens: list[str]) -> list[str]:
    """
    Return the subset of tokens that are operators (not EOS/EQ/numbers/perms).
    Sorted longest-first so greedy matching picks the most specific operator.
    """
    skip = {"<|eos|>", "="}
    # Numbers: purely numeric strings
    # Permutation tokens: exactly 5 chars, all digits (e.g. '01234')
    def is_operand(t):
        if t in skip:
            return True
        if re.fullmatch(r"\d+", t):        # plain integer
            return True
        if re.fullmatch(r"[0-4]{5}", t):   # s5 permutation
            return True
        return False

    ops = [t for t in tokens if not is_operand(t)]
    # Longest first → greedy match won't confuse e.g. "s5" vs "s5conj"
    ops.sort(key=len, reverse=True)
    return ops


def build_operand_index(tokens: list[str]) -> dict[str, int]:
    """
    Map every non-operator, non-special token to a sequential integer index.
    This covers plain numbers (0, 1, …) and s5 permutation tokens alike.
    """
    skip = {"<|eos|>", "="}
    ops = set(extract_operators(tokens))
    operand_tokens = [t for t in tokens if t not in skip and t not in ops]
    return {t: i for i, t in enumerate(operand_tokens)}


# ---------------------------------------------------------------------------
# Equation parsing
# ---------------------------------------------------------------------------

def parse_equation(eq: str, operators: list[str],
                   operand_index: dict[str, int]):
    """
    Parse one equation line and return (row_idx, col_idx).

    Format expected (spaces are token separators):
        <|eos|> a OP b = c <|eos|>

    Returns None on failure.
    """
    eq = eq.strip()
    eq = re.sub(r"<\|eos\|>", "", eq).strip()
    if not eq:
        return None

    parts = eq.split(" = ")
    if len(parts) < 2:
        return None
    lhs = parts[0].strip()
    c_str = parts[1].strip().split()[0]

    op_found = None
    for op in operators:
        if f" {op} " in lhs:
            op_found = op
            break
    if op_found is None:
        return None

    halves = lhs.split(f" {op_found} ", maxsplit=1)
    if len(halves) != 2:
        return None
    a_str, b_str = halves[0].strip(), halves[1].strip()

    a_idx = operand_index.get(a_str)
    b_idx = operand_index.get(b_str)
    if a_idx is None or b_idx is None:
        return None

    return a_idx, b_idx, c_str


def load_data(txt_path, operators, operand_index):
    results = []
    for line in txt_path.read_text().splitlines():
        r = parse_equation(line, operators, operand_index)
        if r is not None:
            results.append(r)
    return results


def resolve_out(folder, filename):
    p = folder / filename
    try:
        p.touch()
        p.unlink()
        return p
    except OSError:
        fallback = Path.cwd() / filename
        print(f"  (folder is read-only -- saving {filename} to {fallback})")
        return fallback


def build_output_grid(all_data, operand_index, n):
    """
    Returns
        val_grid   : float (n,n), NaN where missing; numeric output value
                     (or vocabulary index for non-integer tokens like s5 perms)
        label_grid : str   (n,n), raw c_str for text display
        numeric    : bool, True when all outputs are plain integers
    """
    val_grid   = np.full((n, n), np.nan)
    label_grid = np.full((n, n), "", dtype=object)
    numeric    = True

    for a, b, c_str in all_data:
        if not (0 <= a < n and 0 <= b < n):
            continue
        label_grid[a, b] = c_str
        try:
            val_grid[a, b] = int(c_str)
        except ValueError:
            c_idx = operand_index.get(c_str)
            val_grid[a, b] = float(c_idx) if c_idx is not None else np.nan
            numeric = False

    return val_grid, label_grid, numeric

def find_dataset_folders(root: Path) -> list[Path]:
    """
    Return all directories (under root, including root itself) that contain
    train.txt, val.txt, and tokenizer.txt.

    Fast path: rglob for tokenizer.txt only, then check that the other two
    required files exist in the same directory.  This avoids enumerating
    every file and directory in the tree.
    """
    others = {"train.txt", "val.txt"}
    matches = []
    for tokenizer in sorted(root.rglob("tokenizer.txt")):
        folder = tokenizer.parent
        if all((folder / name).is_file() for name in others):
            matches.append(folder)
    return matches

# ---------------------------------------------------------------------------
# Figure 1 -- Train / Val split membership
# ---------------------------------------------------------------------------

def plot_splits(train_data, val_data, n, out_path, title="Train / Val Split"):
    """
    Draw an n×n grid coloured by split membership.

    Blue  → train only
    Red   → val only
    Green → overlap (both)
    White → missing
    """
    train_coords = [(a, b) for a, b, _ in train_data]
    val_coords = [(a, b) for a, b, _ in val_data]

    grid = np.full((n, n), fill_value=-1, dtype=np.int8)
    for a, b in train_coords:
        if 0 <= a < n and 0 <= b < n:
            grid[a, b] = 0
    for a, b in val_coords:
        if 0 <= a < n and 0 <= b < n:
            grid[a, b] = 2 if grid[a, b] == 0 else 1

    cmap = {
        -1: (1.00, 1.00, 1.00, 1.0),
        0: (0.20, 0.45, 0.75, 1.0),
        1: (0.85, 0.25, 0.25, 1.0),
        2: (0.20, 0.70, 0.30, 1.0),
    }
    img = np.zeros((n, n, 4))
    for v, color in cmap.items():
        img[grid == v] = color

    n_train = int((grid == 0).sum())
    n_val = int((grid == 1).sum())
    n_both = int((grid == 2).sum())
    n_miss = int((grid == -1).sum())
    total = n * n

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    ax.imshow(img, origin="upper", aspect="equal",
              extent=[-0.5, n - 0.5, n - 0.5, -0.5])
    ax.set_xlabel("b  (column operand)", fontsize=11)
    ax.set_ylabel("a  (row operand)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    patches = [
        mpatches.Patch(color=cmap[0][:3], label=f"Train ({n_train:,})"),
        mpatches.Patch(color=cmap[1][:3], label=f"Val   ({n_val:,})"),
    ]
    if n_both:
        patches.append(mpatches.Patch(color=cmap[2][:3], label=f"Both  ({n_both:,})"))
    if n_miss:
        patches.append(mpatches.Patch(color=(0.9, 0.9, 0.9), label=f"Missing ({n_miss:,})"))
    ax.legend(handles=patches, loc="upper right", framealpha=0.85, fontsize=10)

    ax2 = axes[1]
    ax2.axis("off")
    stats = [
        ("Grid size", f"{n} x {n}  =  {total:,}"),
        ("Train cells", f"{n_train:,}  ({100 * n_train / total:.1f}%)"),
        ("Val cells", f"{n_val:,}  ({100 * n_val / total:.1f}%)"),
        ("Overlap", f"{n_both:,}  ({100 * n_both / total:.1f}%)"),
        ("Missing", f"{n_miss:,}  ({100 * n_miss / total:.1f}%)"),
    ]
    y = 0.85
    for label, value in stats:
        ax2.text(0.05, y, label + ":", fontsize=11, fontweight="bold",
                 transform=ax2.transAxes)
        ax2.text(0.55, y, value, fontsize=11, transform=ax2.transAxes)
        y -= 0.12

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 -- Output value heatmap
# ---------------------------------------------------------------------------

def plot_output_heatmap(all_data, operand_index, n, out_path,
                        title="Output Value Heatmap"):
    """
    Each cell coloured by its output value (viridis scale).
    Grey = no equation for that (a, b) pair.
    """
    val_grid, _, numeric = build_output_grid(all_data, operand_index, n)

    fig, ax = plt.subplots(figsize=(9, 8))

    cmap_img = plt.get_cmap("viridis").copy()
    cmap_img.set_bad(color=(0.85, 0.85, 0.85))   # grey for missing cells

    masked = np.ma.masked_invalid(val_grid)
    im = ax.imshow(masked, origin="upper", aspect="equal",
                   extent=[-0.5, n - 0.5, n - 0.5, -0.5],
                   cmap=cmap_img, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Output value" if numeric else "Output token index", fontsize=11)

    ax.set_xlabel("b  (column operand)", fontsize=11)
    ax.set_ylabel("a  (row operand)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- Output values as numbers in each cell
# ---------------------------------------------------------------------------

def plot_output_numbers(all_data, operand_index, n, out_path,
                        title="Output Values (Numbers)"):
    """
    Prints the output value as text inside each cell.
    Background colour is the same viridis scale as the heatmap.
    White/black text chosen automatically for readability.
    """
    val_grid, label_grid, numeric = build_output_grid(all_data, operand_index, n)

    fontsize = max(3, min(10, int(180 / n)))
    fig_side = max(8, min(24, n * 0.20))
    fig, ax = plt.subplots(figsize=(fig_side, fig_side * 0.92))

    cmap_img = plt.get_cmap("viridis").copy()
    cmap_img.set_bad(color=(0.85, 0.85, 0.85))

    vmin = np.nanmin(val_grid) if not np.all(np.isnan(val_grid)) else 0
    vmax = np.nanmax(val_grid) if not np.all(np.isnan(val_grid)) else 1
    norm = Normalize(vmin=vmin, vmax=vmax)

    masked = np.ma.masked_invalid(val_grid)
    ax.imshow(masked, origin="upper", aspect="equal",
              extent=[-0.5, n - 0.5, n - 0.5, -0.5],
              cmap=cmap_img, norm=norm, interpolation="nearest")

    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap_img)
    for row in range(n):
        for col in range(n):
            label = label_grid[row, col]
            if not label:
                continue
            v = val_grid[row, col]
            if np.isnan(v):
                continue
            rgba = scalar_map.to_rgba(v)
            lum  = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(col, row, label, ha="center", va="center",
                    fontsize=fontsize, color="white" if lum < 0.5 else "black",
                    fontfamily="monospace")

    ax.set_xlabel("b  (column operand)", fontsize=11)
    ax.set_ylabel("a  (row operand)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core: process a single folder
# ---------------------------------------------------------------------------

def process_folder(folder: Path) -> bool:
    """
    Generate the three figures for one dataset folder.
    Returns True on success, False if parsing yields no data.
    """
    train_path     = folder / "train.txt"
    val_path       = folder / "val.txt"
    tokenizer_path = folder / "tokenizer.txt"

    output_path_split_plot = resolve_out(folder, "split_plot.pdf")
    output_path_output_heatmap = resolve_out(folder, "output_heatmap.pdf")
    output_path_output_numbers = resolve_out(folder, "output_numbers.pdf")

    if os.path.exists(output_path_split_plot) and os.path.exists(output_path_output_heatmap) and os.path.exists(output_path_output_numbers):
        return True

    tokens        = load_tokens(tokenizer_path)
    operators     = extract_operators(tokens)
    operand_index = build_operand_index(tokens)
    n             = len(operand_index)

    train_data = load_data(train_path, operators, operand_index)
    val_data   = load_data(val_path,   operators, operand_index)
    all_data   = train_data + val_data

    print(f"  {len(train_data):,} train  |  {len(val_data):,} val  "
          f"|  {len(operators)} operator(s)")

    if not all_data:
        print("  WARNING: no parseable equations found, skipping.")
        return False

    # Infer actual grid size from data
    n = max(max(a, b) for a, b, _ in all_data) + 1


    if not os.path.exists(output_path_split_plot):
        plot_splits(
            train_data, val_data, n=n,
            out_path=output_path_split_plot,
            title=f"Train / Val Split  (grid {n}x{n})",
        )

    if not os.path.exists(output_path_output_heatmap):
        plot_output_heatmap(
            all_data, operand_index, n=n,
            out_path=output_path_output_heatmap,
            title=f"Output Value Heatmap  (grid {n}x{n})",
        )

    if not os.path.exists(output_path_output_numbers):
        plot_output_numbers(
            all_data, operand_index, n=n,
            out_path=output_path_output_numbers,
            title=f"Output Values  (grid {n}x{n})",
        )
    return True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=("Plot train/val split and output-value grids. Pass a folder containing train.txt / val.txt / tokenizer.txt, or use --recursive to scan all matching subfolders."))
    parser.add_argument("folder",help="Root folder to process (directly or recursively)",)
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        sys.exit(f"ERROR: not a directory: {root}")

    folders = find_dataset_folders(root)
    if not folders:
        sys.exit(
            "No subfolders containing train.txt + val.txt + tokenizer.txt "
            f"were found under {root}"
        )
    print(f"Found {len(folders)} dataset folder(s) under {root}\n")

    n_ok = n_fail = 0
    for i, folder in enumerate(folders, 1):
        print(f"[{i}/{len(folders)}] Processing: {folder}")
        try:
            ok = process_folder(folder)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            n_fail += 1
        print()

    print(f"Done. {n_ok} succeeded, {n_fail} failed.")


if __name__ == "__main__":
    main()