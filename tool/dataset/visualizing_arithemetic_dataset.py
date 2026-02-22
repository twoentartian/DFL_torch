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
from itertools import permutations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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

    # Everything left of ' = ' is the LHS
    parts = eq.split(" = ")
    if len(parts) < 2:
        return None
    lhs = parts[0].strip()

    # Find the operator (longest-match, searched as ' OP ' substring)
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

    return a_idx, b_idx


def load_coords(txt_path: Path, operators: list[str],
                operand_index: dict[str, int]) -> list[tuple[int, int]]:
    coords = []
    for line in txt_path.read_text().splitlines():
        result = parse_equation(line, operators, operand_index)
        if result is not None:
            coords.append(result)
    return coords


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_splits(train_coords, val_coords, n: int, out_path: Path,
                title: str = "Train / Val Split"):
    """
    Draw an n×n grid coloured by split membership.

    Blue  → train only
    Red   → val only
    Green → overlap (both)
    White → missing
    """
    grid = np.full((n, n), fill_value=-1, dtype=np.int8)

    for a, b in train_coords:
        if 0 <= a < n and 0 <= b < n:
            grid[a, b] = 0
    for a, b in val_coords:
        if 0 <= a < n and 0 <= b < n:
            grid[a, b] = 2 if grid[a, b] == 0 else 1

    cmap = {
        -1: (1.00, 1.00, 1.00, 1.0),   # white  – missing
         0: (0.20, 0.45, 0.75, 1.0),   # blue   – train
         1: (0.85, 0.25, 0.25, 1.0),   # red    – val
         2: (0.20, 0.70, 0.30, 1.0),   # green  – overlap
    }
    img = np.zeros((n, n, 4))
    for v, color in cmap.items():
        img[grid == v] = color

    n_train = int((grid == 0).sum())
    n_val   = int((grid == 1).sum())
    n_both  = int((grid == 2).sum())
    n_miss  = int((grid == -1).sum())
    total   = n * n

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
        patches.append(mpatches.Patch(color=cmap[2][:3],
                                      label=f"Both  ({n_both:,})"))
    if n_miss:
        patches.append(mpatches.Patch(color=(0.9, 0.9, 0.9),
                                      label=f"Missing ({n_miss:,})"))
    ax.legend(handles=patches, loc="upper right", framealpha=0.85, fontsize=10)

    ax2 = axes[1]
    ax2.axis("off")
    stats = [
        ("Grid size",   f"{n} × {n}  =  {total:,}"),
        ("Train cells", f"{n_train:,}  ({100*n_train/total:.1f}%)"),
        ("Val cells",   f"{n_val:,}  ({100*n_val/total:.1f}%)"),
        ("Overlap",     f"{n_both:,}  ({100*n_both/total:.1f}%)"),
        ("Missing",     f"{n_miss:,}  ({100*n_miss/total:.1f}%)"),
    ]
    y = 0.85
    for label, value in stats:
        ax2.text(0.05, y, label + ":", fontsize=11, fontweight="bold",
                 transform=ax2.transAxes)
        ax2.text(0.55, y, value, fontsize=11, transform=ax2.transAxes)
        y -= 0.12

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot train/val split locations. Pass the folder containing "
                    "train.txt, val.txt, and tokenizer.txt."
    )
    parser.add_argument("folder", help="Folder containing train.txt, val.txt, tokenizer.txt")
    args = parser.parse_args()

    folder = Path(args.folder)
    train_path     = folder / "train.txt"
    val_path       = folder / "val.txt"
    tokenizer_path = folder / "tokenizer.txt"

    for p in (train_path, val_path, tokenizer_path):
        if not p.exists():
            sys.exit(f"ERROR: expected file not found: {p}")

    print(f"Reading vocabulary from {tokenizer_path} …")
    tokens         = load_tokens(tokenizer_path)
    operators      = extract_operators(tokens)
    operand_index  = build_operand_index(tokens)
    n              = len(operand_index)  # grid side length

    print(f"  {len(operators)} operator(s) found: {operators}")
    print(f"  {n} operand tokens → {n}×{n} grid")

    print("Parsing equations …")
    train_coords = load_coords(train_path, operators, operand_index)
    val_coords   = load_coords(val_path,   operators, operand_index)
    print(f"  {len(train_coords):,} train  |  {len(val_coords):,} val")

    # Infer actual grid size from data — avoids bloat when the tokenizer
    # contains both integer tokens (0-96) and s5 permutation tokens (120)
    # but only one type is actually used in the equations.
    all_coords = train_coords + val_coords
    if all_coords:
        n = max(max(a, b) for a, b in all_coords) + 1
        print(f"  Actual grid size inferred from data: {n}×{n}")

    # Save alongside the data if possible, else fall back to cwd
    out_path = folder / "split_plot.png"
    try:
        out_path.touch()
        out_path.unlink()
    except OSError:
        out_path = Path.cwd() / "split_plot.png"
        print(f"  (folder is read-only — saving to {out_path})")

    plot_splits(train_coords, val_coords, n=n, out_path=out_path,
                title=f"Train / Val Split  (grid {n}×{n})")


if __name__ == "__main__":
    main()
