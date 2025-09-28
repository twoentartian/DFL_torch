import argparse, math, random, os, sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.ml_setup_base import dataset as dfl_dataset


def show_batch(images, labels, cols=8, save_path=None):
    """Visualize a batch of CHW images in [0,1]."""
    n = images.size(0)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(n):
        img = images[i].detach().cpu()          # C,H,W
        img = img.permute(1, 2, 0).clamp(0, 1)  # H,W,C
        label = int(labels[i])
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img.numpy())
        ax.set_title(f"label: {label}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Visualize ImageDataset batch")
    ap.add_argument("dataset", help="dataset type, mnist, cifar10, cifar100, ")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cols", type=int, default=8, help="number of columns in the grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="", help="optional path to save the grid as an image")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset in dfl_dataset.name_to_dataset_setup:
        ds = dfl_dataset.name_to_dataset_setup[args.dataset]()
    else:
        print(f"dataset name {args.dataset} not found, available: {dfl_dataset.name_to_dataset_setup.keys()}")
        exit(-1)

    dl = DataLoader(
        ds.training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Grab one batch
    images, labels = next(iter(dl))
    show_batch(images, labels, cols=args.cols, save_path=(args.save or None))


if __name__ == "__main__":
    main()