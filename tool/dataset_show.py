import argparse, math, random, os, sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.ml_setup_base import dataset as dfl_dataset


def show_batch(dl, dataset_name, cols=8, save_path=None):
    it = iter(dl)

    while True:
        target = next(it)
        if target is None:
            break
        """Visualize a batch of CHW images in [0,1]."""
        if dfl_dataset.is_masked_dataset[dataset_name]:
            images, labels, path_img, path_mask = target
            n = images.size(0)
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows*2, cols, figsize=(cols * 2, rows * 4), squeeze=False)

            for i in range(n):
                i_row = i // cols
                i_col = i % cols
                img = images[i].detach().cpu()          # C,H,W
                img = img.permute(1, 2, 0).clamp(0, 1)  # H,W,C
                label = int(labels[i])
                ax: plt.Axes = axes[i_row, i_col]
                ax.imshow(img.numpy())
                ax.set_title(f"label: {label}", fontsize=10)
                ax.axis("off")
                ax: plt.Axes = axes[i_row+rows, i_col]
                raw_img = plt.imread(path_img[i])
                ax.imshow(raw_img)
                ax.set_title(f"label: {label} (raw image)", fontsize=10)
                ax.axis('off')

            fig.tight_layout()
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved to {save_path}")
                input("Press Enter to continue...")
            else:
                plt.show(block=True)
        else:
            images, labels = target
            n = images.size(0)
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)

            for i in range(n):
                i_row = i // cols
                i_col = i % cols
                img = images[i].detach().cpu()          # C,H,W
                img = img.permute(1, 2, 0).clamp(0, 1)  # H,W,C
                label = int(labels[i])
                ax: plt.Axes = axes[i_row, i_col]
                ax.imshow(img.numpy())
                ax.set_title(f"label: {label}", fontsize=10)
                ax.axis("off")

            fig.tight_layout()
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved to {save_path}")
                input("Press Enter to continue...")
            else:
                plt.show(block=True)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize ImageDataset batch")
    ap.add_argument("dataset", help=f"dataset type, candidate: {dfl_dataset.name_to_dataset_setup.keys()}")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cols", type=int, default=8, help="number of columns in the grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="", help="optional path to save the grid as an image")
    ap.add_argument("--unmasked_area_type", default="random", help="optional arg pass to masked dataset")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_name = args.dataset
    if dataset_name in dfl_dataset.name_to_dataset_setup:
        if dfl_dataset.is_masked_dataset[dataset_name]:
            ds = dfl_dataset.name_to_dataset_setup[dataset_name](return_path=True, unmasked_area_type=args.unmasked_area_type)
        else:
            ds = dfl_dataset.name_to_dataset_setup[dataset_name]()
    else:
        print(f"dataset name {dataset_name} not found, available: {dfl_dataset.name_to_dataset_setup.keys()}")
        exit(-1)

    dl = DataLoader(
        ds.training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    show_batch(dl, dataset_name, cols=args.cols, save_path=(args.save or None))


if __name__ == "__main__":
    main()