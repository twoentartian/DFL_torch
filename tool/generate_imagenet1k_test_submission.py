import argparse
import json
import os, sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import importlib.resources as ir

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util, ml_setup

def load_imagenet_class_index() -> List[str]:
    p = ir.files("py_src.ml_setup_base").joinpath("imagenet_class_index.json")
    with p.open("r") as f:
        class_index = json.load(f)
    # Map 0..999 -> wnid
    wnids = [class_index[str(i)][0] for i in range(1000)]
    return wnids

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> List[torch.Tensor]:
    """Compute the top-k accuracies"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  # [B, maxk]
    pred = pred.t()                             # [maxk, B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ImageListDataset(Dataset):
    """
    For ImageNet test set (no labels). It usually contains flat files like:
    ILSVRC2012_test_00000001.JPEG ... ILSVRC2012_test_00050000.JPEG (or 100k for full set)
    We sort filenames lexicographically to match submission order.
    """
    EXT = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted([p for p in self.root.glob("*") if p.suffix in self.EXT])
        if len(self.paths) == 0:
            # Also support nested dirs (rare)
            self.paths = sorted([p for p in self.root.rglob("*") if p.suffix in self.EXT])
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(path.name)  # no label


def build_test_loader(test_dir: str, batch_size: int, workers: int) -> Tuple[DataLoader, List[str]]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    ds = ImageListDataset(test_dir, transform=t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    filenames = [p for p in ds.paths]  # full paths if you need them
    return loader, [Path(p).name for p in filenames]

# ---------- Eval & Submission ----------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = False):
    crit = nn.CrossEntropyLoss()
    model.eval()
    top1_m, top5_m, loss_m, n = 0.0, 0.0, 0.0, 0

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with (torch.amp.autocast(device_type=device.type) if amp else torch.no_grad()):
            output = model(images)
            loss = crit(output, target)

        bsz = images.size(0)
        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m += loss.item() * bsz
        top1_m += top1.item() * bsz
        top5_m += top5.item() * bsz
        n += bsz

    return loss_m / n, top1_m / n, top5_m / n


@torch.no_grad()
def predict_topk(model: nn.Module, loader: DataLoader, device: torch.device, k: int = 5, amp: bool = False):
    model.eval()
    all_topk: List[torch.Tensor] = []
    with (torch.amp.autocast(device_type=device.type) if amp else torch.no_grad()):
        for images, _names in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)              # [B, 1000]
            _, topk_idx = logits.topk(k, dim=1) # [B, k]
            all_topk.append(topk_idx.cpu())
    return torch.cat(all_topk, dim=0)  # [N, k]

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(description="Eval on ImageNet val and create test submission (Top-5 WNIDs).")
    p.add_argument("model_state_file_path", type=str, help="path to the .model.pt file.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("-j", "--workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="Enable amp for speed.")

    args = p.parse_args()

    model_state_file_path = args.model_state_file_path
    batch_size = args.batch_size
    workers = args.workers
    device = args.device
    amp = args.amp

    device = torch.device(device)
    cudnn.benchmark = True if device.type == "cuda" else False
    output = f"{model_state_file_path}.test_submission.txt"

    state_dict, model_name, dataset_name = util.load_model_state_file(model_state_file_path)
    if dataset_name != ml_setup.DatasetType.imagenet1k.name:
        print(f"WARNING: the dataset in model state dict file says the dataset name is {dataset_name}, not as expected {ml_setup.DatasetType.imagenet1k.name}")
    current_ml_setup = ml_setup.get_ml_setup_from_config(model_name, dataset_type=ml_setup.DatasetType.imagenet1k.name)

    # Build model & load weights
    model = current_ml_setup.model
    model.load_state_dict(state_dict)
    model.to(device)

    test_dir = f"{ml_setup.imagenet1k_path}/test"

    # Data loaders
    val_loader = DataLoader(current_ml_setup.testing_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader, test_filenames = build_test_loader(test_dir, batch_size, workers)

    # Evaluate on validation (test set has no labels)
    val_loss, val_top1, val_top5 = evaluate(model, val_loader, device, amp=amp)
    print(f"[VAL] loss: {val_loss:.4f} | top1: {val_top1:.3f}% | top5: {val_top5:.3f}%")

    # Predict on test & write submission
    wnids = load_imagenet_class_index()
    top5_idx = predict_topk(model, test_loader, device, k=5, amp=amp)  # [N,5]
    assert top5_idx.shape[0] == len(test_filenames), "Mismatch between predictions and test images."

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write exactly 5 WNIDs per line, space-separated, in the SAME ORDER as the input listing
    with open(out_path, "w") as f:
        for i in range(top5_idx.shape[0]):
            wnid_line = " ".join(str(int(idx)+1) for idx in top5_idx[i])
            f.write(wnid_line + "\n")

    print(f"[SUBMISSION] Wrote Top-5 predictions to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
