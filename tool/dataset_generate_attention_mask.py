import argparse, os , sys, math, traceback
from PIL import Image
from pathlib import Path
import numpy as np

import torch.cuda
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import py_src.ml_setup as ml_setup
from py_src.ml_setup import ModelType
from py_src.ml_setup_base.dataset import default_path_imagenet1k

def load_model_dataloader(model_name, device):
    model_name_type = ModelType[model_name]
    if model_name_type == ModelType.vit_b_16:
        model = models.vit_b_16(progress=False, weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1, num_classes=1000)
        dataset = datasets.ImageNet(root=default_path_imagenet1k, split='train', transform=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False,num_workers=8, pin_memory=device)
    else:
        raise NotImplementedError


    return model, dataloader

def save_mask(mask01: np.ndarray, out_path: Path, out_size):
    im = Image.fromarray((mask01 * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    im = im.resize(out_size, resample=Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path.with_suffix(".png"))

def vit_reshape_transform(tensor):
    # (B, tokens, C) -> (B, C, H, W), drop CLS
    if tensor.dim() != 3:
        raise RuntimeError(f"Unexpected tensor shape: {tensor.shape}")
    B, N, C = tensor.shape
    x = tensor[:, 1:, :]
    H = W = int(math.sqrt(x.shape[1]))
    if H * W != x.shape[1]:
        raise RuntimeError(f"Non-square token grid: {x.shape[1]}")
    return x.permute(0, 2, 1).reshape(B, C, H, W)

def main():
    ap = argparse.ArgumentParser(description="Generate attention mask for ImageNet1k based on a transformer model.")
    ap.add_argument("-d", "--dataset", default="imagenet1k", type=str, help="Dataset type")
    ap.add_argument("-m", "--model", default="vit_b_16", type=str, help="Select model type to extract the attention mask.")
    ap.add_argument("--topk", type=int, default=1, help="Top-k classes to aggregate CAM over.")
    args = ap.parse_args()

    device = torch.device("cuda")
    model, dataloader = load_model_dataloader(args.model, device)
    print(f"model name: {model.__class__.__name__}")

    target_layers = [model.encoder.layers[-1].ln_1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=vit_reshape_transform)

    from tqdm import tqdm
    for batch in tqdm(dataloader, desc="Processing"):
        try:
            x, y, paths, orig_sizes = batch
            x = x.to(device, non_blocking=True)

            # Pick class targets via a quick forward (no grad); CAM will re-forward with grads internally
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)

            targets_list = []
            for i in range(probs.shape[0]):
                k = min(args.topk, probs.shape[1])
                topk = torch.topk(probs[i], k=k).indices
                targets_list.append([ClassifierOutputTarget(int(c)) for c in topk])

            # Compute CAMs (B, Hc, Wc)
            grayscale_cam = cam(input_tensor=x, targets=targets_list)  # numpy float32 in [0,1]

            # Save each item in the batch
            for i in range(grayscale_cam.shape[0]):
                mask = grayscale_cam[i] if args.topk == 1 else np.mean(grayscale_cam[i], axis=0)
                rel = Path(paths[i]).relative_to(args.train_root)
                out_path = args.out_root / rel.with_suffix(".png")
                save_mask(mask, out_path, out_size=orig_sizes[i])

        except Exception as e:
            sys.stderr.write(f"\n[WARN] batch failed: {e}\n")
            traceback.print_exc(limit=1, file=sys.stderr)
            continue

    print(f"Done. Masks saved under: {args.out_root}")


if __name__ == '__main__':
    main()