#!/usr/bin/env python3
"""
Create SAM masks for ImageNet train images using provided Pascal-VOC XML bboxes.
Saves masks (binary, 0/255 PNG) under out_root with same folder layout.

Example:
python imagenet_sam_batch.py \
  --train_root /path/to/train \
  --annotations_root /path/to/annotations_or_same_as_train \
  --out_root /path/to/training_mask \
  --sam_ckpt /path/to/sam_vit_h_4b8939.pth \
  --model_type vit_h \
  --multimask

"""

import argparse, os, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry, SamPredictor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src.util import expand_path

def find_xml_for_image(img_path: Path, annotations_root: Path):
    # Try same folder first (img.jpg -> img.xml)
    xml_same = img_path.with_suffix('.xml')
    if xml_same.exists():
        return xml_same
    # Try annotations_root with same relative path: replace train_root with annotations_root
    try:
        rel = img_path.relative_to(args.train_root)
        xml_candidate = annotations_root / rel
        xml_candidate = xml_candidate.with_suffix('.xml')
        if xml_candidate.exists():
            return xml_candidate
    except Exception:
        pass
    # fallback: look for xml in the parent label folder with same basename
    xml_in_parent = img_path.parent / (img_path.stem + '.xml')
    if xml_in_parent.exists():
        return xml_in_parent
    return None

def load_bbox_from_pascal(xml_path: Path):
    tree = ET.parse(str(xml_path))
    # If multiple objects, return first object. You can change to iterate.
    obj = tree.find('object')
    if obj is None:
        raise RuntimeError(f'No <object> in {xml_path}')
    b = obj.find('bndbox')
    xmin = int(float(b.find('xmin').text))
    ymin = int(float(b.find('ymin').text))
    xmax = int(float(b.find('xmax').text))
    ymax = int(float(b.find('ymax').text))
    return np.array([xmin, ymin, xmax, ymax], dtype=int)

def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None  # empty mask
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])  # xmin,ymin,xmax,ymax

def iou_xyxy(a, b):
    # a,b: [xmin,ymin,xmax,ymax]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1 + 1); ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area_a = max(0, (ax2-ax1+1)) * max(0, (ay2-ay1+1))
    area_b = max(0, (bx2-bx1+1)) * max(0, (by2-by1+1))
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0

def pick_best_mask(masks, scores, box_xyxy):
    # masks: (N,H,W) boolean
    if masks.shape[0] == 0:
        return None, None
    # compute IoU of mask bbox with input box
    best_idx = 0
    best_score = -1.0
    for i in range(masks.shape[0]):
        m = masks[i].astype(np.uint8)
        mb = mask_bbox(m)
        if mb is None:
            cur = -1.0
        else:
            cur = iou_xyxy(mb, box_xyxy)
        # combine with SAM confidence score if provided
        combined = cur if scores is None else (cur + float(scores[i])) / 2.0
        if combined > best_score:
            best_score = combined
            best_idx = i
    return masks[best_idx].astype(np.uint8), float(scores[best_idx]) if scores is not None and len(scores)>best_idx else None

def overlay_and_save(img, mask, save_path_mask, save_overlay=False):
    # mask: 0/1 array same H,W
    save_path_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path_mask), (mask * 255).astype(np.uint8))
    if save_overlay:
        overlay = img.copy()
        # make colored overlay (red) where mask==1
        alpha = 0.35
        overlay[mask==1] = (overlay[mask==1] * (1-alpha) + np.array([0,0,255]) * alpha).astype(np.uint8)
        out_overlay = save_path_mask.with_name(save_path_mask.stem + "_overlay.png")
        cv2.imwrite(str(out_overlay), overlay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', type=Path, default='~/dataset/imagenet1k/train', help='root folder of train (e.g. /.../train)')
    parser.add_argument('--annotations_root', type=Path, default='~/dataset/imagenet1k/train_bounding_box', help='root of xml annotations (if absent, will try next to images)')
    parser.add_argument('--out_root', default='~/dataset/imagenet1k/train_sam_mask', type=Path, help='root to save masks (mirrored structure)')
    parser.add_argument('--sam_ckpt', default='~/dataset/imagenet1k/sam_vit_h.pth', type=Path, help='SAM checkpoint .pth')
    parser.add_argument('--model_type', default='vit_h', choices=['vit_h','vit_l','vit_b'])
    parser.add_argument('--multimask', action='store_true', help='let SAM produce multiple masks and select best')
    parser.add_argument('--save_overlay', action='store_true', help='also save overlay pngs (mask applied to image)')
    parser.add_argument('--ext', default='.JPEG', help='image extension to search for (.JPEG or .jpg)')
    args = parser.parse_args()

    args.train_root = expand_path(args.train_root)
    args.annotations_root = expand_path(args.annotations_root)
    args.out_root = expand_path(args.out_root)
    args.sam_ckpt = expand_path(args.sam_ckpt)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(device.index)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available — falling back to CPU (very slow).")

    # Prepare SAM
    print("Loading SAM model:", args.model_type)
    sam = sam_model_registry[args.model_type](checkpoint=str(args.sam_ckpt))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Collect image paths
    imgs = sorted(args.train_root.rglob(f'*{args.ext}'))
    if len(imgs) == 0:
        imgs = sorted(args.train_root.rglob('*.jpg')) + sorted(args.train_root.rglob('*.jpeg'))
    print(f"Found {len(imgs)} images under {args.train_root}")

    # annotations_root fallback
    annotations_root = args.annotations_root if args.annotations_root is not None else args.train_root

    # process loop
    for imgp in tqdm(imgs, desc="images"):
        try:
            # corresponding output path
            rel = imgp.relative_to(args.train_root)
            out_mask_path = args.out_root / rel.parent / (imgp.stem + '.png')

            xmlp = find_xml_for_image(imgp, annotations_root)
            if xmlp is None:
                # no annotation -> skip
                # tqdm.write(f"NO_XML skip: {imgp}")
                continue

            # read inputs
            img_bgr = cv2.imread(str(imgp))
            if img_bgr is None:
                tqdm.write(f"BAD_IMAGE skip: {imgp}")
                continue
            box = load_bbox_from_pascal(xmlp)  # xmin,ymin,xmax,ymax

            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)  # internally uses sam.to(device)

            # run SAM
            with torch.no_grad():
                if device.type == 'cuda':
                    # AMP autocast may reduce memory and speed up some ops
                    with torch.amp.autocast('cuda'):
                        masks, scores, logits = predictor.predict(
                            box=box[None, :],
                            multimask_output=args.multimask
                        )
                else:
                    masks, scores, logits = predictor.predict(
                        box=box[None, :],
                        multimask_output=args.multimask
                    )

            masks = masks.astype(np.uint8)  # (N,H,W)
            best_mask, best_score = pick_best_mask(masks, scores, box)
            if best_mask is None:
                tqdm.write(f"NO_MASK produced: {imgp}")
                continue

            overlay_and_save(img_bgr, best_mask, out_mask_path, save_overlay=args.save_overlay)

            del masks, scores, logits, best_mask
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            tqdm.write(f"ERROR {imgp}: {e}")
            continue

    print("done")
