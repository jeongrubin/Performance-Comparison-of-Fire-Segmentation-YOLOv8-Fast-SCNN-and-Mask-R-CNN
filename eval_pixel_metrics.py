"""Pixel-level Precision/Recall/F1/IoU evaluation shared across all three models.

This is the metric definition used for the README's comparison table: a fire/
not-fire label per pixel, aggregated over the whole test set (not per-instance
detection AP). Ground-truth masks are binary PNGs (0=background, >0=fire).

Usage:
    python eval_pixel_metrics.py --model yolo --weights fire_seg_yolov8n.pt \
        --images path/to/test/images --labels path/to/test/labels_or_masks

Each model needs slightly different label parsing (YOLO polygon .txt vs PNG
masks), so this script is meant as a reference implementation rather than a
one-line CLI for every checkpoint format.
"""
import argparse
import os

import cv2
import numpy as np
import torch


def pixel_metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def yolo_poly_to_mask(label_path, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = list(map(float, parts[1:]))
            pts = np.array(
                [[coords[i] * w, coords[i + 1] * h] for i in range(0, len(coords), 2)],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 1)
    return mask


def eval_yolo(weights, images_dir, labels_dir):
    from ultralytics import YOLO

    model = YOLO(weights)
    tp = fp = fn = 0
    for fname in sorted(os.listdir(images_dir)):
        img = cv2.imread(os.path.join(images_dir, fname))
        h, w = img.shape[:2]
        gt = yolo_poly_to_mask(os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt"), h, w)

        results = model.predict(img, verbose=False)
        pred = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks is not None:
            for m in results[0].masks.data.cpu().numpy():
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                pred = np.maximum(pred, (m_resized > 0.5).astype(np.uint8))

        tp += int(((pred == 1) & (gt == 1)).sum())
        fp += int(((pred == 1) & (gt == 0)).sum())
        fn += int(((pred == 0) & (gt == 1)).sum())
    return pixel_metrics(tp, fp, fn)


def eval_mask_rcnn(weights, images_dir, masks_dir, threshold=0.5):
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.transforms import functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device).eval()

    tp = fp = fn = 0
    with torch.no_grad():
        for fname in sorted(os.listdir(images_dir)):
            img = cv2.imread(os.path.join(images_dir, fname))
            h, w = img.shape[:2]
            gt_path = os.path.join(masks_dir, os.path.splitext(fname)[0] + ".png")
            if not os.path.exists(gt_path):
                continue
            gt = (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

            tensor = F.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            preds = model(tensor)
            masks, scores = preds[0]["masks"], preds[0]["scores"]
            pred = np.zeros((h, w), dtype=np.uint8)
            for i in range(len(masks)):
                if scores[i] > threshold:
                    pred = np.maximum(pred, (masks[i, 0].cpu().numpy() > 0.5).astype(np.uint8))

            tp += int(((pred == 1) & (gt == 1)).sum())
            fp += int(((pred == 1) & (gt == 0)).sum())
            fn += int(((pred == 0) & (gt == 1)).sum())
    return pixel_metrics(tp, fp, fn)


def eval_fast_scnn(weights, images_dir, masks_dir):
    from models.fast_scnn_fire import FastSCNN
    import torchvision.transforms as T
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSCNN(num_classes=2)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device).eval()

    tp = fp = fn = 0
    with torch.no_grad():
        for fname in sorted(os.listdir(images_dir)):
            img = Image.open(os.path.join(images_dir, fname)).convert("RGB")
            w, h = img.size
            gt_path = os.path.join(masks_dir, os.path.splitext(fname)[0] + ".png")
            if not os.path.exists(gt_path):
                continue
            gt = (np.array(Image.open(gt_path).convert("L")) > 0).astype(np.uint8)

            tensor = T.Compose([T.Resize((512, 512)), T.ToTensor()])(img).unsqueeze(0).to(device)
            out = model(tensor)
            pred = torch.argmax(out.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

            tp += int(((pred == 1) & (gt == 1)).sum())
            fp += int(((pred == 1) & (gt == 0)).sum())
            fn += int(((pred == 0) & (gt == 1)).sum())
    return pixel_metrics(tp, fp, fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolo", "mask_rcnn", "fast_scnn"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True, help="YOLO: label .txt dir. mask_rcnn/fast_scnn: mask .png dir")
    args = parser.parse_args()

    if args.model == "yolo":
        metrics = eval_yolo(args.weights, args.images, args.labels)
    elif args.model == "mask_rcnn":
        metrics = eval_mask_rcnn(args.weights, args.images, args.labels)
    else:
        metrics = eval_fast_scnn(args.weights, args.images, args.labels)

    print(metrics)
