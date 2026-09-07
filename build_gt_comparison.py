"""Build Original / Ground Truth / Prediction comparison panels for all three models.

Paths below point at each model's own held-out test split (see README
'데이터셋 및 전처리' for how these directories are structured). Adjust the
path constants for your own environment before running.
"""
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.fast_scnn_fire import FastSCNN

random.seed(0)

YOLO_WEIGHTS = "fire_seg_yolov8n.pt"
YOLO_TEST_IMAGES = "data/yolo_test/images"
YOLO_TEST_LABELS = "data/yolo_test/labels"

MASK_RCNN_WEIGHTS = "mask_rcnn_model.pth"
MASK_RCNN_TEST_IMAGES = "data/test/images"
MASK_RCNN_TEST_MASKS = "data/test/masks"

FAST_SCNN_WEIGHTS = "fast_scnn_fire.pth"
FAST_SCNN_TEST_IMAGES = "data/fastscnn_test/images"
FAST_SCNN_TEST_MASKS = "data/fastscnn_test/masks"

OUT_DIR = "assets/predictions"


def gt_from_yolo_label(label_path, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = list(map(float, parts[1:]))
            pts = np.array([[coords[i] * w, coords[i + 1] * h] for i in range(0, len(coords), 2)], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def overlay(img_bgr, mask, color=(0, 0, 255), alpha=0.5):
    colored = np.zeros_like(img_bgr)
    colored[mask == 1] = color
    return cv2.addWeighted(img_bgr, 1, colored, alpha, 0)


def make_panel(rows_data, title, out_path):
    n = len(rows_data)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.6 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    fig.suptitle(title, fontsize=13)
    col_titles = ["Original", "Ground Truth", "Prediction"]
    for i, (orig, gt, pred, _name) in enumerate(rows_data):
        for j, im in enumerate([orig, gt, pred]):
            axes[i, j].imshow(im)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"saved {out_path}")


def build_yolo(n_samples=6):
    from ultralytics import YOLO

    model = YOLO(YOLO_WEIGHTS)
    files = sorted(os.listdir(YOLO_TEST_IMAGES))
    random.shuffle(files)

    rows = []
    for fn in files[:n_samples]:
        img = cv2.imread(os.path.join(YOLO_TEST_IMAGES, fn))
        h, w = img.shape[:2]
        gt = gt_from_yolo_label(os.path.join(YOLO_TEST_LABELS, os.path.splitext(fn)[0] + ".txt"), h, w)
        results = model.predict(img, verbose=False)
        pred = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks is not None:
            for m in results[0].masks.data.cpu().numpy():
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                pred = np.maximum(pred, (m_resized > 0.5).astype(np.uint8))
        rows.append((
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(overlay(img, gt), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(overlay(img, pred), cv2.COLOR_BGR2RGB),
            fn,
        ))
    make_panel(rows, "YOLOv8-Seg: Original / Ground Truth / Prediction (test set)",
               os.path.join(OUT_DIR, "yolo_predicted.png"))


def build_mask_rcnn(n_samples=6, threshold=0.5):
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.transforms import functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(MASK_RCNN_WEIGHTS, map_location=device, weights_only=True))
    model.to(device).eval()

    files = sorted(os.listdir(MASK_RCNN_TEST_IMAGES))
    random.shuffle(files)

    rows = []
    with torch.no_grad():
        for fn in files[:n_samples]:
            img = cv2.imread(os.path.join(MASK_RCNN_TEST_IMAGES, fn))
            h, w = img.shape[:2]
            gt_path = os.path.join(MASK_RCNN_TEST_MASKS, os.path.splitext(fn)[0] + ".png")
            gt = (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8) if os.path.exists(gt_path) else np.zeros((h, w), np.uint8)

            tensor = F.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            preds = model(tensor)
            masks, scores = preds[0]["masks"], preds[0]["scores"]
            pred = np.zeros((h, w), dtype=np.uint8)
            for i in range(len(masks)):
                if scores[i] > threshold:
                    pred = np.maximum(pred, (masks[i, 0].cpu().numpy() > 0.5).astype(np.uint8))

            rows.append((
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(overlay(img, gt), cv2.COLOR_BGR2RGB),
                cv2.cvtColor(overlay(img, pred), cv2.COLOR_BGR2RGB),
                fn,
            ))
    make_panel(rows, "Mask R-CNN: Original / Ground Truth / Prediction (test set)",
               os.path.join(OUT_DIR, "mask_rcnn_predicted.png"))


def build_fast_scnn(n_samples=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSCNN(num_classes=2)
    model.load_state_dict(torch.load(FAST_SCNN_WEIGHTS, map_location=device, weights_only=True))
    model.to(device).eval()

    files = sorted(os.listdir(FAST_SCNN_TEST_IMAGES))
    random.shuffle(files)

    rows = []
    with torch.no_grad():
        for fn in files[:n_samples]:
            img = cv2.imread(os.path.join(FAST_SCNN_TEST_IMAGES, fn), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            gt_path = os.path.join(FAST_SCNN_TEST_MASKS, os.path.splitext(fn)[0] + ".png")
            gt = (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8) if os.path.exists(gt_path) else np.zeros((h, w), np.uint8)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (512, 512))
            tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            out = model(tensor)
            pred = torch.argmax(out.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

            rows.append((
                img_rgb,
                cv2.cvtColor(overlay(img, gt), cv2.COLOR_BGR2RGB),
                cv2.cvtColor(overlay(img, pred), cv2.COLOR_BGR2RGB),
                fn,
            ))
    make_panel(rows, "Fast-SCNN (retrained): Original / Ground Truth / Prediction (test set)",
               os.path.join(OUT_DIR, "fast_scnn_predicted.png"))


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    build_yolo()
    build_mask_rcnn()
    build_fast_scnn()
