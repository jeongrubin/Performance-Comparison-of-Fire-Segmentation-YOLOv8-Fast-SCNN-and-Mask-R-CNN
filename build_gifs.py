"""Build short animated GIF demos for all three models.

YOLO/Fast-SCNN run on the same source video; Mask R-CNN uses a separate
video because it does not generalize to the aerial smoke scene used for
the other two (see README '테스트 이미지 모델별 Ground Truth vs Prediction 비교').
Adjust the path constants for your own environment before running.
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image

from models.fast_scnn_fire import FastSCNN

YOLO_WEIGHTS = "fire_seg_yolov8n.pt"
FAST_SCNN_WEIGHTS = "fast_scnn_fire.pth"
MASK_RCNN_WEIGHTS = "mask_rcnn_model.pth"

YOLO_FASTSCNN_VIDEO = "sample_videos/fire_test.mp4"
MASK_RCNN_VIDEO = "sample_videos/armored_vehicle_fire.mp4"

OUT_DIR = "assets/demo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def frames_to_gif(frames_rgb, out_path, width=480, duration=90):
    pil_frames = []
    for f in frames_rgb:
        img = Image.fromarray(f)
        w, h = img.size
        new_h = int(h * width / w)
        img = img.resize((width, new_h), Image.LANCZOS)
        img = img.convert("P", palette=Image.ADAPTIVE, colors=128)
        pil_frames.append(img)
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                        duration=duration, loop=0, optimize=True)
    print(f"saved {out_path} ({len(pil_frames)} frames, {os.path.getsize(out_path) / 1e6:.2f} MB)")


def build_yolo_gif(n_frames=45, stride=3):
    from ultralytics import YOLO

    model = YOLO(YOLO_WEIGHTS)
    cap = cv2.VideoCapture(YOLO_FASTSCNN_VIDEO)
    frames, i = [], 0
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % stride == 0:
            plotted = model.predict(frame, verbose=False)[0].plot()
            frames.append(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    frames_to_gif(frames, os.path.join(OUT_DIR, "yolo_demo.gif"))


def build_fastscnn_gif(n_frames=45, stride=3):
    model = FastSCNN(num_classes=2)
    model.load_state_dict(torch.load(FAST_SCNN_WEIGHTS, map_location=device, weights_only=True))
    model.to(device).eval()

    cap = cv2.VideoCapture(YOLO_FASTSCNN_VIDEO)
    frames, i = [], 0
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % stride == 0:
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(cv2.resize(img_rgb, (512, 512))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = torch.argmax(model(tensor).squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
            pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = frame.copy()
            overlay[pred_full == 1] = [0, 0, 255]
            blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            frames.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    frames_to_gif(frames, os.path.join(OUT_DIR, "fastscnn_demo.gif"))


def build_mask_rcnn_gif(n_frames=45, stride=3, threshold=0.5):
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.transforms import functional as F

    model = maskrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(MASK_RCNN_WEIGHTS, map_location=device, weights_only=True))
    model.to(device).eval()

    cap = cv2.VideoCapture(MASK_RCNN_VIDEO)
    frames, i = [], 0
    with torch.no_grad():
        while len(frames) < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if i % stride == 0:
                tensor = F.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                preds = model(tensor)
                masks, boxes, scores = preds[0]["masks"], preds[0]["boxes"], preds[0]["scores"]
                vis = frame.copy()
                for j in range(len(masks)):
                    if scores[j] > threshold:
                        m = masks[j, 0].mul(255).byte().cpu().numpy()
                        box = boxes[j].cpu().numpy().astype(int)
                        vis[m > 128] = [0, 0, 255]
                        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
                frames.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            i += 1
    cap.release()
    frames_to_gif(frames, os.path.join(OUT_DIR, "maskrcnn_demo.gif"))


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    build_yolo_gif()
    build_fastscnn_gif()
    build_mask_rcnn_gif()
