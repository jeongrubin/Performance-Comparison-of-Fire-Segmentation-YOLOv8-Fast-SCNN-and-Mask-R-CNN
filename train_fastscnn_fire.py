"""Train/evaluate the FastSCNN architecture actually used for the fire checkpoint.

Data layout expected (see README '데이터셋 및 전처리'):
  data/{train,val,test}/images/*.jpg
  data/{train,val,test}/masks/*.png   (binary, 0=background/255=fire)
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from models.fast_scnn_fire import FastSCNN

DATA_ROOT = "data"
CKPT_OUT = "fast_scnn_fire.pth"
LOG_OUT = "train_log.json"
NUM_EPOCHS = 30


class FireDataset(Dataset):
    def __init__(self, split):
        self.img_dir = os.path.join(DATA_ROOT, split, "images")
        self.mask_dir = os.path.join(DATA_ROOT, split, "masks")
        self.files = sorted(os.listdir(self.img_dir))
        self.img_tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        base = os.path.splitext(fn)[0]
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, base + ".png")).convert("L")
        img = self.img_tf(img)
        mask = mask.resize((64, 64), Image.NEAREST)
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.int64))
        return img, mask


def compute_metrics(model, loader, device):
    model.eval()
    tp = fp = fn_ = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            pred = torch.argmax(model(images), dim=1)
            tp += ((pred == 1) & (masks == 1)).sum().item()
            fp += ((pred == 1) & (masks == 0)).sum().item()
            fn_ += ((pred == 0) & (masks == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn_ + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn_ + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = FireDataset("train"), FireDataset("val"), FireDataset("test")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    model = FastSCNN(num_classes=2).to(device)
    # fire pixels are a small minority of each frame -> weight the fire class higher
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_iou, history = -1, []
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        val_metrics = compute_metrics(model, val_loader, device)
        print(f"epoch {epoch+1}/{NUM_EPOCHS} loss={running_loss/len(train_loader):.4f} "
              f"val_iou={val_metrics['iou']:.4f} val_f1={val_metrics['f1']:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": running_loss / len(train_loader), **val_metrics})

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), CKPT_OUT)

    model.load_state_dict(torch.load(CKPT_OUT, map_location=device, weights_only=True))
    test_metrics = compute_metrics(model, test_loader, device)
    print(f"FINAL TEST METRICS: {test_metrics}")

    with open(LOG_OUT, "w") as f:
        json.dump({"history": history, "test_metrics": test_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
