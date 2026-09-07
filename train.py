import os
import argparse
import time
import shutil

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision import transforms

from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.loss import MixSoftmaxCrossEntropyLoss
from utils.lr_scheduler import LRScheduler
from utils.metric import SegmentationMetric


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Fast-SCNN on PyTorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='model name (default: fast_scnn)')
    parser.add_argument('--dataset', type=str, default='fire',
                        help='dataset name (default: fire)')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=8,
                        metavar='N', help='input batch size for training (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    # checkpoint
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume training')
    parser.add_argument('--save-folder', default='./weights',
                        help='directory for saving checkpoint models')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print(args)
    return args


class Trainer:
    def __init__(self, args):
        self.args = args

        # Image transformations
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        ])

        # Dataset and DataLoader
        train_dataset = get_segmentation_dataset(args.dataset, split='train', transform=input_transform)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', transform=input_transform)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)

        # Create model
        self.model = get_fast_scnn(num_classes=2, aux=args.aux)  # 2 classes: background + fire
        self.model.to(args.device)

        # Resume from checkpoint if specified
        if args.resume and os.path.isfile(args.resume):
            print(f"Resuming training from {args.resume}...")
            self.model.load_state_dict(torch.load(args.resume, map_location=args.device))

        # Loss function
        self.criterion = MixSoftmaxCrossEntropyLoss(aux=args.aux, aux_weight=args.aux_weight).to(args.device)

        # Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # Learning rate scheduler
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_loader), power=0.9)

        # Metrics
        self.metric = SegmentationMetric(2)  # 2 classes: background + fire
        self.best_pred = 0.0

    def train(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()
            total_loss = 0.0

            for i, (images, targets) in enumerate(self.train_loader):
                images, targets = images.to(self.args.device), targets.to(self.args.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{self.args.epochs}], Step [{i + 1}/{len(self.train_loader)}], "
                          f"Loss: {loss.item():.4f}")

            # Validation
            if not self.args.no_val:
                self.validate()

            # Save checkpoint
            save_checkpoint(self.model, self.args, is_best=False)

    def validate(self):
        self.metric.reset()
        self.model.eval()
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(images)
                preds = torch.argmax(outputs[0], dim=1).cpu().numpy()
                self.metric.update(preds, targets.cpu().numpy())

            pixAcc, mIoU = self.metric.get()
            print(f"Validation Pixel Accuracy: {pixAcc * 100:.2f}%, mIoU: {mIoU * 100:.2f}%")

            return pixAcc, mIoU


def save_checkpoint(model, args, is_best=False):
    """Save model checkpoint."""
    directory = args.save_folder
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = f"{args.model}_{args.dataset}.pth"
    filepath = os.path.join(directory, filename)
    torch.save(model.state_dict(), filepath)

    if is_best:
        best_filepath = os.path.join(directory, f"{args.model}_{args.dataset}_best.pth")
        shutil.copy(filepath, best_filepath)


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)

    if args.eval:
        print(f"Evaluating model from {args.resume}")
        trainer.validate()
    else:
        print(f"Starting training from epoch {args.start_epoch} to {args.epochs}")
        trainer.train()
