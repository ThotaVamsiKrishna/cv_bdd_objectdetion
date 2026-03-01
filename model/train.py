"""
Training pipeline for BDD100K object detection using Faster R-CNN.

Architecture choice — Faster R-CNN with ResNet-50 + FPN backbone:
    - Two-stage detector: Region Proposal Network (RPN) + RoI classification.
    - FPN (Feature Pyramid Network) allows multi-scale detection — crucial for
      BDD100K which contains objects at vastly different scales (pedestrian
      at 30px vs truck at 500px).
    - Pre-trained on COCO; fine-tuned on BDD100K's 10 classes.
    - Strong mAP baseline while remaining explainable and well-documented.

Usage:
    python train.py \\
        --train_images /data/bdd100k/images/100k/train \\
        --train_labels /data/bdd100k/labels/det_20/det_train.json \\
        --val_images   /data/bdd100k/images/100k/val \\
        --val_labels   /data/bdd100k/labels/det_20/det_val.json \\
        --epochs 1 --subset 500 --batch_size 4 --output_dir ./outputs
"""

import argparse
import logging
import os
import time
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataloader.bdd_dataset import BDD100KDataset, BDD_CLASSES, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

NUM_CLASSES = len(BDD_CLASSES) + 1  # +1 for background


def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """
    Build a Faster R-CNN model with ResNet-50 FPN backbone.

    The classification head is replaced to match num_classes (BDD100K + bg).
    All other weights are initialised from COCO pre-training.

    Args:
        num_classes: Number of output classes including background.
        pretrained:  If True, load COCO pre-trained backbone weights.

    Returns:
        Faster R-CNN model ready for fine-tuning.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 20,
) -> float:
    """
    Run one full training epoch over data_loader.

    Args:
        model:       Detection model in training mode.
        optimizer:   Gradient descent optimiser.
        data_loader: DataLoader yielding (images, targets) batches.
        device:      CUDA or CPU device.
        epoch:       Current epoch index (for logging).
        print_freq:  Log a progress line every this many batches.

    Returns:
        Mean loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0
    start = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in t.items()} for t in targets
        ]

        loss_dict: Dict[str, torch.Tensor] = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += losses.item()

        if (batch_idx + 1) % print_freq == 0:
            elapsed = time.time() - start
            logger.info(
                "Epoch [%d] Batch [%d/%d] | Loss: %.4f | %.1f s/batch",
                epoch,
                batch_idx + 1,
                len(data_loader),
                losses.item(),
                elapsed / (batch_idx + 1),
            )

    mean_loss = total_loss / len(data_loader)
    logger.info("Epoch [%d] complete — Mean Loss: %.4f", epoch, mean_loss)
    return mean_loss


def train(
    train_images: str,
    train_labels: str,
    val_images: str,
    val_labels: str,
    epochs: int,
    batch_size: int,
    lr: float,
    subset: int,
    output_dir: str,
) -> None:
    """
    Full training loop: build model, load data, train, and save weights.

    Args:
        train_images: Directory of training images.
        train_labels: Path to det_train.json.
        val_images:   Directory of validation images.
        val_labels:   Path to det_val.json.
        epochs:       Number of training epochs.
        batch_size:   Samples per mini-batch.
        lr:           Initial learning rate for SGD.
        subset:       If > 0, train on only this many samples (quick run).
        output_dir:   Directory to save checkpoints and loss logs.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # ── Dataset / DataLoader ─────────────────────────────────────────────
    train_dataset = BDD100KDataset(train_images, train_labels)
    if subset > 0:
        train_dataset = Subset(train_dataset, list(range(min(subset, len(train_dataset)))))
        logger.info("Using subset of %d training samples", len(train_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(NUM_CLASSES, pretrained=True).to(device)
    logger.info("Model: Faster R-CNN ResNet-50 FPN | Classes: %d", NUM_CLASSES)

    # ── Optimiser & Scheduler ────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # ── Training loop ────────────────────────────────────────────────────
    loss_history: List[float] = []
    for epoch in range(1, epochs + 1):
        epoch_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        loss_history.append(epoch_loss)
        scheduler.step()

        ckpt_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            },
            ckpt_path,
        )
        logger.info("Checkpoint saved: %s", ckpt_path)

    # Save loss history
    import json
    with open(os.path.join(output_dir, "loss_history.json"), "w") as f:
        json.dump({"epoch_losses": loss_history}, f, indent=2)
    logger.info("Training complete. Outputs: %s", output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    p = argparse.ArgumentParser(
        description="BDD100K Faster R-CNN Training Pipeline"
    )
    p.add_argument("--train_images", required=True)
    p.add_argument("--train_labels", required=True)
    p.add_argument("--val_images", required=True)
    p.add_argument("--val_labels", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument(
        "--subset",
        type=int,
        default=500,
        help="Train on first N samples only (0 = full dataset)",
    )
    p.add_argument("--output_dir", default="./outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        train_images=args.train_images,
        train_labels=args.train_labels,
        val_images=args.val_images,
        val_labels=args.val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subset=args.subset,
        output_dir=args.output_dir,
    )
