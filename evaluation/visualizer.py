"""
Qualitative visualisation of ground truth vs. model predictions on BDD100K.

For each validation image, renders side-by-side panels showing:
    - Left:  Ground truth boxes (green).
    - Right: Model predictions with confidence scores (red).

Also provides a confusion-style class breakdown figure.

Usage:
    python visualizer.py \\
        --checkpoint ../model/outputs/model_epoch_1.pth \\
        --val_images  /data/bdd100k/images/100k/val \\
        --val_labels  /data/bdd100k/labels/det_20/det_val.json \\
        --output_dir  ./qual_results \\
        --n_samples   50
"""

import argparse
import logging
import os
import random
import sys
from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from dataloader.bdd_dataset import BDD100KDataset, IDX_TO_CLASS, collate_fn
from train import build_model, NUM_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

GT_COLOR = "#2ecc71"
PRED_COLOR = "#e74c3c"


def _draw_boxes(
    ax: plt.Axes,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor = None,
    color: str = GT_COLOR,
) -> None:
    """
    Draw labelled bounding boxes on a matplotlib Axes.

    Args:
        ax:     Matplotlib Axes with an image already displayed.
        boxes:  (N, 4) tensor with [x1, y1, x2, y2] coordinates.
        labels: (N,) tensor of class indices.
        scores: Optional (N,) tensor of confidence scores.
        color:  Edge and text colour for the boxes.
    """
    for i, (box, lbl) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.tolist()
        cls_name = IDX_TO_CLASS.get(lbl.item(), "?")
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        label_text = cls_name
        if scores is not None:
            label_text += f" {scores[i].item():.2f}"
        ax.text(
            x1, max(y1 - 4, 0), label_text,
            color="white", fontsize=7,
            bbox=dict(facecolor=color, alpha=0.75, pad=1, edgecolor="none"),
        )


def visualize_prediction(
    image: Image.Image,
    gt: Dict,
    pred: Dict,
    save_path: str,
    score_threshold: float = 0.5,
) -> None:
    """
    Save a side-by-side ground truth / prediction figure.

    Args:
        image:           RGB PIL Image.
        gt:              Target dict with 'boxes' and 'labels' tensors.
        pred:            Prediction dict with 'boxes', 'labels', 'scores'.
        save_path:       Output file path.
        score_threshold: Minimum confidence to display a prediction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax in axes:
        ax.imshow(image)
        ax.axis("off")

    axes[0].set_title(f"Ground Truth ({len(gt['boxes'])} objects)", fontsize=11)
    _draw_boxes(axes[0], gt["boxes"], gt["labels"], color=GT_COLOR)

    mask = pred["scores"] >= score_threshold
    n_pred = mask.sum().item()
    axes[1].set_title(f"Predictions ({n_pred} detections @ ≥{score_threshold})", fontsize=11)
    _draw_boxes(
        axes[1],
        pred["boxes"][mask],
        pred["labels"][mask],
        pred["scores"][mask],
        color=PRED_COLOR,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_qualitative_eval(
    checkpoint: str,
    val_images: str,
    val_labels: str,
    output_dir: str,
    n_samples: int = 50,
    score_threshold: float = 0.5,
) -> None:
    """
    Generate qualitative visualisation images from the validation set.

    Randomly selects n_samples from the validation set and saves side-by-side
    GT vs prediction figures.

    Args:
        checkpoint:      Path to model checkpoint.
        val_images:      Directory of validation images.
        val_labels:      Path to det_val.json.
        output_dir:      Directory to save output images.
        n_samples:       Number of images to visualise.
        score_threshold: Confidence threshold for displayed predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(NUM_CLASSES, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    dataset = BDD100KDataset(val_images, val_labels)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    to_tensor = T.ToTensor()
    logger.info("Visualising %d samples…", len(indices))

    for i, idx in enumerate(indices):
        sample_dict = dataset.samples[idx]
        image = dataset._load_image(sample_dict["image_name"])

        tensor = to_tensor(image).unsqueeze(0).to(device)
        outputs = model(tensor)[0]

        pred = {
            "boxes": outputs["boxes"].cpu(),
            "labels": outputs["labels"].cpu(),
            "scores": outputs["scores"].cpu(),
        }

        import torch as _torch
        gt = {
            "boxes": _torch.tensor(sample_dict["boxes"], dtype=_torch.float32),
            "labels": _torch.tensor(sample_dict["labels"], dtype=_torch.int64),
        }

        save_path = os.path.join(output_dir, f"sample_{idx:05d}.jpg")
        visualize_prediction(image, gt, pred, save_path, score_threshold)

        if (i + 1) % 10 == 0:
            logger.info("  Saved %d / %d", i + 1, len(indices))

    logger.info("Qualitative results saved to: %s", output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="BDD100K Qualitative Visualisation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val_images", required=True)
    p.add_argument("--val_labels", required=True)
    p.add_argument("--output_dir", default="./qual_results")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--score_threshold", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_qualitative_eval(
        checkpoint=args.checkpoint,
        val_images=args.val_images,
        val_labels=args.val_labels,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        score_threshold=args.score_threshold,
    )
