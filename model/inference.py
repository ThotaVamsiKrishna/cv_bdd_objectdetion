"""
Inference script for the trained BDD100K Faster R-CNN model.

Loads a saved checkpoint and runs detection on a single image or a directory
of images, saving annotated output images.

Usage:
    # Single image
    python inference.py \\
        --checkpoint outputs/model_epoch_1.pth \\
        --input path/to/image.jpg \\
        --output_dir ./predictions

    # Directory of images
    python inference.py \\
        --checkpoint outputs/model_epoch_1.pth \\
        --input path/to/images/ \\
        --output_dir ./predictions
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms as T

from dataloader.bdd_dataset import BDD_CLASSES, IDX_TO_CLASS
from train import build_model

NUM_CLASSES = len(BDD_CLASSES) + 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_COLORS = [
    "#e6194b", "#f58231", "#3cb44b", "#4363d8", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#ffffff",
]


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load a Faster R-CNN model from a saved checkpoint.

    Args:
        checkpoint_path: Path to a .pth file saved by train.py.
        device:          Device to load the model onto.

    Returns:
        Model in evaluation mode.
    """
    num_classes = len(BDD_CLASSES) + 1
    model = build_model(num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(
        "Loaded checkpoint from epoch %d (loss=%.4f)",
        checkpoint.get("epoch", "?"),
        checkpoint.get("loss", float("nan")),
    )
    return model


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    score_threshold: float = 0.5,
) -> Dict:
    """
    Run inference on a single PIL image.

    Args:
        model:           Model in eval mode.
        image:           RGB PIL Image.
        device:          Device to run on.
        score_threshold: Discard predictions below this confidence score.

    Returns:
        Dict with filtered 'boxes', 'labels', 'scores' tensors on CPU.
    """
    transform = T.ToTensor()
    tensor = transform(image).unsqueeze(0).to(device)
    outputs = model(tensor)[0]

    mask = outputs["scores"] >= score_threshold
    return {
        "boxes": outputs["boxes"][mask].cpu(),
        "labels": outputs["labels"][mask].cpu(),
        "scores": outputs["scores"][mask].cpu(),
    }


def draw_predictions(
    image: Image.Image,
    predictions: Dict,
    save_path: str,
) -> None:
    """
    Draw predicted bounding boxes on the image and save to disk.

    Args:
        image:       Original RGB PIL Image.
        predictions: Output dict from predict().
        save_path:   Path where the annotated image will be saved.
    """
    fig, ax = plt.subplots(1, figsize=(14, 8))
    ax.imshow(image)
    ax.axis("off")

    for box, label, score in zip(
        predictions["boxes"],
        predictions["labels"],
        predictions["scores"],
    ):
        x1, y1, x2, y2 = box.tolist()
        cls_name = IDX_TO_CLASS.get(label.item(), "unknown")
        color = CLASS_COLORS[label.item() % len(CLASS_COLORS)]

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 5, 0),
            f"{cls_name} {score:.2f}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor=color, alpha=0.75, pad=1, edgecolor="none"),
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_inference(
    checkpoint: str,
    input_path: str,
    output_dir: str,
    score_threshold: float,
) -> None:
    """
    Run inference on a single image or all images in a directory.

    Args:
        checkpoint:      Path to the model checkpoint.
        input_path:      Path to an image file or directory.
        output_dir:      Directory to save annotated images.
        score_threshold: Minimum confidence to keep a detection.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)

    input_path = Path(input_path)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = [
            p for p in input_path.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    logger.info("Running inference on %d image(s)…", len(image_paths))
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        preds = predict(model, image, device, score_threshold)
        out_name = img_path.stem + "_pred.jpg"
        save_path = os.path.join(output_dir, out_name)
        draw_predictions(image, preds, save_path)
        logger.info(
            "  %s → %d detections → %s",
            img_path.name,
            len(preds["boxes"]),
            out_name,
        )

    logger.info("Inference complete. Results saved to: %s", output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inference script."""
    p = argparse.ArgumentParser(
        description="BDD100K Faster R-CNN Inference"
    )
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument(
        "--input",
        required=True,
        help="Path to a single image or a directory of images",
    )
    p.add_argument("--output_dir", default="./predictions")
    p.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score to keep a prediction",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        checkpoint=args.checkpoint,
        input_path=args.input,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
    )
