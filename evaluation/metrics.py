"""
Quantitative evaluation metrics for BDD100K object detection.

Computes mean Average Precision (mAP) using torchmetrics, per-class AP,
and generates a summary table with visualisations.

Metrics rationale (documented here and in docs/model_selection.md):
    - mAP@IoU=0.50       : Standard PASCAL VOC metric; industry baseline for
                           BDD100K leaderboard comparison.
    - mAP@IoU=0.50:0.95  : COCO-style stricter metric; penalises imprecise
                           localisation; better reflects real-world usability.
    - Per-class AP        : Reveals which classes the model handles poorly,
                           enabling targeted data collection or augmentation.

Usage:
    python metrics.py \\
        --checkpoint ../model/outputs/model_epoch_1.pth \\
        --val_images  /data/bdd100k/images/100k/val \\
        --val_labels  /data/bdd100k/labels/det_20/det_val.json \\
        --output_dir  ./eval_results
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from dataloader.bdd_dataset import BDD100KDataset, BDD_CLASSES, collate_fn
from train import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

NUM_CLASSES = len(BDD_CLASSES) + 1


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.05,
) -> Dict:
    """
    Evaluate the model on a validation DataLoader and compute mAP metrics.

    Args:
        model:            Trained detection model in eval mode.
        data_loader:      DataLoader over the validation dataset.
        device:           CUDA or CPU device.
        score_threshold:  Minimum score for a prediction to be included in
                          the metric computation (low threshold for mAP).

    Returns:
        Dict of computed metric tensors from torchmetrics MeanAveragePrecision.
    """
    model.eval()
    metric = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=[0.5, 0.75],
        class_metrics=True,
    )

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = [
                {
                    "boxes": o["boxes"][o["scores"] >= score_threshold].cpu(),
                    "labels": o["labels"][o["scores"] >= score_threshold].cpu(),
                    "scores": o["scores"][o["scores"] >= score_threshold].cpu(),
                }
                for o in outputs
            ]
            tgts = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(preds, tgts)

            if (batch_idx + 1) % 50 == 0:
                logger.info("  Evaluated %d batches…", batch_idx + 1)

    return metric.compute()


def format_results(results: Dict) -> pd.DataFrame:
    """
    Format torchmetrics output into a readable summary DataFrame.

    Args:
        results: Dict returned by MeanAveragePrecision.compute().

    Returns:
        DataFrame with one row per metric and a per-class AP table.
    """
    summary = {
        "mAP@0.50": results["map_50"].item(),
        "mAP@0.75": results["map_75"].item(),
        "mAP@0.50:0.95": results["map"].item(),
        "mAP_small": results.get("map_small", torch.tensor(float("nan"))).item(),
        "mAP_medium": results.get("map_medium", torch.tensor(float("nan"))).item(),
        "mAP_large": results.get("map_large", torch.tensor(float("nan"))).item(),
    }

    per_class_ap = {}
    if "map_per_class" in results:
        for i, ap in enumerate(results["map_per_class"]):
            cls_name = BDD_CLASSES[i] if i < len(BDD_CLASSES) else f"class_{i}"
            per_class_ap[cls_name] = ap.item()

    return summary, per_class_ap


def plot_per_class_ap(per_class_ap: Dict[str, float], save_path: str = None) -> plt.Figure:
    """
    Horizontal bar chart of per-class Average Precision.

    Args:
        per_class_ap: Dict mapping class name to AP value.
        save_path:    If provided, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    df = pd.DataFrame(
        sorted(per_class_ap.items(), key=lambda x: x[1]),
        columns=["class", "AP@0.50"],
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#e74c3c" if v < 0.3 else "#f39c12" if v < 0.6 else "#27ae60"
              for v in df["AP@0.50"]]
    ax.barh(df["class"], df["AP@0.50"], color=colors)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Average Precision @ IoU=0.50")
    ax.set_title("Per-Class Average Precision", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def run_evaluation(
    checkpoint: str,
    val_images: str,
    val_labels: str,
    output_dir: str,
    batch_size: int = 4,
    max_samples: int = 0,
) -> None:
    """
    Full evaluation pipeline: load model, run inference, compute and save metrics.

    Args:
        checkpoint:  Path to .pth model checkpoint.
        val_images:  Directory of validation images.
        val_labels:  Path to det_val.json.
        output_dir:  Directory to write results JSON, CSV, and figures.
        batch_size:  Batch size for the validation DataLoader.
        max_samples: Limit evaluation to first N samples (0 = full val set).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating on device: %s", device)

    model = build_model(NUM_CLASSES, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    val_dataset = BDD100KDataset(
        val_images, val_labels,
        max_samples=max_samples if max_samples > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )
    logger.info("Validation samples: %d", len(val_dataset))

    logger.info("Running evaluation…")
    results = evaluate(model, val_loader, device)
    summary, per_class_ap = format_results(results)

    logger.info("─" * 50)
    logger.info("Evaluation Results:")
    for k, v in summary.items():
        logger.info("  %-25s %.4f", k, v)
    logger.info("─" * 50)
    logger.info("Per-Class AP:")
    for cls, ap in sorted(per_class_ap.items(), key=lambda x: -x[1]):
        logger.info("  %-20s %.4f", cls, ap)

    # Save results
    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
        json.dump({"summary": summary, "per_class_ap": per_class_ap}, f, indent=2)

    pd.DataFrame(
        list(per_class_ap.items()), columns=["class", "AP@0.50"]
    ).to_csv(os.path.join(output_dir, "per_class_ap.csv"), index=False)

    fig = plot_per_class_ap(
        per_class_ap,
        save_path=os.path.join(output_dir, "per_class_ap.png"),
    )
    plt.close(fig)
    logger.info("Evaluation results saved to: %s", output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="BDD100K Object Detection Evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val_images", required=True)
    p.add_argument("--val_labels", required=True)
    p.add_argument("--output_dir", default="./eval_results")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument(
        "--max_samples", type=int, default=0,
        help="Limit to N val samples (0 = full set)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        checkpoint=args.checkpoint,
        val_images=args.val_images,
        val_labels=args.val_labels,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
