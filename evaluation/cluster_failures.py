"""
Failure case clustering for BDD100K object detection evaluation.

Identifies images where the model performs poorly (low IoU between predicted
and ground-truth boxes), extracts deep features via a ResNet-50 backbone,
and clusters these failures using KMeans to surface common failure patterns
such as night-time scenes, heavily occluded objects, or rare classes.

Usage:
    python cluster_failures.py \\
        --checkpoint  ../model/outputs/model_epoch_1.pth \\
        --val_images  /data/bdd100k/images/100k/val \\
        --val_labels  /data/bdd100k/labels/det_20/det_val.json \\
        --output_dir  ./failure_clusters \\
        --n_clusters  5
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from dataloader.bdd_dataset import BDD100KDataset, IDX_TO_CLASS, collate_fn
from train import build_model, NUM_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_max_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> float:
    """
    Compute the maximum IoU between any predicted box and any ground-truth box.

    A value close to 0 indicates a complete miss; a value close to 1 indicates
    a near-perfect localisation.

    Args:
        pred_boxes: (M, 4) predicted boxes in [x1, y1, x2, y2].
        gt_boxes:   (N, 4) ground-truth boxes in [x1, y1, x2, y2].

    Returns:
        Max IoU as a float, or 0.0 if either tensor is empty.
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0.0

    max_iou = 0.0
    for pb in pred_boxes:
        for gb in gt_boxes:
            ix1 = max(pb[0], gb[0])
            iy1 = max(pb[1], gb[1])
            ix2 = min(pb[2], gb[2])
            iy2 = min(pb[3], gb[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_p = (pb[2] - pb[0]) * (pb[3] - pb[1])
            area_g = (gb[2] - gb[0]) * (gb[3] - gb[1])
            union = area_p + area_g - inter
            iou = inter / union if union > 0 else 0.0
            if iou > max_iou:
                max_iou = iou
    return max_iou


def identify_failures(
    model: nn.Module,
    dataset: BDD100KDataset,
    device: torch.device,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.5,
    max_samples: int = 1000,
) -> List[Tuple[str, float]]:
    """
    Scan the dataset and collect (image_name, max_iou) for failure cases.

    A sample is a 'failure' if the maximum IoU between predictions and ground
    truth is below iou_threshold.

    Args:
        model:           Detection model in eval mode.
        dataset:         BDD100K dataset instance.
        device:          CUDA or CPU.
        iou_threshold:   IoU below which a sample is marked as a failure.
        score_threshold: Minimum prediction confidence to evaluate.
        max_samples:     Limit to first N samples for speed.

    Returns:
        List of (image_name, max_iou) for failure samples.
    """
    model.eval()
    to_tensor = T.ToTensor()
    failures = []

    for i in range(min(max_samples, len(dataset))):
        sample_dict = dataset.samples[i]
        image = dataset._load_image(sample_dict["image_name"])
        tensor = to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)[0]

        mask = outputs["scores"] >= score_threshold
        pred_boxes = outputs["boxes"][mask].cpu()
        gt_boxes = torch.tensor(sample_dict["boxes"], dtype=torch.float32)

        max_iou = compute_max_iou(pred_boxes, gt_boxes)
        if max_iou < iou_threshold:
            failures.append((sample_dict["image_name"], max_iou))

        if (i + 1) % 100 == 0:
            logger.info("  Scanned %d / %d  |  Failures so far: %d",
                        i + 1, max_samples, len(failures))

    return failures


def extract_embeddings(
    image_names: List[str],
    image_dir: str,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract 2048-d ResNet-50 feature embeddings for a list of images.

    Uses global average pooling output from ResNet-50 pre-trained on ImageNet
    as a rich visual descriptor for clustering.

    Args:
        image_names: List of image filenames (relative to image_dir).
        image_dir:   Base directory containing the images.
        device:      CUDA or CPU.
        batch_size:  Images processed per forward pass.

    Returns:
        NumPy array of shape (N, 2048).
    """
    backbone = tv_models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    backbone.eval().to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(image_names), batch_size):
            batch_names = image_names[start: start + batch_size]
            tensors = []
            for name in batch_names:
                path = os.path.join(image_dir, name)
                img = Image.open(path).convert("RGB")
                tensors.append(transform(img))
            batch = torch.stack(tensors).to(device)
            embs = backbone(batch).squeeze(-1).squeeze(-1).cpu().numpy()
            all_embeddings.append(embs)

    return np.vstack(all_embeddings)


def cluster_and_visualize(
    embeddings: np.ndarray,
    image_names: List[str],
    iou_scores: List[float],
    n_clusters: int,
    output_dir: str,
) -> np.ndarray:
    """
    KMeans cluster the failure embeddings and produce PCA scatter plots.

    Args:
        embeddings:   (N, D) feature matrix.
        image_names:  Corresponding image filenames.
        iou_scores:   Max IoU for each failure sample.
        n_clusters:   Number of KMeans clusters.
        output_dir:   Directory for output figures and JSON.

    Returns:
        (N,) array of cluster assignments.
    """
    scaler = StandardScaler()
    normed = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normed)

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(normed)

    # ── Scatter plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sc = axes[0].scatter(
        reduced[:, 0], reduced[:, 1],
        c=labels, cmap="tab10", alpha=0.7, s=20,
    )
    plt.colorbar(sc, ax=axes[0], label="Cluster")
    axes[0].set_title(
        f"Failure Cases — {n_clusters} Clusters (PCA of ResNet-50 features)",
        fontsize=11,
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    iou_arr = np.array(iou_scores)
    sc2 = axes[1].scatter(
        reduced[:, 0], reduced[:, 1],
        c=iou_arr, cmap="RdYlGn", alpha=0.7, s=20,
        vmin=0, vmax=1,
    )
    plt.colorbar(sc2, ax=axes[1], label="Max IoU")
    axes[1].set_title("Failure Cases — Coloured by Max IoU", fontsize=11)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "failure_clusters.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ── Per-cluster summary ────────────────────────────────────────────────
    cluster_summary = {}
    for c in range(n_clusters):
        mask = labels == c
        cluster_summary[f"cluster_{c}"] = {
            "size": int(mask.sum()),
            "mean_iou": float(iou_arr[mask].mean()),
            "sample_images": [
                image_names[i] for i in np.where(mask)[0][:5]
            ],
        }

    with open(os.path.join(output_dir, "cluster_summary.json"), "w") as f:
        json.dump(cluster_summary, f, indent=2)

    logger.info("Cluster summary:")
    for k, v in cluster_summary.items():
        logger.info(
            "  %-12s  size=%-5d  mean_iou=%.3f",
            k, v["size"], v["mean_iou"],
        )

    return labels


def run_failure_clustering(
    checkpoint: str,
    val_images: str,
    val_labels: str,
    output_dir: str,
    n_clusters: int = 5,
    iou_threshold: float = 0.3,
    max_samples: int = 1000,
) -> None:
    """
    End-to-end failure clustering pipeline.

    Args:
        checkpoint:    Path to model checkpoint.
        val_images:    Directory of validation images.
        val_labels:    Path to det_val.json.
        output_dir:    Output directory for figures and JSONs.
        n_clusters:    Number of KMeans clusters.
        iou_threshold: IoU below which a sample is a failure.
        max_samples:   Scan at most this many validation samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(NUM_CLASSES, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    dataset = BDD100KDataset(val_images, val_labels)

    logger.info("Identifying failure cases (IoU < %.2f)…", iou_threshold)
    failures = identify_failures(
        model, dataset, device,
        iou_threshold=iou_threshold,
        max_samples=max_samples,
    )
    logger.info("Found %d failure cases out of %d scanned", len(failures), max_samples)

    if len(failures) < n_clusters:
        logger.warning("Too few failures (%d) for %d clusters. Skipping.", len(failures), n_clusters)
        return

    fail_names = [f[0] for f in failures]
    fail_ious = [f[1] for f in failures]

    logger.info("Extracting ResNet-50 embeddings…")
    embeddings = extract_embeddings(fail_names, val_images, device)

    logger.info("Clustering %d failures into %d groups…", len(embeddings), n_clusters)
    cluster_and_visualize(embeddings, fail_names, fail_ious, n_clusters, output_dir)
    logger.info("Failure analysis saved to: %s", output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="BDD100K Failure Case Clustering")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val_images", required=True)
    p.add_argument("--val_labels", required=True)
    p.add_argument("--output_dir", default="./failure_clusters")
    p.add_argument("--n_clusters", type=int, default=5)
    p.add_argument("--iou_threshold", type=float, default=0.3)
    p.add_argument("--max_samples", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_failure_clustering(
        checkpoint=args.checkpoint,
        val_images=args.val_images,
        val_labels=args.val_labels,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        iou_threshold=args.iou_threshold,
        max_samples=args.max_samples,
    )
