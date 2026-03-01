"""
Main entry point for BDD100K data analysis pipeline.

Parses the dataset, computes statistics, saves CSVs for the dashboard,
and generates static visualisation figures.

Usage:
    python main.py \\
        --train_labels /data/bdd100k/labels/det_20/det_train.json \\
        --val_labels   /data/bdd100k/labels/det_20/det_val.json \\
        --train_images /data/bdd100k/images/100k/train \\
        --val_images   /data/bdd100k/images/100k/val \\
        --output_dir   /data/precomputed
"""

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")

from analysis.anomaly_detection import summarize_anomalies
from analysis.class_distribution import (
    compute_bbox_stats,
    compute_class_distribution,
    compute_cooccurrence_matrix,
    compute_images_per_class,
    compute_scene_distribution,
    compute_train_val_comparison,
    plot_class_distribution,
    plot_cooccurrence_heatmap,
)
from parser.bdd_parser import BDDParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_analysis(
    train_labels: str,
    val_labels: str,
    train_images: str,
    val_images: str,
    output_dir: str,
) -> None:
    """
    Execute the full data analysis pipeline and persist results.

    Steps:
        1. Parse train and val JSON labels.
        2. Compute and save class distribution CSVs.
        3. Compute and save bounding box statistics.
        4. Compute and save scene metadata distributions.
        5. Detect and log anomalies.
        6. Save static figures (PNG) for the report.

    Args:
        train_labels: Path to det_train.json.
        val_labels:   Path to det_val.json.
        train_images: Directory containing training images.
        val_images:   Directory containing validation images.
        output_dir:   Directory where CSV files and figures are written.
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ── 1. Parse ─────────────────────────────────────────────────────────────
    logger.info("Parsing training labels from %s", train_labels)
    train_parser = BDDParser(train_labels, train_images)
    train_samples = train_parser.parse()
    logger.info("  → %d training samples loaded", len(train_samples))

    logger.info("Parsing validation labels from %s", val_labels)
    val_parser = BDDParser(val_labels, val_images)
    val_samples = val_parser.parse()
    logger.info("  → %d validation samples loaded", len(val_samples))

    all_samples = train_samples + val_samples

    # ── 2. Class distribution ─────────────────────────────────────────────────
    logger.info("Computing class distributions …")

    train_dist = compute_class_distribution(train_samples)
    val_dist = compute_class_distribution(val_samples)
    combined_dist = compute_class_distribution(all_samples)

    combined_dist.to_csv(os.path.join(output_dir, "class_dist.csv"), index=False)
    logger.info("  Saved class_dist.csv")

    img_per_class = compute_images_per_class(all_samples)
    img_per_class.to_csv(
        os.path.join(output_dir, "images_per_class.csv"), index=False
    )
    logger.info("  Saved images_per_class.csv")

    tv_compare = compute_train_val_comparison(train_samples, val_samples)
    tv_compare.to_csv(os.path.join(output_dir, "train_val_compare.csv"), index=False)
    logger.info("  Saved train_val_compare.csv")

    # Save distribution figures
    fig = plot_class_distribution(combined_dist, title="All Data — Annotation Count per Class")
    fig.savefig(os.path.join(figures_dir, "class_dist_all.png"), dpi=150)

    fig_train = plot_class_distribution(train_dist, title="Train — Annotation Count per Class")
    fig_train.savefig(os.path.join(figures_dir, "class_dist_train.png"), dpi=150)

    # ── 3. BBox statistics ───────────────────────────────────────────────────
    logger.info("Computing bounding box statistics …")
    bbox_df = compute_bbox_stats(all_samples)
    bbox_df.to_csv(os.path.join(output_dir, "bbox_stats.csv"), index=False)
    logger.info("  Saved bbox_stats.csv (%d rows)", len(bbox_df))

    # ── 4. Scene metadata ────────────────────────────────────────────────────
    logger.info("Computing scene metadata distributions …")
    scene_df = compute_scene_distribution(all_samples)
    scene_df.to_csv(os.path.join(output_dir, "scene_dist.csv"), index=False)
    logger.info("  Saved scene_dist.csv")

    # ── 5. Co-occurrence matrix ──────────────────────────────────────────────
    logger.info("Computing co-occurrence matrix …")
    cooc = compute_cooccurrence_matrix(all_samples)
    cooc.to_csv(os.path.join(output_dir, "cooccurrence.csv"))
    fig_cooc = plot_cooccurrence_heatmap(cooc)
    fig_cooc.savefig(os.path.join(figures_dir, "cooccurrence_heatmap.png"), dpi=150)
    logger.info("  Saved cooccurrence.csv and figure")

    # ── 6. Anomaly summary ───────────────────────────────────────────────────
    logger.info("Running anomaly detection …")
    anomalies = summarize_anomalies(all_samples)
    logger.info("  Anomaly summary:")
    for k, v in anomalies.items():
        logger.info("    %-40s %d", k, v)

    import json
    with open(os.path.join(output_dir, "anomaly_summary.json"), "w") as f:
        json.dump(anomalies, f, indent=2)
    logger.info("  Saved anomaly_summary.json")

    logger.info("Analysis complete. Outputs written to: %s", output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="BDD100K Object Detection — Data Analysis Pipeline"
    )
    p.add_argument(
        "--train_labels",
        required=True,
        help="Path to det_train.json",
    )
    p.add_argument(
        "--val_labels",
        required=True,
        help="Path to det_val.json",
    )
    p.add_argument(
        "--train_images",
        required=True,
        help="Directory containing BDD100K training images",
    )
    p.add_argument(
        "--val_images",
        required=True,
        help="Directory containing BDD100K validation images",
    )
    p.add_argument(
        "--output_dir",
        default="/data/precomputed",
        help="Output directory for CSVs and figures (default: /data/precomputed)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        train_labels=args.train_labels,
        val_labels=args.val_labels,
        train_images=args.train_images,
        val_images=args.val_images,
        output_dir=args.output_dir,
    )
