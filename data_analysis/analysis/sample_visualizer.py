"""
Sample visualization utilities for BDD100K dataset exploration.

Provides functions to draw annotated images, class-specific grids,
and bounding box distribution plots.
"""

import random
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from parser.bdd_parser import Sample, BDD_DETECTION_CLASSES

CLASS_COLORS: Dict[str, str] = {
    "pedestrian": "#e6194b",
    "rider": "#f58231",
    "car": "#3cb44b",
    "truck": "#4363d8",
    "bus": "#911eb4",
    "train": "#42d4f4",
    "motorcycle": "#f032e6",
    "bicycle": "#bfef45",
    "traffic light": "#fabed4",
    "traffic sign": "#469990",
}


def visualize_sample(
    sample: Sample,
    figsize: tuple = (12, 7),
    show_labels: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Draw a single BDD100K image with all its bounding box annotations.

    Args:
        sample:      Sample object containing image path and annotations.
        figsize:     Figure dimensions (width, height) in inches.
        show_labels: If True, overlay class name on each box.
        save_path:   If provided, save figure to this file path.

    Returns:
        Matplotlib Figure with the annotated image.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    image = sample.load_image()
    if image is None:
        raise FileNotFoundError(
            f"Image not found: {sample.image_path}"
        )

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    ax.axis("off")

    title_parts = [sample.image_name]
    if sample.weather:
        title_parts.append(sample.weather)
    if sample.time_of_day:
        title_parts.append(sample.time_of_day)
    ax.set_title(" | ".join(title_parts), fontsize=10)

    for ann in sample.annotations:
        color = CLASS_COLORS.get(ann.category, "#ffffff")
        rect = patches.Rectangle(
            (ann.bbox.x1, ann.bbox.y1),
            ann.bbox.width,
            ann.bbox.height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        if show_labels:
            label = ann.category
            if ann.occluded:
                label += " [occ]"
            ax.text(
                ann.bbox.x1,
                max(ann.bbox.y1 - 4, 0),
                label,
                color="white",
                fontsize=7,
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def visualize_class_grid(
    samples: List[Sample],
    category: str,
    n_samples: int = 9,
    cols: int = 3,
    figsize: tuple = (15, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Display a grid of cropped bounding box patches for a given class.

    Randomly samples up to n_samples annotations of the given category and
    displays them as cropped patches from their source images.

    Args:
        samples:   Full list of Sample objects.
        category:  The class to visualize (must be in BDD_DETECTION_CLASSES).
        n_samples: Number of patches to show.
        cols:      Number of columns in the grid.
        figsize:   Overall figure dimensions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure with the patch grid.
    """
    if category not in BDD_DETECTION_CLASSES:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Choose from {BDD_DETECTION_CLASSES}."
        )

    ann_sample_pairs = [
        (ann, sample)
        for sample in samples
        for ann in sample.annotations
        if ann.category == category
    ]
    random.shuffle(ann_sample_pairs)
    ann_sample_pairs = ann_sample_pairs[:n_samples]

    rows = (len(ann_sample_pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for idx, (ann, sample) in enumerate(ann_sample_pairs):
        image = sample.load_image()
        if image is None:
            axes[idx].axis("off")
            continue
        x1, y1 = int(ann.bbox.x1), int(ann.bbox.y1)
        x2, y2 = int(ann.bbox.x2), int(ann.bbox.y2)
        crop = image.crop((max(x1 - 10, 0), max(y1 - 10, 0), x2 + 10, y2 + 10))
        axes[idx].imshow(crop)
        axes[idx].set_title(
            f"{sample.weather or ''} | {sample.time_of_day or ''}",
            fontsize=8,
        )
        axes[idx].axis("off")

    for idx in range(len(ann_sample_pairs), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"Sample patches — '{category}' ({len(ann_sample_pairs)} shown)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def visualize_bbox_stats(
    bbox_df,
    category: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot bounding box area and aspect ratio distributions.

    Args:
        bbox_df:   DataFrame produced by compute_bbox_stats().
        category:  If provided, filter to this class only.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure with two subplots (area and aspect ratio).
    """
    import seaborn as sns

    df = bbox_df.copy()
    if category:
        df = df[df["class"] == category]
        title_suffix = f" — {category}"
    else:
        title_suffix = " — All Classes"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(data=df, x="area", hue="class", bins=60, ax=axes[0], legend=False)
    axes[0].set_title(f"BBox Area Distribution{title_suffix}")
    axes[0].set_xlabel("Area (px²)")
    axes[0].set_ylabel("Count")

    clipped = df[df["aspect_ratio"].between(0.01, 10)]
    sns.histplot(
        data=clipped, x="aspect_ratio", hue="class", bins=60, ax=axes[1], legend=False
    )
    axes[1].set_title(f"Aspect Ratio Distribution{title_suffix}")
    axes[1].set_xlabel("Aspect Ratio (w/h)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
