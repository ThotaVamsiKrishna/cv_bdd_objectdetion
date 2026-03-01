"""
Class distribution analysis for BDD100K object detection.

Provides functions to compute annotation counts, per-image class coverage,
bounding box statistics, and scene-level breakdowns.
"""

from collections import Counter
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from parser.bdd_parser import Sample, BDD_DETECTION_CLASSES


def compute_class_distribution(samples: List[Sample]) -> pd.DataFrame:
    """
    Count total annotations per class across all samples.

    Args:
        samples: List of Sample objects from the BDDParser.

    Returns:
        DataFrame with columns ['class', 'count'], sorted descending by count.
    """
    counter: Counter = Counter()
    for sample in samples:
        for ann in sample.annotations:
            counter[ann.category] += 1

    df = pd.DataFrame(
        [(cls, counter.get(cls, 0)) for cls in BDD_DETECTION_CLASSES],
        columns=["class", "count"],
    )
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def compute_images_per_class(samples: List[Sample]) -> pd.DataFrame:
    """
    Count images containing at least one instance of each class.

    Args:
        samples: List of Sample objects from the BDDParser.

    Returns:
        DataFrame with columns ['class', 'image_count'], sorted descending.
    """
    image_count: Counter = Counter()
    for sample in samples:
        seen = {ann.category for ann in sample.annotations}
        for cls in seen:
            image_count[cls] += 1

    df = pd.DataFrame(
        [(cls, image_count.get(cls, 0)) for cls in BDD_DETECTION_CLASSES],
        columns=["class", "image_count"],
    )
    return df.sort_values("image_count", ascending=False).reset_index(drop=True)


def compute_bbox_stats(samples: List[Sample]) -> pd.DataFrame:
    """
    Collect per-annotation bounding box statistics for all classes.

    Args:
        samples: List of Sample objects from the BDDParser.

    Returns:
        DataFrame with columns:
        ['class', 'width', 'height', 'area', 'aspect_ratio',
         'occluded', 'truncated', 'weather', 'time_of_day', 'scene'].
    """
    records = []
    for sample in samples:
        for ann in sample.annotations:
            records.append(
                {
                    "class": ann.category,
                    "width": ann.bbox.width,
                    "height": ann.bbox.height,
                    "area": ann.bbox.area,
                    "aspect_ratio": ann.bbox.aspect_ratio,
                    "occluded": ann.occluded,
                    "truncated": ann.truncated,
                    "weather": sample.weather,
                    "time_of_day": sample.time_of_day,
                    "scene": sample.scene,
                }
            )
    return pd.DataFrame(records)


def compute_scene_distribution(samples: List[Sample]) -> pd.DataFrame:
    """
    Compute distribution of weather, scene type, and time of day.

    Args:
        samples: List of Sample objects from the BDDParser.

    Returns:
        DataFrame with columns ['attribute', 'value', 'count'].
    """
    records = []
    for attr in ["weather", "scene", "time_of_day"]:
        counter: Counter = Counter()
        for s in samples:
            val = getattr(s, attr, None)
            if val:
                counter[val] += 1
        for val, cnt in counter.items():
            records.append({"attribute": attr, "value": val, "count": cnt})
    return pd.DataFrame(records)


def compute_cooccurrence_matrix(samples: List[Sample]) -> pd.DataFrame:
    """
    Build a class co-occurrence matrix (how often two classes share an image).

    Args:
        samples: List of Sample objects from the BDDParser.

    Returns:
        Square DataFrame indexed and columned by class name.
    """
    matrix = pd.DataFrame(
        0, index=BDD_DETECTION_CLASSES, columns=BDD_DETECTION_CLASSES
    )
    for sample in samples:
        classes = list({ann.category for ann in sample.annotations})
        for i, c1 in enumerate(classes):
            for c2 in classes[i:]:
                matrix.loc[c1, c2] += 1
                if c1 != c2:
                    matrix.loc[c2, c1] += 1
    return matrix


def plot_class_distribution(
    df: pd.DataFrame,
    title: str = "Annotation Count per Class",
    save_path: str = None,
) -> plt.Figure:
    """
    Bar chart of annotation counts per class.

    Args:
        df:        DataFrame with columns ['class', 'count'].
        title:     Plot title string.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=df, x="class", y="count", ax=ax, palette="viridis")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    for bar in ax.patches:
        ax.annotate(
            f"{int(bar.get_height()):,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_cooccurrence_heatmap(
    matrix: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """
    Heatmap of class co-occurrence counts.

    Args:
        matrix:    Square co-occurrence DataFrame.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title("Class Co-occurrence Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def compute_train_val_comparison(
    train_samples: List[Sample], val_samples: List[Sample]
) -> pd.DataFrame:
    """
    Compare class distributions between train and validation splits.

    Args:
        train_samples: Samples from the training split.
        val_samples:   Samples from the validation split.

    Returns:
        DataFrame with columns ['class', 'train_count', 'val_count',
        'train_pct', 'val_pct'].
    """
    train_df = compute_class_distribution(train_samples).set_index("class")
    val_df = compute_class_distribution(val_samples).set_index("class")

    df = train_df.join(val_df, lsuffix="_train", rsuffix="_val")
    df.columns = ["train_count", "val_count"]
    df["train_pct"] = df["train_count"] / df["train_count"].sum() * 100
    df["val_pct"] = df["val_count"] / df["val_count"].sum() * 100
    return df.reset_index()
