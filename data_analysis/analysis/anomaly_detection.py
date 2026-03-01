"""
Anomaly and pattern detection in the BDD100K dataset.

Identifies unusual samples such as images with extreme bounding box sizes,
heavily occluded scenes, very crowded images, and class imbalance outliers.
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from parser.bdd_parser import Sample, BDD_DETECTION_CLASSES


def find_empty_samples(samples: List[Sample]) -> List[Sample]:
    """
    Return samples that have no valid detection annotations.

    These could be unannotated images or images with only non-detection classes
    such as lane markings or drivable areas.

    Args:
        samples: Full list of Sample objects.

    Returns:
        List of samples with zero annotations.
    """
    return [s for s in samples if s.is_empty()]


def find_extreme_bbox_samples(
    samples: List[Sample],
    area_threshold_low: float = 100.0,
    area_threshold_high: float = 150000.0,
) -> Dict[str, List[Tuple[Sample, float]]]:
    """
    Find samples containing very small or very large bounding boxes.

    Extremely small boxes (< area_threshold_low) may indicate annotation noise
    or very distant objects. Extremely large boxes may indicate unusual scenes.

    Args:
        samples:              Full list of Sample objects.
        area_threshold_low:   Minimum area in px² to be considered tiny.
        area_threshold_high:  Maximum area in px² before considered oversized.

    Returns:
        Dict with keys 'tiny' and 'huge', each a list of (Sample, area) tuples.
    """
    tiny, huge = [], []
    for sample in samples:
        for ann in sample.annotations:
            area = ann.bbox.area
            if area < area_threshold_low:
                tiny.append((sample, area))
            elif area > area_threshold_high:
                huge.append((sample, area))
    tiny.sort(key=lambda x: x[1])
    huge.sort(key=lambda x: x[1], reverse=True)
    return {"tiny": tiny, "huge": huge}


def find_heavily_occluded_samples(
    samples: List[Sample], occluded_ratio_threshold: float = 0.5
) -> List[Tuple[Sample, float]]:
    """
    Return samples where more than occluded_ratio_threshold of annotations
    are marked as occluded.

    Args:
        samples:                  Full list of Sample objects.
        occluded_ratio_threshold: Fraction of occluded boxes to trigger selection.

    Returns:
        List of (Sample, ratio) tuples sorted by descending ratio.
    """
    results = []
    for sample in samples:
        if not sample.annotations:
            continue
        occ_count = sum(1 for ann in sample.annotations if ann.occluded)
        ratio = occ_count / len(sample.annotations)
        if ratio >= occluded_ratio_threshold:
            results.append((sample, ratio))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def find_crowded_samples(
    samples: List[Sample], min_annotations: int = 20
) -> List[Tuple[Sample, int]]:
    """
    Find images with unusually high numbers of annotated objects.

    Crowded samples can reveal model weaknesses in dense scenes.

    Args:
        samples:         Full list of Sample objects.
        min_annotations: Minimum annotation count to classify as crowded.

    Returns:
        List of (Sample, count) tuples sorted by descending annotation count.
    """
    results = [
        (s, s.num_annotations)
        for s in samples
        if s.num_annotations >= min_annotations
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def find_class_imbalance_ratio(samples: List[Sample]) -> pd.DataFrame:
    """
    Compute the imbalance ratio between the most and least frequent classes.

    Args:
        samples: Full list of Sample objects.

    Returns:
        DataFrame with columns ['class', 'count', 'ratio_to_max',
        'ratio_to_min'] sorted descending by count.
    """
    counts = {cls: 0 for cls in BDD_DETECTION_CLASSES}
    for sample in samples:
        for ann in sample.annotations:
            counts[ann.category] += 1

    df = pd.DataFrame(list(counts.items()), columns=["class", "count"])
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    max_count = df["count"].max()
    min_count = df["count"].replace(0, np.nan).min()
    df["ratio_to_max"] = df["count"] / max_count
    df["ratio_to_min"] = df["count"] / min_count
    return df


def find_unusual_aspect_ratios(
    samples: List[Sample],
    ar_low: float = 0.2,
    ar_high: float = 5.0,
) -> List[Tuple[Sample, str, float]]:
    """
    Find annotations with unusual aspect ratios (very wide or very tall boxes).

    Args:
        samples: Full list of Sample objects.
        ar_low:  Lower bound; boxes below this are flagged as too tall.
        ar_high: Upper bound; boxes above this are flagged as too wide.

    Returns:
        List of (Sample, category, aspect_ratio) tuples.
    """
    results = []
    for sample in samples:
        for ann in sample.annotations:
            ar = ann.bbox.aspect_ratio
            if ar < ar_low or ar > ar_high:
                results.append((sample, ann.category, ar))
    results.sort(key=lambda x: abs(x[2] - 1.0), reverse=True)
    return results


def summarize_anomalies(samples: List[Sample]) -> Dict[str, int]:
    """
    Produce a high-level summary count of all detected anomaly types.

    Args:
        samples: Full list of Sample objects.

    Returns:
        Dict with anomaly type as key and count as value.
    """
    empty = find_empty_samples(samples)
    extreme = find_extreme_bbox_samples(samples)
    occluded = find_heavily_occluded_samples(samples)
    crowded = find_crowded_samples(samples)
    unusual_ar = find_unusual_aspect_ratios(samples)

    return {
        "empty_samples": len(empty),
        "tiny_bbox_annotations": len(extreme["tiny"]),
        "huge_bbox_annotations": len(extreme["huge"]),
        "heavily_occluded_samples": len(occluded),
        "crowded_samples": len(crowded),
        "unusual_aspect_ratio_annotations": len(unusual_ar),
    }
