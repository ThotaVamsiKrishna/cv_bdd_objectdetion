"""
Parser and data structures for the BDD100K object detection dataset.

BDD100K contains 100,000 driving images annotated with 10 object detection
classes using bounding boxes, along with scene-level attributes.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from PIL import Image


BDD_DETECTION_CLASSES = [
    "car",
    "traffic sign",
    "traffic light",
    "person",
    "truck",
    "bus",
    "bike",
    "rider",
    "motor",
    "train",
]

# Human-readable display names mapped from raw label categories
DISPLAY_NAMES = {
    "person": "pedestrian",
    "bike": "bicycle",
    "motor": "motorcycle",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "rider": "rider",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Width of the bounding box in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of the bounding box in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height ratio; returns 0 if height is zero."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def center(self):
        """(cx, cy) center coordinates of the box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def is_valid(self) -> bool:
        """Return True if the box has positive area."""
        return self.width > 0 and self.height > 0


@dataclass
class Annotation:
    """Single object annotation with class label and bounding box."""

    category: str
    bbox: BoundingBox
    occluded: bool = False
    truncated: bool = False
    attributes: Dict = field(default_factory=dict)


@dataclass
class Sample:
    """One image sample with all its annotations and scene-level metadata."""

    image_name: str
    annotations: List[Annotation]
    image_path: Optional[str] = None
    weather: Optional[str] = None
    scene: Optional[str] = None
    time_of_day: Optional[str] = None

    def load_image(self) -> Optional[Image.Image]:
        """Load and return the PIL image if the path exists."""
        if self.image_path and os.path.exists(self.image_path):
            return Image.open(self.image_path).convert("RGB")
        return None

    @property
    def num_annotations(self) -> int:
        """Total number of object annotations in this sample."""
        return len(self.annotations)

    @property
    def classes_present(self) -> List[str]:
        """Unique object classes present in this sample."""
        return list({ann.category for ann in self.annotations})

    def is_empty(self) -> bool:
        """Return True if the sample has no annotations."""
        return len(self.annotations) == 0


class BDDParser:
    """
    Parser for BDD100K JSON label files.

    Reads the official BDD100K label JSON and constructs a list of Sample
    objects containing only the 10 object-detection classes with valid box2d
    annotations.

    Args:
        label_path: Path to the BDD100K label JSON file
                    (e.g. det_train.json or det_val.json).
        image_dir:  Directory containing the corresponding images.
    """

    def __init__(self, label_path: str, image_dir: str):
        self.label_path = label_path
        self.image_dir = image_dir
        self.samples: List[Sample] = []

    def parse(self) -> List[Sample]:
        """
        Parse the label JSON and return a list of Sample objects.

        Only samples that contain at least one valid detection-class annotation
        are included.

        Returns:
            List of Sample objects with valid annotations.
        """
        with open(self.label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = []
        for entry in data:
            annotations = self._parse_labels(entry.get("labels", []))
            attrs = entry.get("attributes", {})
            sample = Sample(
                image_name=entry["name"],
                annotations=annotations,
                image_path=os.path.join(self.image_dir, entry["name"]),
                weather=attrs.get("weather"),
                scene=attrs.get("scene"),
                time_of_day=attrs.get("timeofday"),
            )
            self.samples.append(sample)

        return self.samples

    def _parse_labels(self, labels: list) -> List[Annotation]:
        """
        Parse raw label entries into Annotation objects.

        Skips labels that are not in BDD_DETECTION_CLASSES or lack a box2d
        field, and discards boxes with invalid (zero or negative) dimensions.

        Args:
            labels: Raw list of label dicts from the JSON entry.

        Returns:
            List of valid Annotation objects.
        """
        annotations = []
        for label in labels:
            category = label.get("category", "")
            if category not in BDD_DETECTION_CLASSES:
                continue
            box2d = label.get("box2d")
            if box2d is None:
                continue
            bbox = BoundingBox(
                x1=box2d["x1"],
                y1=box2d["y1"],
                x2=box2d["x2"],
                y2=box2d["y2"],
            )
            if not bbox.is_valid():
                continue
            label_attrs = label.get("attributes", {})
            annotation = Annotation(
                category=category,
                bbox=bbox,
                occluded=bool(label_attrs.get("occluded", False)),
                truncated=bool(label_attrs.get("truncated", False)),
                attributes=label_attrs,
            )
            annotations.append(annotation)
        return annotations

    def get_samples_by_class(self, category: str) -> List[Sample]:
        """
        Return all samples that contain at least one instance of the given class.

        Args:
            category: One of BDD_DETECTION_CLASSES.

        Returns:
            Filtered list of Sample objects.
        """
        return [s for s in self.samples if category in s.classes_present]

    def get_annotation_counts(self) -> Dict[str, int]:
        """
        Return total annotation count per class across all samples.

        Returns:
            Dict mapping class name to annotation count.
        """
        counts: Dict[str, int] = {cls: 0 for cls in BDD_DETECTION_CLASSES}
        for sample in self.samples:
            for ann in sample.annotations:
                counts[ann.category] += 1
        return counts
