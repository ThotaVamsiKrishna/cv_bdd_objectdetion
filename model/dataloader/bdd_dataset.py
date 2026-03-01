"""
PyTorch Dataset implementation for BDD100K object detection.

Supports Faster R-CNN / YOLOv8 style target formats and optional
Albumentations-based augmentations.
"""

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

BDD_CLASSES = [
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

# Index 0 is reserved for background (required by Faster R-CNN)
CLASS_TO_IDX: Dict[str, int] = {cls: i + 1 for i, cls in enumerate(BDD_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
IDX_TO_CLASS[0] = "background"


def collate_fn(batch: list) -> Tuple:
    """
    Custom collate function for variable-length target lists.

    Required by torch.utils.data.DataLoader when each item has a different
    number of bounding boxes.

    Args:
        batch: List of (image_tensor, target_dict) tuples.

    Returns:
        Tuple of (images_tuple, targets_tuple).
    """
    return tuple(zip(*batch))


class BDD100KDataset(Dataset):
    """
    BDD100K object detection dataset compatible with torchvision detection models.

    Each item returns:
        - image: FloatTensor of shape (3, H, W) in range [0, 1].
        - target: dict with keys:
            - 'boxes'  : FloatTensor (N, 4) in [x1, y1, x2, y2] format.
            - 'labels' : Int64Tensor (N,) with 1-based class indices.
            - 'image_id': Int64Tensor scalar.
            - 'area'   : FloatTensor (N,) bounding box areas.
            - 'iscrowd': Uint8Tensor (N,) all zeros (BDD has no crowd flags).

    Args:
        image_dir:    Directory of JPEG images.
        label_path:   Path to det_train.json or det_val.json.
        transforms:   Optional callable applied to the PIL image before
                      conversion to tensor. Use torchvision.transforms or
                      albumentations-compatible callables.
        max_samples:  If set, truncate the dataset to this many samples.
                      Useful for quick smoke tests.
    """

    def __init__(
        self,
        image_dir: str,
        label_path: str,
        transforms: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = image_dir
        self.transforms = transforms
        self._to_tensor = T.ToTensor()
        self.samples = self._load_labels(label_path, max_samples, image_dir)

    # ── Public interface ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        sample = self.samples[idx]
        image = self._load_image(sample["image_name"])

        boxes = torch.as_tensor(sample["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(sample["labels"], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros(len(labels), dtype=torch.uint8),
        }

        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = self._to_tensor(image)

        return image, target

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_labels(
        self, label_path: str, max_samples: Optional[int], image_dir: str
    ) -> List[Dict]:
        """
        Parse the BDD100K label JSON and build a list of sample dicts.

        Only entries with at least one valid detection annotation are included.

        Args:
            label_path:  Path to the JSON label file.
            max_samples: Truncate to this count after parsing.

        Returns:
            List of dicts with keys 'image_name', 'boxes', 'labels'.
        """
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for entry in data:
            boxes, labels = [], []
            for label in entry.get("labels", []):
                cat = label.get("category", "")
                if cat not in CLASS_TO_IDX:
                    continue
                box2d = label.get("box2d")
                if box2d is None:
                    continue
                x1, y1, x2, y2 = (
                    box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
                )
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_TO_IDX[cat])

            if boxes:
                samples.append(
                    {
                        "image_name": entry["name"],
                        "boxes": boxes,
                        "labels": labels,
                    }
                )

        # Keep only samples whose image files actually exist on disk
        samples = [s for s in samples if os.path.exists(os.path.join(image_dir, s["image_name"]))]
        print(f"  [DataLoader] {len(samples)} samples available on disk")

        if max_samples is not None:
            samples = samples[:max_samples]

        return samples

    def _load_image(self, image_name: str) -> Image.Image:
        """
        Load an image from disk as an RGB PIL Image.

        Args:
            image_name: Filename relative to self.image_dir.

        Returns:
            PIL Image in RGB mode.

        Raises:
            FileNotFoundError: If the image file cannot be located.
        """
        path = os.path.join(self.image_dir, image_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    def get_class_name(self, idx: int) -> str:
        """Return the class name for a given index (1-based)."""
        return IDX_TO_CLASS.get(idx, "unknown")
