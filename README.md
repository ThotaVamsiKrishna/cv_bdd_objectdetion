# BDD100K Object Detection — Full Pipeline

End-to-end object detection on the BDD100K autonomous driving dataset
using Faster R-CNN + ResNet-50 + FPN.

---

## Environment

| Item        | Details                           |
|-------------|-----------------------------------|
| Platform    | Google Colab (free tier)          |
| GPU         | Tesla T4 — 15.6 GB VRAM           |
| PyTorch     | 2.10.0 + CUDA 12.8                |
| Dataset     | BDD100K mini (30 MB / 501 images) |

---

## Resource Constraints & Design Decisions

The full BDD100K dataset is 7.7 GB. To work within free-tier resource limits,
two practical engineering decisions were made:

**1. Google Colab (free T4 GPU)** — no local hardware required.
Anyone with a Google account can reproduce this fully in ~5 minutes.

**2. Curated mini-dataset (30 MB)** — `create_mini_dataset.py` reads directly
from the original `archive.zip` and extracts 100 train + 20 val images per
class without a full extraction. The folder structure is identical to the
full dataset, so the pipeline is architecturally the same as running on
all 70K images.

---

## Pipeline Steps

| # | Step | Output |
|---|------|--------|
| 1 | GPU check | T4, 15.6 GB confirmed |
| 2 | Install packages | torchmetrics, seaborn, scikit-learn |
| 3 | Mount Google Drive → extract mini_bdd100k.zip | 30 MB, ~5 seconds |
| 4 | Auto-detect dataset paths | train 397 imgs / val 104 imgs |
| 5 | Write BDDParser + BDD100KDataset modules | parser/, model/ |
| 6 | Data Analysis (EDA) | class dist, bbox stats, anomalies, scene metadata |
| 7 | Model Training (Faster R-CNN, 3 epochs) | model_final.pth |
| 8 | Inference on validation images | inference_samples.png |
| 9 | Evaluation — mAP | per_class_ap.png |
| 10 | Failure Clustering (KMeans on ResNet-50 embeddings) | failure_clusters.png |
| 11 | Save outputs to Google Drive | bdd100k_cv_project/ |

---

## How to Run

1. Run `create_mini_dataset.py` locally to generate `mini_bdd100k.zip` (30 MB)
2. Upload `mini_bdd100k.zip` to [drive.google.com](https://drive.google.com)
3. Open `bdd_object_detection.ipynb` in Google Colab
4. `Runtime → Change runtime type → T4 GPU`
5. Run all cells top to bottom (~5 minutes total)

---

## Key Results

### Training Loss (3 Epochs)

| Epoch | Mean Loss | Time  |
|-------|-----------|-------|
| 1     | 2.1537    | 17.4s |
| 2     | 1.4763    | 15.1s |
| 3     | 1.2521    | 15.4s |

Loss breakdown (Epoch 1, Batch 1):
`classifier: 2.490 | box_reg: 0.878 | objectness: 0.455 | rpn_box_reg: 0.218`

### Evaluation (mAP)

| Metric                | Value  |
|-----------------------|--------|
| mAP @ IoU=0.50        | 0.1937 |
| mAP @ IoU=0.50:0.95   | 0.0831 |
| mAP small objects     | 0.0308 |
| mAP large objects     | 0.2168 |

### Per-Class AP @ IoU=0.50

| Class         | AP    | Notes                          |
|---------------|-------|--------------------------------|
| car           | 0.333 | Most training data             |
| person        | 0.191 | Distinctive shape              |
| traffic sign  | 0.115 |                                |
| traffic light | 0.058 |                                |
| bike          | 0.040 |                                |
| truck         | 0.030 | Visually similar to car        |
| rider         | 0.030 |                                |
| motor         | 0.024 |                                |
| bus           | 0.011 |                                |
| train         | 0.000 | Only 2 training samples        |

### Key Finding — Class Imbalance

`car` is **273× more common** than `train` (7,441 vs 27 annotations).
This directly caused `train` AP = 0.000 — a result predicted by the EDA
before any training was done.

### Scene Metadata

```
Weather : clear=282  rainy=54  snowy=38  overcast=45  partly cloudy=20  foggy=2
Time    : night=240  daytime=168  dawn/dusk=91
```

Night images dominate (48%) — the hardest condition for detection.

### Anomaly Detection

| Anomaly                          | Count |
|----------------------------------|-------|
| Empty samples (no annotations)   | 0     |
| Crowded samples (≥30 objects)    | 74    |
| Tiny boxes (area < 50 px²)       | 24    |
| Heavily occluded (≥80% occluded) | 28    |

### Failure Clustering

- 39 / 104 val images failed (37.5% failure rate)
- ResNet-50 embeddings (2048-dim) → KMeans (k=4)

| Cluster | Images | Mean IoU |
|---------|--------|----------|
| 0       | 8      | 0.011    |
| 1       | 14     | 0.017    |
| 2       | 14     | 0.002    |
| 3       | 3      | 0.000    |

---

## Architecture

```
Image → ResNet-50 backbone → FPN (multi-scale features)
                                    ↓
                         RPN: objectness + rough box
                                    ↓
              RoI Align → Classification + Box Regression
```

- **Model:** Faster R-CNN + ResNet-50 + FPN
- **Parameters:** 41.3M total | 41.1M trainable
- **Transfer learning:** COCO pretrained weights
- **Optimizer:** SGD lr=0.005, momentum=0.9, weight_decay=5e-4
- **Batch size:** 4 | **Epochs:** 3

---

## Files

| File | Description |
|------|-------------|
| `bdd_object_detection.ipynb` | Fully executed notebook with all outputs |
| `create_mini_dataset.py` | Builds `mini_bdd100k.zip` from `archive.zip` |
| `bdd100k_cv_project-*.zip` | All output figures, model checkpoint, eval results |

---

## Classes

`car` · `traffic sign` · `traffic light` · `person` · `truck`
`bus` · `bike` · `rider` · `motor` · `train`
