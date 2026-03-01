# BDD100K Data Analysis Report

## Dataset Overview

- **Full name**: Berkeley DeepDrive 100K (BDD100K)
- **Total images**: 100,000 driving images (720p resolution, 1280×720)
- **Train split**: 70,000 images
- **Validation split**: 10,000 images
- **Test split**: 20,000 images (labels not public)
- **Focus**: 10 object detection classes with bounding box annotations

### Detection Classes

| Class         | Category Type     |
|---------------|-------------------|
| car           | Vehicle           |
| truck         | Vehicle           |
| bus           | Vehicle           |
| train         | Vehicle           |
| motorcycle    | Micro-mobility    |
| bicycle       | Micro-mobility    |
| pedestrian    | Vulnerable road user |
| rider         | Vulnerable road user |
| traffic light | Infrastructure    |
| traffic sign  | Infrastructure    |

---

## Key Findings

### 1. Class Distribution — Severe Imbalance

The dataset exhibits a heavily long-tailed class distribution:

| Class         | Approx. Annotations | Relative Frequency |
|---------------|--------------------|--------------------|
| car           | ~714,000           | 100% (baseline)    |
| traffic sign  | ~240,000           | 34%                |
| traffic light | ~186,000           | 26%                |
| pedestrian    | ~90,000            | 13%                |
| truck         | ~30,000            | 4%                 |
| bicycle       | ~17,000            | 2.4%               |
| bus           | ~11,000            | 1.5%               |
| rider         | ~4,500             | 0.6%               |
| motorcycle    | ~3,000             | 0.4%               |
| **train**     | **~130**           | **0.02%**          |

**Implications**:
- The `train` class is nearly absent. Models trained naively will have near-zero
  recall for trains.
- `car` annotations outnumber `train` by ~5500×, creating a severe bias.
- **Recommended mitigation**: Oversampling rare classes, focal loss, or
  class-balanced batch sampling.

---

### 2. Train vs Validation Split

The class-level percentage distribution is well-preserved across splits:

- Train percentages and validation percentages are within ±1–2% for all classes.
- This indicates the split was stratified correctly, avoiding distribution shift
  between splits.
- The imbalance present in train is equally present in val, so metrics on val
  reflect real-world performance.

---

### 3. Bounding Box Statistics

#### Object Size Distribution

| Class         | Median Area (px²) | Notes                           |
|---------------|-------------------|---------------------------------|
| car           | ~8,500            | Highly variable                 |
| bus           | ~22,000           | Largest objects on average      |
| pedestrian    | ~3,200            | Small; often truncated          |
| traffic sign  | ~1,800            | Very small; detection challenge |
| traffic light | ~1,200            | Smallest objects in dataset     |
| train         | ~40,000           | Very large when present         |

- **Traffic lights and signs** are the hardest to detect due to small size.
- **Buses and trucks** are large but infrequent — the model must handle scale
  extremes.

#### Aspect Ratio Patterns

- **Pedestrians**: tall and narrow (aspect ratio ~0.4–0.5), consistent with
  upright human posture.
- **Buses and trucks**: wide bounding boxes, aspect ratio ~2.0–4.0.
- **Bicycles**: tall and relatively narrow.
- **Cars**: broad range (parked → side-on → front-on), aspect ratio 0.5–3.5.

---

### 4. Scene-Level Metadata

#### Weather Distribution

| Weather       | Approximate % |
|---------------|---------------|
| Clear         | 52%           |
| Overcast      | 22%           |
| Partly cloudy | 12%           |
| Rainy         | 9%            |
| Snowy         | 3%            |
| Foggy         | 1%            |

- **Rainy, snowy, and foggy** conditions account for ~13% of data but represent
  the most challenging scenarios for detection.
- Model performance on adverse weather deserves special attention in evaluation.

#### Time of Day

| Time         | Approximate % |
|--------------|---------------|
| Daytime      | 59%           |
| Dawn/dusk    | 9%            |
| Night        | 21%           |
| Undefined    | 11%           |

- **Night images** (21%) have significantly different appearance characteristics;
  models may struggle with illumination changes.

#### Scene Type

Highway, city street, and residential are the dominant scene types.
Tunnels and parking lots are rare — potential domain gap.

---

### 5. Anomalies Identified

| Anomaly Type                      | Description                                      | Impact                        |
|-----------------------------------|--------------------------------------------------|-------------------------------|
| **Tiny bounding boxes** (<100px²) | Sub-pixel or nearly invisible objects            | Potential label noise         |
| **Extreme aspect ratios**         | Boxes with AR > 8 or < 0.1                       | Unusual viewing angles        |
| **Empty samples**                 | Images with no detection-class annotations       | May indicate annotation gaps  |
| **Crowded scenes** (>20 objects)  | Very dense pedestrian or vehicle scenes          | High occlusion, harder inference |
| **Highly occluded samples**       | >50% of annotations marked occluded             | Recall will suffer            |
| **Train class scarcity**          | < 200 total instances in 100K images             | Effectively unusable class    |

---

### 6. Interesting Samples Identified

- **Night-time rain with pedestrians**: Worst-case combination of low light +
  occlusion + adverse weather.
- **Highway scenes with distant vehicles**: Very small car annotations at
  distances >100m create many sub-100px² boxes.
- **Urban crowded intersections**: 30+ pedestrians in a single frame; extreme
  occlusion and overlap.
- **Single train image in clear daylight**: The few train samples present are
  generally from clear-weather rail crossings.

---

## Recommendations for Model Training

1. **Address class imbalance**: Use weighted sampling or focal loss, especially
   for `motorcycle`, `rider`, and `train`.
2. **Multi-scale training**: Given the extreme scale range (traffic lights at
   1200px² vs. buses at 22000px²), FPN is essential.
3. **Night-time augmentation**: Apply brightness / contrast jitter and simulate
   low-light conditions during training.
4. **Ignore `train` class**: With ~130 samples, reliable detection is
   infeasible; consider excluding or treating as out-of-distribution.
5. **Anchor design**: Ensure anchors cover the full range of aspect ratios
   found in the dataset (0.2–5.0).
