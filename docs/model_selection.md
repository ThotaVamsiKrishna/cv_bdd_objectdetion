# Model Selection — BDD100K Object Detection

## Chosen Model: Faster R-CNN with ResNet-50 + FPN

### Architecture Overview

Faster R-CNN is a two-stage object detector introduced by Ren et al. (2015).
The pipeline consists of three components:

```
Input Image
     │
     ▼
┌──────────────────────────────────┐
│  Backbone: ResNet-50 + FPN       │  Feature extraction at multiple scales
│  (C2, C3, C4, C5 → P2–P5)       │
└─────────────────┬────────────────┘
                  │  Feature maps
                  ▼
┌──────────────────────────────────┐
│  Region Proposal Network (RPN)   │  Proposes candidate bounding boxes
│  Objectness score + box delta    │  ~2000 proposals per image
└─────────────────┬────────────────┘
                  │  Region proposals (RoIs)
                  ▼
┌──────────────────────────────────┐
│  RoI Pooling / Align             │  Fixed-size feature for each proposal
└─────────────────┬────────────────┘
                  │
                  ▼
┌──────────────────────────────────┐
│  Classification Head             │  Class label (11 classes incl. bg)
│  Box Regression Head             │  Refined bounding box coordinates
└──────────────────────────────────┘
```

#### Feature Pyramid Network (FPN)

FPN adds a top-down pathway and lateral connections to ResNet to produce
multi-scale feature maps (P2–P5). This is critical for BDD100K because:

- **Small objects** (distant cyclists, pedestrians): detected at high-res P2/P3
- **Large objects** (buses, trucks): detected at low-res P4/P5

#### ResNet-50 Backbone

- 50-layer residual network with batch normalisation
- Residual connections prevent vanishing gradients in deep networks
- Pre-trained on ImageNet → rich low-level and mid-level features transfer well
- Fine-tuned on COCO, then further fine-tuned on BDD100K

---

## Why Faster R-CNN over Other Options?

| Model         | mAP (BDD100K) | Speed (FPS) | Complexity | Explainability |
|---------------|--------------|-------------|------------|----------------|
| Faster R-CNN  | ~32 mAP      | ~10 fps     | Medium     | ✅ High         |
| YOLOv8n       | ~28 mAP      | ~100 fps    | Low        | Medium          |
| DETR          | ~34 mAP      | ~8 fps      | High       | Medium          |
| YOLOv8x       | ~35 mAP      | ~25 fps     | High       | Medium          |

**Reasoning for Faster R-CNN:**

1. **Explainability**: The two-stage approach with clear RPN + classification
   heads makes it straightforward to explain each component in detail.
2. **FPN multi-scale detection**: Essential for BDD100K's wide object scale
   range (tiny traffic signs vs. full-frame buses).
3. **Strong pre-trained weights**: Widely available COCO pre-trained weights
   provide a strong starting point, minimising training time.
4. **Well-documented**: Extensive literature allows confident explanation of
   architectural choices.
5. **Trade-off**: While not the fastest, the speed is acceptable for
   offline evaluation and the mAP is competitive.

---

## Training Strategy

- **Pre-training**: COCO weights (80-class detection) → backbone and FPN
- **Fine-tuning**: Replace classification head for 10 BDD100K classes + background
- **Frozen layers**: First 2 ResNet stages frozen for the first epoch to
  preserve low-level features
- **Optimiser**: SGD (lr=0.005, momentum=0.9, weight_decay=5e-4)
- **LR schedule**: StepLR — reduce by ×0.1 every 3 epochs
- **Augmentation**: Random horizontal flip, colour jitter, scale jitter

---

## Limitations and Suggested Improvements

1. **Speed**: Faster R-CNN is ~10 fps, insufficient for real-time deployment.
   Consider YOLOv8 for production.
2. **Small object performance**: mAP_small is lower due to coarse anchor
   sampling. Improvement: use higher-resolution input (1280×720 instead of
   640×384).
3. **Class imbalance**: Cars dominate the dataset (see data analysis report).
   Improvement: focal loss or class-balanced sampling.
4. **Night-time performance**: Night images (20% of dataset) degrade detection.
   Improvement: domain adaptation or night-specific augmentation.
5. **Occluded pedestrians**: High occlusion rate in pedestrian class leads to
   missed detections. Improvement: part-based detection models.
