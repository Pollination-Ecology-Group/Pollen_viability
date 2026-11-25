# Detection Quality Control & Ground-Truth Labeling

Checking all ~265,000 tiles by hand is impossible. Instead, our verification and active learning strategy is divided into two distinct, specialized workflows:
1. **Blinded Quality Control Review (`review_app`)**: A small, stratified, blinded sample is checked by two independent reviewers to quantify error rates and confirm that **detection bias is negligible across ploidies**.
2. **Ground-Truth Labeling & SAM Segmentation (`label_app`)**: Pixel-accurate polygon annotation powered by the Segment Anything Model (SAM), contour vertex editing, and active learning queues to **retrain and improve the YOLO instance segmentation model**.

---

## Web Applications on CESNET Kubernetes

All review and labeling portals are hosted live on the university CERIT-SC Kubernetes cluster (`stenc-ns`), equipped with HTTPS certificates and cross-portal header links:

| Portal | Live Cluster URL | Primary Objective & Phenotype | Interaction Style |
|:---|:---|:---|:---:|
| 🌸 **Pollen Viability QC** | <https://viability-review-stenc.dyn.cloud.e-infra.cz> | **Statistical Validation:** Blinded tile validation across cytotypes. Class correction (<kbd>V</kbd>/<kbd>N</kbd>/<kbd>I</kbd>), false positive filtering (<kbd>X</kbd>), missed grain recovery, relabel flag (<kbd>R</kbd>), and viability bias quantification. | Rapid click-to-correct (no polygon drag) |
| 🏷️ **Pollen Viability Labeler** | <https://viability-label-stenc.dyn.cloud.e-infra.cz> | **Ground-Truth Retraining:** SAM magic wand (positive/negative clicks), full polygon contour vertex editing, active learning queue (QC-flagged tiles prioritized), pre-loaded model predictions for quick confirmation/editing, and one-click YOLO segmentation export. | Deep segmentation & contour editing |
| 🌿 **Stomata Precision Review** | <https://stomata-review-stenc.dyn.cloud.e-infra.cz> | **Vegetative Phenotyping:** Guard cell and pore caliper audits, companion stomata verification, and gigas epidermal scaling checks. | Caliper inspection & measurement |

> 💡 **Cross-Navigation:** All applications include persistent header buttons allowing one-click switching between QC Review, SAM Labeling, and Stomata Audit sessions.

---

## 1. Detection Quality Control Workflow (`review_app`)

### Purpose
To prove for publication that model detection errors are minimal, balanced across cytotypes, and do not introduce systematic viability bias between diploids (`nat_2x`, `ory_2x`) and tetraploids (`neo_4x`, `nat_4x`).

### How the Sample is Built — [`src/build_viability_review_set.py`](../src/build_viability_review_set.py)
1. **Stratified Random Slides**: 6 per cytotype (`nat_2x`, `ory_2x`, `neo_4x`, `nat_4x`). Picks rotate across population × cultivation run.
2. **Density Sampling**: From each slide: one dense tile (90th percentile), one medium (50th), and one sparse (15th).
3. **Flagged Outliers**: Up to 12 outlier slides (2 tiles each) exhibiting atypical viability, extreme staining, or high clump density.
4. **Blinding**: Tiles are randomly shuffled and presented as `Tile N · S1A2B3`. Cytotype, plant ID, and slide names are hidden to eliminate observer bias.
5. **Inter-Rater Overlap**: 30% of tiles (57 tiles) are reviewed in duplicate by both reviewers (`jakub` and `eliska`) to measure inter-rater agreement.

### Reviewer Actions in QC
- **Click an existing outline**: Correct its class (<kbd>V</kbd>: Viable, <kbd>N</kbd>: Non-viable, <kbd>I</kbd>: Intermediate) or mark as false positive (<kbd>X</kbd>: *Not a grain*).
- **Click empty space**: Add a missed pollen grain marker.
- **Flag for Relabeling**: Press <kbd>R</kbd> if the tile contains clumping, ambiguous staining, or novel debris that the model struggled with.
- **Verdict**: <kbd>A</kbd> = Good, <kbd>S</kbd> = Minor issues, <kbd>D</kbd> = Poor / relabel candidate. Press <kbd>Enter</kbd> to save.

---

## 2. Ground-Truth Polygon Labeling Workflow (`src/label_app`)

### Purpose
To create pixel-perfect ground-truth training masks for YOLOv8/YOLO11 instance segmentation retraining, with zero friction and maximum throughput.

### Interactive Features (Mirrored from Pollen Latitude)
- **SAM Magic Wand**: Left-click inside a pollen grain to segment instantly using the Segment Anything Model. Right-click outside to add negative prompt points to steer SAM away from debris or touching grains.
- **Polygon Vertex Editing**:
  - Click and drag any vertex to reshape the contour.
  - Hover over an edge to reveal the green `+` insertion handle and click to split the edge.
  - Press <kbd>Del</kbd> or <kbd>X</kbd> while hovering over a vertex to remove it.
- **Manual Polygon Drawing**: Press <kbd>P</kbd> to enter manual drawing mode, click successive points around the boundary, and press <kbd>Enter</kbd> to close the polygon.
- **Undo History**: Press <kbd>Z</kbd> or <kbd>Ctrl+Z</kbd> to revert any polygon modification or deletion.
- **Class Hotkeys**: <kbd>1</kbd> Viable (green), <kbd>2</kbd> Non-viable (red), <kbd>3</kbd> Intermediate (amber), <kbd>4</kbd> Debris (slate).
- **Active Learning Queue**:
  - Tiles flagged in the QC review (<kbd>R</kbd> or `poor` verdict) appear **first** in the queue with a pink `🏷️` badge.
  - Remaining tiles are prioritized by cytotype deficit balancing and slide diversity.
- **Slide Overview & Zoom**:
  - Mini slide map at the bottom left displays current tile location on the CZI slide.
  - Mouse wheel zooms at cursor point; <kbd>Space</kbd>+drag pans smoothly across 640×640 tiles.

---

## 3. End-to-End Retraining Cycle

When a batch of tiles has been flagged during QC or when new slide batches are introduced:

```bash
# 1. Pull QC-flagged tiles into the relabel queue
python src/prepare_relabel_tiles.py --max-tiles 100

# 2. Open the labeling app locally or on Kubernetes
bash start_label_app.sh
# -> Open http://127.0.0.1:8050 or https://viability-label-stenc.dyn.cloud.e-infra.cz

# 3. Label / refine polygons using SAM and save with Enter (⏎)

# 4. Export ground-truth polygon labels to YOLO segmentation format
python src/export_relabel_dataset.py --output-dir datasets/yolo_retrain --val-split 0.20

# 5. Launch model retraining on CESNET GPU cluster
bash deploy_training.sh
```

---

## Metrics & Statistical Acceptance Criteria (Results Tab)

| Metric | Definition | Scientific Threshold |
|:---|:---|:---:|
| **Precision** | $\frac{\text{detections} - \text{false positives}}{\text{detections}}$ | > 95% |
| **Recall** | $\frac{\text{true detections}}{\text{true detections} + \text{missed grains}}$ | > 92% |
| **Class Accuracy** | $\frac{\text{correctly classed}}{\text{true detections}}$ | > 95% |
| **Viability Bias** | $\text{model viability \%} - \text{human corrected viability \%}$ | **\|Bias\| < 2.5 pp across all cytotypes** |
| **Inter-Rater Agreement** | Mean $|\Delta\text{corrected viability}|$ on overlapping 57 tiles | Within random sampling variance (< 2.0 pp) |

If any cytotype exceeds the 2.5 percentage point bias threshold:
1. Filter QC review for that cytotype's poor tiles.
2. Flag them for relabeling (<kbd>R</kbd>).
3. Annotate using SAM in the Labeling App.
4. Retrain YOLO11m-seg and update `best.pt`.

