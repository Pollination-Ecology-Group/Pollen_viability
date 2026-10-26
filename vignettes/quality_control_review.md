# Detection Quality Control (blinded human review)

Checking all ~265,000 tiles by hand is impossible. Instead, a **small, stratified, blinded sample**
is checked by two people, and the corrections are turned into error rates per cytotype.
What we need to show is that **errors are small and similar across ploidies**. If they are,
detection errors cannot cause the ploidy differences we report.

## Web Apps on CESNET Kubernetes

Both review applications are hosted live on the university CERIT-SC Kubernetes cluster so that collaborators (Eliška and Jakub) can access and evaluate reproductive and vegetative traits side-by-side without installing any local dependencies:

| Portal | Live Cluster URL | Scope & Phenotype | Reviewers |
|:---|:---|:---|:---:|
| 🌸 **Pollen Viability QC** | <https://viability-review-stenc.dyn.cloud.e-infra.cz> | **Reproductive:** Blinded whole-slide tile validation across ploidy cytotypes. Human correction of viability %, false positive filtering, missed grain recovery, and systematic model bias quantification. | `jakub`, `eliska` |
| 🌿 **Stomata Precision Review** | <https://stomata-review-stenc.dyn.cloud.e-infra.cz> | **Vegetative:** Stomatal complex precision audits. Interactive calipers for guard cell length/width, pore dimensions, and verification of epidermal cell gigas scaling. | Open / Lab access |

> 💡 **Quick Navigation:** Both web applications feature cross-navigation badges in their headers, allowing seamless switching between Pollen and Stomata validation sessions in the browser.

### Pollen QC App Details
- **URL:** <https://viability-review-stenc.dyn.cloud.e-infra.cz>
- **Login:** Personal username + password (HTTP basic auth, assigned accounts for `jakub` and `eliska`).
- **Code:** [`review_app/app.py`](../review_app/app.py), [`review_app/index.html`](../review_app/index.html)
- **Deploy / Password updates:** `bash deploy_viability_review.sh` (`--reset-passwords jakub eliska` to regenerate credentials).
- **Cloud Storage:** `s3://peg-bucket/Ostatni/Pollen_viability/Review/` (manifest + one JSON decision record per reviewer and tile).

## How the sample is built — [`src/build_viability_review_set.py`](../src/build_viability_review_set.py)

1. **Stratified random slides**: 6 per cytotype (`nat_2x`, `ory_2x`, `neo_4x`, `nat_4x`). Picks
   rotate across population × cultivation run, and primary slides come before `rep` slides.
2. **Tiles chosen by density**: from each slide, one dense tile (90th percentile of grains per tile),
   one medium (50th) and one sparse but not empty (15th). Errors in crowded and in empty areas
   are different, so we sample both.
3. **Flagged outliers** (up to 12 slides, 2 tiles each), taken from `pollen_slide_summaries.csv`.
   A slide is flagged if:
   - its viability or viable-grain size is atypical *within its cytotype* (robust z > 3), or
   - viability is below 10 % or above 99 %, or
   - fewer than 50 % of grains can be measured (clumps or edges).
4. **Blinding**: the order is shuffled and reviewers see only `Tile N · S1A2B3`. Cytotype, plant
   and slide name are hidden, so what a reviewer expects to see cannot influence the check.
5. **Overlap**: 30 % of tiles go to both reviewers, so inter-rater agreement can be measured. The
   remaining tiles are split between them.

### Production Validation Set (`final_20261026_1614`)
* **Total Sample Size:** **191 tiles** sampled from **72 unique slides** spanning all four cytotypes (`ory_2x`: 52, `nat_2x`: 51, `nat_4x`: 44, `neo_4x`: 44).
* **Composition:** 144 stratified random tiles (unbiased density baseline) + 47 flagged/outlier tiles (stress-testing difficult boundaries and staining extremes).
* **Inter-Rater Reliability:** **57 overlapping tiles (30%)** are assigned blindly to both reviewers (`jakub` and `eliska`), yielding 124 assigned tiles per reviewer.
* **Automatic Label Deduplication:** Model predictions undergo class-agnostic Non-Maximum Suppression ($\text{IoU} = 0.40$) and containment filtering ($\text{IoS} > 0.75$) to automatically suppress duplicate anchor proposals on clumped grains.

## What reviewers do

- **Click a wrong outline** and pick the correct class (<kbd>V</kbd>: Viable, <kbd>N</kbd>: Non-viable, <kbd>I</kbd>: Intermediate), or mark as false positive (<kbd>X</kbd>: *Not a pollen grain*).
- **Click a missed grain** in empty space to add its marker and select its class.
- **Flag for Roboflow Relabeling:** Press <kbd>R</kbd> or toggle *„🏷️ Flag for Relabel“* if a tile exhibits novel debris, ambiguous staining, or dense clumps requiring retraining.
- **Give an overall verdict:** <kbd>A</kbd> = Good, <kbd>S</kbd> = Minor, <kbd>D</kbd> = Poor, optionally tick *image problem*, then save with <kbd>Enter</kbd>.

## Metrics (Results tab, `/api/export.csv`)

| Metric | Definition |
|---|---|
| Precision | (detections − false detections) / detections |
| Recall | true detections / (true detections + missed grains) |
| Class accuracy | correctly classed / true detections |
| **Viability bias** | model viability % − human-corrected viability % (percentage points) |
| Agreement | mean \|Δ corrected viability\| and relative Δ grain count on tiles reviewed by both |

**Acceptance rule (proposed):** |viability bias| < 2–3 pp in every cytotype. The bias must also
not differ systematically between cytotypes, i.e. it should be within the inter-rater difference.
If one cytotype fails this:
1. look at its flagged tiles;
2. add those tiles to training (active learning);
3. retrain and re-run inference.

## Workflow after inference finishes

```bash
python src/pollen_size_analyzer.py --all            # slide summaries (enables outlier flags)
python src/build_viability_review_set.py            # publishes the final set (new set_id)
# send Eliška the URL + her password; follow progress in the Results tab
```

A new `set_id` keeps decisions from earlier or preview sets separate.
