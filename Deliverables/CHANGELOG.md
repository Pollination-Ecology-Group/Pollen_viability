# Pollen Viability & 3D Morphometry Dataset — Archived Versions

All versioned release bundles and compressed single-grain tables are archived both locally under `Deliverables/` and in CESNET S3 object storage under `s3://peg-bucket/Ostatni/Pollen_viability/Deliverables/`.

## v2.1 — 2026-11-28 (`current_production`)

Focus-filtered full-experiment production release adding per-grain contrast-normalized Laplacian sharpness screening (`norm_sharpness = 100 * SD(|Laplacian|) / max(18, P95 - P05) >= 3.50`) across all 1,008 whole-slide scans to eliminate optical Z-defocus blur halos (`22–28 µm` out-of-focus viable grains whose washed-out brownish blur halos mimicked non-viable staining and stretched non-viable diameter violins).

- **Folder / Bundle:** `Deliverables/pollen_viability_v2.1_2026-11-28/`, `pollen_viability_v2.1_2026-11-28.zip`, `pollen_viability_v2.1_2026-11-28_grains_calibrated.csv.gz`
- **Coverage:** **1,008 whole-slide scans** across **504 flowering plants** out of 540 cultivated (`36` non-flowering); **977 sharp QC-passing slides** (`>= 100` calibrated grains AND `>= 100` sharp grains) across **500 plants** (`22` sparse `< 100`-grain slides and `9` wholly Z-defocused `< 100`-sharp-grain slides excluded).
- **Grain totals:** **2,186,245 calibrated pollen grains** (`2,179,117` on the 977 retained slides); **1,925,516 in-focus calibrated grains** (`1,924,830` on retained slides); **1,679,037 in-focus isolated single `viable`/`non_viable` grains** for morphometry; **1,571,294 in-focus hydrated viable grains** in Stage-1/2 3D stereological & closed-mesh unfolding.
- **Improvements vs. v2.0:** Halves sister-slide viability discrepancy (`6.24 pp -> 3.20 pp` mean, `1.48 pp -> 1.01 pp` median; `> 20 pp` discordant pairs cut from `42 -> 13`), restores clean shrunken non-viable microspore medians (`13.14 µm` `ory_2x`, `13.11 µm` `nat_2x`, `15.73 µm` `nat_4x`, `16.52 µm` `neo_4x`), and converges all four cytotypes to homologous `prolate-spheroidal` 3D shapes (`neo_4x` canonical `7,256 µm³` vs. `nat_2x` `3,674 µm³` = `+97.5%` / `1.98x` exact 1:1 karyoplasmic doubling; `nat_4x` `5,203 µm³` = `+41.6%` downsizing).

## v2.0 — 2026-11-28 (`archived_superseded_by_v2.1`)

Complete 1,008-slide unfiltered release after fixing unpaginated `s3.list_objects_v2()` folder discovery in `src/pollen_size_analyzer.py` and filtering 109 legacy Oct-21 orphan `_det.json` files across 6 slides (prior to per-grain optical focus screening).

- **Folder / Bundle:** `Deliverables/pollen_viability_v2.0_2026-11-28/`, `pollen_viability_v2.0_2026-11-28.zip`, `pollen_viability_v2.0_2026-11-28_grains_calibrated.csv.gz`
- **Coverage:** 1,008 whole-slide scans across 504 flowering plants; 986 slides (`>= 100` grains) across 501 plants; 2,186,245 calibrated pollen grains (`1,885,728` isolated single grains).

## v1.0 — 2026-11-25 (`archived_superseded_by_v2.0`)

First two-reviewer validated + 3D-corrected release (commit `885344f0`), covering the first alphabetical S3 page of 502 whole-slide scans before Eliška flagged the missing second page of slides.

- **Folder / Bundle:** `Deliverables/pollen_viability_v1.0_2026-11-25/`, `pollen_viability_v1.0_2026-11-25.zip`
- **Coverage:** 502 scanned slides, 491 QC-passing slides (`>= 100` grains) across 251 plants; 968,195 calibrated grains (`825,310` isolated single grains).
- **Model & QC:** Final post-mosaic `YOLOv11m-seg` checkpoint `last_20261122_r3.pt` (`MD5: 1ea93497105eee085848310df127e42f`), two-reviewer blind audit (`jakub` 100 + `eliska` 100 = 200 reviews across 154 unseen tiles; 98.0% precision, 99.0% recall, 99.3% 3-class accuracy, 97.5% polygon accuracy), plus 96-grain size-extremes audit.
