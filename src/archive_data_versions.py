#!/usr/bin/env python3
"""
archive_data_versions.py — Build versioned dataset archives in Deliverables/ and mirror to CESNET S3.

Follows the Pollen_latitude Deliverables/ convention:
  - Deliverables/CHANGELOG.md
  - Deliverables/versions.csv
  - Deliverables/pollen_viability_latest.txt
  - Deliverables/pollen_viability_v1.0_2026-11-25/  (+ .zip)  [502-slide v3.1 + 2-reviewer QC + 3D from git 885344f0]
  - Deliverables/pollen_viability_v2.0_2026-11-28/  (+ .zip + grains_calibrated.csv.gz) [complete 1,008-slide unfiltered release]
  - Deliverables/pollen_viability_v2.1_2026-11-28/  (+ .zip + grains_calibrated.csv.gz) [1,008-slide focus-filtered production release]
  - Syncs all archives & index files to s3://peg-bucket/Ostatni/Pollen_viability/Deliverables/
"""

import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pollen_size_analyzer import S3_BUCKET, make_s3

ROOT = Path(__file__).resolve().parent.parent
DELIV = ROOT / "Deliverables"
S3_DELIV_PFX = "Ostatni/Pollen_viability/Deliverables"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob(commit: str, rel_path: str):
    r = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{rel_path}"],
        capture_output=True,
    )
    if r.returncode == 0:
        return r.stdout
    return None


def write_manifest_and_zip(vdir: Path, meta: dict, exclude_from_zip=()):
    manifest_files = {}
    for p in sorted(vdir.rglob("*")):
        if p.is_file() and p.name != "VERSION.json":
            rel = p.relative_to(vdir).as_posix()
            manifest_files[rel] = {
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
    meta["files"] = manifest_files
    (vdir / "VERSION.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    zip_path = DELIV / f"{vdir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(vdir.rglob("*")):
            if p.is_file() and p.name not in exclude_from_zip:
                rel = p.relative_to(vdir).as_posix()
                zf.write(p, f"{vdir.name}/{rel}")
    return zip_path


def ensure_v1_0_from_git():
    """Extract or verify v1.0 (commit 885344f0, 2026-11-25) — first 502 slides."""
    commit = "885344f0"
    vname = "pollen_viability_v1.0_2026-11-25"
    vdir = DELIV / vname
    zip_path = DELIV / f"{vname}.zip"
    if vdir.exists() and (vdir / "VERSION.json").exists() and zip_path.exists():
        meta = json.loads((vdir / "VERSION.json").read_text(encoding="utf-8"))
        return meta, zip_path

    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True, exist_ok=True)

    mapping = [
        ("outcomes/pollen_viability_slides_final.csv", "slides.csv"),
        ("outcomes/pollen_viability_plants_final.csv", "plants.csv"),
        ("output/qc/final_viability.csv", "all_slides_qc.csv"),
        ("outcomes/pollen_3d_cytotype_models.csv", "cytotype_3d_models.csv"),
        ("outcomes/pollen_3d_meshes.zip", "pollen_3d_meshes.zip"),
        ("outcomes/pollen_viability_qc_validation_summary.csv", "qc_validation_summary.csv"),
        ("output/qc/viability_qc_r3_final.csv", "detection_checks_200reviews.csv"),
        ("output/qc/viability_qc_r3_summary.json", "detection_checks_summary.json"),
        ("output/qc/qc_summary_cytotype.csv", "qc_summary_cytotype.csv"),
        ("output/qc/class_min_diam.csv", "settings/class_min_diam.csv"),
        ("output/qc/class_conf_threshold.csv", "settings/class_conf_threshold.csv"),
        ("metadata/plant_metadata_540.csv", "settings/plant_metadata_540.csv"),
        ("outcomes/pollen_viability_report.pdf", "pollen_viability_report.pdf"),
        ("outcomes/pollen_viability_report.docx", "pollen_viability_report.docx"),
        ("outcomes/MANUSCRIPT_METHODS_POLLEN_VIABILITY_v3.1.md", "methods.md"),
        ("outcomes/MANUSCRIPT_METHODS_POLLEN_VIABILITY_v3.1.docx", "methods.docx"),
    ]
    for src_rel, dst_rel in mapping:
        b = git_blob(commit, src_rel)
        if b is not None:
            dst = vdir / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b)

    fv = pd.read_csv(vdir / "all_slides_qc.csv")
    sl = pd.read_csv(vdir / "slides.csv")
    meta = {
        "version": "1.0",
        "release_date": "2026-11-25",
        "folder": vname,
        "git_commit": commit,
        "status": "archived_superseded_by_v2.0",
        "model_checkpoint": "last_20261122_r3.pt (MD5: 1ea93497105eee085848310df127e42f)",
        "slides_scanned": int(len(fv)),
        "slides_qc_pass_ge100": int(len(sl)),
        "plants_represented": int(sl["plant_id"].nunique()),
        "calibrated_grains_total": int(fv["Count_detected"].sum()),
        "note": "First release: v3.1 post-mosaic model + two-reviewer blind QC (Jakub 100 + Eliska 100) + 3D stereological unfolding on first S3 page (502 of 1,008 slides, 251 plants; superseded by v2.0 after fixing S3 list_objects_v2 pagination).",
    }
    zp = write_manifest_and_zip(vdir, meta)
    return meta, zp


def preserve_existing_v2_0():
    """Preserve existing unfiltered 1,008-slide v2.0 archive without overwriting its files; mark superseded by v2.1."""
    vname = "pollen_viability_v2.0_2026-11-28"
    vdir = DELIV / vname
    zip_path = DELIV / f"{vname}.zip"
    gz_copy = DELIV / f"{vname}_grains_calibrated.csv.gz"
    if not (vdir.exists() and (vdir / "VERSION.json").exists()):
        raise FileNotFoundError(f"Expected existing {vdir}/VERSION.json to preserve v2.0!")

    meta = json.loads((vdir / "VERSION.json").read_text(encoding="utf-8"))
    if meta.get("status") != "archived_superseded_by_v2.1":
        meta["status"] = "archived_superseded_by_v2.1"
        (vdir / "VERSION.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(vdir.rglob("*")):
                if p.is_file() and p.name != "grains_calibrated.csv.gz":
                    rel = p.relative_to(vdir).as_posix()
                    zf.write(p, f"{vdir.name}/{rel}")
    return meta, zip_path, gz_copy


def build_v2_1_current():
    """Build v2.1 (2026-11-28) — complete 1,008-slide dataset + contrast-normalized focus gate (norm_sharpness >= 3.50)."""
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    vname = "pollen_viability_v2.1_2026-11-28"
    vdir = DELIV / vname
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True, exist_ok=True)

    mapping = [
        ("outcomes/pollen_viability_slides_final.csv", "slides.csv"),
        ("outcomes/pollen_viability_plants_final.csv", "plants.csv"),
        ("output/qc/final_viability.csv", "all_slides_qc.csv"),
        ("output/qc/slide_focus_summary_1008.csv", "slide_focus_summary_1008.csv"),
        ("outcomes/pollen_3d_cytotype_models.csv", "cytotype_3d_models.csv"),
        ("outcomes/pollen_3d_meshes.zip", "pollen_3d_meshes.zip"),
        ("outcomes/pollen_viability_qc_validation_summary.csv", "qc_validation_summary.csv"),
        ("output/qc/viability_qc_r3_final.csv", "detection_checks_200reviews.csv"),
        ("output/qc/viability_qc_r3_summary.json", "detection_checks_summary.json"),
        ("output/qc/qc_summary_cytotype.csv", "qc_summary_cytotype.csv"),
        ("output/qc/qc_summary_cytotype_run.csv", "qc_summary_cytotype_run.csv"),
        ("output/qc/class_min_diam.csv", "settings/class_min_diam.csv"),
        ("output/qc/class_conf_threshold.csv", "settings/class_conf_threshold.csv"),
        ("metadata/plant_metadata_540.csv", "settings/plant_metadata_540.csv"),
        ("outcomes/pollen_viability_report.pdf", "pollen_viability_report.pdf"),
        ("outcomes/pollen_viability_report.docx", "pollen_viability_report.docx"),
        ("outcomes/pollen_viability_report.qmd", "pollen_viability_report.qmd"),
        ("outcomes/MANUSCRIPT_METHODS_POLLEN_VIABILITY_v3.1.md", "methods.md"),
        ("outcomes/MANUSCRIPT_METHODS_POLLEN_VIABILITY_v3.1.docx", "methods.docx"),
    ]
    for src_rel, dst_rel in mapping:
        src = ROOT / src_rel
        if src.exists():
            dst = vdir / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Build calibrated per-grain compressed archive (grains_calibrated.csv.gz) with focus flags & 3D scaling
    gz_path = vdir / "grains_calibrated.csv.gz"
    gz_existing = DELIV / f"{vname}_grains_calibrated.csv.gz"
    if gz_existing.exists() and gz_existing.stat().st_size > 50_000_000:
        print(f"Reusing existing {gz_existing.name} ({gz_existing.stat().st_size / 1e6:.2f} MB)...")
        shutil.copy2(gz_existing, gz_path)
        g_cal = None
    else:
        print("Compressing 2,186,245 calibrated grains (with norm_sharpness, is_in_focus & v2.1 3D columns) into grains_calibrated.csv.gz...")
        md_df = pd.read_csv(ROOT / "output/qc/class_min_diam.csv")[["cytotype", "class_name", "min_diam_um"]]
        ct_df = pd.read_csv(ROOT / "output/qc/class_conf_threshold.csv")[["cytotype", "class_name", "conf_threshold"]]
        m3d = pd.read_csv(ROOT / "outcomes/pollen_3d_cytotype_models.csv")[["cytotype", "k_diam_3d", "k_vol_3d_total"]]
        gf = pd.read_csv(ROOT / "output/qc/grain_focus_flags.csv.gz")[["slide_name", "grain_id", "norm_sharpness"]]
        g = pd.read_csv(ROOT / "output/size_analysis/pollen_grain_measurements.csv")
        g = (
            g.merge(gf, on=["slide_name", "grain_id"], how="left")
            .merge(md_df, on=["cytotype", "class_name"], how="left")
            .merge(ct_df, on=["cytotype", "class_name"], how="left")
            .merge(m3d, on="cytotype", how="left")
        )
        keep = (g["min_diam_um"].isna() | (g["diameter_um"] >= g["min_diam_um"])) & (
            g["conf_threshold"].isna() | (g["conf"] >= g["conf_threshold"])
        )
        g_cal = g[keep].drop(columns=["min_diam_um", "conf_threshold"]).copy()
        g_cal["is_in_focus"] = (g_cal["norm_sharpness"].isna() | (g_cal["norm_sharpness"] >= 3.50)).astype(int)
        is_viab = g_cal["class_name"] == "viable"
        g_cal["diameter_3d_um"] = (g_cal["diameter_um"] * g_cal["k_diam_3d"].where(is_viab, 1.0)).round(3)
        g_cal["volume_3d_um3"] = (
            (3.141592653589793 / 6.0) * (g_cal["diameter_um"] ** 3) * g_cal["k_vol_3d_total"].where(is_viab, 1.0)
        ).round(2)
        g_cal = g_cal.drop(columns=["k_diam_3d", "k_vol_3d_total"])
        g_cal.to_csv(gz_path, index=False, compression="gzip")

    # Document column dictionary (columns.csv)
    col_desc = pd.DataFrame([
        ("slides.csv", "slide_name", "Unique whole-slide CZI scan stem (YYYYMMDD_NNNN_plant_pop_ploidy_[rep_]Pollen_viability)"),
        ("slides.csv", "plant_id", "Unique maternal plant identifier matching plant_metadata_540.csv"),
        ("slides.csv", "cytotype", "Canonical cytotype: nat_2x, ory_2x, nat_4x, neo_4x"),
        ("slides.csv", "cultivation_run", "F2 cultivation cohort: run1_F2 or run3_F2"),
        ("slides.csv", "replicate", "Coverslip aliquot: primary or replicate sister slide"),
        ("slides.csv", "Count_detected", "Total calibrated pollen grains on slide (viable + non_viable + intermediate)"),
        ("slides.csv", "sharp_grains", "Total in-focus calibrated pollen grains on slide (norm_sharpness >= 3.50)"),
        ("slides.csv", "n_defocused", "Number of calibrated grains excluded from viability/size due to Z-defocus (norm_sharpness < 3.50)"),
        ("slides.csv", "Viability_pct_detected", "In-focus pollen viability (%) = 100 * viable_count / sharp_grains"),
        ("slides.csv", "viable_median_diam_um", "2D area-equivalent median diameter of in-focus isolated viable grains (um)"),
        ("slides.csv", "viable_diam_3d_um", "3D stereologically & mesh-corrected median diameter of in-focus viable grains (um)"),
        ("slides.csv", "viable_vol_3d_mesh_median_um3", "3D closed-mesh median grain volume of in-focus viable grains (um^3)"),
        ("slides.csv", "Vol_3d_viable_total_slide_um3", "Total 3D viable pollen biovolume on slide (um^3)"),
        ("plants.csv", "mean_grains_per_slide", "Mean calibrated pollen grains per slide across sister slides of the plant"),
        ("plants.csv", "viability_pct_mean", "Mean in-focus pollen viability (%) across sister slides of the plant"),
        ("plants.csv", "viable_diam_3d_um", "Mean 3D-corrected viable pollen diameter of the plant (um)"),
        ("plants.csv", "viable_vol_3d_mesh_median_um3", "Mean 3D closed-mesh viable grain volume of the plant (um^3)"),
        ("grains_calibrated.csv.gz", "norm_sharpness", "Contrast-normalized grain-local Laplacian sharpness: 100 * SD(|Laplacian|) / max(18, P95 - P05)"),
        ("grains_calibrated.csv.gz", "is_in_focus", "1 if grain passes optical focus gate (norm_sharpness >= 3.50 or verified crisp slide), 0 if Z-defocused"),
    ], columns=["table", "column", "description"])
    col_desc.to_csv(vdir / "columns.csv", index=False)

    fv = pd.read_csv(vdir / "all_slides_qc.csv")
    sl = pd.read_csv(vdir / "slides.csv")
    if g_cal is not None:
        on_sl = g_cal["slide_name"].isin(set(sl["slide_name"]))
        n_iso_sharp = int(((g_cal["is_in_focus"] == 1) & (g_cal["valid_for_size"] == 1) & on_sl & g_cal["class_name"].isin(["viable", "non_viable"])).sum())
    else:
        n_iso_sharp = 1679037

    import tempfile

    readme_qmd = """---
title: "Pollen Viability & 3D Morphometry — Release Package (v2.1)"
subtitle: "Pollination Ecology Group · Complete 1,008-Slide / 500-Plant Dataset (28 November 2026)"
filters:
  - auto_cols.lua
format:
  typst:
    papersize: a4
    margin:
      x: 1.65cm
      y: 1.55cm
    toc: false
    number-sections: false
    include-in-header:
      text: |
        #set text(size: 9.5pt)
        #set table(
          inset: (x: 6.5pt, y: 5pt),
          stroke: (_, y) => (
            top: if y == 0 { 0.85pt + rgb("#374151") } else if y == 1 { 0.6pt + rgb("#374151") } else { 0.35pt + rgb("#e5e7eb") },
            bottom: 0pt,
            left: none,
            right: none,
          ),
          fill: (_, y) => if y == 0 { rgb("#f3f4f6") } else if calc.even(y) { rgb("#f9fafb") } else { none },
        )
        #show table: set text(size: 8.5pt, hyphenate: false)
        #show table: it => block(stroke: (bottom: 0.85pt + rgb("#374151")), inset: 0pt, it)
        #show table.cell: set par(justify: false, leading: 0.48em)
        #show table.cell.where(y: 0): set text(weight: "bold", fill: rgb("#111827"))
---

## 1. Quick Guide to Included Files

- **`pollen_viability_report.pdf`** / **`pollen_viability_report.docx`** — 6-page co-author summary report (Tables 1–7, Figures 1–2, Short Manuscript Methods & Supplementary Methods).
- **`methods.docx`** / **`methods.md`** — Standalone Manuscript Materials & Methods (~450 words) + Acknowledgements + Supplementary Methods.
- **`plants.csv`** — Final plant-level dataset (**500 plants**; sister-slide means for 3D/2D grain diameter, 3D volume, viability %, and grains/slide).
- **`slides.csv`** — Final QC-passing slide-level dataset (**977 whole-slide scans** with $\\ge 100$ calibrated and $\\ge 100$ in-focus grains).
- **`all_slides_qc.csv`** — Full audit table across all **1,008 scanned slides** (including QC flags, `Count_detected`, `sharp_grains`, and `n_defocused`).
- **`cytotype_3d_models.csv`** & **`pollen_3d_meshes.zip`** — Stereological 3D axis/volume parameters and closed genus-0 Wavefront `.obj` meshes per cytotype.
- **`slide_focus_summary_1008.csv`** & **`columns.csv`** — Per-slide optical focus summary and column dictionary.

## 2. Sample, Plant & Focus-Filter Accounting (`540` Cultivated $\\rightarrow$ `500` Plants / `977` Slides)

1. **Plants (`500` in Results):** Out of **540 cultivated plants** (`180` `run1_F2`, `360` `run3_F2`), **36 never flowered** and **504 flowered & were scanned** (`2` sister coverslips per flower = `1,008` scans). Only **4 flowering plants** were excluded (`3` had $<100$ grains on both sister slides; **only 1 plant** was lost to optical blur on both slides), retaining **500 plants** (`477` with both sister slides, `23` with 1 passing slide).
2. **Whole-slide scans (`977` in Results):** Out of **1,008 scans**, **31 scans** (`3.1%`) were excluded: **22 low-count pipetting-failure scans** ($<100$ total grains; 16 rescued at plant level by their paired sister slide) and **9 wholly blurred scans** ($<100$ sharp grains; 7 rescued at plant level by their paired sister slide).
3. **Pollen grains:** **2,186,245 calibrated grains** across all 1,008 scans (**2,179,117** on the 977 retained slides, kept for per-slide abundance so local blur patches do not undercount pollen output). Filtering out **260,729 (`11.9%`) locally defocused grains** (`norm_sharpness < 3.50`) left **1,924,830 sharp grains** for viability % and **1,679,037 sharp isolated single grains** for 2D/3D size (**1,571,294** hydrated viable grains in 3D stereological unfolding).

| Cytotype | Flowered Plants | **Plants in Results** | Scanned Slides | Excluded: Sparse ($<100$ grains) | Excluded: Blurred ($<100$ sharp) | **Slides in Results** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ory_2x` | 89 | **89** | 178 | 8 | 0 | **170** |
| `nat_2x` | 146 | **146** | 292 | 4 | 1 | **287** |
| `nat_4x` | 123 | **119** | 246 | 9 | 2 | **235** |
| `neo_4x` | 146 | **146** | 292 | 1 | 6 | **285** |
| **Total** | **504** | **500** | **1,008** | **22** | **9** | **977** |

*(Note: The 80 MB single-grain table `pollen_viability_v2.1_2026-11-28_grains_calibrated.csv.gz` is archived separately in CESNET S3 under `s3://peg-bucket/Ostatni/Pollen_viability/Deliverables/` to keep this `.zip` lightweight for email.)*
"""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        shutil.copy2(ROOT / "outcomes/auto_cols.lua", td_p / "auto_cols.lua")
        (td_p / "README.qmd").write_text(readme_qmd, encoding="utf-8")
        subprocess.run(["quarto", "render", str(td_p / "README.qmd"), "--to", "typst"], check=True)
        shutil.copy2(td_p / "README.pdf", vdir / "README.pdf")

    meta = {
        "version": "2.1",
        "release_date": "2026-11-28",
        "folder": vname,
        "git_commit": f"{head}+v2.1",
        "status": "current_production",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_checkpoint": "last_20261122_r3.pt (MD5: 1ea93497105eee085848310df127e42f)",
        "focus_filter": "contrast-normalized Laplacian sharpness >= 3.50 (100 * SD(|Laplacian|) / max(18, P95 - P05))",
        "s3_source_czi_prefix": "s3://peg-bucket/Ostatni/Pollen_viability/Source/ (1,008 .czi files)",
        "s3_tiles_prefix": "s3://peg-bucket/Ostatni/Pollen_viability/tiles_640/ (1,008 slide folders, 265,520 tiles)",
        "s3_deliverables_prefix": f"s3://{S3_BUCKET}/{S3_DELIV_PFX}/",
        "plants_cultivated_total": 540,
        "plants_flowered_scanned": int(fv["plant_id"].nunique()),
        "slides_scanned": int(len(fv)),
        "slides_qc_pass_ge100": int(len(sl)),
        "plants_represented": int(sl["plant_id"].nunique()),
        "calibrated_grains_total": int(fv["Count_detected"].sum()),
        "calibrated_sharp_grains_total": int(fv["sharp_grains"].sum()),
        "isolated_morphometry_sharp_grains": n_iso_sharp,
        "note": "Focus-filtered production release across all 1,008 whole-slide scans (504 flowering plants of 540 cultivated; 977 sharp QC-passing slides across 500 plants after excluding 22 sparse <100-grain scans and 9 wholly Z-defocused <100-sharp-grain scans). Eliminates Z-defocus blur halos (22-28 um defocused viable grains misclassified as non_viable), halves sister-slide viability discordance (6.24 -> 3.20 pp, median 1.01 pp), and refits homologous prolate-spheroidal 3D cytotype models.",
    }
    zp = write_manifest_and_zip(vdir, meta, exclude_from_zip=("grains_calibrated.csv.gz",))
    gz_copy = DELIV / f"{vname}_grains_calibrated.csv.gz"
    shutil.copy2(gz_path, gz_copy)
    return meta, zp, gz_copy


def main():
    t0 = time.time()
    DELIV.mkdir(parents=True, exist_ok=True)

    m10, z10 = ensure_v1_0_from_git()
    m20, z20, gz20 = preserve_existing_v2_0()
    m21, z21, gz21 = build_v2_1_current()

    versions_df = pd.DataFrame([
        {
            "version": m["version"],
            "date": m["release_date"],
            "folder": m["folder"],
            "status": m["status"],
            "git_commit": m["git_commit"],
            "slides_scanned": m["slides_scanned"],
            "slides_qc_pass_ge100": m["slides_qc_pass_ge100"],
            "plants_represented": m["plants_represented"],
            "calibrated_grains_total": m["calibrated_grains_total"],
            "note": m["note"],
        }
        for m in (m10, m20, m21)
    ])
    versions_df.to_csv(DELIV / "versions.csv", index=False)
    (DELIV / "pollen_viability_latest.txt").write_text(f"{m21['folder']}\n", encoding="utf-8")

    changelog = f"""# Pollen Viability & 3D Morphometry Dataset — Archived Versions

All versioned release bundles and compressed single-grain tables are archived both locally under `Deliverables/` and in CESNET S3 object storage under `s3://{S3_BUCKET}/{S3_DELIV_PFX}/`.

## v2.1 — 2026-11-28 (`current_production`)

Focus-filtered full-experiment production release adding per-grain contrast-normalized Laplacian sharpness screening (`norm_sharpness = 100 * SD(|Laplacian|) / max(18, P95 - P05) >= 3.50`) across all 1,008 whole-slide scans to eliminate optical Z-defocus blur halos (`22–28 µm` out-of-focus viable grains whose washed-out brownish blur halos mimicked non-viable staining and stretched non-viable diameter violins).

- **Folder / Bundle:** `Deliverables/{m21['folder']}/`, `{z21.name}`, `{gz21.name}`
- **Coverage:** **1,008 whole-slide scans** across **504 flowering plants** out of 540 cultivated (`36` non-flowering); **977 sharp QC-passing slides** (`>= 100` calibrated grains AND `>= 100` sharp grains) across **500 plants** (`22` sparse `< 100`-grain slides and `9` wholly Z-defocused `< 100`-sharp-grain slides excluded).
- **Grain totals:** **2,186,245 calibrated pollen grains** (`2,179,117` on the 977 retained slides); **1,925,516 in-focus calibrated grains** (`1,924,830` on retained slides); **1,679,037 in-focus isolated single `viable`/`non_viable` grains** for morphometry; **1,571,294 in-focus hydrated viable grains** in Stage-1/2 3D stereological & closed-mesh unfolding.
- **Improvements vs. v2.0:** Halves sister-slide viability discrepancy (`6.24 pp -> 3.20 pp` mean, `1.48 pp -> 1.01 pp` median; `> 20 pp` discordant pairs cut from `42 -> 13`), restores clean shrunken non-viable microspore medians (`13.14 µm` `ory_2x`, `13.11 µm` `nat_2x`, `15.73 µm` `nat_4x`, `16.52 µm` `neo_4x`), and converges all four cytotypes to homologous `prolate-spheroidal` 3D shapes (`neo_4x` canonical `7,256 µm³` vs. `nat_2x` `3,674 µm³` = `+97.5%` / `1.98x` exact 1:1 karyoplasmic doubling; `nat_4x` `5,203 µm³` = `+41.6%` downsizing).

## v2.0 — 2026-11-28 (`archived_superseded_by_v2.1`)

Complete 1,008-slide unfiltered release after fixing unpaginated `s3.list_objects_v2()` folder discovery in `src/pollen_size_analyzer.py` and filtering 109 legacy Oct-21 orphan `_det.json` files across 6 slides (prior to per-grain optical focus screening).

- **Folder / Bundle:** `Deliverables/{m20['folder']}/`, `{z20.name}`, `{gz20.name}`
- **Coverage:** 1,008 whole-slide scans across 504 flowering plants; 986 slides (`>= 100` grains) across 501 plants; 2,186,245 calibrated pollen grains (`1,885,728` isolated single grains).

## v1.0 — 2026-11-25 (`archived_superseded_by_v2.0`)

First two-reviewer validated + 3D-corrected release (commit `885344f0`), covering the first alphabetical S3 page of 502 whole-slide scans before Eliška flagged the missing second page of slides.

- **Folder / Bundle:** `Deliverables/{m10['folder']}/`, `{z10.name}`
- **Coverage:** 502 scanned slides, 491 QC-passing slides (`>= 100` grains) across 251 plants; 968,195 calibrated grains (`825,310` isolated single grains).
- **Model & QC:** Final post-mosaic `YOLOv11m-seg` checkpoint `last_20261122_r3.pt` (`MD5: 1ea93497105eee085848310df127e42f`), two-reviewer blind audit (`jakub` 100 + `eliska` 100 = 200 reviews across 154 unseen tiles; 98.0% precision, 99.0% recall, 99.3% 3-class accuracy, 97.5% polygon accuracy), plus 96-grain size-extremes audit.
"""
    (DELIV / "CHANGELOG.md").write_text(changelog, encoding="utf-8")

    # Upload updated index files, v2.0 updated zip/manifest, and v2.1 archive + grains_calibrated.csv.gz to CESNET S3
    s3 = make_s3()
    to_upload = [
        (DELIV / "CHANGELOG.md", f"{S3_DELIV_PFX}/CHANGELOG.md"),
        (DELIV / "versions.csv", f"{S3_DELIV_PFX}/versions.csv"),
        (DELIV / "pollen_viability_latest.txt", f"{S3_DELIV_PFX}/pollen_viability_latest.txt"),
        (z20, f"{S3_DELIV_PFX}/{z20.name}"),
        (DELIV / m20["folder"] / "VERSION.json", f"{S3_DELIV_PFX}/{m20['folder']}_VERSION.json"),
        (z21, f"{S3_DELIV_PFX}/{z21.name}"),
        (gz21, f"{S3_DELIV_PFX}/{gz21.name}"),
        (DELIV / m21["folder"] / "VERSION.json", f"{S3_DELIV_PFX}/{m21['folder']}_VERSION.json"),
    ]
    print(f"\n☁️ Uploading {len(to_upload)} archive artifacts to s3://{S3_BUCKET}/{S3_DELIV_PFX}/ ...")
    for p, key in to_upload:
        print(f"   -> {p.name} ({p.stat().st_size / 1e6:.2f} MB) -> s3://{S3_BUCKET}/{key}")
        s3.upload_file(str(p), S3_BUCKET, key)

    print(f"\n✅ Archived v1.0, v2.0, and v2.1 locally and on S3 in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
