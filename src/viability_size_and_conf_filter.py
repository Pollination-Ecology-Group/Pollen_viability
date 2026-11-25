#!/usr/bin/env python3
"""
viability_size_and_conf_filter.py — Class-specific minimum grain size filter and confidence calibration
for Pollen Viability (adapted from Pollen_latitude/src/size_filter.py & conf_threshold.py).

Why class-specific?
  In Pollen_viability, there is no species difference (all samples are Tripleurospermum inodorum across 4
  cytotypes: `nat_2x`, `ory_2x`, `neo_4x`, `nat_4x`), BUT there is a large biological size difference
  between classes:
    * `viable` grains are plump (~20-28 um diameter)
    * `non_viable` (aborted/shrunken/unstained) grains are significantly smaller (~12-20 um)
    * `intermediate` grains lie in between
  A single size threshold either deletes real shrunken `non_viable` grains (biasing viability % upwards)
  or leaves small dust/stain debris misclassified as `non_viable` (biasing viability % downwards).

This script derives per-(cytotype, class_name) minimum diameter cutoffs (`output/qc/class_min_diam.csv`)
and confidence thresholds (`output/qc/class_conf_threshold.csv`) that lose <= MAX_LOST (2%) of valid
isolated grains of that class while removing small debris and low-confidence artifacts.
"""

import argparse
import os

import numpy as np
import pandas as pd

MAX_LOST = 0.02
CAP = 0.60


def main():
    ap = argparse.ArgumentParser(description="Derive per-(cytotype, class) size and confidence filters for Pollen Viability")
    ap.add_argument("--grains", default="output/size_analysis/pollen_grain_measurements.csv")
    ap.add_argument("--out-size", default="output/qc/class_min_diam.csv")
    ap.add_argument("--out-conf", default="output/qc/class_conf_threshold.csv")
    args = ap.parse_args()

    print(f"📖 Loading per-grain measurements from {args.grains}...")
    use_cols = ["slide_name", "cytotype", "class_name", "conf", "diameter_um", "solidity", "aspect_ratio", "is_outer_edge", "is_clump", "valid_for_size"]
    df = pd.read_csv(args.grains, usecols=lambda c: c in use_cols, low_memory=False)

    # Reference valid grains per (cytotype, class_name): isolated, interior, high solidity, conf >= 0.70
    df["is_ref_grain"] = (
        (df["is_outer_edge"] == 0) &
        (df["is_clump"] == 0) &
        (df["solidity"] >= 0.90) &
        (df["aspect_ratio"] <= 1.30) &
        (df["conf"] >= 0.70)
    )
    # Candidate debris / irregular fragments: low solidity or extreme aspect ratio with low confidence
    df["is_ref_debris"] = ((df["solidity"] < 0.85) | (df["aspect_ratio"] > 1.50)) & (df["conf"] < 0.65)

    size_rows = []
    conf_rows = []

    for (cyt, cls_name), grp in df.groupby(["cytotype", "class_name"]):
        if cyt == "unknown" or len(grp) < 30:
            continue
        ok = grp["is_ref_grain"].values
        bad = grp["is_ref_debris"].values
        dia = grp["diameter_um"].values
        confs = grp["conf"].values

        if ok.sum() < 20:
            continue

        med_d = float(np.median(dia[ok]))
        cap_d = round(CAP * med_d, 2)

        best_d = None
        for t in np.linspace(4.0, cap_d, 100):
            lost = (ok & (dia < t)).sum() / ok.sum()
            if lost <= MAX_LOST:
                best_d = round(float(t), 2)
            else:
                break

        if best_d is not None:
            keep = dia >= best_d
            size_rows.append({
                "cytotype": cyt,
                "class_name": cls_name,
                "min_diam_um": best_d,
                "class_median_diam_um": round(med_d, 2),
                "cap_diam_um": cap_d,
                "small_removed": int((~keep).sum()),
                "debris_removed": int((bad & ~keep).sum()),
                "ref_grain_lost_share": round(float((ok & ~keep).sum() / ok.sum()), 4),
            })

        best_c = 0.55
        for c_thr in (0.55, 0.58, 0.60, 0.62, 0.65, 0.67):
            # For nat_4x non_viable, allow 0.67 to eliminate borderline stain-ring FP (conf 0.661)
            max_loss_c = 0.03 if (cyt == "nat_4x" and cls_name == "non_viable") else MAX_LOST
            lost_c = (ok & (confs < c_thr)).sum() / ok.sum()
            if lost_c <= max_loss_c:
                best_c = c_thr
            else:
                break

        conf_rows.append({
            "cytotype": cyt,
            "class_name": cls_name,
            "conf_threshold": best_c,
            "low_conf_removed": int((confs < best_c).sum()),
            "debris_removed_at_conf": int((bad & (confs < best_c)).sum()),
            "ref_grain_lost_share": round(float((ok & (confs < best_c)).sum() / ok.sum()), 4),
            "median_conf": round(float(np.median(confs[ok])), 3),
        })

    os.makedirs(os.path.dirname(args.out_size) or ".", exist_ok=True)
    out_size_df = pd.DataFrame(size_rows).sort_values(["cytotype", "class_name"])
    out_size_df.to_csv(args.out_size, index=False)

    out_conf_df = pd.DataFrame(conf_rows).sort_values(["cytotype", "class_name"])
    out_conf_df.to_csv(args.out_conf, index=False)

    print("=== Class-Specific Minimum Grain Size Filter (`class_min_diam.csv`) ===")
    print(out_size_df.to_string(index=False))
    print("\n=== Class-Specific Confidence Thresholds (`class_conf_threshold.csv`) ===")
    print(out_conf_df.to_string(index=False))
    print(f"\n✅ Saved {args.out_size} and {args.out_conf}")


if __name__ == "__main__":
    main()

