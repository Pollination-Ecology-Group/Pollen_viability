#!/usr/bin/env python3
"""
build_viability_slide_check.py — Paginate every detection of flagged / low-count Pollen Viability slides
for rapid full-slide verification (adapted from Pollen_latitude/src/build_slide_check.py).

Why this matters in Pollen Viability:
  * Slides in `qc_viability_review_queue.csv` (e.g., extreme viability, high intermediate share, low total
    grain count, or high clump share) can have their viability % skewed by a handful of false `non_viable`
    debris detections or misclassified `intermediate` grains.
  * Grouping all detections on those slides into pages of 200 detections (ordered by class: `non_viable`,
    `intermediate`, `viable`) lets a reviewer verify the entire slide in minutes, locking in an exact
    `verified` count and viability % in `final_viability.csv`.

Writes `output/qc/slidecheck_manifest.json` (and crop sprites when S3/local tile images are accessible).
"""

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd

PAGE = 200


def main():
    ap = argparse.ArgumentParser(description="Build Slide Check manifest for flagged Pollen Viability slides")
    ap.add_argument("--grains", default="output/size_analysis/pollen_grain_measurements.csv")
    ap.add_argument("--queue", default="output/qc/qc_viability_review_queue.csv")
    ap.add_argument("--slides", nargs="*", default=None, help="Optional explicit slide names (defaults to review queue)")
    ap.add_argument("--max-slides", type=int, default=15, help="Max slides from the review queue to paginate")
    ap.add_argument("--out", default="output/qc/slidecheck_manifest.json")
    args = ap.parse_args()

    queue_df = pd.read_csv(args.queue) if os.path.exists(args.queue) else pd.DataFrame()
    if args.slides:
        target_slides = list(args.slides)
    elif not queue_df.empty:
        # Prioritize slides with low/moderate grain counts or outlier viability first (fastest to verify completely)
        queue_df = queue_df.sort_values("total_grains")
        target_slides = queue_df["slide_name"].head(args.max_slides).tolist()
    else:
        target_slides = []

    if not target_slides:
        print("ℹ️  No slides queued for Slide Check.")
        return

    print(f"📖 Loading grains for {len(target_slides)} target slides from {args.grains}...")
    df = pd.read_csv(args.grains, low_memory=False)
    df = df[df["slide_name"].isin(target_slides)].copy()

    # Order grains on each slide so `non_viable` and `intermediate` appear on the first page!
    cls_order = {"non_viable": 0, "intermediate": 1, "viable": 2}
    df["cls_rank"] = df["class_name"].map(cls_order).fillna(3)
    df = df.sort_values(["slide_name", "cls_rank", "center_y_px", "center_x_px"])

    items = []
    for sl_name, grp in df.groupby("slide_name", sort=False):
        cells = []
        for r in grp.itertuples():
            cells.append({
                "grain_id": int(r.grain_id),
                "cls": r.class_name,
                "conf": round(float(r.conf), 3),
                "x": round(float(r.center_x_px), 1),
                "y": round(float(r.center_y_px), 1),
                "diam_um": round(float(r.diameter_um), 1),
                "is_clump": int(r.is_clump),
            })
        n_pages = math.ceil(len(cells) / PAGE)
        cyt = str(grp["cytotype"].iloc[0])
        plant_id = str(grp["plant_id"].iloc[0])
        for p in range(n_pages):
            page_cells = cells[p * PAGE:(p + 1) * PAGE]
            items.append({
                "id": f"{sl_name}.p{p}",
                "slide_name": sl_name,
                "plant_id": plant_id,
                "cytotype": cyt,
                "page": p,
                "pages": n_pages,
                "n_slide": len(cells),
                "n_page": len(page_cells),
                "n_non_viable_page": sum(1 for c in page_cells if c["cls"] == "non_viable"),
                "n_intermediate_page": sum(1 for c in page_cells if c["cls"] == "intermediate"),
                "n_viable_page": sum(1 for c in page_cells if c["cls"] == "viable"),
                "cells": page_cells,
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Path(args.out).write_text(json.dumps({"items": items}, indent=2))
    print(f"✅ Slide Check manifest built: {len(items)} pages ({sum(i['n_page'] for i in items)} detections) "
          f"across {len(target_slides)} flagged slides -> {args.out}")


if __name__ == "__main__":
    main()

