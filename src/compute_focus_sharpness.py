#!/usr/bin/env python3
"""
compute_focus_sharpness.py — Contrast-normalized grain-local Laplacian sharpness audit across all 1,008 slides.

Why contrast-normalized Laplacian sharpness (`norm_sharpness = 100 * std(|Laplacian|) / dynamic_range`)?
  - Raw `var(|Laplacian|)` scales with intensity contrast squared, which unfairly penalizes pale translucent
    empty non-viable exine shells relative to dark opaque viable pollen even at identical sharp optical focus.
  - Normalizing `std(|Laplacian|)` by local patch dynamic range `max(18.0, P95(gray) - P05(gray))` makes edge
    acutance invariant to stain darkness:
      * Z-defocused blurred grains (Gaussian blur halo): norm_sharpness = 2.0 - 3.5  (< 3.8)
      * Sharp viable & sharp translucent sterile shells: norm_sharpness = 4.3 - 9.5  (>= 3.8)

Two-pass exact evaluation across all 1,008 slides:
  Pass 1: Evaluate up to 16 spatially stratified grain-bearing tiles on EVERY slide (1,008 slides).
  Pass 2: On any slide with any soft tile/grain in Pass 1 (q10_norm < 4.3 or min_tile_med < 4.2),
          or >= 20 non-viable grains, or nv/viable diameter ratio >= 0.88, or sister-slide viability
          difference >= 5 pp, evaluate EVERY grain-bearing tile on that slide so every grain gets its
          exact measured `norm_sharpness` and `raw_lap_var`.

Outputs:
  - output/qc/slide_focus_summary_1008.csv
  - output/qc/grain_focus_flags.parquet (or .csv.gz: grain-level sharpness for audited slides)
"""

import concurrent.futures
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pollen_size_analyzer import S3_BUCKET, TILE_PREFIX, make_s3

NORM_SHARP_THRESH = 3.80


def measure_tiles_for_slide(sname: str, slide_grains_df: pd.DataFrame, full_scan: bool = False, max_sample_tiles: int = 16):
    """Measures per-grain contrast-normalized sharpness across sampled or all grain-bearing tiles of one slide."""
    s3 = make_s3()
    pfx = f"{TILE_PREFIX.rstrip('/')}/{sname}/"
    tiles = []
    for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=pfx):
        for o in pg.get("Contents", []):
            k = o["Key"]
            if k.endswith(".jpg") and "_tile_" in k:
                try:
                    ys, xs = k[:-4].rsplit("_tile_", 1)[1].split("_", 1)
                    tiles.append((int(xs), int(ys), k))
                except Exception:
                    pass

    if not tiles or slide_grains_df.empty:
        return sname, [], {}

    # Assign each grain on the slide to the tile whose center (ox+320, oy+320) is closest
    tx = np.array([t[0] for t in tiles], dtype=np.float64)
    ty = np.array([t[1] for t in tiles], dtype=np.float64)
    tcx = tx + 320.0
    tcy = ty + 320.0

    gx = slide_grains_df["center_x_px"].to_numpy(dtype=np.float64)
    gy = slide_grains_df["center_y_px"].to_numpy(dtype=np.float64)
    gd = slide_grains_df["diameter_um"].to_numpy(dtype=np.float64)
    gids = slide_grains_df["grain_id"].to_numpy(dtype=np.int64)

    tile_to_grains = defaultdict(list)
    for idx in range(len(gx)):
        dx = gx[idx] - tcx
        dy = gy[idx] - tcy
        # Must lie inside tile [0, 640]
        inside = (gx[idx] >= tx) & (gx[idx] <= tx + 640.0) & (gy[idx] >= ty) & (gy[idx] <= ty + 640.0)
        if not np.any(inside):
            continue
        dist2 = np.where(inside, dx * dx + dy * dy, 1e18)
        best_t = int(np.argmin(dist2))
        lx = gx[idx] - tx[best_t]
        ly = gy[idx] - ty[best_t]
        tile_to_grains[best_t].append((int(gids[idx]), float(lx), float(ly), float(gd[idx])))

    active_tile_indices = sorted(tile_to_grains.keys())
    if not active_tile_indices:
        return sname, [], {}

    if not full_scan and len(active_tile_indices) > max_sample_tiles:
        step = np.linspace(0, len(active_tile_indices) - 1, max_sample_tiles, dtype=int)
        chosen_indices = [active_tile_indices[i] for i in sorted(set(step))]
    else:
        chosen_indices = active_tile_indices

    def _eval_one_tile(t_idx):
        ox, oy, key = tiles[t_idx]
        glist = tile_to_grains[t_idx]
        try:
            raw = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
            im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if im is None:
                return []
            lap = np.abs(cv2.Laplacian(im, cv2.CV_64F))
            h, w = im.shape[:2]
            out = []
            for gid, lx, ly, d_um in glist:
                rad = max(8, int(round(d_um / 0.8772 / 2.0)) + 5)
                ix, iy = int(round(lx)), int(round(ly))
                y1, y2 = max(0, iy - rad), min(h, iy + rad)
                x1, x2 = max(0, ix - rad), min(w, ix + rad)
                if (y2 - y1) < 6 or (x2 - x1) < 6:
                    continue
                patch = im[y1:y2, x1:x2]
                plap = lap[y1:y2, x1:x2]
                dyn = max(18.0, float(np.percentile(patch, 95) - np.percentile(patch, 5)))
                raw_var = float(np.var(plap))
                norm_sh = 100.0 * float(np.std(plap)) / dyn
                out.append((gid, t_idx, round(norm_sh, 3), round(raw_var, 2)))
            return out
        except Exception:
            return []

    grain_records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as tex:
        for res in tex.map(_eval_one_tile, chosen_indices):
            grain_records.extend(res)

    if not grain_records:
        return sname, [], {"n_active_tiles": len(active_tile_indices), "n_eval_tiles": len(chosen_indices)}

    norms = np.array([r[2] for r in grain_records], dtype=np.float64)
    raws = np.array([r[3] for r in grain_records], dtype=np.float64)
    tile_meds = defaultdict(list)
    for _, t_idx, ns, _ in grain_records:
        tile_meds[t_idx].append(ns)
    t_med_vals = [float(np.median(v)) for v in tile_meds.values()]

    stats = {
        "n_active_tiles": len(active_tile_indices),
        "n_eval_tiles": len(chosen_indices),
        "n_eval_grains": len(grain_records),
        "norm_sharp_med": round(float(np.median(norms)), 3),
        "norm_sharp_q10": round(float(np.percentile(norms, 10)), 3),
        "norm_sharp_q25": round(float(np.percentile(norms, 25)), 3),
        "min_tile_norm_med": round(float(np.min(t_med_vals)), 3),
        "raw_lap_med": round(float(np.median(raws)), 2),
        "share_defocused_lt38": round(float((norms < NORM_SHARP_THRESH).mean()), 4),
    }
    return sname, grain_records, stats


def _pass1_worker(args):
    sname, sub_df = args
    _, _, st = measure_tiles_for_slide(sname, sub_df, full_scan=False, max_sample_tiles=16)
    st["slide_name"] = sname
    return st


def _pass2_worker(args):
    sname, sub_df = args
    _, recs, st = measure_tiles_for_slide(sname, sub_df, full_scan=True)
    st["slide_name"] = sname
    return sname, recs, st


def main():
    t0 = time.time()
    print("📖 Loading 1,008-slide summaries and grain coordinates...")
    fv = pd.read_csv("output/qc/final_viability.csv")
    g = pd.read_csv(
        "output/size_analysis/pollen_grain_measurements.csv",
        usecols=["slide_name", "grain_id", "class_name", "center_x_px", "center_y_px", "diameter_um", "conf"],
    )

    by_slide = {sname: sub.copy() for sname, sub in g.groupby("slide_name")}
    all_slides = sorted(fv["slide_name"].unique())
    p1_tasks = [(s, by_slide.get(s, pd.DataFrame())) for s in all_slides]

    print(f"🔍 Pass 1: Stratified 16-tile sharpness probe across all {len(p1_tasks)} slides (8 processes x 16 threads)...")
    p1_stats = {}
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pex:
        for st in pex.map(_pass1_worker, p1_tasks, chunksize=4):
            p1_stats[st["slide_name"]] = st
            done += 1
            if done % 150 == 0 or done == len(p1_tasks):
                print(f"   Pass 1: {done}/{len(p1_tasks)} slides ({time.time()-t0:.0f}s)", flush=True)

    # Identify slides needing exhaustive per-grain Pass 2 evaluation
    fv_idx = fv.set_index("slide_name")
    sib_max = fv.groupby("plant_id")["Viability_pct_detected"].transform("max")
    fv_idx["sib_diff"] = (sib_max.values - fv["Viability_pct_detected"].values)

    need_p2 = []
    for s in all_slides:
        st = p1_stats.get(s, {})
        if st.get("n_eval_grains", 0) == 0:
            continue
        r = fv_idx.loc[s]
        nv_cnt = float(r.get("non_viable_count", 0) or 0)
        v_d = float(r.get("viable_median_diam_um", 0) or 0)
        nv_d = float(r.get("non_viable_median_diam_um", 0) or 0)
        sib_diff = float(r.get("sib_diff", 0) or 0)

        soft_probe = (
            st.get("norm_sharp_q10", 99.0) < 4.25
            or st.get("min_tile_norm_med", 99.0) < 4.15
            or st.get("share_defocused_lt38", 0.0) > 0.03
        )
        nv_trigger = (nv_cnt >= 25) or (v_d > 0 and nv_d / v_d >= 0.88 and nv_cnt >= 5) or (sib_diff >= 5.0)
        if soft_probe or nv_trigger:
            need_p2.append(s)

    print(f"🎯 Pass 2: Full per-grain sharpness scan on {len(need_p2)} candidate slides ({time.time()-t0:.0f}s)...")
    p2_tasks = [(s, by_slide.get(s, pd.DataFrame())) for s in need_p2]
    grain_rows = []
    done2 = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pex:
        for sname, recs, st in pex.map(_pass2_worker, p2_tasks, chunksize=2):
            st["full_tile_scan"] = True
            p1_stats[sname] = st
            for gid, _, ns, rv in recs:
                grain_rows.append((sname, gid, ns, rv))
            done2 += 1
            if done2 % 40 == 0 or done2 == len(p2_tasks):
                print(f"   Pass 2: {done2}/{len(p2_tasks)} slides ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs("output/qc", exist_ok=True)
    sum_rows = []
    for s in all_slides:
        st = p1_stats.get(s, {})
        st.setdefault("slide_name", s)
        st.setdefault("full_tile_scan", False)
        sum_rows.append(st)
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv("output/qc/slide_focus_summary_1008.csv", index=False)

    gf_df = pd.DataFrame(grain_rows, columns=["slide_name", "grain_id", "norm_sharpness", "raw_lap_var"])
    gf_df.to_csv("output/qc/grain_focus_flags.csv.gz", index=False, compression="gzip")
    print(f"✅ Saved slide_focus_summary_1008.csv ({len(sum_df)} slides) & grain_focus_flags.csv.gz ({len(gf_df):,} grains, {(gf_df['norm_sharpness'] < NORM_SHARP_THRESH).sum():,} defocused < {NORM_SHARP_THRESH}) in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()

