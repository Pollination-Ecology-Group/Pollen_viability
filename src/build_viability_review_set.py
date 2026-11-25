#!/usr/bin/env python3
"""
build_viability_review_set.py — select a blinded, stratified QC sample for human review
═══════════════════════════════════════════════════════════════════════════════════════
Feeds the Pollen Viability QC review app (review_app/). Separate from Pollen_latitude.

Selection
  A) Stratified random  : N slides per cytotype (nat_2x, ory_2x, neo_4x, nat_4x), spread
                          round-robin over population x cultivation_run, primary slides
                          preferred over 'rep'. From each slide K tiles chosen by grain
                          density (dense / medium / sparse-but-not-empty).
  B) Flagged outliers   : slides that look anomalous within their cytotype in
                          output/size_analysis/pollen_slide_summaries.csv (if present):
                          viability or viable-size robust z > 3, viability < 10 % or > 99 %,
                          or low fraction of grains usable for size (clumping / edges).
Assignment
  OVERLAP fraction of items goes to every reviewer (inter-rater agreement); the rest is
  split round-robin. Items are shuffled so cytotypes are interleaved, and the app never
  shows cytotype or slide name to reviewers (blind review).

Only slides whose detections are newer than --min-det-date are eligible, so a set can be
built safely while inference is still running.

Usage
  python src/build_viability_review_set.py                      # final set
  python src/build_viability_review_set.py --set-prefix preview # test set
Writes s3://peg-bucket/Ostatni/Pollen_viability/Review/manifest.json (+ local copy).
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import concurrent.futures
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pollen_size_analyzer import (  # noqa: E402
    make_s3, parse_slide_metadata, S3_BUCKET, TILE_PREFIX, DET_CACHE_DIR, DET_CACHE_TAG,
)

import cv2
import numpy as np

REVIEW_PREFIX = os.environ.get("REVIEW_PREFIX", "Ostatni/Pollen_viability/Review")
CYTOTYPES = ["nat_2x", "ory_2x", "neo_4x", "nat_4x"]
CONF = 0.50
MIN_GRAIN_FOCUS_LAP = 35.0


def load_training_excluded_bases(s3) -> set:
    """Collect normalized tile basenames used anywhere in training so the QC set stays 100% blind."""
    from train_model import strip_tile_base
    excl = set()
    for pfx in (
        "Ostatni/Pollen_viability/Relabel/Labels/",
        "Ostatni/Pollen_viability/active_learning/hard_positives/",
        "Ostatni/Pollen_viability/active_learning/hard_negatives/",
    ):
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=pfx):
            for o in page.get("Contents", []):
                if not o["Key"].endswith("/"):
                    excl.add(strip_tile_base(o["Key"]))
    for root, _, files in os.walk("Pollen_viability"):
        if "datasets" in root:
            continue
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".txt")):
                excl.add(strip_tile_base(f))
    return excl


def is_tile_sharp_and_interior(s3, img_key: str, det_key: str) -> bool:
    """Verify interior coverslip coordinates and grain-local Laplacian sharpness >= 35.0."""
    base = os.path.basename(img_key)
    if "_tile_0_" in base or base.endswith("_tile_0.jpg"):
        return False
    try:
        img_bytes = s3.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg_med = float(np.median(gray))
        if not (175.0 <= bg_med <= 246.0):
            return False
        boxes = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=det_key)["Body"].read()).get("boxes", [])
        boxes = [b for b in boxes if float(b[4]) >= CONF]
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        bvars = [
            float(np.var(lap[max(0, int(b[1]) - 4):min(640, int(b[3]) + 4), max(0, int(b[0]) - 4):min(640, int(b[2]) + 4)]))
            for b in boxes if int(b[2]) - int(b[0]) >= 8 and int(b[3]) - int(b[1]) >= 8
        ]
        focus = float(np.median(bvars)) if bvars else float(np.var(lap))
        return focus >= MIN_GRAIN_FOCUS_LAP
    except Exception:
        return False


def list_slides(s3):
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=TILE_PREFIX, Delimiter="/")
    return [p["Prefix"] for p in resp.get("CommonPrefixes", [])]


def list_slide_objects(s3, prefix):
    """Return {tile_stem: {"img": key, "det": key, "det_mtime": dt}} for one slide."""
    tiles = defaultdict(dict)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            k = o["Key"]
            if k.endswith("_det.json"):
                stem = k[: -len("_det.json")]
                tiles[stem]["det"] = k
                tiles[stem]["det_mtime"] = o["LastModified"]
            elif k.lower().endswith((".jpg", ".jpeg", ".png")):
                tiles[k.rsplit(".", 1)[0]]["img"] = k
    return {s: t for s, t in tiles.items() if "img" in t and "det" in t}


def grain_count(s3, slide_name, det_key):
    """Number of detections >= CONF in a tile, using the local det cache when available."""
    local = Path(DET_CACHE_DIR) / slide_name / os.path.basename(det_key)
    try:
        raw = local.read_bytes() if local.exists() else \
            s3.get_object(Bucket=S3_BUCKET, Key=det_key)["Body"].read()
        boxes = json.loads(raw).get("boxes", [])
        return sum(1 for b in boxes if b[4] >= CONF)
    except Exception:
        return None


def pick_tiles(s3, slide_name, tiles, k, rng, excluded_bases=None, sample_n=180):
    """Pick k sharp, leak-free interior tiles spanning density: dense (p90), medium (p50), sparse non-empty (p15)."""
    from train_model import strip_tile_base
    excluded_bases = excluded_bases or set()
    stems = [
        s for s in tiles
        if "_tile_0_" not in s and strip_tile_base(s) not in excluded_bases
    ]
    rng.shuffle(stems)
    stems = stems[:sample_n]
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        counts = list(ex.map(lambda s: grain_count(s3, slide_name, tiles[s]["det"]), stems))
    nonempty = sorted([(c, s) for c, s in zip(counts, stems) if c], key=lambda x: x[0])
    if not nonempty:
        return []
    targets = [("dense", 0.90), ("medium", 0.50), ("sparse", 0.15)][:k]
    chosen, used = [], set()
    for label, q in targets:
        idx = min(len(nonempty) - 1, int(round(q * (len(nonempty) - 1))))
        # walk outwards to an unused sharp interior tile
        for off in range(len(nonempty)):
            found = False
            for j in (idx - off, idx + off):
                if 0 <= j < len(nonempty) and nonempty[j][1] not in used:
                    c, s = nonempty[j]
                    if is_tile_sharp_and_interior(s3, tiles[s]["img"], tiles[s]["det"]):
                        used.add(s)
                        chosen.append({"stem": s, "density": label, "n_pred": c})
                        found = True
                        break
            if found:
                break
    return chosen


def robust_z(values):
    vals = sorted(v for v in values if v is not None)
    if len(vals) < 5:
        return lambda x: 0.0
    med = vals[len(vals) // 2]
    mad = sorted(abs(v - med) for v in vals)[len(vals) // 2] or 1e-6
    return lambda x: 0.0 if x is None else (x - med) / (1.4826 * mad)


def flag_outliers(summary_csv, eligible):
    """Return [(slide_name, reason, score)] sorted by severity."""
    if not os.path.exists(summary_csv):
        print(f"ℹ️  {summary_csv} not found — skipping flagged outliers (run the analyzer first).")
        return []
    rows = [r for r in csv.DictReader(open(summary_csv)) if r["slide_name"] in eligible]
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else None
    flags = []
    for cyt in CYTOTYPES:
        grp = [r for r in rows if r["cytotype"] == cyt]
        zv = robust_z([f(r, "viability_pct") for r in grp])
        zd = robust_z([f(r, "viable_median_diam_um") or None for r in grp])
        for r in grp:
            reasons, score = [], 0.0
            v, d = f(r, "viability_pct"), f(r, "viable_median_diam_um")
            if v is not None and (v < 10 or v > 99):
                reasons.append(f"extreme viability {v:.0f}%"); score = max(score, 5)
            if abs(zv(v)) > 3:
                reasons.append(f"viability atypical for cytotype (z={zv(v):+.1f})"); score = max(score, abs(zv(v)))
            if d and abs(zd(d)) > 3:
                reasons.append(f"viable size atypical for cytotype (z={zd(d):+.1f})"); score = max(score, abs(zd(d)))
            vc, vs = f(r, "viable_count"), f(r, "viable_valid_size")
            if vc and vc > 50 and vs is not None and vs / vc < 0.5:
                reasons.append(f"only {100*vs/vc:.0f}% grains measurable (clumps/edges)"); score = max(score, 3.5)
            if reasons:
                flags.append((r["slide_name"], "; ".join(reasons), score))
    return sorted(flags, key=lambda x: -x[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-cytotype", type=int, default=6, help="stratified slides per cytotype")
    ap.add_argument("--tiles-per-slide", type=int, default=3)
    ap.add_argument("--n-flagged", type=int, default=12, help="max flagged outlier slides")
    ap.add_argument("--flagged-tiles", type=int, default=2)
    ap.add_argument("--reviewers", default="jakub,eliska")
    ap.add_argument("--overlap", type=float, default=0.3, help="fraction reviewed by everyone")
    ap.add_argument("--target-per-reviewer", type=int, default=0,
                    help="if >0, top up / trim so every reviewer gets exactly this many tiles")
    ap.add_argument("--min-det-date", default="2026-10-24T05:29:00+00:00",
                    help="only use slides whose detections were all written after this (new model)")
    ap.add_argument("--summary-csv", default="output/size_analysis/pollen_slide_summaries.csv")
    ap.add_argument("--set-prefix", default="qc", help="'preview' for a throw-away test set")
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--dry-run", action="store_true", help="write local manifest only")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    reviewers = [r.strip() for r in args.reviewers.split(",") if r.strip()]
    min_date = datetime.fromisoformat(args.min_det_date)
    s3 = make_s3()
    excluded_bases = load_training_excluded_bases(s3)
    print(f"🔒 Excluding {len(excluded_bases)} training/relabel tiles from blind QC selection.")

    # ── 1. inventory: slides with complete, up-to-date detections ──────────────────────
    print("📋 Listing slides…")
    prefixes = list_slides(s3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        objs = dict(zip(prefixes, ex.map(lambda p: list_slide_objects(s3, p), prefixes)))
    slides = {}
    for p, tiles in objs.items():
        if not tiles:
            continue
        if any(t["det_mtime"] < min_date for t in tiles.values()):
            continue  # not (fully) re-detected with the new model yet
        name = p.rstrip("/").split("/")[-1]
        slides[name] = {"prefix": p, "tiles": tiles, "meta": parse_slide_metadata(name)}
    by_cyt = defaultdict(list)
    for name, s in slides.items():
        by_cyt[s["meta"]["cytotype"]].append(name)
    print(f"   {len(slides)}/{len(prefixes)} slides eligible: " +
          ", ".join(f"{c}={len(by_cyt[c])}" for c in CYTOTYPES))

    # ── 2. stratified random slides ────────────────────────────────────────────────────
    selected = []  # (slide_name, selection, reason, n_tiles)
    unused_by_cyt = defaultdict(list)
    for cyt in CYTOTYPES:
        strata = defaultdict(list)
        for name in by_cyt[cyt]:
            m = slides[name]["meta"]
            strata[(m["population"], m["cultivation_run"])].append(name)
        for lst in strata.values():
            rng.shuffle(lst)
            lst.sort(key=lambda n: slides[n]["meta"]["replicate"] != "primary")  # primaries first
        keys = list(strata)
        rng.shuffle(keys)
        ordered_cyt = []
        while any(strata[k] for k in keys):
            for k in keys:
                if strata[k]:
                    ordered_cyt.append(strata[k].pop(0))
        picked = ordered_cyt[:args.per_cytotype]
        unused_by_cyt[cyt] = ordered_cyt[args.per_cytotype:]
        if len(picked) < args.per_cytotype:
            print(f"   ⚠️  only {len(picked)} eligible slides for {cyt}")
        selected += [(n, "random", "", args.tiles_per_slide) for n in picked]

    # ── 3. flagged outliers ────────────────────────────────────────────────────────────
    taken = {s[0] for s in selected}
    for name, reason, _ in flag_outliers(args.summary_csv, set(slides)):
        if len([s for s in selected if s[1] == "flagged"]) >= args.n_flagged:
            break
        if name not in taken:
            selected.append((name, "flagged", reason, args.flagged_tiles))
            taken.add(name)

    # Compute exact total unique items needed if --target-per-reviewer is set
    n_rev = max(1, len(reviewers))
    exact_overlap = None
    exact_total = None
    if args.target_per_reviewer > 0:
        # Solve: n_ov + n_single = T, total = n_ov + K * n_single, n_ov / total ≈ overlap
        best_pair = None
        for n_single in range(args.target_per_reviewer + 1):
            n_ov = args.target_per_reviewer - n_single
            tot = n_ov + n_rev * n_single
            diff = abs((n_ov / max(1, tot)) - args.overlap)
            if best_pair is None or diff < best_pair[0]:
                best_pair = (diff, n_ov, tot)
        _, exact_overlap, exact_total = best_pair
        print(f"🎯 Target {args.target_per_reviewer} tiles/reviewer ({n_rev} reviewers): "
              f"{exact_total} unique sharp tiles ({exact_overlap} shared overlap + "
              f"{args.target_per_reviewer - exact_overlap} unique each).")

    def append_slide_tiles(items_out, name, selection, reason, k):
        s = slides[name]
        for t in pick_tiles(s3, name, s["tiles"], k, rng, excluded_bases=excluded_bases):
            tile = s["tiles"][t["stem"]]
            m = s["meta"]
            items_out.append({
                "id": hashlib.sha1(tile["img"].encode()).hexdigest()[:10],
                "tile_key": tile["img"], "det_key": tile["det"],
                "slide_name": name,
                "slide_code": "S" + hashlib.sha1(name.encode()).hexdigest()[:5].upper(),
                "plant_id": m["sample_id"], "population": m["population"],
                "cytotype": m["cytotype"], "cultivation_run": m["cultivation_run"],
                "replicate": m["replicate"],
                "selection": selection, "flag_reason": reason,
                "density": t["density"], "n_pred": t["n_pred"],
            })

    # ── 4. tiles per slide ─────────────────────────────────────────────────────────────
    items = []
    for i, (name, selection, reason, k) in enumerate(selected, 1):
        s = slides[name]
        print(f"   [{i}/{len(selected)}] {selection:7s} {s['meta']['cytotype']:7s} {name}")
        append_slide_tiles(items, name, selection, reason, k)

    if exact_total is not None:
        # Top up from remaining stratified slides (always filling the least-represented cytotype first)
        while len(items) < exact_total:
            cyt_counts = {c: sum(1 for it in items if it["cytotype"] == c) for c in CYTOTYPES}
            avail_cyts = [c for c in CYTOTYPES if any(n not in taken for n in unused_by_cyt[c])]
            if not avail_cyts:
                break
            cyt = min(avail_cyts, key=lambda c: cyt_counts[c])
            next_name = next(n for n in unused_by_cyt[cyt] if n not in taken)
            taken.add(next_name)
            need_k = min(args.tiles_per_slide, exact_total - len(items))
            print(f"   [+topup] random  {cyt:7s} {next_name}")
            append_slide_tiles(items, next_name, "random", "", need_k)

        # Trim excess random items from the most-represented cytotype if over exact_total
        while len(items) > exact_total:
            cyt_counts = {c: sum(1 for it in items if it["cytotype"] == c) for c in CYTOTYPES}
            top_cyt = max(CYTOTYPES, key=lambda c: cyt_counts[c])
            drop_idx = next(
                (i for i in range(len(items) - 1, -1, -1)
                 if items[i]["cytotype"] == top_cyt and items[i]["selection"] == "random"),
                len(items) - 1,
            )
            items.pop(drop_idx)

    # ── 5. blind shuffle + assignment ──────────────────────────────────────────────────
    rng.shuffle(items)
    n_overlap = exact_overlap if exact_overlap is not None else int(round(args.overlap * len(items)))
    for j, it in enumerate(items):
        it["order"] = j + 1
        it["assigned"] = reviewers if j < n_overlap else [reviewers[(j - n_overlap) % len(reviewers)]]
    # Shuffle presentation order again so overlap items aren't clustered at 1..n_overlap
    rng.shuffle(items)
    for j, it in enumerate(items):
        it["order"] = j + 1

    set_id = f"{args.set_prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    manifest = {
        "set_id": set_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": DET_CACHE_TAG, "conf": CONF, "reviewers": reviewers,
        "overlap_items": n_overlap, "items": items,
    }
    out = Path("output/review"); out.mkdir(parents=True, exist_ok=True)
    local = out / f"manifest_{set_id}.json"
    local.write_text(json.dumps(manifest, indent=1))
    summary = defaultdict(int)
    for it in items:
        summary[(it["cytotype"], it["selection"])] += 1
    print(f"\n✅ {len(items)} tiles in set {set_id} ({n_overlap} reviewed by everyone)")
    for (c, sel), n in sorted(summary.items()):
        print(f"   {c:7s} {sel:7s} {n}")
    print(f"   local copy: {local}")
    if not args.dry_run:
        s3.put_object(Bucket=S3_BUCKET, Key=f"{REVIEW_PREFIX}/manifest.json",
                      Body=json.dumps(manifest).encode(), ContentType="application/json")
        print(f"   uploaded → s3://{S3_BUCKET}/{REVIEW_PREFIX}/manifest.json")


if __name__ == "__main__":
    main()
