#!/usr/bin/env python3
"""
qc_viability_samples.py — Automatic quality control of Pollen Viability slides (one scan = one sample),
adapted from Pollen_latitude/src/qc_samples.py.

Key differences from Pollen_latitude:
  1. Single species (Tripleurospermum inodorum) across 4 cytotypes (`nat_2x`, `ory_2x`, `neo_4x`, `nat_4x`)
     and cultivation runs (`run1_F2`, `run2_F2`, `run3_F2`). The natural stratum is `cytotype` (and
     `cytotype x cultivation_run`).
  2. 3-class detection (`viable`, `non_viable`, `intermediate`) instead of 1-class total count:
     * Primary biological trait 1: Pollen viability % = 100 * viable / (viable + non_viable + intermediate)
       and binary viability % = 100 * viable / (viable + non_viable).
     * Primary biological trait 2: Viable pollen grain diameter (`viable_median_diam_um`) for ploidy sizing.
  3. Class-specific size filter (`class_min_diam.csv`): because shrunken `non_viable` grains are much smaller
     than plump `viable` grains, the minimum diameter cutoff is applied per `(cytotype, class_name)`.
  4. Final Viability Correction (`final_viability.csv`): combines detection precision/recall with the
     human-measured 3-class reclassification bias (`viability_bias_pp` from `review_app`) and full-slide
     Slide Check verifications.

Hard exclusions (slide not usable at all):
  NO_METADATA       cytotype or sample_id missing
  NO_GRAINS         evaluated, but 0 grains detected
Viability:
  VIAB_UNRELIABLE   < MIN_GRAINS_RELIABLE (30) total counted grains (binomial SE > ~9 percentage points) -> not used
  VIAB_LOW_PREC     < MIN_GRAINS_GOOD (100) total counted grains (binomial SE 5-9 pp)                    -> used, flagged
  HIGH_INTERMEDIATE intermediate class > 15% of total grains                                             -> used, flagged
Size (viable grain diameter):
  SIZE_TOO_FEW      < MIN_SIZE_GRAINS (30) isolated, non-edge, non-clump viable grains                   -> not used
Sent to human review (still used until a reviewer rejects them):
  OUTLIER_VIABILITY viability % far from other slides of the cytotype (robust |z| > 3.0 or viability < 10%)
  OUTLIER_SIZE      median viable grain diameter far from other slides of the cytotype (robust |z| > 3.5)
  MANY_CLUMPS       clump / irregular grain share > 0.25 above the cytotype's typical share
  LOW_CONFIDENCE    detector unsure about most grains on the slide (median conf < 0.65)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

MIN_GRAINS_RELIABLE = 30
MIN_GRAINS_GOOD = 100
MIN_SIZE_GRAINS = 30
OUTLIER_VIAB_Z = 3.0
OUTLIER_SIZE_Z = 3.5
MIN_GROUP_FOR_OUTLIERS = 5
CLUMP_EXCESS = 0.25
LOW_CONF_MEDIAN = 0.65
HIGH_INTERMEDIATE_SHARE = 0.15
MIN_DETECTION_PRECISION = 0.90
MIN_DETECTION_RECALL = 0.90


def robust_z(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if mad > 1e-9:
        return (x - med) / mad
    return pd.Series(0.0, index=x.index)


def apply_class_filters(df_grains: pd.DataFrame, min_diam_path: str = None, conf_path: str = None):
    """Applies per-(cytotype, class_name) minimum diameter and confidence filters."""
    g = df_grains.copy()
    key_ser = g["cytotype"].astype(str) + "|" + g["class_name"].astype(str)

    min_diam_map = {}
    removed_small = pd.Series(0, index=g["slide_name"].unique())
    if min_diam_path and os.path.exists(min_diam_path) and os.path.getsize(min_diam_path) > 0:
        md = pd.read_csv(min_diam_path)
        md["key"] = md["cytotype"].astype(str) + "|" + md["class_name"].astype(str)
        md_idx = md.set_index("key")
        min_diam_map = md_idx["min_diam_um"].to_dict()
        thr = key_ser.map(min_diam_map)
        keep = thr.isna() | (g["diameter_um"] >= thr)
        removed_small = g[~keep].groupby("slide_name").size()
        print(f"[class size filter] {len(md)} (cytotype, class) rules, {int((~keep).sum())} small detections removed")
        g = g[keep].copy()
        key_ser = g["cytotype"].astype(str) + "|" + g["class_name"].astype(str)

    conf_map = {}
    removed_conf = pd.Series(0, index=g["slide_name"].unique())
    if conf_path and os.path.exists(conf_path) and os.path.getsize(conf_path) > 0:
        ct = pd.read_csv(conf_path)
        ct["key"] = ct["cytotype"].astype(str) + "|" + ct["class_name"].astype(str)
        conf_map = ct.set_index("key")["conf_threshold"].to_dict()
        thr_c = key_ser.map(conf_map)
        keep_c = thr_c.isna() | (g["conf"] >= thr_c)
        removed_conf = g[~keep_c].groupby("slide_name").size()
        print(f"[class conf filter] {len(ct)} (cytotype, class) rules, {int((~keep_c).sum())} low-conf detections removed")
        g = g[keep_c].copy()

    return g, min_diam_map, removed_small, conf_map, removed_conf


def per_slide_metrics(g: pd.DataFrame) -> pd.DataFrame:
    """Recomputes per-slide counts, viability %, clump share, confidence, and size metrics from filtered grains."""
    counts = g.groupby(["slide_name", "class_name"]).size().unstack(fill_value=0)
    for c in ("viable", "non_viable", "intermediate"):
        if c not in counts.columns:
            counts[c] = 0

    total = counts["viable"] + counts["non_viable"] + counts["intermediate"]
    denom_bin = counts["viable"] + counts["non_viable"]

    gen = g.groupby("slide_name").agg(
        conf_median=("conf", "median"),
        clump_share=("is_clump", "mean"),
        edge_share=("is_outer_edge", "mean"),
    )

    v_size = g[(g["class_name"] == "viable") & (g["valid_for_size"] == 1)].groupby("slide_name")["diameter_um"].agg(
        viable_valid_size="size",
        viable_median_diam_um="median",
        viable_mean_diam_um="mean",
        viable_q25_diam_um=lambda s: s.quantile(0.25),
        viable_q75_diam_um=lambda s: s.quantile(0.75),
    )

    nv_size = g[(g["class_name"] == "non_viable") & (g["valid_for_size"] == 1)].groupby("slide_name")["diameter_um"].agg(
        non_viable_valid_size="size",
        non_viable_median_diam_um="median",
    )

    out = pd.DataFrame({
        "total_grains": total,
        "viable_count": counts["viable"],
        "non_viable_count": counts["non_viable"],
        "intermediate_count": counts["intermediate"],
        "viability_pct": np.where(total > 0, 100.0 * counts["viable"] / total, np.nan).round(2),
        "viab_ratio_binary": np.where(denom_bin > 0, 100.0 * counts["viable"] / denom_bin, np.nan).round(2),
        "intermediate_share": np.where(total > 0, counts["intermediate"] / total, 0.0).round(4),
    })
    out = out.join(gen, how="left").join(v_size, how="left").join(nv_size, how="left")
    out["viable_valid_size"] = out["viable_valid_size"].fillna(0).astype(int)
    out["non_viable_valid_size"] = out["non_viable_valid_size"].fillna(0).astype(int)
    return out


def load_review_quality_viability(review_csv: str = None, min_diam_path: str = None, conf_path: str = None):
    """
    Loads per-cytotype precision, recall, 3-class accuracy, and viability bias (in percentage points)
    from the review app's `/api/export.csv` or S3 decisions, filtering out blurred / image_issue tiles
    and applying the calibrated per-(cytotype, class) size and confidence filter.
    """
    if review_csv and os.path.exists(review_csv) and os.path.getsize(review_csv) > 0:
        rv = pd.read_csv(review_csv)
        if "image_issue" in rv.columns:
            rv = rv[~rv["image_issue"].astype(bool)].copy()
    else:
        # Try fetching from S3 decisions if credentials exist and evaluate with calibrated filter
        try:
            import math
            from pollen_size_analyzer import make_s3, S3_BUCKET, DEFAULT_UM_PER_PX
            s3 = make_s3()
            md = pd.read_csv(min_diam_path).set_index(["cytotype", "class_name"])["min_diam_um"].to_dict() if (min_diam_path and os.path.exists(min_diam_path)) else {}
            ct = pd.read_csv(conf_path).set_index(["cytotype", "class_name"])["conf_threshold"].to_dict() if (conf_path and os.path.exists(conf_path)) else {}
            CLS_MAP = {0: "viable", 1: "non_viable", 2: "intermediate"}
            RECLASS_MAP = {"viable": 0, "to_viable": 0, "non_viable": 1, "to_non_viable": 1, "intermediate": 2, "to_intermediate": 2}

            pfx = "Ostatni/Pollen_viability/Review"
            m = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=f"{pfx}/manifest.json")["Body"].read())
            items_by_id = {it["id"]: it for it in m.get("items", [])}
            set_id = m.get("set_id")
            rows = []
            if set_id:
                dec_pfx = f"{pfx}/decisions/{set_id}/"
                for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=dec_pfx):
                    for o in page.get("Contents", []):
                        if not o["Key"].endswith(".json"):
                            continue
                        d = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=o["Key"])["Body"].read())
                        if d.get("image_issue"):
                            continue
                        it = items_by_id.get(d.get("item_id"))
                        if not it:
                            continue
                        cyt = it["cytotype"]
                        det = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=it["det_key"])["Body"].read())
                        boxes = det.get("boxes", [])
                        grains_override = d.get("grains", {})
                        missed_list = d.get("missed", [])

                        n_pred = 0
                        fp = 0
                        n_rec = 0
                        pred_v = 0
                        corr_v = 0
                        n_true = 0

                        for idx, b in enumerate(boxes):
                            x1, y1, x2, y2, conf, cls_id = b[0], b[1], b[2], b[3], float(b[4]), int(b[5])
                            cls_name = CLS_MAP.get(cls_id, "viable")
                            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
                            diam_um = 2.0 * math.sqrt((w * h * math.pi / 4.0) * (DEFAULT_UM_PER_PX ** 2) / math.pi)
                            edge = (x1 <= 8 or y1 <= 8 or x2 >= 632 or y2 >= 632)
                            min_d = md.get((cyt, cls_name), 10.0)
                            min_c = ct.get((cyt, cls_name), 0.65)
                            if edge or diam_um < min_d or conf < min_c:
                                continue

                            raw_act = str(grains_override.get(str(idx), "ok"))
                            act = raw_act[:-9] if raw_act.endswith("+bad_poly") else ("ok" if raw_act == "bad_poly" else raw_act)
                            if act in ("blurred", "uncertain"):
                                continue
                            n_pred += 1
                            if cls_id == 0:
                                pred_v += 1
                            if act == "fp":
                                fp += 1
                            else:
                                n_true += 1
                                final_cid = RECLASS_MAP.get(act, cls_id)
                                if final_cid != cls_id:
                                    n_rec += 1
                                if final_cid == 0:
                                    corr_v += 1

                        n_miss = len(missed_list)
                        for ms in missed_list:
                            n_true += 1
                            if RECLASS_MAP.get(ms.get("cls", "non_viable"), 1) == 0:
                                corr_v += 1

                        pv = 100.0 * pred_v / n_pred if n_pred else np.nan
                        cv = 100.0 * corr_v / n_true if n_true else np.nan
                        rows.append({
                            "cytotype": cyt,
                            "slide_name": it["slide_name"],
                            "n_pred": n_pred,
                            "n_true": n_true,
                            "fp": fp,
                            "missed": n_miss,
                            "reclassified": n_rec,
                            "viability_bias_pp": pv - cv if pd.notna(pv) and pd.notna(cv) else np.nan,
                        })
            rv = pd.DataFrame(rows)
        except Exception as e:
            print(f"⚠️ Failed loading S3 review quality: {e}")
            rv = pd.DataFrame()

    if rv.empty:
        return pd.DataFrame()

    out_rows = []
    for cyt, g in rv.groupby("cytotype"):
        n_pred = g["n_pred"].sum()
        fp = g["fp"].sum()
        tp = n_pred - fp
        miss = g["missed"].sum()
        recl = g["reclassified"].sum()
        prec = tp / n_pred if n_pred > 0 else np.nan
        rec = tp / (tp + miss) if (tp + miss) > 0 else np.nan
        cls_acc = (tp - recl) / tp if tp > 0 else np.nan
        bias_pp = g["viability_bias_pp"].dropna().mean() if "viability_bias_pp" in g and g["viability_bias_pp"].notna().any() else 0.0
        out_rows.append({
            "cytotype": cyt,
            "Det_tiles_checked": len(g),
            "Det_precision": round(float(prec), 3) if pd.notna(prec) else np.nan,
            "Det_recall": round(float(rec), 3) if pd.notna(rec) else np.nan,
            "Det_class_accuracy": round(float(cls_acc), 3) if pd.notna(cls_acc) else np.nan,
            "Det_viability_bias_pp": round(float(bias_pp), 3),
        })
    return pd.DataFrame(out_rows).set_index("cytotype")


def write_final_viability(out_df: pd.DataFrame, det_quality: pd.DataFrame, folder: str, verified_slides: dict = None):
    """
    final_viability.csv: per slide the detected counts & viability %, the cytotype's measured
    detection quality (precision, recall, class accuracy, viability_bias_pp), and the final
    corrected counts and viability %.
    """
    f = out_df.copy()
    if not det_quality.empty:
        f = f.merge(det_quality, left_on="cytotype", right_index=True, how="left")
    else:
        for col in ("Det_tiles_checked", "Det_precision", "Det_recall", "Det_class_accuracy", "Det_viability_bias_pp"):
            f[col] = np.nan

    p = f["Det_precision"]
    r = f["Det_recall"].fillna(1.0)
    ca = f["Det_class_accuracy"].fillna(1.0)
    lower = np.minimum(np.minimum(p, r), ca)

    f["Detection_tier"] = np.select(
        [p.isna(), lower >= MIN_DETECTION_PRECISION, lower >= MIN_DETECTION_PRECISION - 0.03, lower >= 0.80],
        ["unchecked", "ok", "ok_within_uncertainty", "close"],
        default="low"
    )
    corrected = f["Detection_tier"].isin(["ok_within_uncertainty", "close", "low"]) & (r > 0)
    f["Count_detected"] = f["total_grains"]
    f["Count_final"] = np.where(corrected, f["total_grains"] * p / r.where(r > 0, 1.0), f["total_grains"]).round(1)

    # Viability % correction: subtract measured model viability bias (Model - True = bias -> True = Model - bias)
    f["Viability_pct_detected"] = f["viability_pct"]
    f["Viability_pct_final"] = np.where(
        corrected & f["Det_viability_bias_pp"].notna(),
        np.clip(f["viability_pct"] - f["Det_viability_bias_pp"], 0.0, 100.0),
        f["viability_pct"]
    ).round(2)
    f["Viability_method"] = np.where(corrected, "corrected (confusion bias + precision/recall)", "detected")

    if verified_slides:
        for sl_name, v_info in verified_slides.items():
            mask = f["slide_name"] == sl_name
            if mask.any():
                f.loc[mask, "Count_final"] = v_info["total_true"]
                f.loc[mask, "Viability_pct_final"] = round(v_info["viability_pct_true"], 2)
                f.loc[mask, "Viability_method"] = "verified (all slide detections checked)"

    cols = [
        "slide_name", "plant_id", "cultivation_number", "population", "accession",
        "cytotype", "ploidy_str", "numeric_ploidy", "cultivation_run", "replicate",
        "Detection_model", "Count_detected", "n_removed_small", "n_removed_conf",
        "viable_count", "non_viable_count", "intermediate_count",
        "Det_precision", "Det_recall", "Det_class_accuracy", "Det_viability_bias_pp",
        "Detection_tier", "Viability_method", "Count_final",
        "Viability_pct_detected", "Viability_pct_final", "viab_ratio_binary",
        "viable_valid_size", "viable_median_diam_um", "non_viable_valid_size", "non_viable_median_diam_um",
        "QC_status", "QC_flags", "Use_viability", "Use_size"
    ]
    f[[c for c in cols if c in f.columns]].to_csv(f"{folder}/final_viability.csv", index=False)
    print("final viability tiers:", f["Detection_tier"].value_counts().to_dict())
    return f


def main():
    ap = argparse.ArgumentParser(description="Automatic QC of Pollen Viability slides")
    ap.add_argument("--grains", default="output/size_analysis/pollen_grain_measurements.csv")
    ap.add_argument("--summaries", default="output/size_analysis/pollen_slide_summaries.csv")
    ap.add_argument("--min-diam", default="output/qc/class_min_diam.csv")
    ap.add_argument("--conf-threshold", default="output/qc/class_conf_threshold.csv")
    ap.add_argument("--review-csv", default=None, help="Optional export.csv from Pollen Viability review_app")
    ap.add_argument("--out", default="output/qc")
    args = ap.parse_args()

    print(f"📖 Loading slide summaries from {args.summaries} and grains from {args.grains}...")
    meta_df = pd.read_csv(args.summaries, low_memory=False)
    grains_raw = pd.read_csv(args.grains, low_memory=False)

    grains, min_diam_map, removed_small, conf_map, removed_conf = apply_class_filters(
        grains_raw, args.min_diam, args.conf_threshold
    )
    slide_stats = per_slide_metrics(grains)

    meta_cols = [
        "slide_name", "plant_id", "cultivation_number", "population", "accession",
        "cytotype", "ploidy_str", "numeric_ploidy", "cultivation_run", "replicate"
    ]
    base = meta_df[[c for c in meta_cols if c in meta_df.columns]].drop_duplicates("slide_name").set_index("slide_name")
    s = base.join(slide_stats, how="left")

    for col in ("total_grains", "viable_count", "non_viable_count", "intermediate_count", "viable_valid_size", "non_viable_valid_size"):
        s[col] = s[col].fillna(0).astype(int)

    s["n_removed_small"] = s.index.map(removed_small).fillna(0).astype(int)
    s["n_removed_conf"] = s.index.map(removed_conf).fillna(0).astype(int)
    s["Detection_model"] = "model_20261122_r3"

    # Binomial standard error of viability % (in percentage points)
    p_hat = (s["viability_pct"] / 100.0).clip(0.001, 0.999)
    s["Viability_SE_pp"] = np.where(
        s["total_grains"] > 0,
        100.0 * np.sqrt(p_hat * (1.0 - p_hat) / s["total_grains"].clip(lower=1)),
        np.nan
    ).round(2)

    # Exclude slides where reviewed tiles in blurred_tiles_filter.csv are predominantly blurred
    blur_csv = os.path.join(args.out, "blurred_tiles_filter.csv")
    blurred_slides = set()
    if os.path.exists(blur_csv):
        try:
            bdf = pd.read_csv(blur_csv)
            if "is_blurred_filtered" in bdf.columns and "slide_name" in bdf.columns:
                sl_blur = bdf.groupby("slide_name")["is_blurred_filtered"].agg(["count", "mean"])
                blurred_slides = set(sl_blur[(sl_blur["mean"] >= 0.5) & (sl_blur["count"] >= 1)].index)
        except Exception:
            pass

    flags = {k: pd.Series(False, index=s.index) for k in [
        "NO_METADATA", "NO_GRAINS", "BLURRED_IMAGE",
        "VIAB_UNRELIABLE", "VIAB_LOW_PREC", "HIGH_INTERMEDIATE",
        "SIZE_TOO_FEW", "OUTLIER_VIABILITY", "OUTLIER_SIZE",
        "MANY_CLUMPS", "LOW_CONFIDENCE"
    ]}

    flags["NO_METADATA"] = s["cytotype"].isna() | (s["cytotype"].astype(str) == "unknown")
    flags["NO_GRAINS"] = s["total_grains"] == 0
    flags["BLURRED_IMAGE"] = s.index.isin(blurred_slides)
    flags["VIAB_UNRELIABLE"] = (s["total_grains"] > 0) & (s["total_grains"] < MIN_GRAINS_RELIABLE)
    flags["VIAB_LOW_PREC"] = (s["total_grains"] >= MIN_GRAINS_RELIABLE) & (s["total_grains"] < MIN_GRAINS_GOOD)
    flags["HIGH_INTERMEDIATE"] = (s["total_grains"] >= MIN_GRAINS_RELIABLE) & (s["intermediate_share"] > HIGH_INTERMEDIATE_SHARE)
    flags["SIZE_TOO_FEW"] = (s["total_grains"] > 0) & (s["viable_valid_size"] < MIN_SIZE_GRAINS)

    typical_clump = s[s["total_grains"] >= MIN_GRAINS_RELIABLE].groupby("cytotype")["clump_share"].median()
    flags["MANY_CLUMPS"] = (
        (s["total_grains"] >= MIN_GRAINS_RELIABLE) &
        ((s["clump_share"] - s["cytotype"].map(typical_clump)) > CLUMP_EXCESS)
    )
    flags["LOW_CONFIDENCE"] = (s["total_grains"] > 0) & (s["conf_median"] < LOW_CONF_MEDIAN)

    hard = flags["NO_METADATA"] | flags["NO_GRAINS"] | flags["BLURRED_IMAGE"]
    s["Use_viability"] = (~hard) & (~flags["VIAB_UNRELIABLE"]) & (s["total_grains"] > 0)
    s["Use_size"] = (~hard) & (~flags["SIZE_TOO_FEW"]) & (s["viable_valid_size"] > 0)

    # Outliers within cytotype x cultivation_run (fallback to cytotype)
    s["z_viability"], s["z_size"] = np.nan, np.nan
    for col, zcol, use_col in [
        ("viability_pct", "z_viability", "Use_viability"),
        ("viable_median_diam_um", "z_size", "Use_size"),
    ]:
        val = s[col].where(s[use_col] & s[col].notna())
        for keys in (["cytotype", "cultivation_run"], ["cytotype"]):
            todo = s[zcol].isna() & val.notna()
            if not todo.any():
                continue
            grp = s[todo].groupby(keys)
            sizes = grp[col].transform("size")
            ok = sizes[sizes >= MIN_GROUP_FOR_OUTLIERS].index
            if len(ok):
                s.loc[ok, zcol] = val.loc[ok].groupby([s.loc[ok, k] for k in keys]).transform(robust_z)

    flags["OUTLIER_VIABILITY"] = (s["z_viability"].abs() > OUTLIER_VIAB_Z) | (s["Use_viability"] & (s["viability_pct"] < 10.0))
    flags["OUTLIER_SIZE"] = s["z_size"].abs() > OUTLIER_SIZE_Z

    s["QC_flags"] = pd.DataFrame(flags).apply(lambda r: ";".join(k for k, v in r.items() if v), axis=1)
    review = flags["OUTLIER_VIABILITY"] | flags["OUTLIER_SIZE"] | flags["MANY_CLUMPS"] | flags["LOW_CONFIDENCE"] | flags["HIGH_INTERMEDIATE"]
    s["QC_status"] = np.select([hard, review], ["exclude", "review"], default="pass")

    os.makedirs(args.out, exist_ok=True)
    out = s.rename_axis("slide_name").reset_index()
    out.to_csv(f"{args.out}/qc_viability_samples.csv", index=False)
    out[out["QC_status"] == "review"].sort_values(["cytotype", "cultivation_run", "plant_id"]).to_csv(
        f"{args.out}/qc_viability_review_queue.csv", index=False
    )

    det_quality = load_review_quality_viability(args.review_csv, args.min_diam, args.conf_threshold)
    final_df = write_final_viability(out, det_quality, args.out)

    # Cytotype summary
    def summarise_cyt(d: pd.DataFrame) -> pd.Series:
        v = d[d["Use_viability"]]
        z = d[d["Use_size"]]
        return pd.Series({
            "n_slides": len(d),
            "n_slides_viability": len(v),
            "n_slides_size": len(z),
            "n_review": int((d["QC_status"] == "review").sum()),
            "n_exclude": int((d["QC_status"] == "exclude").sum()),
            "grains_total": int(d["total_grains"].sum()),
            "Viability_detected_median_pct": round(float(v["Viability_pct_detected"].median()), 2) if len(v) else np.nan,
            "Viability_detected_mean_pct": round(float(v["Viability_pct_detected"].mean()), 2) if len(v) else np.nan,
            "Viability_final_median_pct": round(float(v["Viability_pct_final"].median()), 2) if len(v) else np.nan,
            "Viability_final_mean_pct": round(float(v["Viability_pct_final"].mean()), 2) if len(v) else np.nan,
            "Viable_diam_median_um": round(float(z["viable_median_diam_um"].median()), 2) if len(z) else np.nan,
            "Non_viable_diam_median_um": round(float(z["non_viable_median_diam_um"].dropna().median()), 2) if z["non_viable_median_diam_um"].notna().any() else np.nan,
            "size_grains_total": int(z["viable_valid_size"].sum()),
        })

    for name, keys in [("cytotype_run", ["cytotype", "cultivation_run"]), ("cytotype", ["cytotype"])]:
        sum_df = final_df.groupby(keys).apply(summarise_cyt, include_groups=False).reset_index()
        if not det_quality.empty:
            sum_df = sum_df.merge(det_quality, left_on="cytotype", right_index=True, how="left")
        sum_df.to_csv(f"{args.out}/qc_summary_{name}.csv", index=False)

    print("\n=== Pollen Viability Slide QC Summary ===")
    print(out["QC_status"].value_counts().to_string())
    print("\nFlag counts:")
    print(pd.DataFrame(flags).sum().sort_values(ascending=False).to_string())
    print(f"\nUsable slides for viability: {out['Use_viability'].sum()}/{len(out)} | for size: {out['Use_size'].sum()}/{len(out)}")
    print(f"✅ Saved Pollen Viability QC tables to {args.out}/")


if __name__ == "__main__":
    main()

