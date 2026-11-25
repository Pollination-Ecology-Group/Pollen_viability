#!/usr/bin/env python3
"""
compare_models_viability.py — Old vs new 3-class segmentation model & calibrated pipeline comparison
per cytotype for Pollen Viability, including human-confirmed Ground Truth precision and recall on
`viable` and `non_viable` pollen across all 4 cytotypes (`nat_2x`, `ory_2x`, `neo_4x`, `nat_4x`),
with blurred / out-of-focus tiles filtered out.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import boto3
import cv2
import numpy as np
import pandas as pd
from botocore.client import Config

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "1Y920BKC0SAWPNDE8RD6")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD")
SCALE_UM = 0.8772419513050967
MIN_BOX_LAPLACIAN_VAR = 12.0


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def evaluate_human_confirmed_qc(min_diam_path: str, conf_path: str, out_dir: str, run: str, model_weights: str = None):
    """
    Evaluates human-confirmed precision and recall separately for `viable` and `non_viable` pollen
    across all non-blurred human-reviewed QC tiles (filtering out `image_issue == True` or
    out-of-focus tiles with grain-local Laplacian variance < 12.0).
    If `model_weights` is provided and exists locally, runs live YOLO inference on the reviewed tiles
    to evaluate the newly retrained model directly against human ground truth!
    """
    s3 = get_s3()
    md = pd.read_csv(min_diam_path).set_index(["cytotype", "class_name"])["min_diam_um"].to_dict() if os.path.exists(min_diam_path) else {}
    ct = pd.read_csv(conf_path).set_index(["cytotype", "class_name"])["conf_threshold"].to_dict() if os.path.exists(conf_path) else {}

    manifest = json.loads(s3.get_object(Bucket=S3_BUCKET, Key="Ostatni/Pollen_viability/Review/manifest.json")["Body"].read())
    m_by_id = {m["id"]: m for m in manifest.get("items", [])}
    set_id = manifest.get("set_id", "final_20261026_1614")

    dec_keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=f"Ostatni/Pollen_viability/Review/decisions/{set_id}/"):
        for o in page.get("Contents", []):
            if o["Key"].endswith(".json"):
                dec_keys.append(o["Key"])

    yolo_model = None
    if model_weights and os.path.exists(model_weights):
        from ultralytics import YOLO
        print(f"🔬 Loading YOLO model weights from {model_weights} for live QC evaluation...")
        yolo_model = YOLO(model_weights)

    CLS_MAP = {0: "viable", 1: "non_viable", 2: "intermediate"}
    RECLASS_MAP = {"viable": 0, "to_viable": 0, "non_viable": 1, "to_non_viable": 1, "intermediate": 2, "to_intermediate": 2}

    stats_raw = defaultdict(lambda: {"pred_v": 0, "tp_v": 0, "fp_v": 0, "fn_v": 0, "pred_nv": 0, "tp_nv": 0, "fp_nv": 0, "fn_nv": 0, "tiles": 0, "blurred_filtered": 0})
    stats_cal = defaultdict(lambda: {"pred_v": 0, "tp_v": 0, "fp_v": 0, "fn_v": 0, "pred_nv": 0, "tp_nv": 0, "fp_nv": 0, "fn_nv": 0, "tiles": 0, "blurred_filtered": 0})

    for dk in sorted(dec_keys):
        d = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=dk)["Body"].read())
        tid = d.get("item_id")
        m = m_by_id.get(tid)
        if not m:
            continue
        cyt = m.get("cytotype", "unknown")
        verdict = d.get("verdict", "good")
        image_issue = bool(d.get("image_issue", False))

        det = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=m["det_key"])["Body"].read())
        orig_boxes = det.get("boxes", [])

        # Check blur
        is_blurred = image_issue
        img_bgr = None
        if not is_blurred and ((verdict == "poor" and len(orig_boxes) > 1) or yolo_model is not None):
            img_bytes = s3.get_object(Bucket=S3_BUCKET, Key=m["tile_key"])["Body"].read()
            img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if not is_blurred and verdict == "poor" and len(orig_boxes) > 1:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
                blaps = [float(np.var(lap[max(0,int(b[1])-4):min(640,int(b[3])+4), max(0,int(b[0])-4):min(640,int(b[2])+4)])) for b in orig_boxes if int(b[2])>int(b[0]) and int(b[3])>int(b[1])]
                if blaps and float(np.median(blaps)) < MIN_BOX_LAPLACIAN_VAR:
                    is_blurred = True

        if is_blurred:
            for k in (cyt, "ALL"):
                stats_raw[k]["blurred_filtered"] += 1
                stats_cal[k]["blurred_filtered"] += 1
            continue

        for k in (cyt, "ALL"):
            stats_raw[k]["tiles"] += 1
            stats_cal[k]["tiles"] += 1

        grains_override = d.get("grains", {})
        missed_list = d.get("missed", [])

        # Build human Ground Truth boxes on this tile from orig_boxes + overrides + missed
        gt_items = []  # (cx, cy, r, true_cls_id, orig_idx, is_edge, diam_um, orig_conf)
        for idx, b in enumerate(orig_boxes):
            x1, y1, x2, y2, conf, cls_id = b[0], b[1], b[2], b[3], float(b[4]), int(b[5])
            act = grains_override.get(str(idx), "ok")
            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
            diam_um = 2.0 * math.sqrt((w * h * math.pi / 4.0) * (SCALE_UM ** 2) / math.pi)
            edge = (x1 <= 8 or y1 <= 8 or x2 >= 632 or y2 >= 632)

            # Raw evaluation of baseline predictions
            for k in (cyt, "ALL"):
                if cls_id == 0:
                    stats_raw[k]["pred_v"] += 1
                    if act in ("ok", "to_viable"):
                        stats_raw[k]["tp_v"] += 1
                    else:
                        stats_raw[k]["fp_v"] += 1
                elif cls_id == 1:
                    stats_raw[k]["pred_nv"] += 1
                    if act in ("ok", "to_non_viable"):
                        stats_raw[k]["tp_nv"] += 1
                    else:
                        stats_raw[k]["fp_nv"] += 1

            if act != "fp":
                true_cid = RECLASS_MAP.get(act, cls_id)
                gt_items.append((0.5*(x1+x2), 0.5*(y1+y2), 0.25*(w+h), true_cid, idx, edge, diam_um, conf))

        for ms in missed_list:
            mcid = RECLASS_MAP.get(ms.get("cls", "non_viable"), 1)
            mx, my = float(ms.get("x", 320.0)), float(ms.get("y", 320.0))
            edge = (mx <= 12 or my <= 12 or mx >= 628 or my >= 628)
            for k in (cyt, "ALL"):
                if mcid == 0:
                    stats_raw[k]["fn_v"] += 1
                elif mcid == 1:
                    stats_raw[k]["fn_nv"] += 1
            gt_items.append((mx, my, 10.0, mcid, -1, edge, 14.0 if mcid == 1 else 20.0, 1.0))

        if yolo_model is not None and img_bgr is not None:
            # Evaluate live YOLO model predictions + calibrated filter against gt_items
            res = yolo_model.predict(img_bgr, conf=0.45, iou=0.45, agnostic_nms=True, verbose=False)[0]
            p_boxes = res.boxes.data.cpu().numpy() if res.boxes is not None else np.zeros((0, 6))
            matched_gt = set()
            for pb in p_boxes:
                x1, y1, x2, y2, conf, cls_id = float(pb[0]), float(pb[1]), float(pb[2]), float(pb[3]), float(pb[4]), int(pb[5])
                cls_name = CLS_MAP.get(cls_id, "viable")
                w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
                diam_um = 2.0 * math.sqrt((w * h * math.pi / 4.0) * (SCALE_UM ** 2) / math.pi)
                edge = (x1 <= 8 or y1 <= 8 or x2 >= 632 or y2 >= 632)
                min_d = md.get((cyt, cls_name), 10.0)
                min_c = ct.get((cyt, cls_name), 0.65)
                if edge or diam_um < min_d or conf < min_c:
                    continue
                px, py = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                # Match to nearest GT within 18 px
                best_g = None
                best_dist = 18.0
                for g_i, g in enumerate(gt_items):
                    if g_i in matched_gt:
                        continue
                    dist = math.hypot(px - g[0], py - g[1])
                    if dist <= best_dist:
                        best_dist = dist
                        best_g = g_i
                is_tp = False
                if best_g is not None:
                    matched_gt.add(best_g)
                    if gt_items[best_g][3] == cls_id:
                        is_tp = True
                for k in (cyt, "ALL"):
                    if cls_id == 0:
                        stats_cal[k]["pred_v"] += 1
                        if is_tp:
                            stats_cal[k]["tp_v"] += 1
                        else:
                            stats_cal[k]["fp_v"] += 1
                    elif cls_id == 1:
                        stats_cal[k]["pred_nv"] += 1
                        if is_tp:
                            stats_cal[k]["tp_nv"] += 1
                        else:
                            stats_cal[k]["fp_nv"] += 1
            for g_i, g in enumerate(gt_items):
                if g_i not in matched_gt and not g[5]:
                    for k in (cyt, "ALL"):
                        if g[3] == 0:
                            stats_cal[k]["fn_v"] += 1
                        elif g[3] == 1:
                            stats_cal[k]["fn_nv"] += 1
        else:
            # Evaluate calibrated filter on stored detections
            for idx, b in enumerate(orig_boxes):
                x1, y1, x2, y2, conf, cls_id = b[0], b[1], b[2], b[3], float(b[4]), int(b[5])
                cls_name = CLS_MAP.get(cls_id, "viable")
                w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
                diam_um = 2.0 * math.sqrt((w * h * math.pi / 4.0) * (SCALE_UM ** 2) / math.pi)
                edge = (x1 <= 8 or y1 <= 8 or x2 >= 632 or y2 >= 632)
                min_d = md.get((cyt, cls_name), 10.0)
                min_c = ct.get((cyt, cls_name), 0.65)
                if edge or diam_um < min_d or conf < min_c:
                    continue
                act = grains_override.get(str(idx), "ok")
                for k in (cyt, "ALL"):
                    if cls_id == 0:
                        stats_cal[k]["pred_v"] += 1
                        if act in ("ok", "to_viable"):
                            stats_cal[k]["tp_v"] += 1
                        else:
                            stats_cal[k]["fp_v"] += 1
                    elif cls_id == 1:
                        stats_cal[k]["pred_nv"] += 1
                        if act in ("ok", "to_non_viable"):
                            stats_cal[k]["tp_nv"] += 1
                        else:
                            stats_cal[k]["fp_nv"] += 1
            for ms in missed_list:
                mcid = RECLASS_MAP.get(ms.get("cls", "non_viable"), 1)
                for k in (cyt, "ALL"):
                    if mcid == 0:
                        stats_cal[k]["fn_v"] += 1
                    elif mcid == 1:
                        stats_cal[k]["fn_nv"] += 1

    qc_rows = []
    for k in ["nat_2x", "ory_2x", "nat_4x", "neo_4x", "ALL"]:
        sr = stats_raw[k]
        sc = stats_cal[k]
        raw_pv = 100.0 * sr["tp_v"] / max(1, sr["pred_v"])
        raw_pnv = 100.0 * sr["tp_nv"] / max(1, sr["pred_nv"])
        cal_pv = 100.0 * sc["tp_v"] / max(1, sc["pred_v"])
        cal_pnv = 100.0 * sc["tp_nv"] / max(1, sc["pred_nv"])
        cal_rv = 100.0 * sc["tp_v"] / max(1, sc["tp_v"] + sc["fn_v"])
        cal_rnv = 100.0 * sc["tp_nv"] / max(1, sc["tp_nv"] + sc["fn_nv"])
        qc_rows.append({
            "run": run,
            "cytotype": k,
            "clean_tiles": sc["tiles"],
            "blurred_filtered": sc["blurred_filtered"],
            "raw_viable_prec_pct": round(raw_pv, 2),
            "cal_viable_prec_pct": round(cal_pv, 2),
            "cal_viable_rec_pct": round(cal_rv, 2),
            "cal_viable_tp_fp": f"{sc['tp_v']}/{sc['pred_v']} (FP={sc['fp_v']})",
            "raw_non_viable_prec_pct": round(raw_pnv, 2),
            "cal_non_viable_prec_pct": round(cal_pnv, 2),
            "cal_non_viable_rec_pct": round(cal_rnv, 2),
            "cal_non_viable_tp_fp": f"{sc['tp_nv']}/{sc['pred_nv']} (FP={sc['fp_nv']})",
        })

    qc_df = pd.DataFrame(qc_rows)
    os.makedirs(out_dir, exist_ok=True)
    qc_path = f"{out_dir}/human_confirmed_precision_{run}.csv"
    qc_df.to_csv(qc_path, index=False)
    print(f"\n=== Human-Confirmed Precision & Recall (Blurred Filtered Out) [{run}] ===")
    print(qc_df.to_string(index=False))
    print(f"✅ Saved {qc_path}")
    return qc_df


def evaluate_filter_effect(grains_path: str, min_diam_path: str, conf_path: str, out_dir: str, run: str):
    """
    Compares baseline evaluation (`old`: flat conf >= 0.55, no class-specific min diameter) against
    the calibrated pipeline (`new`: class-specific min_diam_um + calibrated confidence threshold)
    across all 4 cytotypes.
    """
    df = pd.read_csv(grains_path, low_memory=False)
    md = pd.read_csv(min_diam_path).set_index(["cytotype", "class_name"])["min_diam_um"].to_dict() if os.path.exists(min_diam_path) else {}
    ct = pd.read_csv(conf_path).set_index(["cytotype", "class_name"])["conf_threshold"].to_dict() if os.path.exists(conf_path) else {}

    # Reference valid grains vs debris/clump artifacts
    df["is_valid_grain"] = (df["is_outer_edge"] == 0) & (df["solidity"] >= 0.88) & (df["aspect_ratio"] <= 1.40)
    df["min_d"] = [md.get((c, cls), 0.0) for c, cls in zip(df["cytotype"], df["class_name"])]
    df["min_c"] = [ct.get((c, cls), 0.55) for c, cls in zip(df["cytotype"], df["class_name"])]
    df["keep_new"] = (df["diameter_um"] >= df["min_d"]) & (df["conf"] >= df["min_c"])

    rows = []
    for cyt, g in sorted(df.groupby("cytotype")):
        if cyt == "unknown":
            continue
        n_old = len(g)
        tp_old = int(g["is_valid_grain"].sum())
        g_new = g[g["keep_new"]]
        n_new = len(g_new)
        tp_new = int(g_new["is_valid_grain"].sum())

        old_prec = tp_old / max(1, n_old)
        new_prec = tp_new / max(1, n_new)
        old_rec = 1.0
        new_rec = tp_new / max(1, tp_old)

        old_viab = 100.0 * (g["class_name"] == "viable").sum() / max(1, n_old)
        new_viab = 100.0 * (g_new["class_name"] == "viable").sum() / max(1, n_new)
        ref_viab = 100.0 * ((g["class_name"] == "viable") & g["is_valid_grain"]).sum() / max(1, tp_old)

        old_bias = old_viab - ref_viab
        new_bias = new_viab - ref_viab

        rows.append({
            "run": run,
            "cytotype": cyt,
            "n_slides": g["slide_name"].nunique(),
            "old_grains": n_old,
            "new_grains": n_new,
            "removed_artifacts": n_old - n_new,
            "old_precision_est": round(old_prec, 4),
            "new_precision_est": round(new_prec, 4),
            "new_recall_vs_valid": round(new_rec, 4),
            "old_viability_pct": round(old_viab, 2),
            "new_viability_pct": round(new_viab, 2),
            "ref_clean_viability_pct": round(ref_viab, 2),
            "old_viability_bias_pp": round(old_bias, 3),
            "new_viability_bias_pp": round(new_bias, 3),
            "decision": "adopt calibrated filter" if (new_prec > old_prec and new_rec >= 0.93) else "keep baseline",
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/model_compare_{run}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"=== Pollen Viability Slide-Level Filter Comparison ({run}) ===")
    print(out_df.to_string(index=False))
    print(f"\n✅ Saved {out_path}")


def evaluate_end_to_end_qc_benchmark(model_weights: str, out_dir: str, run: str, conf_thr: float = 0.55):
    """
    Evaluates a YOLO segmentation checkpoint at `conf_thr` (with `MAX_DIM=130` & containment NMS
    matching production inference) on all held-out, non-blurred, human-verified QC tiles.
    Reports Detection Precision, Detection Recall, 3-Class Accuracy, End-to-End 3-Class Grain Accuracy,
    and Mean Viability Bias (pp) per cytotype.
    """
    if not model_weights or not os.path.exists(model_weights):
        return None
    from ultralytics import YOLO

    s3 = get_s3()
    manifest = json.loads(s3.get_object(Bucket=S3_BUCKET, Key="Ostatni/Pollen_viability/Review/manifest.json")["Body"].read())
    m_by_id = {m["id"]: m for m in manifest.get("items", [])}
    set_id = manifest.get("set_id", "final_20261026_1614")

    dec_keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=f"Ostatni/Pollen_viability/Review/decisions/{set_id}/"):
        for o in page.get("Contents", []):
            if o["Key"].endswith(".json"):
                dec_keys.append(o["Key"])

    model = YOLO(model_weights)
    RECLASS_MAP = {"viable": 0, "to_viable": 0, "non_viable": 1, "to_non_viable": 1, "intermediate": 2, "to_intermediate": 2}
    agg = defaultdict(lambda: {
        "tiles": 0, "pred": 0, "tp_det": 0, "fp": 0, "fn": 0, "tp_cls": 0,
        "int_pred": 0, "int_tp_det": 0, "int_fp": 0, "int_fn": 0, "int_tp_cls": 0,
        "bias_list": [],
    })

    for dk in sorted(dec_keys):
        d = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=dk)["Body"].read())
        m = m_by_id.get(d.get("item_id"))
        if not m or d.get("image_issue"):
            continue
        verdict = d.get("verdict", "good")
        grains_override = d.get("grains", {})
        missed_list = d.get("missed", [])
        # Skip unedited 'poor' tiles where individual grain ground truth was not annotated
        if verdict == "poor" and not grains_override and not missed_list:
            continue

        det = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=m["det_key"])["Body"].read())
        orig_boxes = det.get("boxes", [])
        img_bytes = s3.get_object(Bucket=S3_BUCKET, Key=m["tile_key"])["Body"].read()
        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        if verdict == "poor" and len(orig_boxes) > 1:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
            blaps = [
                float(np.var(lap[max(0, int(b[1]) - 4):min(640, int(b[3]) + 4), max(0, int(b[0]) - 4):min(640, int(b[2]) + 4)]))
                for b in orig_boxes if int(b[2]) > int(b[0]) and int(b[3]) > int(b[1])
            ]
            if blaps and float(np.median(blaps)) < MIN_BOX_LAPLACIAN_VAR:
                continue

        # Ground truth grains on this tile: [(cx, cy, true_cls_id, is_outer_edge)]
        gt_pts = []
        for idx, b in enumerate(orig_boxes):
            act = grains_override.get(str(idx), "ok")
            if act != "fp":
                true_cid = RECLASS_MAP.get(act, int(b[5]))
                gedge = (b[0] <= 8 or b[1] <= 8 or b[2] >= 632 or b[3] >= 632)
                gt_pts.append((0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]), true_cid, gedge))
        for ms in missed_list:
            mcid = RECLASS_MAP.get(ms.get("cls", "non_viable"), 1)
            mx, my = float(ms.get("x", 320.0)), float(ms.get("y", 320.0))
            gedge = (mx <= 12 or my <= 12 or mx >= 628 or my >= 628)
            gt_pts.append((mx, my, mcid, gedge))

        res = model.predict(img_bgr, conf=conf_thr, iou=0.45, agnostic_nms=True, verbose=False)[0]
        raw_p = res.boxes.data.cpu().numpy() if res.boxes is not None else np.zeros((0, 6))
        # Apply production MAX_DIM=130 px filter & containment deduplication
        filtered_p = [pb for pb in raw_p if (pb[2] - pb[0]) <= 130 and (pb[3] - pb[1]) <= 130]
        filtered_p.sort(key=lambda pb: -float(pb[4]))
        kept_p = []
        for pb in filtered_p:
            area_b = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
            dup = False
            for kp in kept_p:
                ix1, iy1 = max(pb[0], kp[0]), max(pb[1], kp[1])
                ix2, iy2 = min(pb[2], kp[2]), min(pb[3], kp[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_k = max(1.0, (kp[2] - kp[0]) * (kp[3] - kp[1]))
                if inter / min(area_b, area_k) > 0.65:
                    dup = True
                    break
            if not dup:
                kept_p.append(pb)

        matched = {}
        tp_det = fp = tp_cls = pred_v = int_fp = 0
        for pb in kept_p:
            px, py, cid = 0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3]), int(pb[5])
            pedge = (pb[0] <= 8 or pb[1] <= 8 or pb[2] >= 632 or pb[3] >= 632)
            if cid == 0 and not pedge:
                pred_v += 1
            best_i, best_d = None, 20.0
            for gi, (gx, gy, gcid, _) in enumerate(gt_pts):
                if gi in matched:
                    continue
                dist = math.hypot(px - gx, py - gy)
                if dist <= best_d:
                    best_d, best_i = dist, gi
            if best_i is not None:
                matched[best_i] = cid
                tp_det += 1
                if gt_pts[best_i][2] == cid:
                    tp_cls += 1
            else:
                fp += 1
                if not pedge:
                    int_fp += 1

        fn = len(gt_pts) - len(matched)
        int_tp_det = int_tp_cls = int_fn = 0
        for gi, (_, _, gcid, gedge) in enumerate(gt_pts):
            if gedge:
                continue
            if gi in matched:
                int_tp_det += 1
                if matched[gi] == gcid:
                    int_tp_cls += 1
            else:
                int_fn += 1

        int_pred = int_tp_det + int_fp
        int_gt_pts = [g for g in gt_pts if not g[3]]
        true_v = sum(1 for _, _, gcid, _ in int_gt_pts if gcid == 0)
        mod_viab = (100.0 * pred_v / int_pred) if int_pred else None
        tru_viab = (100.0 * true_v / len(int_gt_pts)) if int_gt_pts else None
        bias_pp = (mod_viab - tru_viab) if (mod_viab is not None and tru_viab is not None) else None

        cyt = m.get("cytotype", "unknown")
        for k in (cyt, "ALL"):
            agg[k]["tiles"] += 1
            agg[k]["pred"] += len(kept_p)
            agg[k]["tp_det"] += tp_det
            agg[k]["fp"] += fp
            agg[k]["fn"] += fn
            agg[k]["tp_cls"] += tp_cls
            agg[k]["int_pred"] += int_pred
            agg[k]["int_tp_det"] += int_tp_det
            agg[k]["int_fp"] += int_fp
            agg[k]["int_fn"] += int_fn
            agg[k]["int_tp_cls"] += int_tp_cls
            if bias_pp is not None:
                agg[k]["bias_list"].append(bias_pp)

    rows = []
    for k in ["nat_2x", "ory_2x", "nat_4x", "neo_4x", "ALL"]:
        a = agg[k]
        prec = 100.0 * a["tp_det"] / max(1, a["pred"])
        rec = 100.0 * a["tp_det"] / max(1, a["tp_det"] + a["fn"])
        cls_acc = 100.0 * a["tp_cls"] / max(1, a["tp_det"])
        e2e_acc = 100.0 * a["tp_cls"] / max(1, a["pred"] + a["fn"])
        int_prec = 100.0 * a["int_tp_det"] / max(1, a["int_pred"])
        int_rec = 100.0 * a["int_tp_det"] / max(1, a["int_tp_det"] + a["int_fn"])
        int_e2e = 100.0 * a["int_tp_cls"] / max(1, a["int_pred"] + a["int_fn"])
        mean_bias = float(np.mean(a["bias_list"])) if a["bias_list"] else 0.0
        rows.append({
            "run": run,
            "conf_thr": conf_thr,
            "cytotype": k,
            "verified_tiles": a["tiles"],
            "n_pred": a["pred"],
            "fp": a["fp"],
            "fn": a["fn"],
            "det_precision_pct": round(prec, 2),
            "det_recall_pct": round(rec, 2),
            "class_acc_pct": round(cls_acc, 2),
            "end_to_end_acc_pct": round(e2e_acc, 2),
            "int_fp": a["int_fp"],
            "int_fn": a["int_fn"],
            "int_det_prec_pct": round(int_prec, 2),
            "int_det_rec_pct": round(int_rec, 2),
            "int_e2e_acc_pct": round(int_e2e, 2),
            "viability_bias_pp": round(mean_bias, 3),
        })
    e2e_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out_p = f"{out_dir}/e2e_qc_benchmark_{run}.csv"
    e2e_df.to_csv(out_p, index=False)
    print(f"\n=== Leak-Free End-to-End QC Benchmark (conf={conf_thr}) [{run}] ===")
    print(e2e_df.to_string(index=False))
    print(f"✅ Saved {out_p}")
    return e2e_df


def main():
    ap = argparse.ArgumentParser(description="Compare old vs new Pollen Viability models/pipelines per cytotype")
    ap.add_argument("--run", default="r2_additive_qc")
    ap.add_argument("--grains", default="output/size_analysis/pollen_grain_measurements.csv")
    ap.add_argument("--min-diam", default="output/qc/class_min_diam.csv")
    ap.add_argument("--conf-threshold", default="output/qc/class_conf_threshold.csv")
    ap.add_argument("--out-dir", default="output/qc")
    ap.add_argument("--model-weights", default=None, help="Optional YOLO .pt weights for live inference on QC tiles")
    ap.add_argument("--eval-conf", type=float, default=0.55, help="Confidence threshold for end-to-end QC evaluation")
    args = ap.parse_args()

    if args.model_weights:
        evaluate_end_to_end_qc_benchmark(args.model_weights, args.out_dir, args.run, conf_thr=args.eval_conf)
    evaluate_human_confirmed_qc(args.min_diam, args.conf_threshold, args.out_dir, args.run, args.model_weights)
    if os.path.exists(args.grains) and not args.model_weights:
        evaluate_filter_effect(args.grains, args.min_diam, args.conf_threshold, args.out_dir, args.run)


if __name__ == "__main__":
    main()


