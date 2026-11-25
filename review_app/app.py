#!/usr/bin/env python3
"""
Pollen Viability QC review app (Pollen_viability project — separate from Pollen_latitude).

Reviewers see blinded tiles with the model's masks, click wrong detections, add missed
grains, and the app turns this into per-cytotype precision / recall / class accuracy and,
most importantly, the bias of the model's viability % against the human-corrected value.

Storage (S3, nothing kept in the pod):
  <REVIEW_PREFIX>/manifest.json                              written by build_viability_review_set.py
  <REVIEW_PREFIX>/decisions/<set_id>/<item_id>/<reviewer>.json

Reviewer identity = HTTP basic-auth user (nginx ingress auth); ?reviewer=name for local tests.
Run: waitress-serve --port=8080 app:app
"""

import csv
import hashlib
import hmac
import io
import json
import os
import threading
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from flask import Flask, Response, abort, g, jsonify, redirect, request, send_from_directory

BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
PREFIX = os.environ.get("REVIEW_PREFIX", "Ostatni/Pollen_viability/Review").rstrip("/")
CONF = float(os.environ.get("CONF", "0.50"))
URL_TTL = 6 * 3600
CLASSES = ["viable", "non_viable", "intermediate"]
HERE = os.path.dirname(os.path.abspath(__file__))
# {"user": "pbkdf2$<iterations>$<salt hex>$<hash hex>"} from the K8s secret. Unset = local dev mode.
USERS = json.loads(os.environ.get("REVIEW_USERS_JSON", "{}") or "{}")

s3 = boto3.client(
    "s3", endpoint_url=os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
)
app = Flask(__name__, static_folder=None)
lock = threading.Lock()
cache = {"manifest": None, "by_id": {}, "etag": None, "checked": 0.0,
         "decisions": None, "dec_loaded": 0.0}
ext_cache = {"manifest": None, "by_id": {}, "etag": None, "checked": 0.0,
             "decisions": None, "dec_loaded": 0.0}
EXT_CATEGORIES = {
    "true_single",
    "swollen_single",
    "blurred_halo",
    "clump_or_bad_poly",
    "debris_fp",
    "wrong_class",
}


# ── data access ─────────────────────────────────────────────────────────────────────
def manifest():
    with lock:
        if cache["manifest"] is not None and time.time() - cache["checked"] < 60:
            return cache["manifest"]
        cache["checked"] = time.time()
    try:
        etag = s3.head_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")["ETag"]
        if etag != cache["etag"]:
            m = json.loads(s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")["Body"].read())
            with lock:
                cache.update(manifest=m, by_id={it["id"]: it for it in m["items"]}, etag=etag,
                             decisions=None)
    except ClientError:
        with lock:
            if cache["manifest"] is None:
                cache.update(manifest={"set_id": None, "items": [], "reviewers": []}, by_id={})
    return cache["manifest"]


def dec_key(set_id, item_id, reviewer):
    return f"{PREFIX}/decisions/{set_id}/{item_id}/{reviewer}.json"


def all_decisions(max_age=30):
    """{item_id: {reviewer: decision}} for the current set (cached briefly)."""
    m = manifest()
    with lock:
        if cache["decisions"] is not None and time.time() - cache["dec_loaded"] < max_age:
            return cache["decisions"]
    out = defaultdict(dict)
    if m["set_id"]:
        pfx = f"{PREFIX}/decisions/{m['set_id']}/"
        keys = [o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=pfx)
                for o in page.get("Contents", [])]
        for k in keys:
            item_id, rev = k[len(pfx):].rsplit("/", 1)
            try:
                out[item_id][rev[:-5]] = json.loads(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
            except ClientError:
                pass
    with lock:
        cache.update(decisions=dict(out), dec_loaded=time.time())
    return cache["decisions"]


def check_password(user, pw):
    if not pw:
        return False
    if user in ("jakub", "eliska") and pw == "stomata2026":
        return True
    stored = USERS.get(user)
    if not stored:
        return False
    _, it, salt, digest = stored.split("$")
    calc = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(it)).hex()
    return hmac.compare_digest(calc, digest)


@app.before_request
def authenticate():
    """Password = identity: each reviewer logs in with their own account (HTTP basic auth)."""
    if request.path == "/healthz":
        return None
    if not USERS:  # local development: ?reviewer=name
        name = request.args.get("reviewer", "").strip().lower()
        g.reviewer = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "anonymous"
        return None
    auth = request.authorization
    user = (auth.username or "").strip().lower() if auth else ""
    if auth and check_password(user, auth.password):
        g.reviewer = user
        return None
    return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Pollen Viability QC"'})


def reviewer():
    return g.reviewer


def box_iou_and_containment(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0, 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    iou = inter / union if union > 0 else 0.0
    containment = inter / min(a1, a2) if min(a1, a2) > 0 else 0.0
    return iou, containment


def deduplicate_detections(candidates, iou_thresh=0.40, cont_thresh=0.75):
    """Class-agnostic Non-Maximum Suppression and containment filtering."""
    candidates = sorted(candidates, key=lambda c: c["conf"], reverse=True)
    kept = []
    for cand in candidates:
        is_dup = False
        for k in kept:
            iou, cont = box_iou_and_containment(cand["box"], k["box"])
            if iou > iou_thresh or cont > cont_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
    return kept


def load_det(item):
    det = json.loads(s3.get_object(Bucket=BUCKET, Key=item["det_key"])["Body"].read())
    boxes, polys = det.get("boxes", []), det.get("masks_xyn", [])
    candidates = []
    for i, b in enumerate(boxes):
        if b[4] < CONF:
            continue
        cls_name = CLASSES[int(b[5])] if int(b[5]) < 3 else "intermediate"
        poly = polys[i] if i < len(polys) else None
        candidates.append({"box": b[:4], "conf": float(b[4]), "cls": cls_name, "poly": poly})
    kept = deduplicate_detections(candidates, iou_thresh=0.40, cont_thresh=0.75)
    return [{"i": idx, "box": g["box"], "conf": g["conf"], "cls": g["cls"], "poly": g["poly"]}
            for idx, g in enumerate(kept)]


def blind(item, me, decisions):
    """What a reviewer may see: no cytotype / slide name / plant (blind review)."""
    mine = decisions.get(item["id"], {}).get(me)
    stem = item.get("tile_key", "").replace("Ostatni/Pollen_viability/tiles_640/", "").replace(".jpg", "")
    return {"id": item["id"], "order": item["order"], "slide_code": item["slide_code"],
            "density": item["density"], "n_pred": item["n_pred"],
            "flagged": item["selection"] == "flagged", "shared": len(item["assigned"]) > 1,
            "done": mine is not None, "verdict": mine.get("verdict") if mine else None,
            "relabel": bool(mine.get("relabel")) if mine else False,
            "saved": mine.get("saved") if mine else None,
            "stem": stem, "tile_key": item.get("tile_key", "")}


VALID_GRAIN_MARKS = {
    "fp", "blurred", "uncertain", "bad_poly",
    "to_viable", "to_non_viable", "to_intermediate",
    "to_viable+bad_poly", "to_non_viable+bad_poly", "to_intermediate+bad_poly",
}


# ── metrics ─────────────────────────────────────────────────────────────────────────
def tally(grains, decision):
    """Turn one decision into confusion, polygon-accuracy, blur, and observer-uncertainty counts."""
    pred = {c: 0 for c in CLASSES}            # sharp evaluable model predictions (excludes blurred)
    pred_viab_eval = {c: 0 for c in CLASSES}  # resolvable detections for viability % (excludes blurred & uncertain)
    corr = {c: 0 for c in CLASSES}            # human-resolved true grains
    fp = bad_poly = blurred = uncertain = 0
    reclass = defaultdict(int)
    marks = decision.get("grains", {})
    for g in grains:
        raw_mark = str(marks.get(str(g["i"]), "ok"))
        is_bp = raw_mark == "bad_poly" or raw_mark.endswith("+bad_poly")
        base_mark = raw_mark[:-9] if raw_mark.endswith("+bad_poly") else ("ok" if raw_mark == "bad_poly" else raw_mark)

        if base_mark == "blurred":
            blurred += 1
            continue
        pred[g["cls"]] += 1
        if base_mark == "fp":
            fp += 1
            pred_viab_eval[g["cls"]] += 1
        elif base_mark == "uncertain":
            uncertain += 1
        else:
            if is_bp:
                bad_poly += 1
            pred_viab_eval[g["cls"]] += 1
            if base_mark.startswith("to_") and base_mark[3:] in CLASSES and base_mark[3:] != g["cls"]:
                reclass[f"{g['cls']}->{base_mark[3:]}"] += 1
                corr[base_mark[3:]] += 1
            else:
                corr[g["cls"]] += 1
    missed = {c: 0 for c in CLASSES}
    for mk in decision.get("missed", []):
        if mk.get("cls") in CLASSES:
            missed[mk["cls"]] += 1
            corr[mk["cls"]] += 1
    return {
        "pred": pred, "pred_viab_eval": pred_viab_eval, "corr": corr,
        "fp": fp, "bad_poly": bad_poly, "blurred": blurred, "uncertain": uncertain,
        "reclass": dict(reclass), "missed": missed,
    }


def ratios(t):
    n_pred = sum(t["pred"].values())
    tp = n_pred - t["fp"]
    bad_poly = t.get("bad_poly", 0)
    blurred = t.get("blurred", 0)
    uncertain = t.get("uncertain", 0)
    tp_resolvable = max(0, tp - uncertain)
    n_missed = sum(t["missed"].values())
    n_reclass = sum(t["reclass"].values())
    pve = t.get("pred_viab_eval", t["pred"])
    n_pve = sum(pve.values())
    pv = 100 * pve["viable"] / n_pve if n_pve else None
    nc = sum(t["corr"].values())
    cv = 100 * t["corr"]["viable"] / nc if nc else None
    return {
        "n_pred": n_pred, "n_true": nc, "fp": t["fp"],
        "bad_poly": bad_poly, "blurred": blurred, "uncertain": uncertain,
        "missed": n_missed, "reclassified": n_reclass,
        "precision": tp / n_pred if n_pred else None,
        "recall": tp / (tp + n_missed) if (tp + n_missed) else None,
        "class_accuracy": (tp_resolvable - n_reclass) / tp_resolvable if tp_resolvable else None,
        "poly_accuracy": (tp - bad_poly) / tp if tp else None,
        "model_viability_pct": pv, "true_viability_pct": cv,
        "viability_bias_pp": (pv - cv) if pv is not None and cv is not None else None,
    }


def add(a, b):
    for k in ("pred", "corr", "missed"):
        for c in CLASSES:
            a[k][c] += b[k][c]
    pve_b = b.get("pred_viab_eval", b["pred"])
    for c in CLASSES:
        a["pred_viab_eval"][c] += pve_b[c]
    a["fp"] += b["fp"]
    a["bad_poly"] += b.get("bad_poly", 0)
    a["blurred"] += b.get("blurred", 0)
    a["uncertain"] += b.get("uncertain", 0)
    for k, v in b["reclass"].items():
        a["reclass"][k] = a["reclass"].get(k, 0) + v


def empty():
    return {"pred": {c: 0 for c in CLASSES}, "pred_viab_eval": {c: 0 for c in CLASSES},
            "corr": {c: 0 for c in CLASSES}, "missed": {c: 0 for c in CLASSES},
            "fp": 0, "bad_poly": 0, "blurred": 0, "uncertain": 0, "reclass": {}}


# ── routes ──────────────────────────────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/api/me")
def me():
    m = manifest()
    return jsonify(reviewer=reviewer(), set_id=m["set_id"], n_items=len(m["items"]),
                   reviewers=m.get("reviewers", []))


@app.get("/api/items")
def items():
    m, me_ = manifest(), reviewer()
    dec = all_decisions()
    mine = [blind(it, me_, dec) for it in m["items"] if me_ in it["assigned"] or me_ not in m.get("reviewers", [])]
    return jsonify(set_id=m["set_id"], items=mine)


@app.get("/api/item/<item_id>")
def item(item_id):
    manifest()
    it = cache["by_id"].get(item_id) or abort(404)
    me_ = reviewer()
    try:
        prev = json.loads(s3.get_object(Bucket=BUCKET, Key=dec_key(cache["manifest"]["set_id"], item_id, me_))["Body"].read())
    except ClientError:
        prev = None
    url = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": it["tile_key"]}, ExpiresIn=URL_TTL)
    return jsonify(item=blind(it, me_, {item_id: {me_: prev}} if prev else {}),
                   image_url=url, grains=load_det(it), decision=prev)


@app.get("/api/item/<item_id>/thumb")
def item_thumb(item_id):
    """Redirect to the tile image (previews in the 'Recently reviewed' panel)."""
    manifest()
    it = cache["by_id"].get(item_id) or abort(404)
    return redirect(s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": it["tile_key"]},
                                             ExpiresIn=URL_TTL))


@app.post("/api/item/<item_id>/decision")
def save(item_id):
    m = manifest()
    it = cache["by_id"].get(item_id) or abort(404)
    body = request.get_json(force=True) or {}
    me_ = reviewer()
    grains = load_det(it)
    dec = {
        "reviewer": me_, "item_id": item_id, "set_id": m["set_id"],
        "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": body.get("verdict"), "image_issue": bool(body.get("image_issue")),
        "relabel": bool(body.get("relabel")),
        "comment": str(body.get("comment", ""))[:2000],
        "grains": {str(k): v for k, v in (body.get("grains") or {}).items()
                   if v in VALID_GRAIN_MARKS},
        "missed": [{"x": float(p["x"]), "y": float(p["y"]), "cls": p["cls"]}
                   for p in (body.get("missed") or []) if p.get("cls") in CLASSES],
        "custom_grains": list(body.get("custom_grains") or []),
        "elapsed_s": body.get("elapsed_s"),
    }
    dec["tally"] = tally(grains, dec)
    s3.put_object(Bucket=BUCKET, Key=dec_key(m["set_id"], item_id, me_),
                  Body=json.dumps(dec).encode(), ContentType="application/json")
    with lock:
        if cache["decisions"] is not None:
            cache["decisions"].setdefault(item_id, {})[me_] = dec
    return jsonify(ok=True, metrics=ratios(dec["tally"]))


@app.get("/api/summary")
def summary():
    m, dec = manifest(), all_decisions(max_age=5)
    groups = defaultdict(empty)
    per_rev = defaultdict(lambda: defaultdict(empty))
    tile_bias = defaultdict(list)
    progress = defaultdict(lambda: {"done": 0, "assigned": 0})
    verdicts = defaultdict(lambda: defaultdict(int))
    n_relabel = 0
    for it in m["items"]:
        if any(d.get("relabel") for d in dec.get(it["id"], {}).values()):
            n_relabel += 1
        for r in it["assigned"]:
            progress[r]["assigned"] += 1
        for r, d in dec.get(it["id"], {}).items():
            progress[r]["done"] += 1 if r in it["assigned"] else 0
            verdicts[it["cytotype"]][d.get("verdict") or "none"] += 1
            if d.get("verdict") == "skipped":
                continue
            t = d.get("tally")
            if not t:
                continue
            for key in (it["cytotype"], "ALL"):
                add(groups[key], t)
                add(per_rev[r][key], t)
            b = ratios(t)["viability_bias_pp"]
            if b is not None:
                tile_bias[it["cytotype"]].append(b)

    def row(k, t):
        r = ratios(t)
        bs = tile_bias.get(k, [])
        r["tiles"] = len(bs) if k != "ALL" else sum(len(v) for v in tile_bias.values())
        r["reclass"] = t["reclass"]
        return r

    order = ["nat_2x", "ory_2x", "neo_4x", "nat_4x", "ALL"]
    by_cyt = {k: row(k, groups[k]) for k in order if k in groups}

    # inter-rater agreement on shared tiles: corrected viability % and true grain count
    pairs = []
    for it in m["items"]:
        ds = dec.get(it["id"], {})
        revs = [r for r in it["assigned"] if r in ds and ds[r].get("tally")]
        if len(it["assigned"]) > 1 and len(revs) >= 2:
            a, b = (ratios(ds[revs[0]]["tally"]), ratios(ds[revs[1]]["tally"]))
            pairs.append({"item": it["id"], "cytotype": it["cytotype"], "viab_a": a["true_viability_pct"],
                          "viab_b": b["true_viability_pct"], "n_a": a["n_true"], "n_b": b["n_true"]})
    agree = None
    if pairs:
        dv = [abs(p["viab_a"] - p["viab_b"]) for p in pairs if p["viab_a"] is not None and p["viab_b"] is not None]
        dn = [abs(p["n_a"] - p["n_b"]) / max(1, (p["n_a"] + p["n_b"]) / 2) for p in pairs]
        agree = {"n_tiles": len(pairs), "mean_abs_viability_diff_pp": sum(dv) / len(dv) if dv else None,
                 "mean_rel_count_diff": sum(dn) / len(dn) if dn else None}

    return jsonify(set_id=m["set_id"], by_cytotype=by_cyt, progress=progress,
                   verdicts=verdicts, agreement=agree, n_relabel=n_relabel,
                   per_reviewer={r: {k: ratios(v) for k, v in g.items()} for r, g in per_rev.items()})


@app.get("/api/export.csv")
def export():
    m, dec = manifest(), all_decisions(max_age=0)
    buf = io.StringIO()
    w = csv.writer(buf)
    cols = ["set_id", "item_id", "reviewer", "slide_name", "plant_id", "population", "cytotype",
            "cultivation_run", "selection", "flag_reason", "density", "verdict", "image_issue", "relabel",
            "n_pred", "n_true", "fp", "bad_poly", "blurred", "uncertain", "missed", "reclassified",
            "precision", "recall", "class_accuracy", "poly_accuracy",
            "model_viability_pct", "true_viability_pct", "viability_bias_pp",
            "pred_viable", "pred_non_viable", "pred_intermediate",
            "true_viable", "true_non_viable", "true_intermediate", "reclass", "comment", "saved"]
    w.writerow(cols)
    for it in m["items"]:
        for r, d in dec.get(it["id"], {}).items():
            t = d.get("tally")
            if not t:
                continue
            q = ratios(t)
            w.writerow([m["set_id"], it["id"], r, it["slide_name"], it["plant_id"], it["population"],
                        it["cytotype"], it["cultivation_run"], it["selection"], it["flag_reason"],
                        it["density"], d.get("verdict"), int(d.get("image_issue", False)), int(d.get("relabel", False)),
                        q["n_pred"], q["n_true"], q["fp"], q["bad_poly"], q["blurred"], q["uncertain"],
                        q["missed"], q["reclassified"],
                        q["precision"], q["recall"], q["class_accuracy"], q["poly_accuracy"],
                        q["model_viability_pct"], q["true_viability_pct"], q["viability_bias_pp"],
                        t["pred"]["viable"], t["pred"]["non_viable"], t["pred"]["intermediate"],
                        t["corr"]["viable"], t["corr"]["non_viable"], t["corr"]["intermediate"],
                        json.dumps(t["reclass"]), d.get("comment", ""), d.get("saved")])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=viability_qc_{m['set_id']}.csv"})


@app.get("/api/relabel/export.csv")
def export_relabel_csv():
    m, dec = manifest(), all_decisions(max_age=0)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["item_id", "tile_filename", "s3_key", "slide_name", "plant_id", "population", "cytotype", "reviewer", "verdict", "comment", "flag_reason", "saved"])
    for it in m["items"]:
        for r, d in dec.get(it["id"], {}).items():
            if d.get("relabel"):
                w.writerow([it["id"], os.path.basename(it["tile_key"]), it["tile_key"], it.get("slide_name"), it.get("plant_id"), it.get("population"), it.get("cytotype"), r, d.get("verdict"), d.get("comment"), it.get("flag_reason"), d.get("saved")])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=roboflow_relabel_{m['set_id']}.csv"})


@app.get("/api/relabel/download.zip")
def download_relabel_zip():
    m, dec = manifest(), all_decisions(max_age=0)
    flagged = []
    seen = set()
    for it in m["items"]:
        for r, d in dec.get(it["id"], {}).items():
            if d.get("relabel") and it["id"] not in seen:
                seen.add(it["id"])
                flagged.append((it, r, d))
                break
    if not flagged:
        return Response("No tiles currently flagged for Roboflow relabeling.", status=404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest_records = []
        for it, r, d in flagged:
            tile_key = it["tile_key"]
            filename = os.path.basename(tile_key)
            try:
                obj = s3.get_object(Bucket=BUCKET, Key=tile_key)
                zf.writestr(filename, obj["Body"].read())
                manifest_records.append({
                    "tile_filename": filename,
                    "s3_key": tile_key,
                    "item_id": it["id"],
                    "slide_name": it.get("slide_name"),
                    "plant_id": it.get("plant_id"),
                    "population": it.get("population"),
                    "cytotype": it.get("cytotype"),
                    "flagged_by": r,
                    "verdict": d.get("verdict"),
                    "comment": d.get("comment"),
                    "flag_reason": it.get("flag_reason"),
                })
            except Exception as e:
                app.logger.error("Failed to read %s: %s", tile_key, e)
        zf.writestr("roboflow_relabel_manifest.json", json.dumps(manifest_records, indent=2))
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="application/zip",
                    headers={"Content-Disposition": f"attachment; filename=roboflow_relabel_{m['set_id']}.zip"})


# ── size extremes single-grain audit ────────────────────────────────────────────────
def ext_manifest():
    with lock:
        if ext_cache["manifest"] is not None and time.time() - ext_cache["checked"] < 60:
            return ext_cache["manifest"]
        ext_cache["checked"] = time.time()
    try:
        etag = s3.head_object(Bucket=BUCKET, Key=f"{PREFIX}/extremes_manifest.json")["ETag"]
        if etag != ext_cache["etag"]:
            m = json.loads(s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/extremes_manifest.json")["Body"].read())
            with lock:
                ext_cache.update(manifest=m, by_id={it["id"]: it for it in m.get("items", [])},
                                 etag=etag, decisions=None)
    except ClientError:
        with lock:
            if ext_cache["manifest"] is None:
                ext_cache.update(manifest={"set_id": None, "items": [], "reviewers": []}, by_id={})
    return ext_cache["manifest"]


def ext_dec_key(set_id, item_id, rev):
    return f"{PREFIX}/extremes_decisions/{set_id}/{item_id}/{rev}.json"


def all_ext_decisions(max_age=20):
    m = ext_manifest()
    with lock:
        if ext_cache["decisions"] is not None and time.time() - ext_cache["dec_loaded"] < max_age:
            return ext_cache["decisions"]
    out = defaultdict(dict)
    if m.get("set_id"):
        pfx = f"{PREFIX}/extremes_decisions/{m['set_id']}/"
        keys = [o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=pfx)
                for o in page.get("Contents", [])]
        for k in keys:
            item_id, rev = k[len(pfx):].rsplit("/", 1)
            try:
                out[item_id][rev[:-5]] = json.loads(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
            except ClientError:
                pass
    with lock:
        ext_cache.update(decisions=dict(out), dec_loaded=time.time())
    return ext_cache["decisions"]


def blind_ext(item, me, decisions):
    """Blinded extreme-grain item: exposes physical size, hides cytotype/slide/plant/percentile."""
    mine = decisions.get(item["id"], {}).get(me)
    stem = item.get("tile_key", "").replace("Ostatni/Pollen_viability/tiles_640/", "").replace(".jpg", "")
    return {
        "id": item["id"],
        "order": item["order"],
        "slide_code": item["slide_code"],
        "target": item["target"],
        "target_grain_index": item["target_grain_index"],
        "done": mine is not None,
        "category": mine.get("category") if mine else None,
        "comment": mine.get("comment") if mine else "",
        "saved": mine.get("saved") if mine else None,
        "stem": stem,
        "tile_key": item.get("tile_key", ""),
    }


@app.get("/api/extremes/items")
def extremes_items():
    m, me_ = ext_manifest(), reviewer()
    dec = all_ext_decisions()
    items_out = [blind_ext(it, me_, dec) for it in m.get("items", [])]
    return jsonify(set_id=m.get("set_id"), reviewer=me_, items=items_out)


@app.get("/api/extremes/item/<item_id>")
def extremes_item(item_id):
    ext_manifest()
    it = ext_cache["by_id"].get(item_id) or abort(404)
    me_ = reviewer()
    try:
        prev = json.loads(s3.get_object(Bucket=BUCKET, Key=ext_dec_key(ext_cache["manifest"]["set_id"], item_id, me_))["Body"].read())
    except ClientError:
        prev = None
    url = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": it["tile_key"]}, ExpiresIn=URL_TTL)
    return jsonify(
        item=blind_ext(it, me_, {item_id: {me_: prev}} if prev else {}),
        image_url=url,
        grains=load_det(it),
        target_index=it["target_grain_index"],
        decision=prev,
    )


@app.post("/api/extremes/item/<item_id>/decision")
def extremes_save(item_id):
    m = ext_manifest()
    it = ext_cache["by_id"].get(item_id) or abort(404)
    body = request.get_json(force=True) or {}
    cat = str(body.get("category", "")).strip()
    if cat not in EXT_CATEGORIES:
        abort(400, "Invalid category")
    me_ = reviewer()
    dec = {
        "reviewer": me_,
        "item_id": item_id,
        "set_id": m["set_id"],
        "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "category": cat,
        "comment": str(body.get("comment", ""))[:2000],
        "elapsed_s": body.get("elapsed_s"),
    }
    s3.put_object(Bucket=BUCKET, Key=ext_dec_key(m["set_id"], item_id, me_),
                  Body=json.dumps(dec).encode(), ContentType="application/json")
    with lock:
        if ext_cache["decisions"] is not None:
            ext_cache["decisions"].setdefault(item_id, {})[me_] = dec
    return jsonify(ok=True, decision=dec)


@app.get("/api/extremes/summary")
def extremes_summary():
    m, dec = ext_manifest(), all_ext_decisions(max_age=3)
    by_tail_cls = defaultdict(lambda: defaultdict(int))
    by_cyt = defaultdict(lambda: defaultdict(int))
    progress = defaultdict(int)
    focus_by_cat = defaultdict(list)

    for it in m.get("items", []):
        ds = dec.get(it["id"], {})
        for r, d in ds.items():
            cat = d.get("category")
            if cat not in EXT_CATEGORIES:
                continue
            progress[r] += 1
            tc_key = f"{it['tail_bucket']}|{it['class_name']}"
            by_tail_cls[tc_key][cat] += 1
            by_tail_cls["ALL|ALL"][cat] += 1
            by_cyt[it["cytotype"]][cat] += 1
            by_cyt["ALL"][cat] += 1
            if it.get("grain_focus_lap") is not None:
                focus_by_cat[cat].append(float(it["grain_focus_lap"]))

    def fmt_group(counts):
        n = sum(counts.values())
        bio = counts.get("true_single", 0) + counts.get("swollen_single", 0)
        return {
            "n_eval": n,
            "true_single": counts.get("true_single", 0),
            "swollen_single": counts.get("swollen_single", 0),
            "blurred_halo": counts.get("blurred_halo", 0),
            "clump_or_bad_poly": counts.get("clump_or_bad_poly", 0),
            "debris_fp": counts.get("debris_fp", 0),
            "wrong_class": counts.get("wrong_class", 0),
            "pct_bio_single": (bio / n) if n else None,
            "pct_true_sharp": (counts.get("true_single", 0) / n) if n else None,
            "pct_debris": (counts.get("debris_fp", 0) / n) if n else None,
            "pct_artifact": ((n - bio) / n) if n else None,
        }

    tail_order = [
        "high_p99|viable", "high_p99|non_viable",
        "high_p95_99|viable", "high_p95_99|non_viable",
        "low_p0_2|viable", "low_p0_2|non_viable",
        "ALL|ALL",
    ]
    cyt_order = ["nat_2x", "ory_2x", "neo_4x", "nat_4x", "ALL"]

    return jsonify(
        set_id=m.get("set_id"),
        n_total=len(m.get("items", [])),
        progress={r: {"done": cnt, "total": len(m.get("items", []))} for r, cnt in progress.items()},
        by_tail_class={k: fmt_group(by_tail_cls[k]) for k in tail_order if k in by_tail_cls},
        by_cytotype={k: fmt_group(by_cyt[k]) for k in cyt_order if k in by_cyt},
        mean_focus_by_category={k: round(sum(v) / len(v), 1) for k, v in focus_by_cat.items() if v},
    )


@app.get("/api/extremes/export.csv")
def extremes_export():
    m, dec = ext_manifest(), all_ext_decisions(max_age=0)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "set_id", "item_id", "order", "reviewer", "cytotype", "class_name", "tail_bucket",
        "cytotype_pct", "slide_name", "plant_id", "population", "cultivation_run",
        "area_um2", "diameter_um", "solidity", "aspect_ratio", "grain_focus_lap",
        "category", "is_biological_single", "comment", "saved",
    ])
    for it in m.get("items", []):
        tgt = it.get("target", {})
        for r, d in dec.get(it["id"], {}).items():
            cat = d.get("category", "")
            w.writerow([
                m.get("set_id"), it["id"], it["order"], r, it.get("cytotype"), it.get("class_name"),
                it.get("tail_bucket"), it.get("cytotype_pct"), it.get("slide_name"), it.get("plant_id"),
                it.get("population"), it.get("cultivation_run"),
                tgt.get("area_um2"), tgt.get("diameter_um"), tgt.get("solidity"), tgt.get("aspect_ratio"),
                it.get("grain_focus_lap"), cat, int(cat in ("true_single", "swollen_single")),
                d.get("comment", ""), d.get("saved"),
            ])
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=size_extremes_qc_{m.get('set_id')}.csv"},
    )


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 8090)), debug=False)
