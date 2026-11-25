"""Shared helpers for Pollen Viability Labeling App: config, S3 access, key naming."""

import json
import os
from pathlib import Path

import boto3
import yaml
from botocore.client import Config

ROOT = Path(__file__).resolve().parent.parent.parent
_CFG_CANDIDATES = [
    Path(__file__).with_name("project.yaml"),
    ROOT / "config" / "project.yaml",
    ROOT / "src" / "label_app" / "project.yaml",
]


def load_config():
    for p in _CFG_CANDIDATES:
        if p.exists():
            return yaml.safe_load(p.read_text())
    # Fallback default configuration
    return {
        "s3": {
            "endpoint": "https://s3.cl4.du.cesnet.cz",
            "bucket": "peg-bucket",
            "prefix": "Ostatni/Pollen_viability/Relabel/",
            "tiles": "Tiles/",
            "prelabels": "Prelabels/",
            "labels": "Labels/",
            "review": "Review/",
            "tiles_raw": "Ostatni/Pollen_viability/tiles_640/",
            "review_prefix": "Ostatni/Pollen_viability/Review/",
        },
        "imaging": {
            "um_per_px": 0.87724195,
        },
        "classes": ["viable", "non_viable", "intermediate", "debris"],
    }


CFG = load_config()
PREFIX = CFG["s3"]["prefix"]
BUCKET = os.environ.get("S3_BUCKET", CFG["s3"]["bucket"])


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", CFG["s3"]["endpoint"]),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
    )


def key(area, *parts):
    """S3 key under the project prefix, e.g. key('tiles', batch, scan, name)."""
    pfx = PREFIX + CFG["s3"].get(area, area.capitalize() + "/")
    clean_parts = [p.strip("/") for p in parts if p]
    return pfx + "/".join(clean_parts)


def list_keys(s3, prefix, suffix=""):
    out = []
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            out += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(suffix)]
    except Exception as e:
        print(f"Warning: list_keys failed for {prefix}: {e}")
    return out


def put_json(s3, k, obj):
    s3.put_object(Bucket=BUCKET, Key=k, Body=json.dumps(obj, indent=2).encode(), ContentType="application/json")


def get_json(s3, k, default=None):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
    except Exception:
        return default
