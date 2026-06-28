"""
Diagnostic — does pixel content (not just NIfTI header) leak subject identity
in dataset3? Tries center-of-brain block and full-volume hash, both per pool.

leak_refine.py already showed:
  - dataset3 corner pixels are mostly air -> uninformative
  - dataset3 header_blob = 38/77 unique on test (same as affine)
  - dataset2 fully washed; no pixel signal expected there

The bet for d3: a deeper interior block carries genuine brain tissue and might
distinguish subjects even where the affine doesn't (the 39 affine-orphans).

Usage:
  python workers/X/leak_pixel.py
"""
from __future__ import annotations
import hashlib
import os
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


DATA = os.environ.get("DATA_ROOT", "/shared-docker/data")


def center_block_hash(path: str, block: int = 32) -> str:
    """Hash a `block`^3 cube at the volume midpoint. Lazy-loaded slice."""
    img = nib.load(path)
    s = img.shape
    h = block // 2
    sl = tuple(slice(max(0, c // 2 - h), min(d, c // 2 + h))
               for c, d in zip(s, s))
    arr = np.asarray(img.dataobj[sl])
    return hashlib.md5(arr.tobytes()).hexdigest()[:10]


def full_pixel_hash(path: str) -> str:
    """Hash all voxel data. Loads full volume — only call when we mean it."""
    arr = np.asarray(nib.load(path).dataobj)
    return hashlib.md5(arr.tobytes()).hexdigest()[:10]


def check_pool(ds: str, split: str, do_full: bool = True):
    print(f"\n========= {ds} {split} =========")
    qcsv = Path(DATA) / ds / f"{split}_queries.csv"
    gcsv = Path(DATA) / ds / f"{split}_gallery.csv"
    if not qcsv.exists():
        print("  skip (no csv)"); return
    qdf = pd.read_csv(qcsv); gdf = pd.read_csv(gcsv)

    for fp_name, fp_fn in [
        ("center_32^3", center_block_hash),
        *([("full_pixel", full_pixel_hash)] if do_full else []),
    ]:
        qfp = {r.query_id:  fp_fn(f"{DATA}/{r.query_image}")  for r in qdf.itertuples()}
        gfp = {r.target_id: fp_fn(f"{DATA}/{r.target_image}") for r in gdf.itertuples()}
        g_by = {}
        for tid, h in gfp.items():
            g_by.setdefault(h, []).append(tid)
        unique = sum(1 for h in qfp.values() if len(g_by.get(h, [])) == 1)
        dist = Counter(len(g_by.get(h, [])) for h in qfp.values())
        print(f"  {fp_name:>12}: {unique}/{len(qfp)} unique  dist={dict(dist)}")


if __name__ == "__main__":
    # d2 is washed -- center block only, skip full to save time
    for split in ("val", "test"):
        check_pool("dataset2", split, do_full=False)
    # d3 is the bet
    for split in ("val", "test"):
        check_pool("dataset3", split, do_full=True)
