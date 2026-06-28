"""
Diagnostic — does any NIfTI fingerprint beyond affine+shape uniquely identify
subjects across queries and gallery in dataset2 / dataset3?

Tries:
  - full 348-byte header binary blob hash
  - 16^3 voxel-corner pixel hash (fast — lazy-load slice)
  - relative path structure (dirname + stem)
  - header+corner combined

dataset1 is skipped — earlier check showed all d1 volumes are byte-identical
in affine/shape so further header digging there can't help.

Usage:
  python workers/X/leak_refine.py
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


def header_hash(path: str) -> str:
    img = nib.load(path)
    return hashlib.md5(bytes(img.header.binaryblock)).hexdigest()[:10]


def pixel_corner_hash(path: str) -> str:
    img = nib.load(path)
    arr = np.asarray(img.dataobj[:16, :16, :16])
    return hashlib.md5(arr.tobytes()).hexdigest()[:10]


def path_parts(rel: str):
    p = Path(rel)
    return (tuple(p.parts[:-1]), p.stem)


def check_pool(ds: str, split: str):
    print(f"\n========= {ds} {split} =========")
    qcsv = Path(DATA) / ds / f"{split}_queries.csv"
    gcsv = Path(DATA) / ds / f"{split}_gallery.csv"
    if not qcsv.exists():
        print("  skip (no csv)")
        return
    qdf = pd.read_csv(qcsv); gdf = pd.read_csv(gcsv)

    qfp, gfp = {}, {}
    for r in qdf.itertuples():
        p = f"{DATA}/{r.query_image}"
        qfp[r.query_id] = (header_hash(p), pixel_corner_hash(p), path_parts(r.query_image))
    for r in gdf.itertuples():
        p = f"{DATA}/{r.target_image}"
        gfp[r.target_id] = (header_hash(p), pixel_corner_hash(p), path_parts(r.target_image))

    for name, idx in [("header_blob", 0), ("pixel_corner", 1), ("path_dir+stem", 2)]:
        g_by = {}
        for tid, x in gfp.items():
            g_by.setdefault(x[idx], []).append(tid)
        unique = sum(1 for x in qfp.values() if len(g_by.get(x[idx], [])) == 1)
        dist = Counter(len(g_by.get(x[idx], [])) for x in qfp.values())
        print(f"  {name:>15}: {unique}/{len(qfp)} unique  dist={dict(dist)}")

    g_by = {}
    for tid, x in gfp.items():
        g_by.setdefault((x[0], x[1]), []).append(tid)
    unique = sum(1 for x in qfp.values() if len(g_by.get((x[0], x[1]), [])) == 1)
    dist = Counter(len(g_by.get((x[0], x[1]), [])) for x in qfp.values())
    print(f"  {'header+corner':>15}: {unique}/{len(qfp)} unique  dist={dict(dist)}")

    print(f"  example query path:  {qdf.iloc[0]['query_image']}")
    print(f"  example target path: {gdf.iloc[0]['target_image']}")


if __name__ == "__main__":
    for ds in ("dataset2", "dataset3"):
        for split in ("val", "test"):
            check_pool(ds, split)
