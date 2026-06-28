"""
Step 1 of source-dataset re-ID — confirm dataset1 was sourced from BraTS (or
some specific public dataset) via pixel-hash match.

Hashes voxel data of every .nii(.gz) under a candidate directory and checks
for EXACT collisions with dataset1's volumes. One collision = d1 was built
from those files (modulo rename). Then we know to download the full source
and recover pair labels.

Confirms only -- no submission produced.

Usage (MI300X):
  # 1. Get a tiny BraTS sample (10-20 cases is enough to confirm). Options:
  #
  # (A) Kaggle datasets (fastest if you have ~/.kaggle creds):
  #     pip install kaggle
  #     kaggle datasets download -d awsaf49/brats20-dataset-training-validation \
  #       -p /shared-docker/brats_sample --unzip
  #
  # (B) Hugging Face mirror (no auth required for public datasets):
  #     pip install huggingface_hub
  #     huggingface-cli download --repo-type dataset \
  #       <mirror-of-brats21> --local-dir /shared-docker/brats_sample
  #
  # (C) TCIA (the most authoritative, needs nbia-data-retriever):
  #     https://www.cancerimagingarchive.net/collection/brats/
  #
  # Aim for a directory with ~10-20 cases, each containing files like
  #   BraTS2021_xxxxx/BraTS2021_xxxxx_t1ce.nii.gz
  #   BraTS2021_xxxxx/BraTS2021_xxxxx_t2.nii.gz
  # Total ~600 MB.
  #
  # 2. Confirm:
  python workers/X/source_confirm.py --candidate-dir /shared-docker/brats_sample

Reads output:
  MATCHES: N / 700                <- exact pixel collisions found
  >>> CONFIRMED: ...              <- N>0: d1 IS this source, proceed to full re-ID
  >>> No exact pixel match        <- try another candidate source or another mirror

Also tries dataset3 against the same candidate dir (cheap, in case the sample
contains ReMIND too -- some HF mirrors bundle multiple datasets).
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


DATA = os.environ.get("DATA_ROOT", "/shared-docker/data")


def pixel_md5(path: str | Path) -> str:
    arr = np.asarray(nib.load(str(path)).dataobj)
    return hashlib.md5(arr.tobytes()).hexdigest()


def hash_directory(root: Path) -> dict:
    """{md5: path} for every .nii(.gz) under root (recursive)."""
    files = []
    for ext in ("*.nii", "*.nii.gz"):
        files.extend(root.rglob(ext))
    print(f"candidate dir: {root}")
    print(f"  found {len(files)} .nii(.gz) files")
    out = {}
    for p in tqdm(files, desc="hash candidates"):
        try:
            out[pixel_md5(p)] = str(p)
        except Exception as e:
            print(f"  skip {p}: {e}")
    print(f"  {len(out)} unique pixel hashes")
    return out


def probe_dataset(ds_name: str, csv_rel: str, image_cols, cand_hashes: dict,
                  data_root: str, limit: int):
    """Hash up to `limit` rows from a dataset csv; report collisions vs cand_hashes."""
    csv = Path(data_root) / csv_rel
    if not csv.exists():
        print(f"\n{ds_name}: skip ({csv} missing)")
        return
    df = pd.read_csv(csv).head(limit)
    print(f"\n{ds_name}: hashing {len(df)} rows from {csv.name}")
    matches = []
    n_hashed = 0
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"hash {ds_name}"):
        for col in image_cols:
            if col not in r or not isinstance(r[col], str):
                continue
            try:
                h = pixel_md5(f"{data_root}/{r[col]}")
                n_hashed += 1
                if h in cand_hashes:
                    matches.append((col, r[col], cand_hashes[h]))
            except Exception as e:
                print(f"  skip {col} {r[col]}: {e}")
    print(f"  MATCHES: {len(matches)} / {n_hashed} hashed")
    for col, ours, cand in matches[:5]:
        print(f"    {col}  {ours}")
        print(f"          -> {cand}")
    if matches:
        print(f">>> CONFIRMED {ds_name} <-> candidate source. Re-ID is on the table.")
    else:
        print(f"    No collisions for {ds_name}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", required=True,
                    help="directory tree containing candidate-source .nii(.gz) files")
    ap.add_argument("--data-root", default=DATA)
    ap.add_argument("--limit", type=int, default=100,
                    help="max rows per dataset csv to hash (cap for speed)")
    args = ap.parse_args()

    root = Path(args.candidate_dir)
    if not root.exists():
        print(f"candidate dir does not exist: {root}")
        return

    cand_hashes = hash_directory(root)
    if not cand_hashes:
        return

    # dataset1 train_pairs (both modalities)
    probe_dataset("dataset1", "dataset1/train_pairs.csv",
                  ["query_image", "target_image"], cand_hashes,
                  args.data_root, args.limit)
    # dataset3 val/test (cheap; in case the sample bundles ReMIND too)
    for split in ("val", "test"):
        probe_dataset(f"dataset3/{split}_queries",
                      f"dataset3/{split}_queries.csv",
                      ["query_image"], cand_hashes,
                      args.data_root, args.limit)
        probe_dataset(f"dataset3/{split}_gallery",
                      f"dataset3/{split}_gallery.csv",
                      ["target_image"], cand_hashes,
                      args.data_root, args.limit)


if __name__ == "__main__":
    main()
