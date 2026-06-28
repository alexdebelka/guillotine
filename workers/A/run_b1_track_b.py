"""
Run Sebastien's track_b MIND-SSC over all 6 pools, output:
  - branch_b1.csv      (Nicole's branch format: query_id, target_id, score)
  - holdout: branch_b1_holdout.csv (seed=0 first-50 d1 pairs for local-MRR gate)

Why this exists: track_b.cli emits submission-format (query_id, target_id_ranking),
one CSV per pool. Nicole's RRF wants per-(q,t) score rows across all pools.
This script glues them.

Score = the actual MIND cosine sim (not synthetic rank-based), so RRF gets real info.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# import track_b in-process — way faster than CLI subprocesses (cache reuse, no model reload)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from track_b.mind import mind_vector
from track_b.io import load_nifti_array


def _mv(path: Path, max_side: int):
    """mind_vector wraps load_nifti_array; force ndarray output so np.stack + matmul work."""
    v = mind_vector(load_nifti_array(path), max_side=max_side)
    return v.detach().cpu().numpy() if hasattr(v, "detach") else v


DATA_ROOT = os.environ.get("DATA_ROOT", "/shared-docker/data")
POOLS = [
    ("dataset1", "val"),  ("dataset1", "test"),
    ("dataset2", "val"),  ("dataset2", "test"),
    ("dataset3", "val"),  ("dataset3", "test"),
]


def embed_pool(queries_df: pd.DataFrame, gallery_df: pd.DataFrame, data_root: str,
               max_side: int) -> list:
    """Return list of (qid, tid, score) rows for one pool."""
    g_vecs, g_ids = [], []
    for _, r in tqdm(gallery_df.iterrows(), total=len(gallery_df), desc=" gallery"):
        v = _mv(Path(data_root) / r["target_image"], max_side)
        g_vecs.append(v); g_ids.append(str(r["target_id"]))
    G = np.stack(g_vecs)

    rows = []
    for _, r in tqdm(queries_df.iterrows(), total=len(queries_df), desc=" queries"):
        q = _mv(Path(data_root) / r["query_image"], max_side)
        sims = G @ q  # unit-norm cosine
        qid = str(r["query_id"])
        for tid, s in zip(g_ids, sims):
            rows.append((qid, tid, float(s)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default="workers/A/runs/branch_b1.csv")
    ap.add_argument("--max-side", type=int, default=96)
    ap.add_argument("--holdout-out", default="workers/A/runs/branch_b1_holdout.csv",
                    help="also produce held-out branch CSV for the local gate")
    args = ap.parse_args()

    all_rows = []
    for ds, split in POOLS:
        qcsv = Path(args.data_root) / ds / f"{split}_queries.csv"
        gcsv = Path(args.data_root) / ds / f"{split}_gallery.csv"
        if not qcsv.exists() or not gcsv.exists():
            print(f"skip {ds}/{split}: csvs missing")
            continue
        print(f"\n=== {ds}/{split} ===")
        q_df = pd.read_csv(qcsv)
        g_df = pd.read_csv(gcsv)
        rows = embed_pool(q_df, g_df, args.data_root, args.max_side)
        all_rows.extend(rows)
        print(f"  added {len(rows)} rows")

    df = pd.DataFrame(all_rows, columns=["query_id", "target_id", "score"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows total)")

    # --- held-out branch CSV for the local gate ---
    if args.holdout_out:
        pairs = pd.read_csv(Path(args.data_root) / "dataset1/train_pairs.csv")
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(pairs))
        hold = pairs.iloc[idx[:50]].reset_index(drop=True)
        print(f"\n=== held-out: {len(hold)} pairs (seed=0 first-50) ===")
        # gallery is the 50 held-out targets
        g_vecs, g_ids = [], []
        for _, r in tqdm(hold.iterrows(), total=len(hold), desc=" gallery"):
            v = _mv(Path(args.data_root) / r["target_image"], args.max_side)
            g_vecs.append(v); g_ids.append(str(r["target_id"]))
        G = np.stack(g_vecs)
        rows = []
        for _, r in tqdm(hold.iterrows(), total=len(hold), desc=" queries"):
            q = _mv(Path(args.data_root) / r["query_image"], args.max_side)
            sims = G @ q
            qid = str(r["query_id"])
            for tid, s in zip(g_ids, sims):
                rows.append((qid, tid, float(s)))
        hold_df = pd.DataFrame(rows, columns=["query_id", "target_id", "score"])
        Path(args.holdout_out).parent.mkdir(parents=True, exist_ok=True)
        hold_df.to_csv(args.holdout_out, index=False)
        print(f"wrote {args.holdout_out}  ({len(hold_df)} rows)")

        # standalone MRR sanity
        gt = dict(zip(hold["query_id"].astype(str), hold["target_id"].astype(str)))
        rec = []
        for qid, qv in zip(hold["query_id"].astype(str),
                           [_mv(Path(args.data_root) / p, args.max_side)
                            for p in hold["query_image"]]):
            sims = G @ qv
            order = np.argsort(-sims)
            rank = next(k for k, j in enumerate(order, start=1) if g_ids[j] == gt[qid])
            rec.append(1.0 / rank)
        print(f"B1 standalone MRR (sanity) = {float(np.mean(rec)):.4f}")


if __name__ == "__main__":
    main()
