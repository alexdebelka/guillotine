"""
Convert a Track A embeddings .pkl (B3 or B4) into Nicole's Track-C branch CSV
format: query_id, target_id, score (higher = more similar).

Input pkl structure (from embed_b3.py / b4_shape.py):
    {dataset: {pool_name: {id: vec}, ...}}
    pool_name in {val_queries, val_gallery, test_queries, test_gallery}

Output CSV: one big file with rows for every (q, t, score) WITHIN each of the 6
(dataset, split) pools. Cross-pool combinations are not emitted — challenge rule
says rank queries only against same-dataset same-split gallery.

Then in run_fuse_submit.ipynb:
    df = pd.read_csv("branch_b4.csv")
    branch_b4 = scores_to_rankings(df)
    fused = rrf([branch_b2, branch_baseline, branch_b4])

Usage:
    python workers/A/pkl_to_branch.py \
        --in workers/A/runs/b4_embeddings.pkl \
        --out workers/A/runs/branch_b4.csv
"""
from __future__ import annotations
import argparse
import pickle
import sys
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="input .pkl from embed_b3.py / b4_shape.py")
    ap.add_argument("--out", required=True, help="output branch CSV path")
    args = ap.parse_args()

    with open(args.inp, "rb") as f:
        emb = pickle.load(f)

    rows = []
    for ds, pools in emb.items():
        for split in ("val", "test"):
            q = pools.get(f"{split}_queries", {})
            g = pools.get(f"{split}_gallery", {})
            if not q or not g:
                continue
            q_ids = list(q.keys()); q_vecs = np.stack([q[k] for k in q_ids])
            g_ids = list(g.keys()); g_vecs = np.stack([g[k] for k in g_ids])
            sims = q_vecs @ g_vecs.T  # already unit-norm -> cosine
            for i, qid in enumerate(q_ids):
                for j, tid in enumerate(g_ids):
                    rows.append((qid, tid, float(sims[i, j])))
            print(f"  {ds} {split}: {len(q_ids)} q × {len(g_ids)} g = {len(q_ids)*len(g_ids)} rows")

    df = pd.DataFrame(rows, columns=["query_id", "target_id", "score"])
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  {len(df)} rows total")


if __name__ == "__main__":
    main()
