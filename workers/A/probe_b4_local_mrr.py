"""
Track A — local MRR probe for B4 V1.

Reproduces Track C's Level-1 measurement on the same seed=0 held-out 50 dataset1
pairs, without depending on trackc.py — so we can score B4 the moment it's built
instead of waiting on the full Track C plumbing.

Reference numbers from Track C's baseline (gradient magnitude on a 48^3 downsample):
  Level-1 MRR = 0.7626   (registered, no deformation — easy)
  Level-2 MRR = 0.0981   (independent rigid + nonlinear deformation per target)
  random       = 0.02    (1/50)

If B4 V1 stays comfortably above 0.02 on Level-1 it's worth keeping in RRF.
ponytail: no trackc.py dependency, no class hierarchy, 30 lines of the metric.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b4_shape import shape_fingerprint


DATA_ROOT = os.environ.get("DATA_ROOT", "/shared-docker/data")


def main():
    pairs = pd.read_csv(os.path.join(DATA_ROOT, "dataset1/train_pairs.csv"))
    # match Track C's split: rng seed=0, hold out the last 50 after shuffle.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pairs))
    hold = pairs.iloc[idx[-50:]].reset_index(drop=True)
    print(f"held-out: {len(hold)} pairs (dataset1 train, seed=0)")

    # embed
    q_emb, g_emb = {}, {}
    for i, r in hold.iterrows():
        q_emb[r["query_id"]]  = shape_fingerprint(os.path.join(DATA_ROOT, r["query_image"]))
        g_emb[r["target_id"]] = shape_fingerprint(os.path.join(DATA_ROOT, r["target_image"]))
        if (i + 1) % 10 == 0:
            print(f"  embedded {i+1}/{len(hold)}")

    gt = dict(zip(hold["query_id"], hold["target_id"]))
    gallery_ids = list(g_emb.keys())
    G = np.stack([g_emb[t] for t in gallery_ids])  # (50, 4096)

    # rank + MRR
    reciprocals = []
    for qid, qv in q_emb.items():
        sims = G @ qv  # cosine on unit-norm
        order = np.argsort(-sims)
        rank = next(k for k, j in enumerate(order, start=1) if gallery_ids[j] == gt[qid])
        reciprocals.append(1.0 / rank)
    score = float(np.mean(reciprocals))
    print(f"\nB4 V1 local MRR (Level-1, 50 held-out, no deformation) = {score:.4f}")
    print(f"  reference baseline (gradient-mag) = 0.7626")
    print(f"  random                            = 0.02")
    print(f"  top-1 hits = {sum(1 for r in reciprocals if r == 1.0)}/{len(reciprocals)}")


if __name__ == "__main__":
    main()
