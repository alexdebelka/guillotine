"""
Track A — held-out embedding + branch CSV for Nicole's local-MRR gate.

Matches `trackc.make_local_split(train_pairs, n_holdout=50, seed=0)` exactly
(first 50 of shuffled index). Computes embeddings for the 50 held-out queries
+ targets and writes a within-pool branch CSV (50 q × 50 g = 2500 rows) that
Nicole's `scores_to_rankings + rrf + mrr` consumes identically to the val/test
branch CSVs.

Supports both B3 and B4:
    python workers/A/embed_holdout.py --branch b4
    python workers/A/embed_holdout.py --branch b3 --b3-ckpt workers/A/runs/b3_run1.pt

Output files:
    workers/A/runs/branch_b{3,4}_holdout.csv   <- what Nicole loads
    workers/A/runs/b{3,4}_holdout.pkl          <- pkl for inspection / debug
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


DATA_ROOT = os.environ.get("DATA_ROOT", "/shared-docker/data")


def build_b4_embedder():
    from b4_shape import shape_fingerprint
    def emb(img_rel: str) -> np.ndarray:
        return shape_fingerprint(os.path.join(DATA_ROOT, img_rel))
    return emb


def build_b3_embedder(ckpt_path: str):
    import torch
    import torch.nn.functional as F
    from b3_encoder import build_encoder, INPUT_SIZE
    from train_b3 import ProjHead, pool_features, get_pooled_dim
    from viz_aug import load_and_resize

    model, device = build_encoder()
    pooled_dim = get_pooled_dim(model, device)
    head = ProjHead(in_dim=pooled_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.swinViT.load_state_dict(ckpt["swinViT"])
    head.load_state_dict(ckpt["head"])
    model.eval(); head.eval()
    print(f"loaded {ckpt_path}  pooled_dim={pooled_dim}")

    @torch.no_grad()
    def emb(img_rel: str) -> np.ndarray:
        vol = load_and_resize(img_rel)
        x = torch.from_numpy(vol)[None, None].float().to(device)
        out = model.swinViT(x)
        vec = pool_features(out)
        z = head(vec)
        z = F.normalize(z, dim=1)
        return z.flatten().cpu().numpy().astype(np.float32)

    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=["b3", "b4"], required=True)
    ap.add_argument("--b3-ckpt", default="workers/A/runs/b3_run1.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out-dir", default="workers/A/runs")
    args = ap.parse_args()

    if args.branch == "b3":
        emb_fn = build_b3_embedder(args.b3_ckpt)
    else:
        emb_fn = build_b4_embedder()

    pairs = pd.read_csv(os.path.join(DATA_ROOT, "dataset1/train_pairs.csv"))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))
    hold = pairs.iloc[idx[:args.n]].reset_index(drop=True)
    print(f"held-out {len(hold)} pairs (seed={args.seed}, first-{args.n} split)")

    q_emb, g_emb = {}, {}
    for i, r in hold.iterrows():
        q_emb[str(r["query_id"])]  = emb_fn(r["query_image"])
        g_emb[str(r["target_id"])] = emb_fn(r["target_image"])
        if (i + 1) % 10 == 0:
            print(f"  embedded {i+1}/{len(hold)}")
    gt = dict(zip(hold["query_id"].astype(str), hold["target_id"].astype(str)))

    # pkl for inspection
    pkl_path = os.path.join(args.out_dir, f"{args.branch}_holdout.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"queries": q_emb, "gallery": g_emb, "gt": gt}, f)
    print(f"wrote {pkl_path}")

    # branch CSV — 50 × 50 within-pool cosine sims
    g_ids = list(g_emb.keys())
    G = np.stack([g_emb[t] for t in g_ids])
    rows = []
    for qid, qv in q_emb.items():
        sims = G @ qv
        for j, tid in enumerate(g_ids):
            rows.append((qid, tid, float(sims[j])))
    csv_path = os.path.join(args.out_dir, f"branch_{args.branch}_holdout.csv")
    pd.DataFrame(rows, columns=["query_id", "target_id", "score"]).to_csv(csv_path, index=False)
    print(f"wrote {csv_path}  {len(rows)} rows")

    # quick standalone MRR for sanity
    reciprocals = []
    for qid, qv in q_emb.items():
        sims = G @ qv
        order = np.argsort(-sims)
        rank = next(k for k, j in enumerate(order, start=1) if g_ids[j] == gt[qid])
        reciprocals.append(1.0 / rank)
    print(f"standalone MRR (sanity) = {float(np.mean(reciprocals)):.4f}")


if __name__ == "__main__":
    main()
