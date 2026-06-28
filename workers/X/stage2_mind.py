"""
Plan B — Stage-2 re-rank by MIND-distance on RRF top-k.

Same MIND-SSC math as workers/X/mindssc.py, but here computed per (q, t) pair
WITHOUT any descriptor pooling — we compare full per-voxel MIND maps to score
how structurally similar two volumes are in a contrast-blind way.

Wrapper structure mirrors workers/A/stage2_ncc.py (Alex's NCC re-rank). The
two scripts are interchangeable at the scorer level:
   stage2_ncc.py  -> ncc(z-scored 64^3)        — intensity-coarse, fast
   stage2_mind.py -> -mean ||MIND(A)-MIND(B)||  — contrast-invariant, slower

Why MIND instead of SynthMorph for Plan B:
- SynthMorph install on ROCm is a real risk; voxelmorph/neurite stack pulls TF.
- MIND distance IS what SynthMorph minimizes once you remove the warp field.
- For same-subject pairs the (rigid+nonlinear) deformation in dataset2/3 is
  small enough that voxelwise MIND difference still discriminates strongly.
- One descriptor implementation for both branches — same invariance guarantee.

Two modes (identical CLI shape to stage2_ncc.py):
  python workers/X/stage2_mind.py --branch a.csv --branch b.csv --out sub.csv
  python workers/X/stage2_mind.py --branch a_holdout.csv ... --gate

Speed: 754 unique volumes -> MIND computed once each (cached). Re-rank pass is
top-k pairwise (default k=10) -> ~3700 (q,t) MIND-distance ops total at 96^3 ~
seconds on GPU, ~5 min on CPU. The cache dominates; MIND itself is one conv3d.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mindssc import load_resize_zscore, mind_6ch, RESIZE


DATA_ROOT = os.environ.get("DATA_ROOT", "/shared-docker/data")


# ============================================================ MIND-distance scorer
def mind_map(vol_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """(D,H,W) np.float32 -> (6, D, H, W) float tensor on device."""
    vol_t = torch.from_numpy(vol_np)[None, None].to(device)
    with torch.no_grad():
        m = mind_6ch(vol_t)              # (1, 6, D, H, W)
    return m[0]


def mind_score(mA: torch.Tensor, mB: torch.Tensor) -> float:
    """Higher = more similar. Negative mean squared MIND difference per voxel."""
    with torch.no_grad():
        diff = mA - mB
        # mean over all (channel, voxel) entries
        return float(-(diff * diff).mean().item())


# ============================================================ small helpers (mirror stage2_ncc.py)
def rrf(branch_rankings: list, weights=None, k: int = 60) -> dict:
    weights = weights or [1.0] * len(branch_rankings)
    queries = set().union(*[set(r) for r in branch_rankings]) if branch_rankings else set()
    fused = {}
    for q in queries:
        scores = {}
        for w, ranking in zip(weights, branch_rankings):
            for rank, t in enumerate(ranking.get(q, []), start=1):
                scores[t] = scores.get(t, 0.0) + w / (k + rank)
        fused[q] = [t for t, _ in sorted(scores.items(), key=lambda x: -x[1])]
    return fused


def scores_to_rankings(df: pd.DataFrame) -> dict:
    return {q: list(g.sort_values("score", ascending=False)["target_id"])
            for q, g in df.groupby("query_id")}


def mrr(rankings: dict, ground_truth: dict) -> float:
    rrs = []
    for q, truth in ground_truth.items():
        ranked = rankings.get(q, [])
        rr = 0.0
        for i, t in enumerate(ranked, start=1):
            if t == truth:
                rr = 1.0 / i
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


def load_pools(data_root: str) -> dict:
    pools = {}
    for ds in ("dataset1", "dataset2", "dataset3"):
        for split in ("val", "test"):
            qcsv = os.path.join(data_root, ds, f"{split}_queries.csv")
            gcsv = os.path.join(data_root, ds, f"{split}_gallery.csv")
            if os.path.exists(qcsv) and os.path.exists(gcsv):
                pools[(ds, split)] = {"queries": pd.read_csv(qcsv),
                                       "gallery": pd.read_csv(gcsv)}
    return pools


# ============================================================ re-rank core
def rerank_pool(fused_rankings: dict, q_path: dict, t_path: dict,
                topk: int, mind_cache: Dict[str, torch.Tensor],
                device: torch.device, data_root: str) -> dict:
    """Top-k per query gets re-ranked by MIND-distance; tail keeps fused order."""
    def get_mind(rel: str) -> torch.Tensor:
        if rel not in mind_cache:
            mind_cache[rel] = mind_map(load_resize_zscore(rel, data_root), device)
        return mind_cache[rel]

    new_rankings = {}
    for qid, order in tqdm(fused_rankings.items(), desc="mind rerank"):
        top = order[:topk]
        tail = order[topk:]
        qm = get_mind(q_path[qid])
        scored = [(t, mind_score(qm, get_mind(t_path[t]))) for t in top]
        scored.sort(key=lambda x: -x[1])
        new_rankings[qid] = [t for t, _ in scored] + tail
    return new_rankings


# ============================================================ entry points
def submission_mode(args, device):
    print(f"loading {len(args.branch)} branches:")
    for b in args.branch:
        print(f"  {b}")
    branch_dfs = [pd.read_csv(p) for p in args.branch]

    pools = load_pools(args.data_root)
    print(f"loaded {len(pools)} pools: {list(pools.keys())}")

    mind_cache: Dict[str, torch.Tensor] = {}
    submission_rows = []

    for (ds, split), pool in pools.items():
        qids = set(pool["queries"]["query_id"].astype(str))
        gids = set(pool["gallery"]["target_id"].astype(str))
        pool_branches = []
        for bdf in branch_dfs:
            sub = bdf[bdf["query_id"].astype(str).isin(qids) &
                      bdf["target_id"].astype(str).isin(gids)]
            pool_branches.append(scores_to_rankings(sub))
        fused = rrf(pool_branches)

        q_path = dict(zip(pool["queries"]["query_id"].astype(str),
                          pool["queries"]["query_image"]))
        t_path = dict(zip(pool["gallery"]["target_id"].astype(str),
                          pool["gallery"]["target_image"]))

        print(f"\n{ds}/{split}: {len(fused)} q x {len(gids)} g")
        refined = rerank_pool(fused, q_path, t_path, args.topk,
                              mind_cache, device, args.data_root)
        for qid, order in refined.items():
            submission_rows.append((qid, " ".join(order)))

    out_df = pd.DataFrame(submission_rows, columns=["query_id", "target_id_ranking"])
    out_df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out_df)} queries, {len(mind_cache)} volumes cached)")
    if len(out_df) != 377:
        print(f"WARN: expected 377 rows, got {len(out_df)} — check pool coverage")


def gate_mode(args, device):
    pairs = pd.read_csv(os.path.join(args.data_root, "dataset1/train_pairs.csv"))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pairs))
    hold = pairs.iloc[idx[:50]].reset_index(drop=True)
    qids = set(hold["query_id"].astype(str))
    gids = set(hold["target_id"].astype(str))
    gt = dict(zip(hold["query_id"].astype(str), hold["target_id"].astype(str)))

    print(f"gate: {len(hold)} held-out pairs (seed=0 first-50)")
    print(f"loading {len(args.branch)} branch holdout CSVs:")
    for b in args.branch:
        print(f"  {b}")
    branch_dfs = [pd.read_csv(p) for p in args.branch]

    pool_branches = []
    for bdf in branch_dfs:
        sub = bdf[bdf["query_id"].astype(str).isin(qids) &
                  bdf["target_id"].astype(str).isin(gids)]
        pool_branches.append(scores_to_rankings(sub))
    fused = rrf(pool_branches)
    fused_mrr = mrr(fused, gt)
    print(f"\nfused MRR (RRF only)         = {fused_mrr:.4f}")

    q_path = dict(zip(hold["query_id"].astype(str), hold["query_image"]))
    t_path = dict(zip(hold["target_id"].astype(str), hold["target_image"]))
    refined = rerank_pool(fused, q_path, t_path, args.topk, {}, device, args.data_root)
    refined_mrr = mrr(refined, gt)
    print(f"refined MRR (+ MIND top-{args.topk})  = {refined_mrr:.4f}")
    print(f"delta                        = {refined_mrr - fused_mrr:+.4f}")
    if refined_mrr - fused_mrr > 0.02:
        print("\nGATE PASSED — MIND stage-2 helps. Worth submitting.")
    else:
        print("\nGATE FAILED — MIND stage-2 doesn't help on local d1. "
              "Try larger topk, or skip Plan B and use mindssc.py branch only.")


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", action="append", required=True,
                    help="branch CSV path (repeat per branch); for --gate use *_holdout.csv files")
    ap.add_argument("--out", default="submission_stage2_mind.csv")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    if args.gate:
        gate_mode(args, device)
    else:
        submission_mode(args, device)


if __name__ == "__main__":
    main()
