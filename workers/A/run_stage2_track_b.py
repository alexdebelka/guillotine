"""
Stage 2 V2: gate Sebastien's track_b rerank on the held-out 50 d1 pairs.
If it lifts local MRR by >= +0.02, the same script also writes a full submission CSV.

Pipeline:
  1. Load specified branch CSVs (holdout for gate, val/test for submission).
  2. RRF fuse → top-k per query.
  3. Call track_b.rerank.rerank_ranked_candidates on top-k of each query.
  4. Tail keeps fused order.

Usage:
  # GATE first
  python workers/A/run_stage2_track_b.py --gate \
    --branch workers/A/runs/branch_b3_holdout.csv \
    --branch workers/A/runs/branch_b4_holdout.csv \
    --branch workers/A/runs/branch_b1_holdout.csv

  # If gate passes, write submission
  python workers/A/run_stage2_track_b.py \
    --branch workers/A/runs/branch_b3.csv \
    --branch workers/A/runs/branch_b4.csv \
    --branch workers/A/runs/branch_b1.csv \
    --branch "Track C/branch_baseline.csv" \
    --branch "Track C/branch_b2.csv" \
    --out workers/A/runs/submission_stage2_v2.csv
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from track_b.rerank import rerank_ranked_candidates


DATA_ROOT = os.environ.get("DATA_ROOT", "/shared-docker/data")


# ============================================================ minimal RRF (no trackc.py dep)
def scores_to_rankings(df: pd.DataFrame) -> dict:
    return {q: list(g.sort_values("score", ascending=False)["target_id"].astype(str))
            for q, g in df.groupby("query_id")}


def rrf(branch_rankings: list, weights=None, k: int = 60) -> dict:
    weights = weights or [1.0] * len(branch_rankings)
    queries = set().union(*[set(r) for r in branch_rankings])
    fused = {}
    for q in queries:
        scores = {}
        for w, r in zip(weights, branch_rankings):
            for rank, t in enumerate(r.get(q, []), start=1):
                scores[t] = scores.get(t, 0.0) + w / (k + rank)
        fused[q] = [t for t, _ in sorted(scores.items(), key=lambda x: -x[1])]
    return fused


def mrr(rankings: dict, gt: dict) -> float:
    rrs = []
    for q, t_true in gt.items():
        ranked = rankings.get(q, [])
        rr = 0.0
        for i, t in enumerate(ranked, start=1):
            if t == t_true:
                rr = 1.0 / i
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


# ============================================================ pool helpers
def load_pools(data_root: str) -> dict:
    pools = {}
    for ds in ("dataset1", "dataset2", "dataset3"):
        for split in ("val", "test"):
            qcsv = Path(data_root) / ds / f"{split}_queries.csv"
            gcsv = Path(data_root) / ds / f"{split}_gallery.csv"
            if qcsv.exists() and gcsv.exists():
                pools[(ds, split)] = {"q": pd.read_csv(qcsv), "g": pd.read_csv(gcsv)}
    return pools


def rerank_pool(fused: dict, q_path: dict, t_path: dict, topk: int, max_side: int) -> dict:
    out = {}
    for qid, order in tqdm(fused.items(), desc="rerank"):
        top = order[:topk]
        tail = order[topk:]
        candidates = [(t, t_path[t]) for t in top]
        reranked = rerank_ranked_candidates(
            q_path[qid], candidates, top_k=topk, max_side=max_side,
        )
        new_top = [item.target_id for item in reranked]
        out[qid] = new_top + tail
    return out


# ============================================================ entry points
def gate(args):
    pairs = pd.read_csv(Path(args.data_root) / "dataset1/train_pairs.csv")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pairs))
    hold = pairs.iloc[idx[:50]].reset_index(drop=True)
    qids = set(hold["query_id"].astype(str)); gids = set(hold["target_id"].astype(str))
    gt = dict(zip(hold["query_id"].astype(str), hold["target_id"].astype(str)))

    branch_rs = []
    for p in args.branch:
        df = pd.read_csv(p)
        df = df[df["query_id"].astype(str).isin(qids) & df["target_id"].astype(str).isin(gids)]
        branch_rs.append(scores_to_rankings(df))
    fused = rrf(branch_rs)
    fused_mrr = mrr(fused, gt)
    print(f"fused MRR (RRF of {len(args.branch)} branches) = {fused_mrr:.4f}")

    q_path = {str(r["query_id"]): str(Path(args.data_root) / r["query_image"])
              for _, r in hold.iterrows()}
    t_path = {str(r["target_id"]): str(Path(args.data_root) / r["target_image"])
              for _, r in hold.iterrows()}
    refined = rerank_pool(fused, q_path, t_path, args.topk, args.max_side)
    refined_mrr = mrr(refined, gt)
    print(f"refined MRR (+ track_b rerank top-{args.topk}, max_side={args.max_side}) = {refined_mrr:.4f}")
    print(f"delta = {refined_mrr - fused_mrr:+.4f}")
    if refined_mrr - fused_mrr > 0.02:
        print("\nGATE PASSED — run without --gate to write submission.")
    else:
        print("\nGATE FAILED — rerank doesn't help here. Try different --topk or --max-side.")


def submission(args):
    pools = load_pools(args.data_root)
    print(f"loaded {len(pools)} pools")
    branch_dfs = [pd.read_csv(p) for p in args.branch]

    rows = []
    for (ds, split), pool in pools.items():
        qids = set(pool["q"]["query_id"].astype(str))
        gids = set(pool["g"]["target_id"].astype(str))
        pool_branches = []
        for bdf in branch_dfs:
            sub = bdf[bdf["query_id"].astype(str).isin(qids) &
                      bdf["target_id"].astype(str).isin(gids)]
            pool_branches.append(scores_to_rankings(sub))
        fused = rrf(pool_branches)

        q_path = {str(r["query_id"]): str(Path(args.data_root) / r["query_image"])
                  for _, r in pool["q"].iterrows()}
        t_path = {str(r["target_id"]): str(Path(args.data_root) / r["target_image"])
                  for _, r in pool["g"].iterrows()}
        print(f"\n{ds}/{split}: {len(fused)} queries")
        refined = rerank_pool(fused, q_path, t_path, args.topk, args.max_side)
        for qid, order in refined.items():
            rows.append((qid, " ".join(order)))

    out = pd.DataFrame(rows, columns=["query_id", "target_id_ranking"])
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} queries — expected 377)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", action="append", required=True)
    ap.add_argument("--out", default="workers/A/runs/submission_stage2_v2.csv")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=96)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    if args.gate:
        gate(args)
    else:
        submission(args)


if __name__ == "__main__":
    main()
