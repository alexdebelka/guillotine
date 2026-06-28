"""
Stage-2 V2 with PER-DATASET gating — re-rank dataset1 only, plain RRF for d2/d3.

Why:
  Local gate (dataset1 holdout, 50 pairs) showed Stage-2 V2 lifts d1 by +0.15
  to +0.24 (independent of RRF composition — re-rank is deterministic given the
  top-10 candidate set). Live Kaggle submission (Stage-2 V2 over all 3 datasets)
  scored 0.49294 vs 0.5298 best — net -0.037, meaning d2/d3 took a hit bigger
  than the d1 gain. MIND descriptor at max_side=96 does not survive d3's
  preop->intra-op structural change and may shuffle truth out of rank 1.

Hybrid: re-rank where the gate proved it helps, leave the rest alone.

Imports rerank_pool / scores_to_rankings / rrf / load_pools from Alex's
run_stage2_track_b.py — same code path, just gated.

Usage (MI300X, ~21 min runtime; 140 d1 queries x ~9s/q, d2/d3 are free):
  python workers/X/stage2_v2_d1only.py \
    --branch "Track C/branch_baseline.csv" \
    --branch "Track C/branch_b2.csv" \
    --branch workers/A/runs/branch_b3.csv \
    --branch workers/A/runs/branch_b4.csv \
    --out workers/X/runs/submission_stage2_d1only.csv

Expected macro: 0.5298 + d1 lift (probably +0.02 to +0.04) - whatever Stage-2
hurt d2/d3 by, now zero. Net upside +0.02-0.04. ONE submission to verify.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Reuse Alex's helpers (run_stage2_track_b.py imports the actual track_b package)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "A"))
from run_stage2_track_b import scores_to_rankings, rrf, load_pools, rerank_pool


DATA_ROOT = os.environ.get("DATA_ROOT", "/shared-docker/data")
# ponytail: re-rank only the datasets where re-rank helps (gated empirically by
# Kaggle submission). dataset1: confirmed +0.038 macro (PB 0.56830). dataset2:
# under test -- MIND is provably deformation-tolerant so it MIGHT help, but no
# holdout to gate locally. dataset3: confirmed NOT to re-rank (preop->intra-op
# structural change defeats MIND). Override via --rerank-ds CLI flag.
RERANK_DATASETS = {"dataset1", "dataset2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", action="append", required=True,
                    help="branch CSV (repeat per branch)")
    ap.add_argument("--out", default="workers/X/runs/submission_stage2_d1only.csv")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=96)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--rerank-ds", default=",".join(sorted(RERANK_DATASETS)),
                    help="comma-separated dataset names to re-rank (e.g. 'dataset1,dataset2')")
    args = ap.parse_args()
    rerank_datasets = {s.strip() for s in args.rerank_ds.split(",") if s.strip()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pools = load_pools(args.data_root)
    print(f"loaded {len(pools)} pools  rerank={sorted(rerank_datasets)}")
    branch_dfs = [pd.read_csv(p) for p in args.branch]
    print(f"loaded {len(branch_dfs)} branches:")
    for b in args.branch:
        print(f"  {b}")

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

        if ds in rerank_datasets:
            q_path = {str(r["query_id"]): str(Path(args.data_root) / r["query_image"])
                      for _, r in pool["q"].iterrows()}
            t_path = {str(r["target_id"]): str(Path(args.data_root) / r["target_image"])
                      for _, r in pool["g"].iterrows()}
            print(f"\n{ds}/{split}: re-ranking top-{args.topk} ({len(fused)} queries)")
            final = rerank_pool(fused, q_path, t_path, args.topk, args.max_side)
        else:
            print(f"\n{ds}/{split}: plain RRF, no re-rank ({len(fused)} queries)")
            final = fused

        for qid, order in final.items():
            rows.append((qid, " ".join(order)))

    out = pd.DataFrame(rows, columns=["query_id", "target_id_ranking"])
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} queries -- expected 377)")
    if len(out) != 377:
        print("WARN: row count off -- check pool coverage")


if __name__ == "__main__":
    main()
