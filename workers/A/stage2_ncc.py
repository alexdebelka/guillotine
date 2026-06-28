"""
Track A — Stage 2 re-rank by Normalized Cross-Correlation (NCC) on top-k.

Flow:
  1. Load branch CSVs (`--branch path` repeated).
  2. RRF fuse per pool → top-k candidates per query.
  3. For each (query, top-k candidate): load both volumes at 64³ z-scored,
     compute NCC, re-rank top-k by NCC. Tail keeps fused order.
  4. Write submission CSV in the official format (`query_id,target_id_ranking`).

Two modes:
  python workers/A/stage2_ncc.py --branch a.csv --branch b.csv --out sub.csv
  python workers/A/stage2_ncc.py --branch a_holdout.csv ... --gate    # local MRR

NCC is contrast-coarse (z-score already removes most intensity drift between
sequences). For top-10 of a 100-target gallery, same-subject pairs reliably
beat different-subject pairs by margin. Pure numpy — runs anywhere.

ponytail: V1 is NCC. Swap in MIND distance / SynthMorph residual at the
ponytail: ncc() function — wrapper stays identical.
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm


DATA_ROOT = os.environ.get("DATA_ROOT", "/shared-docker/data")
RESIZE = (64, 64, 64)


# ============================================================ I/O
def load_and_norm(rel_path: str) -> np.ndarray:
    """Load NIfTI, z-score on nonzero brain voxels, resize to RESIZE."""
    img = nib.load(os.path.join(DATA_ROOT, rel_path))
    vol = img.get_fdata().astype(np.float32)
    mask = vol > vol.mean() * 0.1
    if mask.sum() > 0:
        m, s = vol[mask].mean(), vol[mask].std() + 1e-8
        vol = (vol - m) / s
    factors = [t / s for t, s in zip(RESIZE, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


# ============================================================ scoring
def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation on flattened volumes. Higher = more similar."""
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-8
    return float((a * b).sum() / denom)


def rrf(branch_rankings: list, weights=None, k: int = 60) -> dict:
    """Reciprocal Rank Fusion. branch_rankings: list of {qid: [tids ordered]}."""
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


# ============================================================ pool handling
def load_pools(manifest_dir: str) -> dict:
    """Returns {(ds, split): {'queries': df, 'gallery': df}} for val + test."""
    pools = {}
    for ds in ("dataset1", "dataset2", "dataset3"):
        for split in ("val", "test"):
            qcsv = os.path.join(manifest_dir, ds, f"{split}_queries.csv")
            gcsv = os.path.join(manifest_dir, ds, f"{split}_gallery.csv")
            if os.path.exists(qcsv) and os.path.exists(gcsv):
                pools[(ds, split)] = {"queries": pd.read_csv(qcsv),
                                       "gallery": pd.read_csv(gcsv)}
    return pools


def rerank_pool(fused_rankings: dict, q_path: dict, t_path: dict,
                topk: int, cache: dict) -> dict:
    """For each query, take top-k from fused order, re-rank by NCC. Tail unchanged."""
    def get_vol(rel):
        if rel not in cache:
            cache[rel] = load_and_norm(rel)
        return cache[rel]

    new_rankings = {}
    for qid, order in tqdm(fused_rankings.items(), desc="ncc rerank"):
        top = order[:topk]
        tail = order[topk:]
        q_vol = get_vol(q_path[qid])
        scored = [(t, ncc(q_vol, get_vol(t_path[t]))) for t in top]
        scored.sort(key=lambda x: -x[1])
        new_rankings[qid] = [t for t, _ in scored] + tail
    return new_rankings


# ============================================================ entry points
def submission_mode(args):
    """Read branch CSVs, RRF + NCC re-rank, write submission CSV."""
    print(f"loading {len(args.branch)} branches:")
    for b in args.branch:
        print(f"  {b}")
    branch_dfs = [pd.read_csv(p) for p in args.branch]

    pools = load_pools(args.data_root)
    print(f"loaded {len(pools)} pools: {list(pools.keys())}")

    image_cache = {}
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

        print(f"\n{ds}/{split}: {len(fused)} q × {len(gids)} g")
        refined = rerank_pool(fused, q_path, t_path, args.topk, image_cache)
        for qid, order in refined.items():
            submission_rows.append((qid, " ".join(order)))

    out_df = pd.DataFrame(submission_rows, columns=["query_id", "target_id_ranking"])
    out_df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out_df)} queries, {len(image_cache)} volumes cached)")
    if len(out_df) != 377:
        print(f"WARN: expected 377 rows, got {len(out_df)} — check pool coverage")


def gate_mode(args):
    """Run Stage 2 on the seed=0 first-50 held-out d1 pairs, print MRR."""
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
    print(f"\nfused MRR (RRF only)  = {fused_mrr:.4f}")

    q_path = dict(zip(hold["query_id"].astype(str), hold["query_image"]))
    t_path = dict(zip(hold["target_id"].astype(str), hold["target_image"]))
    refined = rerank_pool(fused, q_path, t_path, args.topk, {})
    refined_mrr = mrr(refined, gt)
    print(f"refined MRR (+ NCC re-rank top-{args.topk}) = {refined_mrr:.4f}")
    print(f"delta                                = {refined_mrr - fused_mrr:+.4f}")
    if refined_mrr - fused_mrr > 0.02:
        print("\nGATE PASSED — Stage 2 helps. Worth submitting.")
    else:
        print("\nGATE FAILED — Stage 2 doesn't help on local d1. "
              "Either weights need tuning or NCC isn't the right metric for these pools.")


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", action="append", required=True,
                    help="branch CSV path (repeat per branch); for --gate use *_holdout.csv files")
    ap.add_argument("--out", default="submission_stage2.csv",
                    help="output submission CSV (submission mode only)")
    ap.add_argument("--topk", type=int, default=10, help="top-k to re-rank per query")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--gate", action="store_true",
                    help="held-out MRR mode (uses dataset1 train_pairs seed=0 first-50)")
    args = ap.parse_args()

    if args.gate:
        gate_mode(args)
    else:
        submission_mode(args)


if __name__ == "__main__":
    main()
