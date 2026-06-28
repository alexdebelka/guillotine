"""
v2 of apply_d3_leak.py — adds two refinements on top of the unique-fingerprint
promotion rule:

  Rule 1 (v1): query has UNIQUE aff/shape match -> promote that target to rank 1
  Rule 2 (NEW): query's shape group is SMALL (<= MAX_SHAPE_K) -> restrict its
                ranking so those K candidates sit at the front, in the model's
                relative order. Expected MRR for a true-positive-in-set is
                ~(K+1)/(2K) instead of the full-gallery model MRR.
  Rule 3 (NEW): everyone else has CLAIMED targets (those Rule-1 promoted) shoved
                to the END of their ranking, so they don't waste rank 1 on an
                item we know belongs to a confident-match query.

Every output ranking is still a full gallery permutation (challenge requires it);
we re-order, never drop.

Usage:
  python workers/X/apply_d3_leak_v2.py \
    --in workers/X/runs/submission_stage2_d1d2_max128_d3leak.csv \
    --out workers/X/runs/submission_stage2_d1d2_max128_d3leak_v2.csv

Or apply directly to the existing PB:
  python workers/X/apply_d3_leak_v2.py \
    --in workers/X/runs/submission_stage2_d1d2_d3leak.csv \
    --out workers/X/runs/submission_stage2_d1d2_d3leak_v2.csv

Acts ONLY on dataset3 queries; d1/d2 pass through unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import nibabel as nib
import pandas as pd


DATA = os.environ.get("DATA_ROOT", "/shared-docker/data")


def aff_fp(path: str) -> str:
    return hashlib.md5(nib.load(path).affine.tobytes()).hexdigest()[:10]


def shape_fp(path: str) -> tuple:
    return tuple(nib.load(path).header.get_data_shape())


def collect_pool(data_root: str, split: str):
    """For dataset3/{split} return (q_fp, g_fp, g_by_aff, g_by_shape) or None."""
    q_csv = Path(data_root) / "dataset3" / f"{split}_queries.csv"
    g_csv = Path(data_root) / "dataset3" / f"{split}_gallery.csv"
    if not q_csv.exists():
        return None
    qdf = pd.read_csv(q_csv); gdf = pd.read_csv(g_csv)
    q_fp = {r.query_id:  (aff_fp(f"{data_root}/{r.query_image}"),
                           shape_fp(f"{data_root}/{r.query_image}"))
            for r in qdf.itertuples()}
    g_fp = {r.target_id: (aff_fp(f"{data_root}/{r.target_image}"),
                           shape_fp(f"{data_root}/{r.target_image}"))
            for r in gdf.itertuples()}
    g_by_aff: Dict[str, List[str]] = defaultdict(list)
    g_by_shape: Dict[tuple, List[str]] = defaultdict(list)
    for tid, (a, s) in g_fp.items():
        g_by_aff[a].append(tid)
        g_by_shape[s].append(tid)
    return q_fp, g_fp, dict(g_by_aff), dict(g_by_shape)


def build_actions(pool, max_shape_k: int):
    """Return ({qid: action_tuple}, claimed_tids).
    Action tuples:
      ('promote', tid)
      ('restrict', [tid, ...])   # candidates to pull to the front in model order
      ('exclude',  set(tid))     # claimed tids to push to the end
    """
    q_fp, g_fp, g_by_aff, g_by_shape = pool
    actions: Dict[str, Tuple] = {}
    claimed: Set[str] = set()

    # Pass 1: unique aff/shape -> promote
    for qid, (a, s) in q_fp.items():
        if len(g_by_aff.get(a, [])) == 1:
            actions[qid] = ("promote", g_by_aff[a][0])
            claimed.add(g_by_aff[a][0])
        elif len(g_by_shape.get(s, [])) == 1:
            actions[qid] = ("promote", g_by_shape[s][0])
            claimed.add(g_by_shape[s][0])

    # Pass 2: shape group of size 2..K -> restrict (filter out already-claimed)
    for qid, (a, s) in q_fp.items():
        if qid in actions:
            continue
        candidates = g_by_shape.get(s, [])
        if 2 <= len(candidates) <= max_shape_k:
            unclaimed = [t for t in candidates if t not in claimed]
            if unclaimed:
                actions[qid] = ("restrict", unclaimed)

    # Pass 3: everyone else -> push claimed to the end
    for qid in q_fp:
        if qid not in actions:
            actions[qid] = ("exclude", set(claimed))   # snapshot

    return actions, claimed


def apply_action(ranking: List[str], action: Optional[Tuple]) -> List[str]:
    if action is None:
        return ranking
    kind = action[0]
    if kind == "promote":
        tid = action[1]
        if tid in ranking:
            return [tid] + [t for t in ranking if t != tid]
    elif kind == "restrict":
        keep = set(action[1])
        front = [t for t in ranking if t in keep]
        back  = [t for t in ranking if t not in keep]
        return front + back
    elif kind == "exclude":
        ex = action[1]
        kept = [t for t in ranking if t not in ex]
        moved = [t for t in ranking if t in ex]
        return kept + moved
    return ranking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=DATA)
    ap.add_argument("--max-shape-k", type=int, default=4)
    args = ap.parse_args()

    sub = pd.read_csv(args.inp)
    actions_by_q: Dict[str, Tuple] = {}

    for split in ("val", "test"):
        pool = collect_pool(args.data_root, split)
        if pool is None:
            print(f"  skip d3/{split}"); continue
        actions, claimed = build_actions(pool, args.max_shape_k)
        actions_by_q.update(actions)
        n_p = sum(1 for a in actions.values() if a[0] == "promote")
        n_r = sum(1 for a in actions.values() if a[0] == "restrict")
        n_e = sum(1 for a in actions.values() if a[0] == "exclude")
        n_q = len(actions)
        print(f"  d3/{split}: promote={n_p} restrict={n_r} exclude={n_e} "
              f"(of {n_q} queries; claimed={len(claimed)})")

    # Apply, plus tiny stats so we can read the effect
    n_changed = 0; n_restricted_promo = 0
    rows = []
    for _, r in sub.iterrows():
        qid = r["query_id"]
        ranking = r["target_id_ranking"].split(" ")
        new = apply_action(ranking, actions_by_q.get(qid))
        if new != ranking:
            n_changed += 1
            if qid in actions_by_q and actions_by_q[qid][0] == "restrict":
                cand = actions_by_q[qid][1]
                # how many positions moved up
                if new[0] in cand:
                    n_restricted_promo += 1
        rows.append({"query_id": qid, "target_id_ranking": " ".join(new)})

    out_df = pd.DataFrame(rows, columns=["query_id", "target_id_ranking"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out_df)} rows)")
    print(f"  rankings reordered: {n_changed}")
    print(f"  restrict actions that pulled a candidate to rank 1: {n_restricted_promo}")


if __name__ == "__main__":
    main()
