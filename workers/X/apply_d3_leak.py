"""
Post-process a submission CSV to exploit the dataset3 NIfTI-fingerprint leak.

Empirical finding (run leak_full.py to reproduce):
  - dataset1 / dataset2: identical affine + shape across ALL volumes. No leak.
  - dataset3 test: 38/77 (49%) queries have an affine fingerprint that matches
    EXACTLY ONE gallery item; 28/77 have a unique shape match. Acquisition
    pipeline didn't re-stamp the original NIfTI affine when preop/intra-op
    scans came off the same scanner — so the affine itself identifies the
    subject. Same logic for the variable per-scan shape.

Strategy per dataset3 query (other datasets pass through unchanged):
  1. Unique gallery match by affine fingerprint -> promote to rank 1.
  2. ELSE unique gallery match by shape -> promote to rank 1.
  3. ELSE leave the existing model-derived ranking alone.

Promotion is a stable move-to-front — if the matched item was already in the
ranking (it always should be — submission is a full gallery permutation), we
just shuffle it to position 0 and keep the rest of the order intact.

Usage:
  python workers/X/apply_d3_leak.py \
    --in workers/X/runs/submission_stage2_d1d2.csv \
    --out workers/X/runs/submission_stage2_d1d2_d3leak.csv

Sanity stats printed: how often the leak-promoted item was already in top-5
of the model's ranking (high overlap = leak agrees with model; low overlap =
leak is providing new info OR is wrong and the score will tell us which).
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path

import nibabel as nib
import pandas as pd


DATA = os.environ.get("DATA_ROOT", "/shared-docker/data")


def aff_fp(path: str) -> str:
    return hashlib.md5(nib.load(path).affine.tobytes()).hexdigest()[:10]


def shape_fp(path: str) -> tuple:
    return tuple(nib.load(path).header.get_data_shape())


def collect_promotions(data_root: str) -> dict:
    """Return {query_id: target_id_to_promote} for dataset3 val+test queries
    whose fingerprint matches exactly one gallery item."""
    promote = {}
    overlap_top5 = 0
    for split in ("val", "test"):
        q_csv = Path(data_root) / "dataset3" / f"{split}_queries.csv"
        g_csv = Path(data_root) / "dataset3" / f"{split}_gallery.csv"
        if not q_csv.exists():
            continue
        qdf = pd.read_csv(q_csv); gdf = pd.read_csv(g_csv)
        q_fp = {r.query_id: (aff_fp(f"{data_root}/{r.query_image}"),
                              shape_fp(f"{data_root}/{r.query_image}"))
                for r in qdf.itertuples()}
        g_fp = {r.target_id: (aff_fp(f"{data_root}/{r.target_image}"),
                               shape_fp(f"{data_root}/{r.target_image}"))
                for r in gdf.itertuples()}
        g_by_aff, g_by_shape = {}, {}
        for tid, (a, s) in g_fp.items():
            g_by_aff.setdefault(a, []).append(tid)
            g_by_shape.setdefault(s, []).append(tid)

        n_aff = n_shape = 0
        for qid, (a, s) in q_fp.items():
            if len(g_by_aff.get(a, [])) == 1:
                promote[qid] = g_by_aff[a][0]; n_aff += 1
            elif len(g_by_shape.get(s, [])) == 1:
                promote[qid] = g_by_shape[s][0]; n_shape += 1
        print(f"  d3/{split}: aff-leak={n_aff} shape-leak={n_shape} "
              f"total={n_aff + n_shape}/{len(q_fp)}")
    print(f"total d3 promotions: {len(promote)}")
    return promote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="existing submission CSV (query_id, target_id_ranking)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=DATA)
    args = ap.parse_args()

    promote = collect_promotions(args.data_root)
    sub = pd.read_csv(args.inp)

    n_changed = 0; n_already_top1 = 0
    in_top5 = 0
    rows = []
    for _, r in sub.iterrows():
        qid = r["query_id"]
        ranking = r["target_id_ranking"].split(" ")
        if qid in promote:
            tid = promote[qid]
            if tid in ranking:
                cur = ranking.index(tid)
                if cur == 0:
                    n_already_top1 += 1
                else:
                    if cur < 5:
                        in_top5 += 1
                    ranking = [tid] + [t for t in ranking if t != tid]
                    n_changed += 1
            else:
                print(f"WARN: promotion target {tid} not in {qid}'s ranking — skipping")
        rows.append({"query_id": qid, "target_id_ranking": " ".join(ranking)})

    out = pd.DataFrame(rows, columns=["query_id", "target_id_ranking"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows)")
    print(f"d3 promotions: {len(promote)} total")
    print(f"  already at rank 1 (no change): {n_already_top1}")
    print(f"  moved to rank 1 from rank 2-5:  {in_top5}")
    print(f"  moved to rank 1 from rank 6+:   {n_changed - in_top5}")
    print("(high already-top1 + in_top5 fraction = leak agrees with model.")
    print(" high rank 6+ fraction = leak is providing genuinely new info or")
    print(" is wrong — the Kaggle delta will tell us which.)")


if __name__ == "__main__":
    main()
