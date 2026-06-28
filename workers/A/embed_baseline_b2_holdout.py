"""
Generate held-out branch CSVs for baseline + b2 (Nicole's embedders) so the
Stage 2 V2 gate uses the SAME branches as the production submission.

Why this script exists: Nicole's notebook embeds baseline/b2 on the held-out
50 pairs inline, but never persists branch CSVs in the format trackc-style RRF
consumes. Without these, the rerank gate uses a 3-branch fusion while the
submission uses 5 — different starting MRR, gate doesn't predict production.

Outputs:
  workers/A/runs/branch_baseline_holdout.csv
  workers/A/runs/branch_b2_holdout.csv

Then the gate can use 4 branches (baseline + b2 + b3 + b4) — matching what we
actually ship (we drop `branch_b` since its source is opaque).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

NICOLE = "/shared-docker/Nicole"
DATA_ROOT = "/shared-docker/data"
OUT_DIR = "/shared-docker/work/repo/workers/A/runs"


def main():
    sys.path.insert(0, NICOLE)
    os.chdir(NICOLE)  # b2.download_weights may use cwd
    import trackc  # noqa: E402
    import b2_foundation as b2  # noqa: E402
    from trackc import load_volume, embed_baseline, resize_to, make_local_split  # noqa: E402
    trackc.DATA_ROOT = DATA_ROOT

    print("loading b2 foundation model...")
    b2.download_weights()
    model, device = b2.build_encoder()
    print(f"b2 ready on {device}")

    pairs = pd.read_csv(f"{DATA_ROOT}/dataset1/train_pairs.csv")
    _, hold, _ = make_local_split(pairs, n_holdout=50, seed=0)
    print(f"held-out: {len(hold)} pairs (seed=0 first-50)")

    def base_emb(rel_path):
        return embed_baseline(rel_path, data_root=DATA_ROOT)

    @torch.no_grad()
    def b2_emb(rel_path):
        vol = load_volume(rel_path, resample_1mm=False, zscore=True, data_root=DATA_ROOT)
        vol = resize_to(vol, b2.INPUT_SIZE)
        x = torch.from_numpy(vol)[None, None].float().to(device)
        out = model.swinViT(x)
        feat = out[-1] if isinstance(out, (list, tuple)) else out
        v = torch.nn.functional.adaptive_avg_pool3d(feat, 1).flatten().cpu().numpy()
        return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    for name, fn in [("baseline", base_emb), ("b2", b2_emb)]:
        print(f"\nembedding {name}...")
        q_emb = {str(r["query_id"]): fn(r["query_image"]) for _, r in hold.iterrows()}
        g_emb = {str(r["target_id"]): fn(r["target_image"]) for _, r in hold.iterrows()}
        g_ids = list(g_emb.keys())
        G = np.stack([g_emb[t] for t in g_ids])
        rows = []
        for qid, qv in q_emb.items():
            sims = G @ qv
            for j, tid in enumerate(g_ids):
                rows.append((qid, tid, float(sims[j])))

        # standalone sanity MRR
        gt = dict(zip(hold["query_id"].astype(str), hold["target_id"].astype(str)))
        rec = []
        for qid, qv in q_emb.items():
            sims = G @ qv
            order = np.argsort(-sims)
            rank = next(k for k, j in enumerate(order, start=1) if g_ids[j] == gt[qid])
            rec.append(1.0 / rank)
        print(f"  {name} standalone MRR = {float(np.mean(rec)):.4f}")

        out_path = f"{OUT_DIR}/branch_{name}_holdout.csv"
        pd.DataFrame(rows, columns=["query_id", "target_id", "score"]).to_csv(out_path, index=False)
        print(f"  wrote {out_path}  ({len(rows)} rows)")

    print("\ndone. Now run the 4-branch gate:")
    print("  python workers/A/run_stage2_track_b.py --gate \\")
    print("    --branch workers/A/runs/branch_baseline_holdout.csv \\")
    print("    --branch workers/A/runs/branch_b2_holdout.csv \\")
    print("    --branch workers/A/runs/branch_b3_holdout.csv \\")
    print("    --branch workers/A/runs/branch_b4_holdout.csv --topk 10")


if __name__ == "__main__":
    main()
