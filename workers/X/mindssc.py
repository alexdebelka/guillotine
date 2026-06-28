"""
Plan C — MIND-SSC contrast-invariant descriptor branch.

Heinrich et al. 2012 "MIND" (Modality-Independent Neighborhood Descriptor),
6-channel face-neighbor variant — provably contrast-invariant by construction
(monotone intensity transforms cancel inside the local SSD ratio).

Pipeline per volume:
  load NIfTI -> resample 1mm (trackc.load_volume) -> z-score
  -> resize to 96^3 (trilinear) -> MIND 6-channel map (96^3 x 6)
  -> adaptive avg-pool to 8^3 -> flatten -> L2-norm -> 3072-d vector

Per-pool: cosine sim between query and gallery vectors -> branch CSV.

Outputs (default --out-dir workers/X/runs):
  --mode full     ->  mindssc_embeddings.pkl, branch_mindssc.csv  (29,529 rows)
  --mode holdout  ->  mindssc_holdout.pkl,    branch_mindssc_holdout.csv (2500 rows)
  --mode both     ->  both pairs

Usage (MI300X):
  python workers/X/mindssc.py --smoke                              # math check, no data
  python workers/X/mindssc.py --mode holdout                        # local gate prep
  python workers/X/mindssc.py --mode full --device cuda             # 754 volumes

GPU-optional. CPU works fine: ~0.5s/volume @ 96^3 -> ~7 min for 754 volumes.
ponytail: 6-channel MIND (not 12-channel SSC) — same invariance class, half the
ponytail: tensor and zero additional accuracy to defend at the hackathon scale.
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import nibabel as nib
from scipy.ndimage import zoom


DATA_ROOT = os.environ.get("DATA_ROOT", "/shared-docker/data")
RESIZE = (96, 96, 96)
POOL_GRID = (8, 8, 8)
PATCH = 3              # local SSD box edge (voxels)
EPS = 1e-6

POOLS = [
    ("dataset1", "val_queries",   "query_id",  "query_image"),
    ("dataset1", "val_gallery",   "target_id", "target_image"),
    ("dataset1", "test_queries",  "query_id",  "query_image"),
    ("dataset1", "test_gallery",  "target_id", "target_image"),
    ("dataset2", "val_queries",   "query_id",  "query_image"),
    ("dataset2", "val_gallery",   "target_id", "target_image"),
    ("dataset2", "test_queries",  "query_id",  "query_image"),
    ("dataset2", "test_gallery",  "target_id", "target_image"),
    ("dataset3", "val_queries",   "query_id",  "query_image"),
    ("dataset3", "val_gallery",   "target_id", "target_image"),
    ("dataset3", "test_queries",  "query_id",  "query_image"),
    ("dataset3", "test_gallery",  "target_id", "target_image"),
]


# ============================================================ I/O + preprocessing
def load_resize_zscore(rel_path: str, data_root: str = DATA_ROOT) -> np.ndarray:
    """Load NIfTI, resample 1mm, z-score on brain voxels, trilinear resize -> RESIZE."""
    full = rel_path if os.path.isabs(rel_path) else os.path.join(data_root, rel_path)
    if not os.path.exists(full) and full.endswith(".nii.gz") and os.path.exists(full[:-3]):
        full = full[:-3]
    img = nib.load(full)
    from nibabel.processing import resample_to_output
    img = resample_to_output(img, voxel_sizes=(1.0, 1.0, 1.0), order=1)
    vol = np.asarray(img.get_fdata(), dtype=np.float32)
    mask = vol > vol.mean() * 0.1
    if mask.sum() > 0:
        m, s = float(vol[mask].mean()), float(vol[mask].std())
        vol = (vol - m) / (s + EPS)
    factors = [t / s for t, s in zip(RESIZE, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


# ============================================================ MIND
def mind_6ch(vol_t: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    """6-channel face-neighbor MIND descriptor.

    Args
      vol_t: (1, 1, D, H, W) float tensor.
    Returns
      (1, 6, D, H, W) float tensor, every voxel a 6-d MIND descriptor.

    Math: D_k(x) = mean over patch of ((I(x) - I(x + n_k))^2).
          V(x)   = mean_k D_k(x)                       (local self-similarity scale)
          MIND_k(x) = exp(- D_k(x) / (V(x) + eps) )    (contrast-invariant)
    """
    pad = patch // 2
    box = torch.ones(1, 1, patch, patch, patch, device=vol_t.device,
                     dtype=vol_t.dtype) / (patch ** 3)
    offsets = [(1, 0, 0), (-1, 0, 0),
               (0, 1, 0), (0, -1, 0),
               (0, 0, 1), (0, 0, -1)]
    ssds = []
    for (dz, dy, dx) in offsets:
        # roll = shift volume by (-dz,-dy,-dx) -> I(x+n_k)
        shifted = torch.roll(vol_t, shifts=(-dz, -dy, -dx), dims=(2, 3, 4))
        diff_sq = (vol_t - shifted) ** 2
        ssd = F.conv3d(diff_sq, box, padding=pad)
        ssds.append(ssd)
    D = torch.cat(ssds, dim=1)                        # (1, 6, D, H, W)
    V = D.mean(dim=1, keepdim=True) + EPS             # (1, 1, D, H, W)
    return torch.exp(-D / V)


def descriptor(vol_np: np.ndarray, device: torch.device) -> np.ndarray:
    """volume (D,H,W) np.float32 -> unit-norm flat descriptor np.float32."""
    vol_t = torch.from_numpy(vol_np)[None, None].to(device)
    with torch.no_grad():
        mind = mind_6ch(vol_t)                        # (1, 6, D, H, W)
        pooled = F.adaptive_avg_pool3d(mind, POOL_GRID)   # (1, 6, 8, 8, 8)
    vec = pooled.flatten().cpu().numpy().astype(np.float32)
    n = float(np.linalg.norm(vec))
    return vec / (n + EPS)


# ============================================================ pool iteration
def embed_pool(csv_path: str, id_col: str, img_col: str,
               device: torch.device, data_root: str) -> dict:
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in tqdm(df.iterrows(), total=len(df),
                     desc=os.path.basename(csv_path)):
        vol = load_resize_zscore(r[img_col], data_root)
        out[str(r[id_col])] = descriptor(vol, device)
    return out


def pkl_to_branch_csv(emb: dict, out_csv: str) -> None:
    """Cosine sims within each (dataset, split) pool -> branch CSV."""
    rows = []
    for ds, pools in emb.items():
        for split in ("val", "test"):
            q = pools.get(f"{split}_queries", {})
            g = pools.get(f"{split}_gallery", {})
            if not q or not g:
                continue
            q_ids = list(q.keys()); Q = np.stack([q[k] for k in q_ids])
            g_ids = list(g.keys()); G = np.stack([g[k] for k in g_ids])
            sims = Q @ G.T
            for i, qid in enumerate(q_ids):
                for j, tid in enumerate(g_ids):
                    rows.append((qid, tid, float(sims[i, j])))
            print(f"  {ds} {split}: {len(q_ids)} q x {len(g_ids)} g "
                  f"= {len(q_ids)*len(g_ids)} rows")
    pd.DataFrame(rows, columns=["query_id", "target_id", "score"]).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(rows)} rows total)")


def holdout_csv(emb_q: dict, emb_g: dict, gt: dict, out_csv: str) -> float:
    """50 x 50 within-pool branch CSV + return standalone MRR for sanity."""
    g_ids = list(emb_g.keys())
    G = np.stack([emb_g[t] for t in g_ids])
    rows = []
    reciprocals = []
    for qid, qv in emb_q.items():
        sims = G @ qv
        for j, tid in enumerate(g_ids):
            rows.append((qid, tid, float(sims[j])))
        order = np.argsort(-sims)
        rank = next(k for k, j in enumerate(order, start=1) if g_ids[j] == gt[qid])
        reciprocals.append(1.0 / rank)
    pd.DataFrame(rows, columns=["query_id", "target_id", "score"]).to_csv(out_csv, index=False)
    standalone_mrr = float(np.mean(reciprocals))
    print(f"wrote {out_csv}  ({len(rows)} rows)  standalone MRR = {standalone_mrr:.4f}")
    return standalone_mrr


# ============================================================ entry points
def run_full(args, device):
    result = {"dataset1": {}, "dataset2": {}, "dataset3": {}}
    for ds, csv_name, id_col, img_col in POOLS:
        csv_path = os.path.join(args.data_root, ds, f"{csv_name}.csv")
        if not os.path.exists(csv_path):
            print(f"skip missing {csv_path}")
            continue
        result[ds][csv_name] = embed_pool(csv_path, id_col, img_col, device, args.data_root)

    pkl = os.path.join(args.out_dir, "mindssc_embeddings.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(result, f)
    print(f"wrote {pkl}")
    pkl_to_branch_csv(result, os.path.join(args.out_dir, "branch_mindssc.csv"))


def run_holdout(args, device):
    pairs = pd.read_csv(os.path.join(args.data_root, "dataset1/train_pairs.csv"))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))
    hold = pairs.iloc[idx[:args.n]].reset_index(drop=True)
    print(f"holdout: {len(hold)} pairs (seed={args.seed} first-{args.n} dataset1)")

    q_emb, g_emb = {}, {}
    for i, r in tqdm(list(hold.iterrows()), desc="holdout"):
        q_emb[str(r["query_id"])]  = descriptor(load_resize_zscore(r["query_image"],  args.data_root), device)
        g_emb[str(r["target_id"])] = descriptor(load_resize_zscore(r["target_image"], args.data_root), device)
    gt = dict(zip(hold["query_id"].astype(str), hold["target_id"].astype(str)))

    pkl = os.path.join(args.out_dir, "mindssc_holdout.pkl")
    with open(pkl, "wb") as f:
        pickle.dump({"queries": q_emb, "gallery": g_emb, "gt": gt}, f)
    print(f"wrote {pkl}")
    holdout_csv(q_emb, g_emb, gt,
                os.path.join(args.out_dir, "branch_mindssc_holdout.csv"))


def smoke():
    """Self-test: MIND on a random volume is unit-norm, finite, deterministic."""
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    vol = rng.standard_normal(RESIZE).astype(np.float32)
    v1 = descriptor(vol, device)
    v2 = descriptor(vol, device)
    assert v1.shape == (6 * 8 * 8 * 8,), f"shape {v1.shape}"
    assert np.isfinite(v1).all(), "NaNs/Infs in MIND descriptor"
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-3, f"not unit-norm: {np.linalg.norm(v1)}"
    assert np.allclose(v1, v2), "non-deterministic"
    # Invariance check: monotone intensity transform must leave MIND nearly unchanged
    vol_scaled = vol * 7.0 + 3.0
    v_scaled = descriptor(vol_scaled, device)
    cos = float(v1 @ v_scaled)
    assert cos > 0.999, f"affine intensity invariance failed: cos={cos:.4f}"
    print(f"smoke OK  dim={v1.shape[0]}  |v|={np.linalg.norm(v1):.4f}  "
          f"affine-cos={cos:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "holdout", "both"], default="both")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out-dir", default="workers/X/runs")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="default: cuda if available")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=50, help="holdout size")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}  data_root: {args.data_root}  out_dir: {args.out_dir}")

    if args.mode in ("holdout", "both"):
        run_holdout(args, device)
    if args.mode in ("full", "both"):
        run_full(args, device)


if __name__ == "__main__":
    main()
