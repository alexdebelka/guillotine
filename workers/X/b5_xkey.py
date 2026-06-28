"""
B5 -- CrossKEY-style patch-level cross-modal contrastive descriptor.

Why this exists:
  CrossKEY off-the-shelf was dead (no public checkpoint, requires synthetic-US
  per brain). The RECIPE is the value: ResNet 3D on 32^3 patches, trained
  contrastively on cross-modal positive pairs. Adapt: skip synthetic-US -- we
  have 350 REAL ceT1/T2 pairs in dataset1, registered. Skip SIFT3D -- the
  registration gives us patch correspondence at any random voxel for free.

Pipeline:
  TRAIN
    - Cache all 350 (or 300 train + 50 holdout) (ceT1, T2) pairs at 128^3 in
      RAM (~1.8 GB float32 -- fits trivially).
    - Per step: sample B=64 pairs, pick a random foreground voxel in each
      ceT1 volume, crop 32^3 patches at the SAME coord in both views.
    - Forward ceT1 batch + T2 batch through shared ResNet18 3D + BN + linear
      projection (256-d L2-norm).
    - InfoNCE loss (temp 0.1, symmetric) with in-batch negatives.
    - B3 collapse-fix recipe: BatchNorm at the projection input,
      weight_decay=0, temp=0.1. (We learned this the hard way on B3.)
    - 2000 steps batch 64 -> ~30-60 min on MI300X.

  EMBED
    - For each of the 754 val/test volumes (and the 50 holdout): load,
      resize to 128^3, sample K=128 foreground 32^3 patches, encode each,
      mean-pool, L2-norm -> 256-d per-volume vector.
    - Cosine sim within each pool -> branch_b5.csv (29,529 rows) +
      branch_b5_holdout.csv.

Usage:
  python workers/X/b5_xkey.py train --steps 2000 --ckpt workers/X/runs/b5.pt
  python workers/X/b5_xkey.py embed --ckpt workers/X/runs/b5.pt --mode both

Plug-in: add branch_b5.csv to Nicole's RRF dict; gate via local-MRR on
branch_b5_holdout.csv before any Kaggle submission.
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from tqdm import tqdm


DATA = os.environ.get("DATA_ROOT", "/shared-docker/data")
DIM = 256
VOL_SIZE = 128
PATCH = 32
EPS = 1e-8

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


# ============================================================ data
def load_vol(rel_path: str) -> np.ndarray:
    """NIfTI -> resample 1mm -> z-score on brain voxels -> resize to VOL_SIZE^3."""
    full = rel_path if os.path.isabs(rel_path) else f"{DATA}/{rel_path}"
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
    factors = [VOL_SIZE / s for s in vol.shape]
    return zoom(vol, factors, order=1).astype(np.float32)


def foreground_coords(vol: np.ndarray, margin: int) -> np.ndarray:
    """Return coords of foreground voxels that fit a `margin`-padded patch."""
    mask = vol > vol.mean() * 0.1
    valid = np.zeros_like(mask, dtype=bool)
    valid[margin:-margin, margin:-margin, margin:-margin] = \
        mask[margin:-margin, margin:-margin, margin:-margin]
    coords = np.argwhere(valid)
    if len(coords) == 0:
        coords = np.argwhere(mask)
        if len(coords) == 0:
            coords = np.array([[VOL_SIZE // 2] * 3])
    return coords


def crop_patch(vol: np.ndarray, z: int, y: int, x: int) -> np.ndarray:
    m = PATCH // 2
    return vol[z-m:z+m, y-m:y+m, x-m:x+m][None]  # (1, PATCH, PATCH, PATCH)


def sample_batch(cache, B: int, rng: np.random.Generator):
    """Sample B (ceT1_patch, T2_patch) pairs from cached volume pairs.
    Same voxel coord in both views (dataset1 is registered)."""
    idx = rng.choice(len(cache), B)
    ce_patches, t2_patches = [], []
    margin = PATCH // 2
    for i in idx:
        ce_vol, t2_vol = cache[i]
        coords = foreground_coords(ce_vol, margin)
        z, y, x = coords[rng.integers(len(coords))]
        ce_patches.append(crop_patch(ce_vol, z, y, x))
        t2_patches.append(crop_patch(t2_vol, z, y, x))
    return (torch.from_numpy(np.stack(ce_patches)).float(),
            torch.from_numpy(np.stack(t2_patches)).float())


# ============================================================ model
class B5(nn.Module):
    """ResNet18 3D backbone + BN-at-input projection head."""
    def __init__(self, dim: int = DIM):
        super().__init__()
        from monai.networks.nets import resnet18
        self.backbone = resnet18(spatial_dims=3, n_input_channels=1, num_classes=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.bn(h)
        z = self.proj(h)
        return F.normalize(z, dim=1)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    sim = z1 @ z2.T / temp                              # (B, B)
    labels = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(sim, labels) +
            F.cross_entropy(sim.T, labels)) / 2


# ============================================================ training
def cmd_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    pairs = pd.read_csv(f"{DATA}/dataset1/train_pairs.csv")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))
    train_df = pairs.iloc[idx[args.n_holdout:]].reset_index(drop=True)
    print(f"training on {len(train_df)} pairs (held out {args.n_holdout})")

    t0 = time.time()
    cache = []
    for _, r in tqdm(train_df.iterrows(), total=len(train_df), desc="cache"):
        cache.append((load_vol(r["query_image"]), load_vol(r["target_image"])))
    print(f"cached in {time.time()-t0:.1f}s  (~{(len(cache)*VOL_SIZE**3*4*2)/1e9:.2f} GB)")

    model = B5().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    rng = np.random.default_rng(args.seed + 1)
    for step in range(args.steps):
        ce, t2 = sample_batch(cache, args.batch, rng)
        ce, t2 = ce.to(device), t2.to(device)
        z_ce, z_t2 = model(ce), model(t2)
        loss = info_nce(z_ce, z_t2, args.temp)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                diag = (z_ce * z_t2).sum(dim=1).mean().item()
                full = (z_ce @ z_t2.T)
                off = (full.sum() - full.diag().sum()) / (args.batch * (args.batch - 1))
            print(f"step {step:4d}  loss {loss.item():.4f}  "
                  f"diag {diag:+.4f}  offdiag {off.item():+.4f}  "
                  f"delta {diag - off.item():+.4f}")
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}},
               args.ckpt)
    print(f"saved {args.ckpt}")


# ============================================================ inference
def grid_coords(vol: np.ndarray, n_per_axis: int) -> np.ndarray:
    """n_per_axis^3 deterministic coords uniformly inside the brain bounding box."""
    margin = PATCH // 2
    mask = vol > vol.mean() * 0.1
    if mask.sum() == 0:
        bbox = [(margin, VOL_SIZE - margin)] * 3
    else:
        nz = np.where(mask)
        bbox = [(max(margin, int(d.min())), min(VOL_SIZE - margin, int(d.max())))
                for d in nz]
    axes = [np.linspace(lo, hi, n_per_axis, dtype=int) for lo, hi in bbox]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T


@torch.no_grad()
def embed_volume(model, device, vol: np.ndarray, K: int,
                 rng: np.random.Generator, grid: bool = False,
                 grid_n: int = 8) -> np.ndarray:
    """Sample patches, encode, mean-pool, L2-norm -> (DIM,) float32.
    grid=True: deterministic grid_n^3 coords (subject-comparable). Else K random."""
    margin = PATCH // 2
    if grid:
        coords = grid_coords(vol, grid_n)
    else:
        fg = foreground_coords(vol, margin)
        pick = rng.choice(len(fg), min(K, len(fg)), replace=(len(fg) < K))
        coords = fg[pick]
    patches = np.stack([crop_patch(vol, *c) for c in coords])  # (N, 1, P, P, P)
    batch = torch.from_numpy(patches).float().to(device)
    z = model(batch)                       # (N, DIM)
    v = F.normalize(z.mean(dim=0), dim=0)  # (DIM,)
    return v.cpu().numpy().astype(np.float32)


def embed_pool(model, device, csv_path: str, id_col: str, img_col: str,
               K: int, rng: np.random.Generator,
               grid: bool = False, grid_n: int = 8) -> dict:
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in tqdm(df.iterrows(), total=len(df),
                     desc=os.path.basename(csv_path)):
        vol = load_vol(r[img_col])
        out[str(r[id_col])] = embed_volume(model, device, vol, K, rng, grid, grid_n)
    return out


def pkl_to_branch_csv(emb: dict, out_csv: str):
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
            print(f"  {ds} {split}: {len(q_ids)} q x {len(g_ids)} g")
    pd.DataFrame(rows, columns=["query_id", "target_id", "score"]).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(rows)} rows)")


def cmd_embed(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = B5().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt}")

    rng = np.random.default_rng(0)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # holdout
    if args.mode in ("holdout", "both"):
        pairs = pd.read_csv(f"{DATA}/dataset1/train_pairs.csv")
        idx = np.random.default_rng(args.seed).permutation(len(pairs))
        hold = pairs.iloc[idx[:args.n_holdout]].reset_index(drop=True)
        q_emb, g_emb = {}, {}
        for _, r in tqdm(list(hold.iterrows()), desc="holdout"):
            q_emb[str(r["query_id"])]  = embed_volume(model, device, load_vol(r["query_image"]),  args.K, rng, args.grid, args.grid_n)
            g_emb[str(r["target_id"])] = embed_volume(model, device, load_vol(r["target_image"]), args.K, rng, args.grid, args.grid_n)
        gt = dict(zip(hold["query_id"].astype(str), hold["target_id"].astype(str)))
        with open(out_dir / "b5_holdout.pkl", "wb") as f:
            pickle.dump({"queries": q_emb, "gallery": g_emb, "gt": gt}, f)
        # 50 x 50 CSV + standalone MRR
        g_ids = list(g_emb.keys())
        G = np.stack([g_emb[t] for t in g_ids])
        rows, reciprocals = [], []
        for qid, qv in q_emb.items():
            sims = G @ qv
            for j, tid in enumerate(g_ids):
                rows.append((qid, tid, float(sims[j])))
            order = np.argsort(-sims)
            rank = next(k for k, j in enumerate(order, start=1) if g_ids[j] == gt[qid])
            reciprocals.append(1.0 / rank)
        csv = out_dir / "branch_b5_holdout.csv"
        pd.DataFrame(rows, columns=["query_id", "target_id", "score"]).to_csv(csv, index=False)
        print(f"wrote {csv}  standalone MRR = {float(np.mean(reciprocals)):.4f}")

    # full pools
    if args.mode in ("full", "both"):
        result = {"dataset1": {}, "dataset2": {}, "dataset3": {}}
        for ds, csv_name, id_col, img_col in POOLS:
            csv_path = f"{DATA}/{ds}/{csv_name}.csv"
            if not os.path.exists(csv_path):
                print(f"skip {csv_path}"); continue
            result[ds][csv_name] = embed_pool(model, device, csv_path, id_col, img_col,
                                              args.K, rng, args.grid, args.grid_n)
        with open(out_dir / "b5_embeddings.pkl", "wb") as f:
            pickle.dump(result, f)
        pkl_to_branch_csv(result, str(out_dir / "branch_b5.csv"))


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--ckpt", default="workers/X/runs/b5.pt")
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--temp", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n-holdout", type=int, default=50)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("embed")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--mode", choices=["full", "holdout", "both"], default="both")
    e.add_argument("--out-dir", default="workers/X/runs")
    e.add_argument("--K", type=int, default=128)
    e.add_argument("--grid", action="store_true",
                   help="use deterministic grid_n^3 coords (subject-comparable)")
    e.add_argument("--grid-n", type=int, default=8,
                   help="grid size per axis when --grid; total patches = grid_n^3")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--n-holdout", type=int, default=50)
    e.set_defaults(func=cmd_embed)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
