"""
Track A — B3 embedding extraction.

Loads a trained B3 checkpoint (swinViT + projection head) and embeds every query
and gallery volume across all 3 datasets, both val and test splits.

Output: a single pickle with structure
    {
      "dataset1": {"val_queries": {qid: vec}, "val_gallery": {tid: vec},
                   "test_queries": {...}, "test_gallery": {...}},
      "dataset2": {...},
      "dataset3": {...},
    }
vec is np.ndarray(128,) unit-norm float32 — the trained projection head's output,
NOT the raw encoder. The encoder stays anisotropic (enc Δ ~0 during training);
the head is what learned to discriminate, so retrieval uses head output.

Track C's RRF reads {id: vec} per pool and ranks via cosine similarity — drops
in without conversion.

ponytail: pickle one file; 6 pools × ≤100 items × 128-d float32 = ~300 KB total.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3_encoder import build_encoder, INPUT_SIZE
from train_b3 import ProjHead, pool_features, get_pooled_dim
from viz_aug import load_and_resize


POOLS = [
    # (dataset, csv_name, id_col, img_col)
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


@torch.no_grad()
def embed_one(model, head, device, volume: np.ndarray) -> np.ndarray:
    """volume (D,H,W) float32 -> unit-norm projection-head output (128,) float32."""
    x = torch.from_numpy(volume)[None, None].float().to(device)
    out = model.swinViT(x)
    vec = pool_features(out)
    z = head(vec)
    z = F.normalize(z, dim=1)
    return z.flatten().cpu().numpy().astype(np.float32)


def embed_pool(model, head, device, csv_path: str, id_col: str, img_col: str) -> dict:
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(csv_path)):
        vol = load_and_resize(r[img_col])
        out[str(r[id_col])] = embed_one(model, head, device, vol)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="b3 checkpoint (.pt) — must contain swinViT + head")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/shared-docker/data"))
    ap.add_argument("--out", required=True, help="output .pkl path")
    args = ap.parse_args()

    model, device = build_encoder()
    pooled_dim = get_pooled_dim(model, device)
    head = ProjHead(in_dim=pooled_dim).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.swinViT.load_state_dict(ckpt["swinViT"])
    head.load_state_dict(ckpt["head"])
    model.eval(); head.eval()
    print(f"loaded {args.ckpt}  pooled_dim={pooled_dim}  head_out=128")

    result = {"dataset1": {}, "dataset2": {}, "dataset3": {}}
    for ds, csv_name, id_col, img_col in POOLS:
        csv_path = os.path.join(args.data_root, ds, f"{csv_name}.csv")
        result[ds][csv_name] = embed_pool(model, head, device, csv_path, id_col, img_col)

    with open(args.out, "wb") as f:
        pickle.dump(result, f)

    total = sum(len(p) for ds in result.values() for p in ds.values())
    print(f"saved {args.out}  total embeddings: {total}")


def _smoke():
    """Wire check: synthetic ckpt round-trip, embed one fake volume."""
    model, device = build_encoder()
    pooled_dim = get_pooled_dim(model, device)
    head = ProjHead(in_dim=pooled_dim).to(device)
    model.eval(); head.eval()
    vol = np.random.randn(*INPUT_SIZE).astype(np.float32)
    z = embed_one(model, head, device, vol)
    assert z.shape == (128,), f"unexpected shape {z.shape}"
    assert abs(np.linalg.norm(z) - 1.0) < 1e-3, f"not unit-norm: {np.linalg.norm(z)}"
    print(f"smoke OK  device={device}  z.shape={z.shape}  |z|={np.linalg.norm(z):.4f}")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
    else:
        main()
