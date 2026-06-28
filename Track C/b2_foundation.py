"""
B2 — frozen foundation-model embeddings (Track C modeling branch).

Backbone: MONAI SwinUNETR encoder (3D Swin Transformer) initialised from the public
self-supervised pretrained weights `model_swinvit.pt`, then FROZEN. We pool the deepest
encoder feature map to a single 768-d vector per volume and rank the gallery by cosine
similarity. Output obeys the team branch contract: [query_id, target_id, score].

Why this backbone: it loads with certainty on the ROCm/MONAI env, needs no skull-strip or
MNI registration (which would destroy subject identity), and adds an orthogonal 'deep 3D
structure' signal to RRF. Swappable for BrainIAC/M3Ret later behind the same interface.

Depends on trackc.py (load_volume, resize_to, rank_by_embeddings, embeddings_to_scores_df).
"""
from __future__ import annotations
import os
import urllib.request

import numpy as np
import pandas as pd
import torch

import trackc

SSL_URL = ("https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/"
           "0.8.1/model_swinvit.pt")
SSL_PATH = "model_swinvit.pt"
INPUT_SIZE = (96, 96, 96)          # SwinUNETR needs dims divisible by 32
FEATURE_SIZE = 48                  # -> deepest channels = 48*16 = 768


# --------------------------------------------------------------------------- model
def download_weights(url=SSL_URL, path=SSL_PATH):
    if not os.path.exists(path):
        print("downloading", url)
        urllib.request.urlretrieve(url, path)
    print("weights:", path, f"({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def build_encoder(weights_path=SSL_PATH, device=None, feature_size=FEATURE_SIZE):
    """Build SwinUNETR, load SSL weights into its swinViT encoder (frozen).
    Prints how many tensors matched so we can confirm the load actually worked."""
    from monai.networks.nets import SwinUNETR
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    common = dict(in_channels=1, out_channels=1, feature_size=feature_size, use_checkpoint=False)
    try:
        model = SwinUNETR(img_size=INPUT_SIZE, **common)   # MONAI <= 1.3
    except TypeError:
        model = SwinUNETR(**common)                         # MONAI >= 1.4 (img_size removed)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for p in ("module.", "swinViT.", "swin_vit.", "backbone."):
            if nk.startswith(p):
                nk = nk[len(p):]
        cleaned[nk] = v
    enc_sd = model.swinViT.state_dict()
    matched = {k: v for k, v in cleaned.items() if k in enc_sd and v.shape == enc_sd[k].shape}
    missing, unexpected = model.swinViT.load_state_dict({**enc_sd, **matched}, strict=False)
    print(f"SSL load: matched {len(matched)}/{len(enc_sd)} encoder tensors "
          f"(unmatched-from-ckpt: {len(cleaned) - len(matched)})")
    if len(matched) < 0.5 * len(enc_sd):
        print("  WARNING: <50% matched — encoder is largely random; check checkpoint keys.")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, device


# --------------------------------------------------------------------------- embedding
@torch.no_grad()
def embed_volume(model, device, path, data_root=None):
    """ceT1/T2 volume -> L2-normalised 768-d embedding from the deepest Swin stage."""
    data_root = data_root or trackc.DATA_ROOT
    vol = trackc.load_volume(path, resample_1mm=True, zscore=True, data_root=data_root)
    vol = trackc.resize_to(vol, INPUT_SIZE)
    x = torch.from_numpy(vol)[None, None].float().to(device)   # (1,1,D,H,W)
    out = model.swinViT(x)                                      # list of hidden states
    feat = out[-1] if isinstance(out, (list, tuple)) else out   # deepest map
    vec = torch.nn.functional.adaptive_avg_pool3d(feat, 1).flatten().cpu().numpy()
    n = np.linalg.norm(vec)
    return (vec / (n + 1e-8)).astype(np.float32)


def embed_dataframe(model, device, df, id_col, img_col, data_root=None):
    """Embed every row -> {id: vector}. Prints a heartbeat every 25 volumes."""
    emb = {}
    for i, (_, r) in enumerate(df.iterrows()):
        emb[r[id_col]] = embed_volume(model, device, r[img_col], data_root)
        if (i + 1) % 25 == 0:
            print(f"  embedded {i + 1}/{len(df)}")
    return emb


def write_latents(emb_by_role: dict, path="b2_latents.csv"):
    """medgemma-style latents.csv for UMAP sanity. emb_by_role:
    {(dataset, split, role): {id: vec}}  with role in {'query','gallery'}."""
    rows = []
    for (ds, split, role), emb in emb_by_role.items():
        for _id, vec in emb.items():
            rows.append({"id": _id, "dataset": ds, "split": split, "role": role,
                         **{f"e{i}": float(x) for i, x in enumerate(vec)}})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print("wrote", path, df.shape)
    return df


def branch_scores(query_emb, gallery_emb) -> pd.DataFrame:
    """The team branch contract: [query_id, target_id, score] for RRF / submission."""
    return trackc.embeddings_to_scores_df(query_emb, gallery_emb)
