"""
Track A — B3 training loop: ceT1 ↔ T2 contrastive on dataset1 pairs.

Each pair is augmented INDEPENDENTLY (see augmenter.py — that's where the contrast +
deformation invariances are manufactured). Both views are encoded through SwinUNETR and
trained with symmetric InfoNCE (CLIP-style). The positive for q_aug[i] is t_aug[i]; all
other t_aug[j] in the batch are negatives.

Run real training:
    python workers/A/train_b3.py --pairs-csv $DATA_ROOT/dataset1/train_pairs.csv

Smoke-check the wiring (no data, no GPU required):
    python workers/A/train_b3.py --smoke

ponytail: one InfoNCE, no temperature schedule, no warmup, no projection head.
ponytail: upgrade path → add a 2-layer MLP projection if linear probe on val plateaus.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3_encoder import build_encoder, load_ssl_weights, INPUT_SIZE
from augmenter import make_augmenter, augment_pair
from viz_aug import load_and_resize


class ProjHead(nn.Module):
    """SimCLR-style 2-layer projection head. in_dim is set at startup from pool_features dim."""
    def __init__(self, in_dim: int, hidden: int = 1024, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PairDataset(Dataset):
    def __init__(self, pairs_csv: str | None, aug, synthetic_n: int | None = None):
        self.synthetic_n = synthetic_n
        self.aug = aug
        self.df = None if synthetic_n else pd.read_csv(pairs_csv).reset_index(drop=True)

    def __len__(self):
        return self.synthetic_n if self.synthetic_n else len(self.df)

    def __getitem__(self, i):
        if self.synthetic_n:
            rng = np.random.default_rng(i)
            q = rng.standard_normal(INPUT_SIZE).astype(np.float32)
            t = rng.standard_normal(INPUT_SIZE).astype(np.float32)
        else:
            r = self.df.iloc[i]
            q = load_and_resize(r["query_image"])
            t = load_and_resize(r["target_image"])
        q_aug, t_aug = augment_pair(self.aug, q, t)
        return torch.from_numpy(q_aug)[None], torch.from_numpy(t_aug)[None]  # (1,D,H,W)


def pool_features(features) -> torch.Tensor:
    """Avg + max pool concat over ALL stages from swinViT.
    Deepest stage alone is too abstract (97% sim across different brains at init);
    fusing all stages captures both low-level (anatomy) and high-level (semantic) signals.
    Input: list/tuple of (B, C_i, D_i, H_i, W_i). Output: (B, 2*sum(C_i))."""
    if not isinstance(features, (list, tuple)):
        features = [features]
    pooled = []
    for f in features:
        pooled.append(F.adaptive_avg_pool3d(f, 1).flatten(1))
        pooled.append(F.adaptive_max_pool3d(f, 1).flatten(1))
    return torch.cat(pooled, dim=1)


def encode(model, batch: torch.Tensor, head: nn.Module | None = None,
           return_features: bool = False):
    """batch: (B,1,D,H,W) -> (B,C) unit-norm. If head given, project before normalizing
    (training path). Without head, returns raw pooled encoder features (retrieval path).
    If return_features, also returns the pre-projection pooled features (unit-norm) for diagnostics."""
    out = model.swinViT(batch)
    vec = pool_features(out)
    enc_vec = F.normalize(vec, dim=1)
    if head is not None:
        vec = head(vec)
    out_vec = F.normalize(vec, dim=1)
    if return_features:
        return out_vec, enc_vec
    return out_vec


def get_pooled_dim(model, device) -> int:
    """Run one dummy forward to detect total pooled dim across all stages."""
    model.eval()
    with torch.no_grad():
        probe = torch.zeros(1, 1, *INPUT_SIZE, device=device)
        out = model.swinViT(probe)
        dim = pool_features(out).shape[1]
    model.train()
    return dim


def info_nce(zq: torch.Tensor, zt: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE on already-normalized embeddings."""
    logits = zq @ zt.t() / temp
    labels = torch.arange(zq.size(0), device=zq.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--ssl-weights", default="/shared-docker/work/weights/model_swinvit.pt")
    ap.add_argument("--severity", default="medium", choices=["mild", "medium", "heavy"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=3e-4, help="higher LR for projection head")
    ap.add_argument("--temp", type=float, default=0.1,
                    help="InfoNCE temperature. 0.07 saturates softmax at init and stalls.")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="0 by default — nonzero wd on the projection head collapses InfoNCE.")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default="b3_ckpt.pt")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    aug = make_augmenter(spatial_size=INPUT_SIZE, severity=args.severity)
    ds = PairDataset(args.pairs_csv, aug)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)

    model, device = build_encoder()
    if os.path.exists(args.ssl_weights):
        load_ssl_weights(model, args.ssl_weights)
    else:
        print(f"WARN: ssl weights not found at {args.ssl_weights} — training from scratch")
    pooled_dim = get_pooled_dim(model, device)
    print(f"pooled feature dim (all stages, avg+max): {pooled_dim}")
    head = ProjHead(in_dim=pooled_dim).to(device)
    model.train(); head.train()

    opt = torch.optim.AdamW([
        {"params": model.swinViT.parameters(), "lr": args.lr},
        {"params": head.parameters(),          "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    step, t0, losses = 0, time.time(), []
    while step < args.steps:
        for q, t in loader:
            q, t = q.to(device, non_blocking=True), t.to(device, non_blocking=True)
            zq, eq = encode(model, q, head, return_features=True)
            zt, et = encode(model, t, head, return_features=True)
            loss = info_nce(zq, zt, args.temp)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if step % args.log_every == 0:
                with torch.no_grad():
                    esim = eq @ et.t()
                    e_diag = esim.diag().mean().item()
                    e_off = (esim.sum() - esim.diag().sum()).item() / (esim.numel() - esim.size(0))
                    sim = zq @ zt.t()
                    diag = sim.diag().mean().item()
                    off = (sim.sum() - sim.diag().sum()).item() / (sim.numel() - sim.size(0))
                    inp_std = q.std(dim=0).mean().item()  # variance across batch — should be > 0 if inputs differ
                print(f"step {step:5d}  loss {loss.item():.4f}  "
                      f"avg10 {np.mean(losses[-10:]):.4f}  "
                      f"inp_std {inp_std:.3f}  "
                      f"enc Δ {e_diag-e_off:+.3f} (d{e_diag:+.3f}/o{e_off:+.3f})  "
                      f"head Δ {diag-off:+.3f} (d{diag:+.3f}/o{off:+.3f})  "
                      f"t {time.time()-t0:.1f}s")
            step += 1
            if step >= args.steps:
                break

    torch.save({"swinViT": model.swinViT.state_dict(),
                "head": head.state_dict(),
                "args": vars(args)}, args.out)
    print(f"saved {args.out}  final_loss={losses[-1]:.4f}")


def _smoke():
    """assert-based wire check: dataset shapes, encoder unit-norm, finite InfoNCE, gradients flow."""
    aug = make_augmenter(INPUT_SIZE, "medium")
    ds = PairDataset(None, aug, synthetic_n=4)
    q0, t0 = ds[0]
    assert q0.shape == (1, *INPUT_SIZE) and t0.shape == (1, *INPUT_SIZE), \
        f"bad dataset shape: {q0.shape} {t0.shape}"
    model, device = build_encoder()
    pooled_dim = get_pooled_dim(model, device)
    print(f"pooled feature dim (all stages, avg+max): {pooled_dim}")
    head = ProjHead(in_dim=pooled_dim).to(device)
    model.train(); head.train()
    qb = torch.stack([ds[i][0] for i in range(2)]).to(device)
    tb = torch.stack([ds[i][1] for i in range(2)]).to(device)
    zq, zt = encode(model, qb, head), encode(model, tb, head)
    assert zq.shape[0] == 2 and zq.shape == zt.shape, f"encode shape: {zq.shape} vs {zt.shape}"
    assert torch.allclose(zq.norm(dim=1), torch.ones(2, device=device), atol=1e-4), \
        "projected output not unit-norm"
    loss = info_nce(zq, zt)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.swinViT.parameters())
    head_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert enc_grad and head_grad, f"grads — encoder:{enc_grad} head:{head_grad}"
    print(f"smoke OK  device={device}  zq.shape={tuple(zq.shape)}  loss={loss.item():.4f}")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
    else:
        main()
