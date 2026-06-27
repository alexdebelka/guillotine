"""
Eyeball check for the augmenter (Task #5 acceptance).

Loads N pairs from dataset1 train_pairs.csv, applies the augmenter to each view INDEPENDENTLY,
and saves a PNG mosaic showing: [original_q | aug_q | original_t | aug_t] for one middle slice.

If a row's "aug_q" looks like a totally different scan of the SAME brain as "original_q",
the augmenter is doing its job.

Usage:
    python3 viz_aug.py [--n 8] [--severity medium] [--out aug_check.png]
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augmenter import make_augmenter, augment_pair

# minimal volume loader — same conventions as trackc.load_volume
import nibabel as nib
from scipy.ndimage import zoom

DATA_ROOT = os.environ.get("DATA_ROOT", "/shared-docker/data")
TARGET = (96, 96, 96)


def load_and_resize(rel_path: str) -> np.ndarray:
    img = nib.load(os.path.join(DATA_ROOT, rel_path))
    vol = img.get_fdata().astype(np.float32)
    # z-score on brain voxels (nonzero)
    mask = vol > vol.mean() * 0.1
    if mask.sum() > 0:
        m, s = vol[mask].mean(), vol[mask].std() + 1e-8
        vol = (vol - m) / s
    # resize to TARGET
    factors = [t / s for t, s in zip(TARGET, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


def mid_slice(vol: np.ndarray) -> np.ndarray:
    return vol[:, :, vol.shape[2] // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--severity", default="medium", choices=["mild", "medium", "heavy"])
    ap.add_argument("--out", default="aug_check.png")
    ap.add_argument("--pairs-csv", default=os.path.join(DATA_ROOT, "dataset1/train_pairs.csv"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = pd.read_csv(args.pairs_csv).sample(n=args.n, random_state=0).reset_index(drop=True)
    aug = make_augmenter(spatial_size=TARGET, severity=args.severity, seed=0)

    fig, axes = plt.subplots(args.n, 4, figsize=(12, 3 * args.n))
    if args.n == 1:
        axes = axes[None, :]

    for i, r in pairs.iterrows():
        q = load_and_resize(r["query_image"])
        t = load_and_resize(r["target_image"])
        q_aug, t_aug = augment_pair(aug, q, t)
        for ax, im, title in zip(
            axes[i],
            [q, q_aug, t, t_aug],
            [f"q (ceT1) [{r['query_id'][:8]}]", "q aug", f"t (T2) [{r['target_id'][:8]}]", "t aug"],
        ):
            ax.imshow(mid_slice(im), cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=80, bbox_inches="tight")
    print(f"wrote {args.out} — {args.n} rows of [orig_q | aug_q | orig_t | aug_t]")


if __name__ == "__main__":
    main()
