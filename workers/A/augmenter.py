"""
Track A — Task #5: B3 augmenter (synth-contrast + independent deformation per view).

Manufactures the two invariances B3 must learn:
  L1 (contrast): RandHistogramShift + RandAdjustContrast + RandBiasField → simulates a
                 different MRI sequence (T2 from ceT1 input, etc.) without paired data.
  L2 (geometry): RandAffine (rigid+scale) + Rand3DElastic (nonlinear) applied INDEPENDENTLY
                 to query and target → kills the registered-grid shortcut.

API:
    make_augmenter(spatial_size, severity='medium') -> Callable[[np.ndarray], np.ndarray]
    augment_pair(aug, q_volume, t_volume) -> (q_aug, t_aug)  # INDEPENDENT calls

ponytail: MONAI transforms do all the work. No SynthMorph weights needed for v1; if the
ponytail: histogram-shift approximation is too weak we'll swap in a label-driven synth later.
"""
from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandHistogramShiftd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandAffined,
    Rand3DElasticd,
    NormalizeIntensityd,
)


SEVERITY = {
    "mild":   dict(hist_p=0.6, gamma=(0.8, 1.3), bias_p=0.15, bias_coeff=(0.0, 0.15),
                   affine_p=0.6, rot=0.05, trans=3, scale=0.03,
                   elastic_p=0.3, sigma=(5, 8), mag=(30, 60)),
    "medium": dict(hist_p=0.8, gamma=(0.7, 1.5), bias_p=0.25, bias_coeff=(0.0, 0.2),
                   affine_p=0.8, rot=0.10, trans=5, scale=0.05,
                   elastic_p=0.5, sigma=(5, 8), mag=(50, 100)),
    "heavy":  dict(hist_p=0.9, gamma=(0.5, 2.0), bias_p=0.4, bias_coeff=(0.0, 0.25),
                   affine_p=0.9, rot=0.15, trans=8, scale=0.08,
                   elastic_p=0.7, sigma=(6, 10), mag=(80, 150)),
}


def make_augmenter(spatial_size: Tuple[int, int, int] = (96, 96, 96),
                   severity: str = "medium",
                   seed: int | None = None) -> Compose:
    cfg = SEVERITY[severity]
    aug = Compose([
        # contrast — simulates a different MRI sequence
        RandHistogramShiftd(keys="image", num_control_points=8, prob=cfg["hist_p"]),
        RandAdjustContrastd(keys="image", gamma=cfg["gamma"], prob=cfg["hist_p"]),
        RandBiasFieldd(keys="image", coeff_range=cfg["bias_coeff"], prob=cfg["bias_p"]),
        # geometry — independent rigid + nonlinear per view
        RandAffined(
            keys="image",
            rotate_range=(cfg["rot"], cfg["rot"], cfg["rot"]),
            translate_range=(cfg["trans"], cfg["trans"], cfg["trans"]),
            scale_range=(cfg["scale"], cfg["scale"], cfg["scale"]),
            spatial_size=spatial_size,
            mode="bilinear",
            padding_mode="border",
            prob=cfg["affine_p"],
        ),
        Rand3DElasticd(
            keys="image",
            sigma_range=cfg["sigma"],
            magnitude_range=cfg["mag"],
            spatial_size=spatial_size,
            mode="bilinear",
            padding_mode="border",
            prob=cfg["elastic_p"],
        ),
        # renormalize after geometric/intensity perturbations
        NormalizeIntensityd(keys="image", nonzero=True),
    ])
    if seed is not None:
        aug.set_random_state(seed)
    return aug


def _apply(aug: Compose, volume: np.ndarray) -> np.ndarray:
    """volume: (D,H,W) float32. Returns (D,H,W) float32 after augmenter."""
    x = volume[None]  # add channel -> (1,D,H,W)
    out = aug({"image": x})["image"]
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
    return np.asarray(out)[0].astype(np.float32)


def augment_pair(aug: Compose,
                 q_volume: np.ndarray,
                 t_volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two INDEPENDENT augmentations (different random state per call)."""
    return _apply(aug, q_volume), _apply(aug, t_volume)


# ---------------- self-check ----------------
if __name__ == "__main__":
    import sys
    print("self-test: medium severity, 96^3 volume")
    aug = make_augmenter(severity="medium", seed=0)
    vol = np.random.randn(96, 96, 96).astype(np.float32)
    q, t = augment_pair(aug, vol, vol)  # same input, independent aug
    assert q.shape == (96, 96, 96) and t.shape == (96, 96, 96), "shape wrong"
    diff = float(np.linalg.norm(q - t) / (np.linalg.norm(vol) + 1e-8))
    print(f"two augs of the SAME input differ by relative L2 = {diff:.3f}")
    assert diff > 0.05, "augs are too similar — check that prob>0 and state is independent"
    print("OK")
