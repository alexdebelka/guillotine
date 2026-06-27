from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .io import load_nifti_array, nonzero_mask, zscore_volume


DEFAULT_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def _as_tensor(volume: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))
    return tensor


def _resize_to_max_side(volume: torch.Tensor, max_side: int) -> torch.Tensor:
    depth, height, width = volume.shape
    longest = max(depth, height, width)
    if longest <= max_side:
        return volume
    scale = max_side / float(longest)
    new_shape = (
        max(1, int(round(depth * scale))),
        max(1, int(round(height * scale))),
        max(1, int(round(width * scale))),
    )
    resized = F.interpolate(
        volume[None, None],
        size=new_shape,
        mode="trilinear",
        align_corners=False,
    )
    return resized[0, 0]


def _shift(volume: torch.Tensor, offset: tuple[int, int, int]) -> torch.Tensor:
    dz, dy, dx = offset
    depth, height, width = volume.shape
    pad = (
        max(dx, 0), max(-dx, 0),
        max(dy, 0), max(-dy, 0),
        max(dz, 0), max(-dz, 0),
    )
    padded = F.pad(volume[None, None], pad, mode="replicate")[0, 0]
    z0 = max(-dz, 0)
    y0 = max(-dy, 0)
    x0 = max(-dx, 0)
    return padded[z0 : z0 + depth, y0 : y0 + height, x0 : x0 + width]


def mind_descriptor(
    volume: np.ndarray | torch.Tensor,
    *,
    patch_size: int = 3,
    max_side: int = 96,
    offsets: Iterable[tuple[int, int, int]] = DEFAULT_OFFSETS,
    eps: float = 1e-6,
) -> torch.Tensor:
    if isinstance(volume, np.ndarray):
        vol = _as_tensor(volume)
    else:
        vol = volume.to(dtype=torch.float32, copy=True)
    if vol.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {tuple(vol.shape)}")
    vol_np = vol.cpu().numpy()
    vol = _as_tensor(zscore_volume(vol_np, mask=nonzero_mask(vol_np)))
    vol = _resize_to_max_side(vol, max_side=max_side)
    vol = vol[None, None]

    padding = patch_size // 2
    ssd_maps = []
    base = vol[0, 0]
    for offset in offsets:
        shifted = _shift(base, offset)
        diff2 = (base - shifted).pow(2)[None, None]
        pooled = F.avg_pool3d(diff2, kernel_size=patch_size, stride=1, padding=padding)
        ssd_maps.append(pooled)
    ssd = torch.cat(ssd_maps, dim=1)
    variance = ssd.mean(dim=1, keepdim=True).clamp_min(eps)
    mind = torch.exp(-ssd / variance)
    mind = mind / mind.sum(dim=1, keepdim=True).clamp_min(eps)
    return mind[0]


def mind_vector(
    volume: np.ndarray | torch.Tensor,
    *,
    pool_shape: tuple[int, int, int] = (4, 4, 4),
    patch_size: int = 3,
    max_side: int = 96,
    offsets: Iterable[tuple[int, int, int]] = DEFAULT_OFFSETS,
) -> torch.Tensor:
    descriptor = mind_descriptor(
        volume,
        patch_size=patch_size,
        max_side=max_side,
        offsets=offsets,
    )
    pooled = F.adaptive_avg_pool3d(descriptor[None], output_size=pool_shape)[0]
    vector = pooled.flatten()
    return F.normalize(vector, dim=0)


@dataclass(frozen=True)
class DescriptorCache:
    root: Path

    def path_for(self, volume_path: str | Path, *, suffix: str = ".pt") -> Path:
        stem = Path(volume_path).name.replace(".nii.gz", "").replace(".nii", "")
        return self.root / f"{stem}{suffix}"

    def get(self, volume_path: str | Path, **kwargs) -> torch.Tensor:
        cache_path = self.path_for(volume_path)
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = mind_vector(load_nifti_array(volume_path), **kwargs)
        torch.save(descriptor, cache_path)
        return descriptor


def similarity_matrix(query_vectors: list[torch.Tensor], gallery_vectors: list[torch.Tensor]) -> np.ndarray:
    query = torch.stack(query_vectors)
    gallery = torch.stack(gallery_vectors)
    query = F.normalize(query, dim=1)
    gallery = F.normalize(gallery, dim=1)
    scores = query @ gallery.T
    return scores.cpu().numpy()
