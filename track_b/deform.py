from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .io import load_nifti_array


def _seed_from_key(key: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _random_rotation_matrix(generator: torch.Generator, max_degrees: float) -> torch.Tensor:
    angles = (torch.rand(3, generator=generator) * 2.0 - 1.0) * math.radians(max_degrees)
    cx, cy, cz = torch.cos(angles)
    sx, sy, sz = torch.sin(angles)
    cx, cy, cz = float(cx), float(cy), float(cz)
    sx, sy, sz = float(sx), float(sy), float(sz)

    rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=torch.float32,
    )
    ry = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=torch.float32,
    )
    rz = torch.tensor(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return rz @ ry @ rx


def _random_affine_theta(
    generator: torch.Generator,
    *,
    max_degrees: float,
    translation: float,
    scale_range: tuple[float, float],
) -> torch.Tensor:
    rotation = _random_rotation_matrix(generator, max_degrees=max_degrees)
    scale = torch.empty(3).uniform_(scale_range[0], scale_range[1], generator=generator)
    theta = rotation * scale.unsqueeze(0)
    shift = (torch.rand(3, generator=generator) * 2.0 - 1.0) * translation
    affine = torch.cat([theta, shift[:, None]], dim=1)
    return affine


def _smooth_displacement(
    shape: tuple[int, int, int],
    generator: torch.Generator,
    *,
    coarse_size: tuple[int, int, int],
    magnitude: float,
) -> torch.Tensor:
    noise = torch.randn((1, 3, *coarse_size), generator=generator)
    displacement = F.interpolate(noise, size=shape, mode="trilinear", align_corners=False)[0]
    displacement = displacement / displacement.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return displacement * magnitude


def deform_volume(
    volume: np.ndarray,
    *,
    seed: int,
    max_degrees: float = 12.0,
    translation: float = 0.08,
    scale_range: tuple[float, float] = (0.92, 1.08),
    coarse_size: tuple[int, int, int] = (6, 6, 6),
    elastic_magnitude: float = 0.12,
) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {tuple(volume.shape)}")

    tensor = torch.from_numpy(np.asarray(volume, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    generator = torch.Generator().manual_seed(seed)
    theta = _random_affine_theta(
        generator,
        max_degrees=max_degrees,
        translation=translation,
        scale_range=scale_range,
    ).unsqueeze(0)
    grid = F.affine_grid(theta, tensor.shape, align_corners=False)
    elastic = _smooth_displacement(volume.shape, generator, coarse_size=coarse_size, magnitude=elastic_magnitude)
    grid = grid + elastic.permute(1, 2, 3, 0).unsqueeze(0)
    warped = F.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return warped[0, 0].cpu().numpy().astype(np.float32, copy=False)


def write_deformed_nifti(
    source_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    max_degrees: float = 12.0,
    translation: float = 0.08,
    scale_range: tuple[float, float] = (0.92, 1.08),
    coarse_size: tuple[int, int, int] = (6, 6, 6),
    elastic_magnitude: float = 0.12,
) -> Path:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("nibabel is required to write deformed NIfTI files") from exc

    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = nib.load(str(source_path))
    volume = load_nifti_array(source_path)
    deformed = deform_volume(
        volume,
        seed=seed,
        max_degrees=max_degrees,
        translation=translation,
        scale_range=scale_range,
        coarse_size=coarse_size,
        elastic_magnitude=elastic_magnitude,
    )
    nib.save(nib.Nifti1Image(deformed, affine=image.affine, header=image.header), str(output_path))
    return output_path
