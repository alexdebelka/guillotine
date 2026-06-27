from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .io import load_nifti_array, nonzero_mask, zscore_volume
from .mind import mind_descriptor


def _downsample_for_rerank(volume: np.ndarray, max_side: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))
    depth, height, width = tensor.shape
    longest = max(depth, height, width)
    if longest > max_side:
        scale = max_side / float(longest)
        tensor = F.interpolate(
            tensor[None, None],
            size=(
                max(1, int(round(depth * scale))),
                max(1, int(round(height * scale))),
                max(1, int(round(width * scale))),
            ),
            mode="trilinear",
            align_corners=False,
        )[0, 0]
    return tensor


def _phase_correlation(reference: torch.Tensor, moving: torch.Tensor) -> tuple[tuple[int, int, int], float]:
    if reference.shape != moving.shape:
        raise ValueError("phase correlation expects equal shapes")
    ref_fft = torch.fft.fftn(reference)
    mov_fft = torch.fft.fftn(moving)
    cross_power = ref_fft * torch.conj(mov_fft)
    cross_power = cross_power / cross_power.abs().clamp_min(1e-8)
    corr = torch.fft.ifftn(cross_power).real
    flat_index = int(torch.argmax(corr).item())
    z, y, x = np.unravel_index(flat_index, tuple(corr.shape))
    shifts = []
    for axis, size in zip((z, y, x), corr.shape):
        shift = int(axis)
        if shift > size // 2:
            shift -= size
        shifts.append(shift)
    return (shifts[0], shifts[1], shifts[2]), float(corr.reshape(-1)[flat_index].item())


def _apply_integer_shift(volume: torch.Tensor, shift: tuple[int, int, int]) -> torch.Tensor:
    dz, dy, dx = shift
    if volume.ndim == 3:
        working = volume.unsqueeze(0)
    elif volume.ndim == 4:
        working = volume
    else:
        raise ValueError(f"expected 3D or 4D tensor, got shape {tuple(volume.shape)}")

    depth, height, width = working.shape[-3:]
    pad = (
        max(dx, 0), max(-dx, 0),
        max(dy, 0), max(-dy, 0),
        max(dz, 0), max(-dz, 0),
    )
    padded = F.pad(working.unsqueeze(0), pad, mode="replicate")[0]
    z0 = max(-dz, 0)
    y0 = max(-dy, 0)
    x0 = max(-dx, 0)
    shifted = padded[..., z0 : z0 + depth, y0 : y0 + height, x0 : x0 + width]
    return shifted[0] if volume.ndim == 3 else shifted


def _centered_ncc(reference: torch.Tensor, moving: torch.Tensor, eps: float = 1e-6) -> float:
    ref = reference - reference.mean()
    mov = moving - moving.mean()
    denom = ref.norm() * mov.norm()
    if float(denom) < eps:
        return 0.0
    return float((ref.flatten() @ mov.flatten()) / denom)


def rerank_pair(
    query_path: str | Path,
    candidate_path: str | Path,
    *,
    max_side: int = 96,
    mind_pool_shape: tuple[int, int, int] = (4, 4, 4),
    search_radius: int = 4,
) -> float:
    query_raw = load_nifti_array(query_path)
    candidate_raw = load_nifti_array(candidate_path)
    query_volume = zscore_volume(query_raw, mask=nonzero_mask(query_raw))
    candidate_volume = zscore_volume(candidate_raw, mask=nonzero_mask(candidate_raw))
    query_tensor = _downsample_for_rerank(query_volume, max_side=max_side)
    candidate_tensor = _downsample_for_rerank(candidate_volume, max_side=max_side)

    if query_tensor.shape != candidate_tensor.shape:
        shared_shape = tuple(min(a, b) for a, b in zip(query_tensor.shape, candidate_tensor.shape))
        query_tensor = F.interpolate(
            query_tensor[None, None],
            size=shared_shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]
        candidate_tensor = F.interpolate(
            candidate_tensor[None, None],
            size=shared_shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]

    query_descriptor = mind_descriptor(query_tensor.numpy(), max_side=max_side)
    candidate_descriptor = mind_descriptor(candidate_tensor.numpy(), max_side=max_side)

    query_energy = query_descriptor.mean(dim=0)
    candidate_energy = candidate_descriptor.mean(dim=0)
    shift, corr_score = _phase_correlation(query_energy, candidate_energy)
    shift = tuple(int(np.clip(axis, -search_radius, search_radius)) for axis in shift)
    aligned_candidate = _apply_integer_shift(candidate_descriptor, shift)

    query_feature = F.adaptive_avg_pool3d(query_descriptor[None], output_size=mind_pool_shape)[0].flatten()
    aligned_feature = F.adaptive_avg_pool3d(aligned_candidate[None], output_size=mind_pool_shape)[0].flatten()
    query_feature = F.normalize(query_feature, dim=0)
    aligned_feature = F.normalize(aligned_feature, dim=0)
    mind_score = float(query_feature @ aligned_feature)
    intensity_score = _centered_ncc(query_tensor, _apply_integer_shift(candidate_tensor, shift))
    return 0.65 * mind_score + 0.20 * intensity_score + 0.15 * corr_score


@dataclass(frozen=True)
class RankedCandidate:
    target_id: str
    score: float


def rerank_ranked_candidates(
    query_path: str | Path,
    ranked_candidates: Iterable[tuple[str, str | Path]],
    *,
    top_k: int = 10,
    max_side: int = 96,
    mind_pool_shape: tuple[int, int, int] = (4, 4, 4),
) -> list[RankedCandidate]:
    scored: list[RankedCandidate] = []
    for index, (target_id, candidate_path) in enumerate(ranked_candidates):
        if index >= top_k:
            scored.append(RankedCandidate(target_id=target_id, score=float(top_k - index)))
            continue
        score = rerank_pair(
            query_path,
            candidate_path,
            max_side=max_side,
            mind_pool_shape=mind_pool_shape,
        )
        scored.append(RankedCandidate(target_id=target_id, score=score))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored
