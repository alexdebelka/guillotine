from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ManifestEntry:
    item_id: str
    image_path: Path
    dataset: str


def open_text(path: str | Path, mode: str = "r"):
    file_path = Path(path)
    if file_path.suffix == ".gz":
        return gzip.open(file_path, mode + "t", newline="")
    return file_path.open(mode, newline="")


def read_manifest(path: str | Path, root: str | Path | None = None) -> list[ManifestEntry]:
    root_path = Path(root) if root is not None else None
    entries: list[ManifestEntry] = []
    with open_text(path, "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_key = "query_image" if "query_image" in row else "target_image"
            item_key = "query_id" if "query_id" in row else "target_id"
            image_path = Path(row[image_key])
            if root_path is not None and not image_path.is_absolute():
                image_path = root_path / image_path
            entries.append(
                ManifestEntry(
                    item_id=row[item_key],
                    image_path=image_path,
                    dataset=row.get("dataset", ""),
                )
            )
    return entries


def load_nifti_array(path: str | Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("nibabel is required to load NIfTI files") from exc
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {data.shape} for {path}")
    return np.ascontiguousarray(data)


def zscore_volume(volume: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    values = volume[mask] if mask is not None else volume[np.isfinite(volume)]
    if values.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std())
    if std < eps:
        return (volume - mean).astype(np.float32)
    return ((volume - mean) / std).astype(np.float32)


def nonzero_mask(volume: np.ndarray) -> np.ndarray:
    mask = np.isfinite(volume) & (np.abs(volume) > 0)
    if mask.any():
        return mask
    return np.isfinite(volume)


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    resolved = Path(path)
    if root is not None and not resolved.is_absolute():
        resolved = Path(root) / resolved
    return resolved


def write_ranking_csv(rows: Iterable[tuple[str, list[str]]], path: str | Path) -> None:
    with open_text(path, "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "target_id_ranking"])
        for query_id, ranking in rows:
            writer.writerow([query_id, " ".join(ranking)])
