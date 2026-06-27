"""
Track A — B4 shape fingerprint (V1).

Goal: orthogonal-to-B3 signal that's contrast-agnostic by construction. B3 uses
deep semantic features (anisotropic, contrast-trained). B4 uses pure GEOMETRY:
where is brain tissue in space, regardless of intensity.

V1 recipe — simplest contrast-agnostic fingerprint:
  1. Threshold the volume to a brain mask (mean*0.1 cutoff — crude but works).
  2. Crop tightly to the brain bbox  (kills translation).
  3. Resample the cropped mask to a fixed 16³ cube  (kills scale).
  4. Flatten → 4096-d, L2-normalize.

What this captures: the "shape envelope" of the brain at coarse resolution.
What it ignores: intensity, contrast, absolute position, absolute size.

ponytail: pure numpy/scipy, no GPU, no extra deps. SynthSeg + per-region features
ponytail: would be a stronger V2 — add if V1 doesn't lift RRF on local MRR.

V1 weakness: not rotation-invariant. Fine for dataset1 (registered) and partially
covered by RRF fusion on dataset2/3. If rotation kills it, the upgrade path is
PCA-canonicalize-orientation before resampling, or Hu/Zernike moments.

Output format matches embed_b3.py exactly so Track C's RRF takes both branches
identically.
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

SHAPE_GRID = 16  # 16^3 = 4096 dims
FEATURE_DIM = SHAPE_GRID ** 3


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


def shape_fingerprint(volume_path: str) -> np.ndarray:
    """Read NIfTI, return unit-norm float32 (4096,) shape fingerprint."""
    img = nib.load(volume_path)
    vol = img.get_fdata().astype(np.float32)
    # crude brain mask. 0.1*mean works because raw MRI intensities are positive
    # and brain voxels are well above the background mean.
    thresh = vol.mean() * 0.1
    mask = vol > thresh
    if mask.sum() < 100:  # empty or near-empty volume
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    coords = np.argwhere(mask)
    mins, maxs = coords.min(axis=0), coords.max(axis=0) + 1
    crop = mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]].astype(np.float32)
    factors = [SHAPE_GRID / s for s in crop.shape]
    cube = zoom(crop, factors, order=1)  # linear interp keeps soft mask values
    vec = cube.flatten()
    return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)


def embed_pool(csv_path: str, data_root: str, id_col: str, img_col: str) -> dict:
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(csv_path)):
        vol_path = os.path.join(data_root, r[img_col])
        out[str(r[id_col])] = shape_fingerprint(vol_path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/shared-docker/data"))
    ap.add_argument("--out", required=True, help="output .pkl path")
    args = ap.parse_args()

    print(f"B4 shape fingerprint  grid={SHAPE_GRID}^3  dim={FEATURE_DIM}")
    result = {"dataset1": {}, "dataset2": {}, "dataset3": {}}
    for ds, csv_name, id_col, img_col in POOLS:
        csv_path = os.path.join(args.data_root, ds, f"{csv_name}.csv")
        result[ds][csv_name] = embed_pool(csv_path, args.data_root, id_col, img_col)

    with open(args.out, "wb") as f:
        pickle.dump(result, f)

    total = sum(len(p) for ds in result.values() for p in ds.values())
    print(f"saved {args.out}  total embeddings: {total}")


def _smoke():
    """assert-based wire check on a synthetic brain-like blob."""
    # synthesize a 3D ellipsoid (brain-like)
    grid = np.indices((96, 96, 96), dtype=np.float32)
    cx, cy, cz = 48, 48, 48
    rx, ry, rz = 30, 40, 25
    blob = (((grid[0]-cx)/rx)**2 + ((grid[1]-cy)/ry)**2 + ((grid[2]-cz)/rz)**2) < 1.0
    blob = blob.astype(np.float32) * 1000.0  # MRI-like intensity range
    # save to temp NIfTI and round-trip
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    tmp.close()
    nib.save(nib.Nifti1Image(blob, np.eye(4)), tmp.name)
    vec = shape_fingerprint(tmp.name)
    os.unlink(tmp.name)
    assert vec.shape == (FEATURE_DIM,), f"unexpected shape: {vec.shape}"
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-3, f"not unit-norm: {np.linalg.norm(vec)}"
    assert vec.sum() > 0, "all-zero fingerprint"
    # second blob with different size — fingerprint should differ
    blob2 = (((grid[0]-cx)/(rx*1.4))**2 + ((grid[1]-cy)/ry)**2 + ((grid[2]-cz)/rz)**2) < 1.0
    blob2 = blob2.astype(np.float32) * 1000.0
    tmp2 = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    tmp2.close()
    nib.save(nib.Nifti1Image(blob2, np.eye(4)), tmp2.name)
    vec2 = shape_fingerprint(tmp2.name)
    os.unlink(tmp2.name)
    sim = float(vec @ vec2)
    assert sim < 0.999, f"different shapes gave identical fingerprint: sim={sim}"
    print(f"smoke OK  dim={vec.shape[0]}  |v|={np.linalg.norm(vec):.4f}  "
          f"two-blob sim={sim:.4f}")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
    else:
        main()
