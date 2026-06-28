"""
Track C engine — EHL Paris 2026 cross-modal MRI retrieval.

Owns the shared infrastructure the whole team plugs into:
  1. Data manifest / robust path resolution         -> load_manifest, resolve_image
  2. Shared preprocessing (resample 1mm + z-score)  -> load_volume
  3. Local MRR harness + 300/50 split + L2 proxy    -> mrr, make_local_split, deform_volume
  4. Reciprocal Rank Fusion                          -> rrf
  5. submission.csv writer + validator               -> write_submission, validate_submission
  6. Cheap baseline branch (pipeline smoke test)     -> embed_baseline, rank_by_embeddings

Branch contract (every branch -> Track C):
    a CSV / DataFrame with columns [query_id, target_id, score]   (higher score = more similar)
RRF and the submission writer are the ONLY things that produce the final file.

Pure-Python where possible; heavy deps (nibabel, scipy) are imported lazily so the
metric self-test runs anywhere. Run `python trackc.py` to execute the self-tests.
"""
from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- config
DATA_ROOT = "data"                       # folder containing dataset1/2/3 (edit if needed)
DATASETS = ["dataset1", "dataset2", "dataset3"]
SPLITS = ["val", "test"]                 # the 6 submission pools = DATASETS x SPLITS
RRF_K = 60                               # RRF damping constant (Cormack 2009 default)
SEED = 0

# Official submission format (from the challenge brief). One combined CSV:
#   query_id,target_id_ranking      with target_ids SPACE-separated, most -> least similar.
SUBMISSION_COLS = ["query_id", "target_id_ranking"]
RANKING_SEP = " "


# ============================================================ 1. manifest & path resolve
def resolve_image(rel_or_abs_path: str, data_root: str = DATA_ROOT) -> str:
    """Return an existing file path. CSVs reference '.nii.gz' but some files are '.nii';
    try the given path, then swap the extension, under data_root and as-is."""
    cands = []
    p = rel_or_abs_path
    for base in (p, os.path.join(data_root, p)):
        cands.append(base)
        if base.endswith(".nii.gz"):
            cands.append(base[:-3])          # -> .nii
        elif base.endswith(".nii"):
            cands.append(base + ".gz")       # -> .nii.gz
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"None of these exist for {rel_or_abs_path!r}: {cands}")


@dataclass
class Pool:
    """One (dataset, split) retrieval pool."""
    dataset: str
    split: str
    queries: pd.DataFrame          # cols: query_id, query_image[, ...]
    gallery: pd.DataFrame          # cols: target_id, target_image[, ...]

    @property
    def query_ids(self):  return self.queries["query_id"].tolist()
    @property
    def gallery_ids(self): return self.gallery["target_id"].tolist()


def load_manifest(data_root: str = DATA_ROOT) -> dict:
    """Read every queries/gallery CSV into a {(dataset, split): Pool} dict.
    Also attaches manifest['train_pairs'] = labeled dataset1 pairs DataFrame."""
    manifest = {}
    for ds in DATASETS:
        for split in SPLITS:
            qcsv = os.path.join(data_root, ds, f"{split}_queries.csv")
            gcsv = os.path.join(data_root, ds, f"{split}_gallery.csv")
            if os.path.exists(qcsv) and os.path.exists(gcsv):
                manifest[(ds, split)] = Pool(
                    ds, split, pd.read_csv(qcsv), pd.read_csv(gcsv)
                )
    tp = os.path.join(data_root, "dataset1", "train_pairs.csv")
    manifest["train_pairs"] = pd.read_csv(tp) if os.path.exists(tp) else None
    return manifest


def manifest_summary(manifest: dict) -> pd.DataFrame:
    rows = []
    for key, pool in manifest.items():
        if key == "train_pairs":
            continue
        rows.append({"dataset": pool.dataset, "split": pool.split,
                     "n_queries": len(pool.queries), "n_gallery": len(pool.gallery)})
    df = pd.DataFrame(rows).sort_values(["dataset", "split"]).reset_index(drop=True)
    total_q = df["n_queries"].sum()
    print(df.to_string(index=False))
    print(f"TOTAL query rows (submission length) = {total_q}  (expect 377)")
    tp = manifest.get("train_pairs")
    print(f"labeled train pairs (dataset1)        = {0 if tp is None else len(tp)}  (expect 350)")
    return df


# ============================================================ 2. preprocessing / loading
def load_volume(path, resample_1mm=True, zscore=True, data_root=DATA_ROOT):
    """Load a NIfTI volume as float32 np.ndarray. Resample to 1mm iso + per-volume z-score.
    Lazy-imports nibabel/scipy so the rest of the module works without them."""
    import nibabel as nib
    img = nib.load(resolve_image(path, data_root))
    if resample_1mm:
        from nibabel.processing import resample_to_output
        img = resample_to_output(img, voxel_sizes=(1.0, 1.0, 1.0), order=1)
    vol = np.asarray(img.get_fdata(), dtype=np.float32)
    if zscore:
        m, s = float(vol.mean()), float(vol.std())
        vol = (vol - m) / (s + 1e-6)
    return vol


def resize_to(vol: np.ndarray, shape=(64, 64, 64)) -> np.ndarray:
    """Trilinear resize to a fixed shape (for cheap fixed-length descriptors)."""
    from scipy.ndimage import zoom
    factors = [t / s for t, s in zip(shape, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


# ============================================================ 3. MRR harness + split + L2
def mrr(rankings: dict, ground_truth: dict) -> float:
    """rankings: {query_id: [target_id ordered most->least similar]}.
    ground_truth: {query_id: true_target_id}. Returns mean reciprocal rank.
    RR = 1/rank of the true target (1-indexed); 0 if absent."""
    rrs = []
    for q, truth in ground_truth.items():
        ranked = rankings.get(q, [])
        rr = 0.0
        for i, t in enumerate(ranked, start=1):
            if t == truth:
                rr = 1.0 / i
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


def make_local_split(train_pairs: pd.DataFrame, n_holdout=50, seed=SEED):
    """Split the 350 labeled dataset1 pairs into train / held-out gallery+query.
    The held-out set is a self-contained retrieval problem: its queries are ranked
    against ONLY its own targets (a clean MRR proxy for Level-1).
    Returns (train_df, holdout_df, ground_truth{query_id: target_id})."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_pairs))
    hold = train_pairs.iloc[idx[:n_holdout]].reset_index(drop=True)
    train = train_pairs.iloc[idx[n_holdout:]].reset_index(drop=True)
    gt = dict(zip(hold["query_id"], hold["target_id"]))
    return train, hold, gt


def deform_volume(vol: np.ndarray, rng=None, max_rot_deg=10.0,
                  max_shift=8.0, elastic_sigma=12.0, elastic_alpha=18.0):
    """Independent random rigid (rotation+shift) + nonlinear (elastic) deformation.
    Apply to held-out volumes to build the Level-2 (deformation) MRR proxy offline."""
    from scipy.ndimage import rotate, shift, gaussian_filter, map_coordinates
    rng = rng or np.random.default_rng()
    v = vol
    # rigid: small rotations about each axis + translation
    for axes in [(0, 1), (0, 2), (1, 2)]:
        ang = rng.uniform(-max_rot_deg, max_rot_deg)
        v = rotate(v, ang, axes=axes, reshape=False, order=1, mode="nearest")
    v = shift(v, rng.uniform(-max_shift, max_shift, size=3), order=1, mode="nearest")
    # nonlinear: smooth random displacement field
    coords = np.meshgrid(*[np.arange(s) for s in v.shape], indexing="ij")
    disp = [gaussian_filter(rng.uniform(-1, 1, v.shape), elastic_sigma) * elastic_alpha
            for _ in range(3)]
    warped = [c + d for c, d in zip(coords, disp)]
    v = map_coordinates(v, warped, order=1, mode="nearest").reshape(v.shape)
    return v.astype(np.float32)


# ============================================================ 4. Reciprocal Rank Fusion
def rrf(branch_rankings: list, weights=None, k=RRF_K) -> dict:
    """Fuse multiple branch rankings. Each element of branch_rankings is
    {query_id: [target_id ordered most->least similar]}. Returns fused
    {query_id: [target_id ...]}. weights: optional list, one per branch."""
    weights = weights or [1.0] * len(branch_rankings)
    assert len(weights) == len(branch_rankings)
    queries = set().union(*[set(r) for r in branch_rankings]) if branch_rankings else set()
    fused = {}
    for q in queries:
        scores = {}
        for w, ranking in zip(weights, branch_rankings):
            for rank, t in enumerate(ranking.get(q, []), start=1):
                scores[t] = scores.get(t, 0.0) + w / (k + rank)
        fused[q] = [t for t, _ in sorted(scores.items(), key=lambda x: -x[1])]
    return fused


def scores_to_rankings(df: pd.DataFrame) -> dict:
    """Convert a branch's [query_id, target_id, score] table into
    {query_id: [target_id ordered by descending score]}."""
    out = {}
    for q, grp in df.sort_values("score", ascending=False).groupby("query_id", sort=False):
        out[q] = grp["target_id"].tolist()
    return out


# ============================================================ 5. submission writer/validator
def write_submission(pool_rankings: dict, manifest: dict, path="submission.csv") -> pd.DataFrame:
    """pool_rankings: {(dataset, split): {query_id: [target_id full ranking]}}.
    Writes the single combined CSV (query_id,target_id_ranking). Returns the DataFrame."""
    rows = []
    for (ds, split), ranking in pool_rankings.items():
        for q, ranked in ranking.items():
            rows.append({"query_id": q, "target_id_ranking": RANKING_SEP.join(map(str, ranked))})
    df = pd.DataFrame(rows, columns=SUBMISSION_COLS)
    df.to_csv(path, index=False)
    return df


def validate_submission(df: pd.DataFrame, manifest: dict) -> bool:
    """Hard checks before burning a Kaggle submission. Raises on any problem."""
    assert list(df.columns) == SUBMISSION_COLS, f"bad columns: {list(df.columns)}"
    assert df["query_id"].is_unique, "duplicate query_id rows"
    # build the set of valid (query_id) and per-query expected gallery from the manifest
    q_to_pool = {}
    for key, pool in manifest.items():
        if key == "train_pairs":
            continue
        for q in pool.query_ids:
            q_to_pool[q] = set(pool.gallery_ids)
    expected_n = sum(len(p.queries) for k, p in manifest.items() if k != "train_pairs")
    assert len(df) == expected_n, f"{len(df)} rows, expected {expected_n}"
    for _, r in df.iterrows():
        q = r["query_id"]
        assert q in q_to_pool, f"unknown query_id {q}"
        ranked = r["target_id_ranking"].split(RANKING_SEP)
        gal = q_to_pool[q]
        assert len(ranked) == len(gal), f"{q}: ranked {len(ranked)} != gallery {len(gal)}"
        assert set(ranked) == gal, f"{q}: ranking is not a permutation of its gallery"
    print(f"submission OK: {len(df)} rows, all rankings are valid same-pool permutations.")
    return True


# ============================================================ 6. baseline branch (smoke test)
def embed_baseline(path, shape=(48, 48, 48), use_gradient=True, data_root=DATA_ROOT):
    """Cheap fixed-length descriptor to validate the pipeline end-to-end.
    Gradient-magnitude (somewhat contrast-robust) of a downsampled volume, L2-normalized.
    NOTE: placeholder — B1 (MIND), B2 (foundation), B3, B4 replace this."""
    vol = load_volume(path, resample_1mm=False, zscore=True, data_root=data_root)
    vol = resize_to(vol, shape)
    if use_gradient:
        gx, gy, gz = np.gradient(vol)
        vol = np.sqrt(gx**2 + gy**2 + gz**2)
    v = vol.ravel().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def rank_by_embeddings(query_emb: dict, gallery_emb: dict) -> dict:
    """Cosine-similarity ranking. query_emb/gallery_emb: {id: vector}.
    Returns {query_id: [target_id ordered most->least similar]}."""
    gids = list(gallery_emb)
    G = np.stack([gallery_emb[g] for g in gids])           # (Ng, D)
    out = {}
    for q, qe in query_emb.items():
        sims = G @ qe
        out[q] = [gids[i] for i in np.argsort(-sims)]
    return out


def embeddings_to_scores_df(query_emb, gallery_emb) -> pd.DataFrame:
    """Emit the branch contract [query_id, target_id, score] for RRF / sharing."""
    gids = list(gallery_emb)
    G = np.stack([gallery_emb[g] for g in gids])
    rows = []
    for q, qe in query_emb.items():
        sims = G @ qe
        for g, s in zip(gids, sims):
            rows.append({"query_id": q, "target_id": g, "score": float(s)})
    return pd.DataFrame(rows)


# ============================================================ self-tests (no data needed)
def _selftest():
    # --- MRR: hand-checked toy ---
    rankings = {
        "q1": ["a", "b", "c"],   # truth a -> rank 1 -> RR 1
        "q2": ["a", "b", "c"],   # truth c -> rank 3 -> RR 1/3
        "q3": ["a", "b", "c"],   # truth x absent -> RR 0
    }
    gt = {"q1": "a", "q2": "c", "q3": "x"}
    got = mrr(rankings, gt)
    exp = (1 + 1/3 + 0) / 3
    assert abs(got - exp) < 1e-9, (got, exp)

    # --- RRF: a target both branches rank highly should win ---
    b1 = {"q1": ["a", "b", "c"]}
    b2 = {"q1": ["b", "a", "c"]}
    fused = rrf([b1, b2])
    assert fused["q1"][0] in ("a", "b") and fused["q1"][-1] == "c", fused

    # --- RRF weighting: heavily up-weighting b2 must pull its #1 (b) to the top ---
    fused_w = rrf([b1, b2], weights=[0.01, 1.0])
    assert fused_w["q1"][0] == "b", fused_w

    # --- scores_to_rankings ---
    df = pd.DataFrame({"query_id": ["q1", "q1"], "target_id": ["a", "b"], "score": [0.1, 0.9]})
    assert scores_to_rankings(df)["q1"] == ["b", "a"]

    print("all self-tests passed ✔")


if __name__ == "__main__":
    _selftest()
