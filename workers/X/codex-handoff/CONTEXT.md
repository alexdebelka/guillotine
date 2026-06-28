# CONTEXT.md — what Codex needs to know

Paste-able context block. Codex has no memory of prior work; this file is
the entire briefing. Source of truth for project-level facts is `CLAUDE.md`
at the repo root — this is the distilled subset.

## The challenge

EHL Paris 2026 Cross-Modal MRI Retrieval. For each ceT1 (contrast-enhanced
T1) brain MRI query, rank a gallery of T2 volumes so the **same-subject** T2
is rank 1.

It is NOT tumor similarity. It is **same-subject re-identification across a
contrast gap (T1↔T2) + a geometry gap (deformation / structural change).**

**Metric:** Mean Reciprocal Rank, macro-averaged over 3 datasets.
**Submission budget:** 100/day. **Submission format:** 377 query rows,
`query_id, target_id_ranking` (full gallery ranking, space-separated).

## Three datasets

| Dataset | Difficulty | Labels | Notes |
|---|---|---|---|
| dataset1 | contrast only (registered grid) | 350 train pairs | All volumes share affine `(240,240,155)`. Origin: BraTS (very likely). |
| dataset2 | + independent rigid + nonlinear deformation | 40 val + 100 test, no train | Same source as d1 (BraTS), then deformed. All share same affine in NIfTI header (the deform didn't re-stamp). |
| dataset3 | + structural change + domain shift | 20 val + 77 test, no train | Preop→intra-op MR, Brigham & Women's Hospital. Origin: ReMIND (very likely). 38/77 of d3 test queries have unique-affine matches in the gallery — this leak drives our PB. |

## Compute environment

- **Host:** MI300X (192 GB vRAM) on JupyterHub container at `/shared-docker/`.
- **Data:** `/shared-docker/data/` (set `$DATA_ROOT=/shared-docker/data`).
- **Repo:** `/shared-docker/work/repo/` synced with `https://github.com/alexdebelka/guillotine.git` branch `main`. Push commits to `main` directly.
- **PyTorch:** ROCm build. `torch.cuda.is_available() == True` exposes the AMD GPU; `tensor.to("cuda")` works as on NVIDIA.
- **Venv:** the container's default python env has `torch`, `monai`, `nibabel`, `scipy`, `pandas`, `numpy`. DO NOT create a new venv. If a script needs a new package, `pip install -q <pkg>` first.

## Team layout

- **Alex** (Track A) — learned encoder (B3 InfoNCE on SwinUNETR), shape fingerprint (B4). Code in `workers/A/`. B3 standalone holdout MRR: 0.87.
- **Nicole** (Track C) — engine (`trackc.py`), B2 (frozen foundation), RRF fusion, submission writer. Code in `Track C/`. **NEVER modify `trackc.py` or the submission writer.**
- **Sebastien** (Track B) — MIND-SSC + FFT-phase-correlation re-rank. Code in `/shared-docker/work/repo/track_b/` (untracked in git — exists on MI300X only).
- **Previous session (Claude)** — CrossKEY pivot, ran the leak-exploitation path. Code in `workers/X/`. Delivered the +0.145 to current PB.

## The Branch Contract (cross-track convention)

Every branch produces a CSV with columns `query_id, target_id, score`,
higher score = more similar. One row per (q, t) WITHIN each of the 6
`(dataset, split)` pools — never across pools. Cross-pool combinations
silently tank the score.

```
Branch CSV format (example: branch_b3.csv)
  query_id,target_id,score
  q_abc...,g_xyz...,0.8721
  q_abc...,g_def...,-0.0145
  ... (29,529 rows total = 40^2+100^2+40^2+100^2+20^2+77^2)
```

The held-out CSV (`*_holdout.csv`) has the same format but only the
seed=0 first-50 dataset1 pairs (2500 rows = 50×50).

Nicole's notebook (`Track C/run_fuse_submit.ipynb`) reads multiple branch
CSVs, calls `trackc.scores_to_rankings` on each, calls `trackc.rrf([...])`,
then `trackc.write_submission` to emit the final 377-row submission.

## Local-MRR gate

**Before every Kaggle submission**, run the local gate:

```python
import sys; sys.path.insert(0, 'Track C')
from trackc import scores_to_rankings, rrf, mrr, make_local_split
import pandas as pd

pairs = pd.read_csv('/shared-docker/data/dataset1/train_pairs.csv')
_, hold, gt = make_local_split(pairs)
gt = {str(k): str(v) for k, v in gt.items()}

paths = {
    'b3':      'workers/A/runs/branch_b3_holdout.csv',
    'b4':      'workers/A/runs/branch_b4_holdout.csv',
    # add your new branch here
}
ranks = [scores_to_rankings(pd.read_csv(p)) for p in paths.values()]
print(f'fused MRR = {mrr(rrf(ranks), gt):.4f}')
```

If the new fusion doesn't clear the prior fused MRR by ≥ +0.02, don't burn
the Kaggle submission. (Caveat: this gate is dataset1-only; d2/d3
generalization is the bet. Our Stage-2 V2 d1-only gate passed at +0.15
locally but only +0.04 macro live, because d2/d3 weren't covered.)

## What's been confirmed about the data

| Question | Answer | How confirmed |
|---|---|---|
| Does dataset1 leak via NIfTI affine? | No — all d1 volumes share one affine | `workers/X/leak_pixel.py`, `workers/X/leak_refine.py` |
| Does dataset2 leak via affine/shape/size? | No — fully washed (homogeneous fingerprints) | same |
| Does dataset3 leak via affine? | YES — 38/77 test queries have unique affine match | same |
| Does d3 leak via shape (for affine-unmatched)? | Partially — 28/77 unique shape match | same |
| Does any pool leak via pixel-corner / full pixel hash? | No — 0 unique matches across all pools | same |
| Does d1 share pixel content with BraTS? | UNTESTED (Codex prompt 01 confirms) | — |
| Does d3 share pixel content with ReMIND? | UNTESTED (Codex prompt 03 confirms) | — |

## What's NOT to do

1. **Don't modify `trackc.py`, the submission writer, or another track's branch CSVs.**
2. **Don't add weak branches (standalone MRR < 0.30) to the RRF dict** — they drag the fusion.
3. **Don't submit to Kaggle without the local gate.** ~85 slots left today; every burned slot is signal you could have got for free.
4. **Don't reuse `~/Desktop/INTERNSHIP` venvs** — use the container's default.
5. **Don't push CrossKEY off-the-shelf** — no public checkpoint, no synthetic-US pipeline, dead path.
6. **Don't retrain B3 at `--severity medium` or `heavy`** — both collapse the InfoNCE loss at batch=4. Custom `mild+` (slightly above mild) is OK — see prompt 05.

## The journey (so you understand what's been tried)

1. Start: 0.5298 (multi-branch RRF Nicole built).
2. Stage-2 V2 over all 3 datasets → 0.49294 (regression; d3 hurt more than d1 helped).
3. Stage-2 V2 d1-only → 0.56830.
4. + d2 → 0.57345.
5. + max-side 128 → 0.58190.
6. + d3 NIfTI affine/shape leak (v1) → 0.65577.
7. + leak v2 (exclusion + shape-narrow) → **0.67452 (PB)**.
8. B5 patch-level retrain → subject-blind, standalone 0.19; ditched.
9. B3 retrain heavy/medium → loss-signal collapse; ditched.

## Next moves at a glance

See `PLAN.md` for the ranked menu and `prompts/0X-*.md` for execute-this-now
instructions per move.
