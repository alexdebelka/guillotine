# Task for Worker B's Claude

## Context (read first)
- 24h hackathon, EHL Paris 2026 cross-modal MRI retrieval (ceT1 → T2 same-subject re-id). See repo `CLAUDE.md`.
- You are **Track B**: training-free CV branch + Stage-2 re-rank. Your code goes in `workers/B/`.
- The metric is **MRR macro-averaged over 3 datasets**, not just dataset1. Per CLAUDE.md, dataset2 (deformation) and dataset3 (post-op + domain shift) are where wins come from — so a branch that holds up on ALL THREE matters more than one that crushes dataset1 alone.
- **Submission budget: 100/day.** Local-MRR gate before every submission.

## What's already done (don't redo)
- **Track A** (Alex) shipped:
  - **B3** learned encoder (SwinUNETR + InfoNCE) — training in background, ~3h ETA. `branch_b3.csv` lands after.
  - **B4 V1** shape fingerprint (centered brain mask 16³) — `workers/A/runs/branch_b4.csv` ready, standalone Level-1 MRR 0.30.
- **Track C** (Nicole) has the full pipeline:
  - `trackc.py` engine: `load_volume`, `make_local_split(seed=0)`, `mrr`, `rrf`, `scores_to_rankings`, `write_submission`. Living at `/shared-docker/Nicole/trackc.py` and `workers/C/trackc.py`.
  - **B2** frozen foundation embeddings + a gradient-magnitude baseline. First submission already on Kaggle.
- **You do NOT touch** `trackc.py`, the submission writer, or anyone else's branch CSVs.

## Your deliverables (priority order)

### P0 — B1 MIND-SSC descriptor → `branch_b1.csv`
**Modality-Independent Neighbourhood Descriptor (MIND-SSC)** is the classical training-free, contrast-invariant descriptor for medical image registration. Paper: Heinrich et al. 2012, 2013.

Why it matters here: it's **contrast-invariant by construction**, so it works on ceT1↔T2 with zero learning. Should give a non-trivial MRR on all 3 datasets from day 1.

Sketch:
1. For each NIfTI: load + resample 1mm (use `trackc.load_volume`) → optionally crop to brain.
2. Compute MIND-SSC at every voxel: a 12-channel descriptor based on patch self-similarity around a small (e.g. 3×3×3) neighborhood. Reference impl in numpy/PyTorch: search "MIND SSC descriptor torch" — multiple open-source versions on GitHub (~50–100 lines of conv ops). Examples: `torch.nn.functional.unfold` + Gaussian-weighted SSD.
3. Aggregate: per-volume vector. Simplest = avg-pool the 12-channel descriptor map over a coarse spatial grid (e.g. 8×8×8 = 6144 dims) → L2 normalize. Alternative: keep dense and do patch-level matching (P1).
4. Output `{dataset:{pool_name:{id:vec}}}` dict — same format as Track A's `.pkl`.
5. Convert to branch CSV via `python workers/A/pkl_to_branch.py --in workers/B/runs/b1_embeddings.pkl --out workers/B/runs/branch_b1.csv`.

### P1 — Stage-2 re-rank on the top-k from RRF
After RRF gives top-10 candidates per query, re-rank with one of:
- **SynthMorph** contrast-invariant registration: register query→candidate, score by inverse residual.
- **MIND distance** post-registration.
- **Segmentation overlap** if SynthSeg is available.

Output: `branch_stage2_rerank.csv` (same format), Nicole fuses with high weight on top-10 only.

### P2 — C-MIR ColBERT-style late interaction
Only if Stage-2 is done and there's headroom. Patch-level dense matching.

## Local check before any submission

Same seed=0 split everyone uses. Reuse Alex's pattern in `workers/A/probe_b4_local_mrr.py` — copy it to `workers/B/probe_b1_local_mrr.py` and replace the embedder. Threshold to beat:
- random = 0.02
- Track C baseline (gradient mag) on Level-1 = **0.76**
- B4 V1 standalone = **0.30**
- B1 (your target) should clear ≥ 0.10 on Level-1 to be worth including. On the Level-2 proxy (apply `trackc.deform_volume` to held-out targets), aim ≥ 0.05 — that's where you justify being in the fleet because B3/B4 fall off there.

## Bootstrap on JupyterHub

```bash
cd /shared-docker/work/repo
git pull
mkdir -p workers/B/runs
# scaffold: copy Alex's pattern as the skeleton
cp workers/A/b4_shape.py workers/B/b1_mind_ssc.py    # then rewrite shape_fingerprint -> mind_ssc descriptor
cp workers/A/probe_b4_local_mrr.py workers/B/probe_b1_local_mrr.py
```

Data is at `/shared-docker/data/` (env: `DATA_ROOT=/shared-docker/data`).

## Hand-off when done

1. Write `workers/B/runs/b1_embeddings.pkl` matching Track A's format.
2. Run `python workers/A/pkl_to_branch.py --in workers/B/runs/b1_embeddings.pkl --out workers/B/runs/branch_b1.csv`.
3. Commit + push.
4. Ping Nicole: she adds `branch_b1` to her `rrf([...])` call, runs the local-MRR gate, submits if it lifts.

## What NOT to do
- Don't reimplement MRR, splits, RRF, or the submission writer — `trackc.py` owns those.
- Don't break the branch CSV contract (`query_id, target_id, score`, within-pool only).
- Don't burn a Kaggle slot to measure something `make_local_split(seed=0)` can measure.
- Don't train a model — Track B is the training-free track. If you find yourself defining a `nn.Module`, you're on Alex's turf.
