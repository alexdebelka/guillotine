# CLAUDE.md — EHL Paris 2026: Contrast-Agnostic Brain MRI Cross-Modal Retrieval

This file orients Claude Code (and the team) for a **24h hackathon**. Read it first, then `docs/RESEARCH_PLAN.md`.

---

## 0. TL;DR for an agent picking this up

- **Task:** for each query **ceT1 (contrast-enhanced T1)** brain MRI volume, rank a gallery of **T2** volumes so the **same-subject** T2 is rank 1.
- **It is NOT tumor similarity.** It is **same-subject re-identification across a contrast gap + a geometry gap.** Identity = the subject's individual anatomy (ventricles, sulci, skull, tumor location/shape).
- **Metric:** Mean Reciprocal Rank (MRR), per-dataset, **macro-averaged over 3 datasets** → generalization beats dataset1 accuracy.
- **The bet:** several *invariant* embedding branches (training-free + foundation + learned) → **Reciprocal Rank Fusion** → label-free **top-k re-rank**. See `docs/RESEARCH_PLAN.md` §5.
- **Biggest lever / fastest win:** the challenge PI's own method **CrossKEY** is open source → clone & adapt. https://github.com/morozovdd/CrossKEY
- **First thing to build:** a **local MRR harness** from the 350 labeled pairs (Kaggle caps submissions at 100/day, val/test labels hidden). See §6.

---

## 1. The challenge (ground truth)

- Organizers: Inria / Paris Brain Institute (ICM) / PRAIRIE. PI **Reuben Dorent**; challenge repo by **Nicolas Stellwag**.
- Repo: https://github.com/NicoStellwag/ehl-paris-2026-medical-retrieval
- Kaggle: https://www.kaggle.com/t/b33ec3e76c3d4e16a6b56852470b3ebf
- **Query modality:** T1 post-contrast (ceT1). **Target modality:** T2. All 3D NIfTI `.nii.gz`, RAS, 1.0³ mm spacing. **No** intensity norm / skull-strip / cropping applied. **Shapes vary** (esp. dataset2/3) — don't assume a fixed shape.

### Three datasets = three difficulty levels (one nuisance each)
| Dataset | Setting | Labeled train? | Invariance required |
|---|---|---|---|
| **dataset1** | preop pairs, **registered to a common grid** | **Yes — 350 pairs** | Contrast only (pure modality gap) |
| **dataset2** | same source, **independent rigid + non-linear deformation** per image | No | Contrast **+ deformation** |
| **dataset3** | **preop→intra-op**, different hospital; tissue shifted/missing | No | Contrast + deformation + **structural change + domain shift** |

### Counts
- dataset1: 350 train pairs · val 40/40 · test 100/100
- dataset2: val 40/40 · test 100/100
- dataset3: val 20/20 · test 77/77
- Submission template = 377 rows total.

### Evaluation & submission
- `score = (MRR_d1 + MRR_d2 + MRR_d3) / 3`. Reciprocal rank = 1/rank of the true target; 0 if absent/omitted.
- Rank a query **only** against its **same-dataset, same-split** gallery. Never mix datasets or val/test.
- Submission CSV: `query_id,target_id_ranking` where ranking is space-separated target IDs, most→least likely, **full gallery length** (40/100/40/100/20/77).
- **100 submissions/team/day** → rely on local eval.

---

## 2. Data layout (as released)

```
dataset1/  train_pairs.csv  val_queries.csv val_gallery.csv  test_queries.csv test_gallery.csv  images/{train,val,test}/
dataset2/  val_*.csv test_*.csv  images/{val,test}/
dataset3/  val_*.csv test_*.csv  images/{val,test}/
sample_submission.csv
```
- `train_pairs.csv`: `pair_id,query_id,target_id,query_image,target_image,query_modality,target_modality,dataset`
- query manifest: `query_id,query_image,query_modality,dataset` · gallery: `target_id,target_image,target_modality,dataset`
- IDs are globally unique across datasets/splits → one combined submission file.
- **Set `DATA_ROOT`** to the Kaggle download root before running anything.

---

## 3. Baseline (what to beat)

`slice_clip_baseline.py` in the challenge repo: dual-encoder CLIP (separate query/target tiny 2D CNNs), in-batch contrastive loss, trained **only on the 350 registered dataset1 pairs**, using **3 axial slices** (z=0.35/0.50/0.65, 96×96), ranks by cosine. MONAI + `PersistentDataset` cache; `uv run`.

**Why it's weak (our edge):** discards 3D; two separate encoders don't structurally bridge the modality gap; trained on registered geometry only → ~random on dataset2/3; tiny capacity; ignores unlabeled pools & foundation priors. The macro-average means its dataset2/3 collapse is fatal.

---

## 4. Our approach (summary — full detail in `docs/RESEARCH_PLAN.md`)

Multi-branch, rank-fused, two-stage:

1. **B1 — MIND-SSC descriptor** (training-free, contrast-invariant by construction) → works on all 3 levels day 1.
2. **B2 — Foundation-model frozen embedding** (BrainIAC / 3D-Neuro-SimCLR / M3Ret) → robust to domain shift (dataset3).
3. **B3 — Learned shared encoder** via **matching-by-synthesis + synthetic contrast + independent deformation aug + supervised contrastive** (adapt **CrossKEY**). Best for dataset1/2.
4. **B4 — Anatomy/shape fingerprint** from contrast-agnostic segmentation (SynthSeg) + SSM shape features. Orthogonal signal, helps dataset2/3.
5. **Fusion:** Reciprocal Rank Fusion across branches (parameter-free; highest-leverage, lowest-risk trick).
6. **Stage-2 re-rank** of top-k (label-free): SynthMorph registration residual / MIND distance / seg overlap, and/or C-MIR ColBERT-style late interaction.

**Design principle:** manufacture each invariance explicitly (synthesis→contrast, deformation aug + keypoints/shape→geometry, big pretraining→domain shift), and fuse heterogeneous branches because the metric is averaged over heterogeneous datasets.

---

## 5. Reuse from `~/Desktop/INTERNSHIP` (Alex's prior work)

| Asset | Path | Use for |
|---|---|---|
| MedGemma latent harness (latents.csv contract; t-SNE/UMAP; **`--residualize covariates`**) | `INTERNSHIP/medgemma/` | Foundation-embedding extraction (B2), latent sanity viz, cross-subject/contrast alignment (subtract nuisance) |
| SwinUNETR 3D encoder | `INTERNSHIP/modeling/oa_progression/models/seg_swinunetr.py`, `INTERNSHIP/modeling/Swin/` | Backbone for B3 (init from MONAI pretrained `model_swinvit.pt`) |
| SSM + Registrations | `INTERNSHIP/modeling/SSM/{Model,Registrations}` | Shape fingerprint (B4) + registration re-rank (Stage 2) |
| Jean Zay deploy scripts | `INTERNSHIP/jean_zay_deploy/` | GPU runs if needed |

---

## 6. Build order / commands (hour-by-hour)

> Use a clean venv for this repo; do **not** reuse INTERNSHIP venvs. Python ≥3.12. Baseline runs with `uv`.

```bash
# H0 — environment + data
export DATA_ROOT=/path/to/kaggle_dataset      # set this!
git clone https://github.com/NicoStellwag/ehl-paris-2026-medical-retrieval
git clone https://github.com/morozovdd/CrossKEY     # the PI's method — our B3 starting point
# pip/uv: monai>=1.5, torch>=2.7, nibabel, numpy, scikit-learn, tqdm

# H0–2 — LOCAL EVAL HARNESS (do this before modeling)
#   - hold out ~50 of the 350 dataset1 pairs as a local query/gallery
#   - implement MRR exactly as the challenge does
#   - add a SYNTHETIC-DEFORMATION proxy for dataset2 (random rigid + nonlinear on held-out)

# H2–4 — B1 (MIND) + B2 (frozen foundation) → first RRF submission
# H4–10 — B3 shared encoder (synth contrast + deformation aug + supervised contrastive)
# H10–14 — Stage-2 re-rank on top-k
# H14–18 — B4 shape fingerprint + tune RRF / per-dataset weights
# H18–22 — full submissions + ablations (per-branch per-dataset MRR)
# H22–24 — freeze best, writeup, slides
```

### Run the baseline (sanity)
```bash
uv run slice_clip_baseline.py --data-root "$DATA_ROOT" \
  --train-pair-csv "$DATA_ROOT/dataset1/train_pairs.csv" \
  --query-csv "$DATA_ROOT/dataset1/val_queries.csv"   --gallery-csv "$DATA_ROOT/dataset1/val_gallery.csv" \
  ... (repeat --query-csv/--gallery-csv for all 6 pools) ... \
  --out slice_clip_submission.csv
```

---

## 7. Conventions / gotchas

- **Always rank within the same dataset+split.** Mixing them silently tanks the score.
- **Variable shapes:** resample to 1 mm (given), pad/crop to a working box; keypoint/shape branches are shape-agnostic — prefer them for dataset2/3.
- **No imputation of intensities**; normalize per-volume (z-score on brain voxels or MONAI `ScaleIntensity`) inside each branch as needed.
- **Local-first:** never burn a Kaggle submission to measure something you can measure locally.
- **Determinism:** fix seeds; the baseline uses seed 20260626.
- **Don't overfit dataset1** — it's only 1/3 of the score and the easy third.

---

## 8. Pointers

- Full plan, ablations, risks: `docs/RESEARCH_PLAN.md`
- Challenge facts distilled: `docs/CHALLENGE_BRIEF.md`
- Annotated literature: `docs/REFERENCES.md` (PDFs in `papers/`)
- Consensus search prompts + saved results: `docs/CONSENSUS_QUERIES.md`, `papers/*.csv`

## 9. Repo state (2026-06-28)

Current Kaggle macro MRR = **0.5298**; B2+B3+B4+baseline RRF plateaued. Active pivot: **CrossKEY** (PI's published method) — see `PIVOT_TASK.md` at repo root for the self-contained handoff. Code split across `workers/<track>/` (per-branch scripts) + `Track C/` (engine + B2 + fusion). Track B still empty.

- `workers/A/` — B3 learned encoder + B4 shape. **Both shipped.**
  - `b3_encoder.py` — SwinUNETR backbone, MONAI SSL weight loader (handles MONAI 1.6 `Mlp.fc{1,2}`→`linear{1,2}` rename), multi-stage avg+max pool (POOLED_DIM=1536), SimCLR projection head, BatchNorm at head input. Embeds to unit-norm `(C,)` float32 at input `(96,96,96)`.
  - `augmenter.py` — synth-contrast + independent deformation per view (MONAI `Compose`).
  - `train_b3.py` — InfoNCE on dataset1 pairs (temp 0.1, no WD, split LRs, logs diag/off-diag sim).
  - `embed_b3.py` — extract B3 embeddings → `.pkl`.
  - `b4_shape.py` — B4 V1: centered brain mask resampled to 16³ = 4096-d unit-norm. **Standalone local MRR 0.30 (Level-1, 8/50 top-1).**
  - `probe_b4_local_mrr.py` — local MRR probe matching Track C's seed=0 split.
  - `pkl_to_branch.py` — adapter: Track A `.pkl` → Track C branch CSV (`query_id,target_id,score`, higher = more similar).
  - `smoke_test.py` — SwinUNETR build + SSL load + forward (`python workers/A/smoke_test.py`).
  - `runs/` — empty locally now (artifacts on compute node). B3 finished; `branch_b3.csv` lives in `Track C/`.
  - SSL weights at `/shared-docker/work/weights/model_swinvit.pt` (`download_swinvit.sh`).
- `workers/B/` — empty. Hand-off doc at `workers/B/WORKER_B_TASK.md` — paste into a fresh Claude session to start Track B (MIND-SSC descriptor B1 + Stage-2 SynthMorph re-rank).
- `workers/X/` — late-stage exploration (NOT in main RRF). `b5_xkey.py` (CrossKEY-style patch-level cross-modal contrastive, Stack C; `--grid` for deterministic inference sampling), `mindssc.py`, `stage2_mind.py` / `stage2_v2_d1only.py` (Stage-2 NCC/MIND re-rank, d1-only by default via `--rerank-ds`), `apply_d3_leak.py` + `leak_pixel.py` + `leak_refine.py` (post-processor exploiting a NIfTI-header/pixel-corner fingerprint leak on dataset3 — diagnostic + exploit).
- `workers/C/` / `Track C/` — Track C engine.
  - `trackc.py` — held-out split (seed=0), MRR eval, `rank_by_embeddings`, RRF, submission writer.
  - `b2_foundation.py` — B2 frozen foundation-model embeddings.
  - `run_b2.ipynb`, `run_pipeline_c.ipynb`, `run_fuse_submit.ipynb` — drivers.
  - `NICOLE_TASK.md` — current handoff: add B4 to the RRF, gate on local MRR, submit.
  - Branch CSVs land here (`branch_baseline.csv`, `branch_b2.csv`, …) → fused into `submission.csv`.

**Cross-track branch contract:** CSV with columns `query_id, target_id, score` (higher = more similar), one row per (q, t) WITHIN each of the 6 (dataset, split) pools — never across. `trackc.scores_to_rankings(df)` converts to the `{qid: [tid…]}` form `rrf()` consumes.

**Embedding contract (Track A intermediate):** `dict[dataset -> {pool_name -> {id_str -> np.ndarray (C,) unit-norm float32}}]` as `.pkl`. `pkl_to_branch.py` converts to the branch CSV.

**Submission flow:** every branch → its CSV → Nicole's `run_fuse_submit.ipynb` reads all CSVs → `rrf([...])` → `write_submission` → `submission.csv`. Only Track C writes the final file. **Local-MRR gate every submission** — never burn one to measure what `make_local_split(seed=0)` can.

No `pyproject.toml`, venv, tests, or lint config at repo root — run scripts directly with the project venv (Python ≥3.12, clean venv at repo root; do **not** reuse `~/Desktop/INTERNSHIP` venvs). The baseline (`slice_clip_baseline.py`) lives in the upstream challenge repo (see §6).

*Last updated 2026-06-28 (Kaggle macro MRR 0.5298; CrossKEY pivot active — see `PIVOT_TASK.md`; Track X dataset3 fingerprint-leak post-processor landed).*
