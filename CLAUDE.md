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

## 9. Repo state (2026-06-27)

- **Track B scaffold added.** Repo now includes a minimal `track_b/` package plus `pyproject.toml` for the training-free descriptor and label-free rerank path.
- The rest of the repo is still planning/docs-heavy; no dataset wiring, no evaluation harness, and no tests for the new code yet.
- When adding code: clean venv at repo root, Python ≥3.12, `uv` for the baseline. Do **not** reuse `~/Desktop/INTERNSHIP` venvs.

*Status: Track B scaffold added; broader implementation still pending. Last updated 2026-06-27.*
