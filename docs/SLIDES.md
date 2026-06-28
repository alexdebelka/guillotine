# EHL Paris 2026 — Hackathon Slides Notes
**Contrast-Agnostic Brain MRI Cross-Modal Retrieval**

This file is the single source of truth for the deck. All numbers, code refs, and lessons learned in one place — pull from here, don't re-derive.

---

## Slide 1 — Title
- **EHL Paris 2026 Cross-Modal MRI Retrieval**
- Same-subject ceT1 → T2 brain MRI retrieval
- Team: Alex, Nicole, (Worker B placeholder)
- Stack: Track A (learned + shape), Track B (training-free), Track C (eval + fusion)

---

## Slide 2 — The problem
- For each **ceT1 (T1 post-contrast)** query, rank a gallery of **T2** volumes so the same subject's T2 is rank 1.
- **Not tumor similarity.** Same-subject re-identification across **contrast gap + geometry gap**.
- 3D NIfTI, RAS, 1.0³ mm spacing, shapes vary (esp. dataset2/3).
- **Metric:** MRR macro-averaged over 3 datasets. Generalization beats single-dataset accuracy.

| Dataset | What | Difficulty | Labels |
|---|---|---|---|
| **dataset1** | preop pairs, registered | contrast only | 350 train pairs + 40 val + 100 test |
| **dataset2** | independent rigid + nonlinear deformation per image | contrast + deformation | 40 val + 100 test (no train) |
| **dataset3** | preop→intra-op, different hospital | contrast + deformation + structural change + domain shift | 20 val + 77 test (no train) |

Submission: 1 combined CSV, 377 query rows total. **Budget: 100/day.**

---

## Slide 3 — Our approach (one diagram)

```
        ┌─ B1 MIND-SSC descriptor      (training-free — Worker B, deferred)
ceT1 ──┐├─ B2 foundation embedding      (frozen SwinUNETR — Nicole)
       ├┤├─ B3 shared encoder           (learned, InfoNCE, SwinUNETR — Alex)
T2  ──┘└─ B4 anatomy/shape fingerprint  (centered brain mask — Alex)
                  │
        Reciprocal Rank Fusion → top-k per query
        (per-dataset weights — chosen via local L1+L2 proxy)
                  │
        Stage-2 re-rank (planned, not shipped)
                  │
            submission.csv
```

**Design principle:** manufacture each invariance explicitly. Synthesis for contrast, deformation aug + shape for geometry, foundation pretraining for domain shift. Fuse heterogeneous branches because the score is macro-averaged over heterogeneous datasets.

---

## Slide 4 — Per-branch local results (seed=0 held-out 50 dataset1 pairs)

| Branch | Level-1 MRR | Level-2 MRR (deformation proxy) | What it captures |
|--------|------------:|-------------------------------:|------------------|
| Random | 0.02 | 0.02 | — |
| Track C baseline (gradient-mag) | **0.7626** | 0.0981 | raw 3D gradient on registered space |
| B2 (frozen foundation) | 0.2152 | — | semantic SwinUNETR SSL features |
| B3 (learned, InfoNCE) | **0.8674** | (tbd in tuning run) | learned contrast-invariance from 350 pairs |
| B4 V1 (centered mask 16³) | 0.2970 | (tbd) | pure geometry, contrast-agnostic |
| RRF (base + b2 d1 wts) | 0.6918 | — | Nicole's first submission combo |
| RRF (base + b2 + b3) | **0.8718** | — | adding B3 lifted gate |
| RRF (base + b2 + b3 + b4) | 0.7923 | — | B4 hurts d1 (weight needs to be 0 here) |

**Kaggle (27% public leaderboard):** macro MRR = **0.5298** with the multi-branch submission. Per-dataset breakdown unknown.

---

## Slide 5 — Key technical lessons (the bugs we hit)

### B3 InfoNCE was collapsing exactly to `ln(B)` for 100+ steps
- Symptom: loss stuck at 2.7726 (= ln(16) for B=16) — every embedding the encoder produced was identical.
- Three independent root causes, fixed in sequence:
  1. **SSL weight load was 94/126** — MONAI 1.6 renamed `Mlp.fc{1,2}` → `linear{1,2}`. Loader patch fixed.
  2. **Weight decay was killing the projection head** — wd pushed head weights to zero → all outputs collapsed. Fixed by `wd=0` and temp 0.1.
  3. **Anisotropy in SSL features (97% similarity across DIFFERENT brains at init)** — the encoder maps all brains to a narrow cone. Fixed by **BatchNorm at the head input** — subtracts the "common brain" direction.
- After these three: loss 2.83 → 0.04, head Δ 0.22 → 0.55, B3 standalone MRR 0.87.

### SSL anisotropy is the deeper insight
- Foundation features trained on segmentation cluster heavily — "this is brain tissue" dominates over "this is THIS brain."
- BatchNorm before contrastive loss is the standard antidote. Without it, no amount of training works.

### The B4 V1 shape mask doesn't survive deformation
- 0.30 on L1 (registered) → near random on L2 (deformed).
- Translation/scale invariant by cropping/resampling, but NOT rotation-invariant.
- V2 plan: add Hu/inertia moments or PCA-canonicalize orientation. Deferred.

---

## Slide 6 — Code organization

```
workers/A/                          (Alex — learned + shape)
  b3_encoder.py        SwinUNETR + SSL loader + multi-stage avg+max pool
  augmenter.py         MONAI Compose: contrast + deformation, independent per view
  train_b3.py          InfoNCE loop, projection head, BN input, diag/off-diag log
  embed_b3.py          val/test pkl
  pkl_to_branch.py     pkl → Track C branch CSV format
  b4_shape.py          V1 fingerprint: centered brain mask 16³
  embed_holdout.py     held-out CSVs for the local-MRR gate (B3 or B4)
  probe_b4_local_mrr.py  standalone B4 sanity

workers/C/                          (cross-team docs)
  NICOLE_TASK.md       hand-off when B3+B4 landed
  TUNE_WEIGHTS.md      cells to add for per-dataset tuning

workers/B/
  WORKER_B_TASK.md     MIND-SSC brief, not yet executed

Track C/                            (Nicole — engine + fusion)
  trackc.py            harness: split, MRR, RRF, submission writer
  b2_foundation.py     frozen-encoder B2
  run_fuse_submit.ipynb  the submission driver
  branch_baseline.csv  branch_b2.csv  branch_b.csv
```

**Cross-track contract:** every branch → CSV `query_id,target_id,score`. RRF eats lists of those. Single submission writer.

---

## Slide 7 — What we'd do with 24 more hours

In priority order, with realistic upside:

1. **Adapt CrossKEY** (https://github.com/morozovdd/CrossKEY) — the PI's own published method, keypoint matching + synthetic contrast. The top teams (1.0 on leaderboard) almost certainly used it. Realistic ceiling: 0.85+ macro.
2. **Stage-2 re-rank** — SynthMorph registration of the top-k from RRF, score by inverse residual. Label-free, big lift expected on d2/d3.
3. **B3 V2: train with `--severity heavy`** — current run was `mild`. Heavier deformation in the augmenter teaches stronger L2 invariance. Same recipe, 3h GPU.
4. **B4 V2: rotation-invariant moments** — fixes the L2 collapse.
5. **MIND-SSC as Worker B's branch** — adds a strong training-free signal on all 3 datasets.

---

## Slide 8 — Team workflow (what worked)

- **Single embedding contract:** `{id: vec}` pkl + branch CSV `{q, t, score}`. Every branch produces the same format. No glue code per branch.
- **`trackc.py` as the single engine** — split, MRR, RRF, submission writer. Touch once, every track benefits.
- **Local-MRR gate before every Kaggle submission** — used the 350 dataset1 pairs (300 train / 50 holdout, seed=0) + a `deform_volume` L2 proxy. Burned zero submissions to noise.
- **Hand-off markdowns** (`NICOLE_TASK.md`, `WORKER_B_TASK.md`) for fresh Claude sessions joining the team — kept context cost cheap.

---

## Slide 9 — One bullet summary
> Multi-branch contrast/deformation-invariant retrieval with parameter-free RRF fusion. Real engineering lessons: SSL feature anisotropy, contrastive collapse modes, and a label-free local gate that costs no submissions. Score: 0.5298 macro on the 27% public leaderboard, with the recipe for getting it materially higher (CrossKEY, Stage-2) documented and unblocked.

---

## Appendix — numbers you might be asked

- B3 training: 1500 steps × batch 16 × 96³, ~8s/step, ~3h total on MI300X.
- B3 final training loss: 0.044 (started at 2.77 = ln(16) random).
- B3 final head Δ (diag − off-diag cosine sim on training batches): +0.92.
- B4: 754 volumes embedded in ~3 minutes CPU.
- Branch CSV size: 29,529 rows (sum of q × g across the 6 pools: 40²+100²+40²+100²+20²+77²).
- Submission rows: 377 (40+100+40+100+20+77 queries).
- Cumulative Kaggle submissions burned: 4 (Nicole's first run + iterations including B3+B4).
- Daily budget: 100. Remaining: 96+.
