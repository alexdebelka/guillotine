# EHL Paris 2026 — 24h Hackathon Roadmap (Team of 3)

## Context

We need a clear, parallelizable 24h plan for a **3-person team** building a contrast-agnostic, same-subject cross-modal MRI retrieval system (ceT1 → T2). The existing `docs/RESEARCH_PLAN.md` is a strong but **sequential, single-author** plan. This file slices that plan into **three independent tracks** with explicit ownership, priorities, and a critical path — so the team works in parallel without colliding and so nobody is blocked waiting for someone else's branch.

Scoring is **MRR macro-averaged across 3 datasets of increasing nuisance** (contrast / +deformation / +resection+domain-shift). Generalization wins, not dataset1 accuracy. See `CLAUDE.md` §1 and `docs/RESEARCH_PLAN.md` §1 for the framing; this file is operational.

## Team & compute

- **Alex** — MD + 3D MRI / MONAI / SwinUNETR / SSM. Owns the heavy-training branch and reuses `~/Desktop/INTERNSHIP` assets.
- **Teammate B (CV-ML, bioimaging)** — Owns classical / training-free CV and Stage-2 registration re-rank.
- **Teammate C (ML, biology)** — Owns foundation-model embedding extraction, the eval harness, and fusion / submission plumbing.
- **Compute:** AMD MI300X (192 GB vRAM), 38 instances, root, near bare-metal. Constraint is **ROCm compatibility**, not VRAM or queue time. We can run B2/B3 in parallel and afford multiple seeds.

## Architecture (recap, one diagram)

```
        ┌─ B1 MIND-SSC descriptor       (training-free; all levels)
ceT1 ──┐├─ B2 foundation embedding      (frozen; robust to L3 domain shift)
       ├┤├─ B3 shared encoder           (learned; CrossKEY-adapted; best L1/L2)
T2  ──┘└─ B4 anatomy/shape fingerprint  (geometry; L2/L3)
                  │
        Reciprocal-Rank Fusion → top-k per query
                  │
        Stage-2 re-rank: SynthMorph residual / MIND distance / seg overlap
                  │
            submission.csv
```

## The three tracks

### Track A — Alex: Learned encoder (B3) + shape fingerprint (B4)
The single biggest lever on dataset1/2 (B3) and an orthogonal signal for dataset2/3 (B4). Requires deep MRI/MONAI knowledge and reuse of `INTERNSHIP/modeling/Swin` and `INTERNSHIP/modeling/SSM`.

- Clone & adapt **CrossKEY** (https://github.com/morozovdd/CrossKEY) — swap MR→US synthesis for ceT1↔T2 synthesis; keep the contrastive descriptor machinery.
- Backbone: SwinUNETR (reuse `INTERNSHIP/modeling/oa_progression/models/seg_swinunetr.py`), init from MONAI `model_swinvit.pt`.
- Augmentation: synthetic contrast (SynthMorph-style generative) + **independent** random rigid + nonlinear deformation per view. This is what manufactures the L1 and L2 invariances.
- Loss: supervised contrastive on the 350 dataset1 pairs (or, once aug is in place, dataset1 pairs treated as one of many augmented views).
- Then B4: SynthSeg → parcellation → shape fingerprint via `INTERNSHIP/modeling/SSM`.

### Track B — CV-ML teammate: Training-free descriptor (B1) + Stage-2 re-rank
Classical CV / image-processing strengths. Produces a non-trivial score on **all 3 datasets** from hour 4, and owns the biggest precision lift in Stage 2.

- B1: MIND-SSC descriptor per voxel → global pooling or patch-level → cosine / MIND distance. No training. Refs `papers/` [4][15].
- Stage 2: SynthMorph (contrast-invariant registration) on the top-k from RRF; rank by registration residual / MIND distance post-registration / segmentation overlap. Label-free.
- Optionally: explore C-MIR ColBERT-style late interaction if time allows (P2).

### Track C — ML-bio teammate: Eval harness + foundation embeddings (B2) + RRF + submission
The infrastructure that **blocks the other two tracks** plus the lowest-effort modeling branch (frozen foundation extractor).

- Local MRR harness (replicates the Kaggle metric exactly).
- Held-out split from the 350 dataset1 pairs (~300/50).
- Synthetic-deformation L2 proxy (apply independent random rigid + nonlinear to the held-out 50; re-measure MRR).
- B2: frozen embedding extraction from **BrainIAC / 3D-Neuro-SimCLR / M3Ret** (pick one available on ROCm first; fall back across the list). Reuse `INTERNSHIP/medgemma/` for the `latents.csv` contract + UMAP + `--residualize covariates`.
- RRF combiner: ingest per-branch rankings, output fused ranking, write submission.csv. Single source of truth for all submissions.

## Critical path & priority ladder

Stop at the first **unfinished** rung and ship; everything later is icing.

| Pri | Item | Owner | Done = |
|-----|------|-------|--------|
| **P0** | ROCm env (PyTorch+MONAI on MI300X), shared repo skeleton, `DATA_ROOT` set | All (H0, 60 min shared sprint) | `python -c "import torch; print(torch.cuda.is_available())"` true on one MI300X, baseline runs |
| **P0** | Local MRR harness + 300/50 dataset1 split + synthetic L2 proxy | C | Re-runs in <60s on a fixed split; numbers match a hand calculation on a toy case |
| **P0** | Submission CSV writer (all 6 query pools → one 377-row file) | C | Sample submission template round-trips byte-equal |
| **P0** | **B1 MIND-SSC** + **B2 frozen foundation** + **RRF** → **first Kaggle submission** | B + C | One submission burned, leaderboard score on record |
| **P1** | **B3 CrossKEY-adapted shared encoder** training | A | Local MRR on held-out dataset1 beats B1+B2 alone |
| **P1** | **Stage-2 re-rank** (SynthMorph residual / MIND) on top-10 from RRF | B | Local MRR jumps on the L1+L2-proxy splits |
| **P2** | **B4 shape fingerprint** (SynthSeg → SSM) | A | Adds non-zero MRR when fused via RRF |
| **P2** | Per-dataset RRF weight tuning (lean B1/B2/B4 for d3; B3+Stage2 for d1) | All | Macro MRR improves on local + leaderboard |
| **P2** | Ablations table (per-branch per-dataset MRR; aug on/off; re-rank on/off) | All | Filled table in writeup |
| **P3** | Slides + writeup + repro script + freeze best submission | All | Final Kaggle submission + 10-slide deck |

## Hour-by-hour (anchored to the critical path)

| Hours | Alex (A: B3+B4) | Teammate B (B1 + Stage-2) | Teammate C (Eval + B2 + RRF) |
|------|------------------|---------------------------|------------------------------|
| 0–2 | ROCm sanity on 1 instance; clone CrossKEY; pull `INTERNSHIP/Swin` weights | ROCm sanity; clone repo; read MIND-SSC ref; sketch descriptor pipeline | **MRR harness, 300/50 split, L2 proxy, submission writer** (blocks A & B) |
| 2–4 | Stand up synth-contrast + deformation augmenter; smoke-train tiny SwinUNETR | **B1 MIND-SSC end-to-end on dataset1 val** → rankings to C | B2 frozen-foundation extractor (BrainIAC first; fall back if ROCm trips); RRF; **first submission** |
| 4–10 | **B3 full training** (multi-instance if needed); checkpoint every hour | Stage-2 SynthMorph on top-10; measure local lift | Monitor leaderboard; help with whichever track is slowest; start B4 prep notebook (SynthSeg env) |
| 10–14 | B3 converged → embeddings to C; start B4 (SynthSeg → SSM features) | Tune Stage-2 (residual vs MIND vs seg-overlap); pick best | Fuse B3 into RRF; **second submission**; collect per-branch per-dataset MRR table |
| 14–18 | B4 → embeddings to C | Stage-2 polish; second pass over B1 (try MIND-SSC scales) | Per-dataset RRF weight tuning; **third submission** |
| 18–22 | Backstop whichever branch is weakest; rerun B3 with best aug recipe if time | Backstop; help with ablations | Ablation submissions (1 per hypothesis, budget 4–6); writeup table |
| 22–24 | Slides on Track A | Slides on Track B | **Freeze final submission**, slides on Track C, repro script |

## Coordination protocol (cheap, mandatory)

- **Single source of truth for rankings:** every branch outputs a CSV `{query_id, target_id, score}` (or pre-sorted ranking). Track C's RRF reads these. No branch writes the submission file directly.
- **Shared `latents.csv` contract** from `INTERNSHIP/medgemma/`: any branch that produces embeddings dumps them in this format so any teammate can sanity-check via UMAP.
- **Submission budget: 100/day.** Allocate: 1 baseline, 1 B1+B2 RRF, 1 +B3, 1 +Stage-2, 1 +B4, ~4–5 for ablations + per-dataset weight tuning, 1 reserved final. Never burn a submission to measure something local MRR can measure.
- **Stand-ups every 4h** (max 10 min): "what's my local MRR, what's blocking me, what do I need from the other tracks."
- **Branch hand-off:** when a branch hits its local-MRR milestone, the owner posts the ranking CSV path in chat; C re-runs RRF and the team decides whether to submit.

## Risks & mitigations (team-specific)

| Risk | Mitigation |
|------|------------|
| ROCm/MI300X kernel breakage in CrossKEY or SynthMorph | A and B each smoke-test their stack in H0–1 on **one** instance before scaling; have CPU fallback paths for B1/B4 (no GPU needed). |
| C's harness slips → A and B can't measure | H0–2 is the **shared sprint**: A and B help C land the harness before going to their own tracks. Non-negotiable. |
| B3 training doesn't converge in time | B1+B2 RRF + Stage-2 alone is already a strong submission — that is our **floor**. B3 is upside, not a dependency. |
| Variable shapes break a branch on d2/d3 | Resample to 1 mm + pad/crop in the shared loader; keypoint/shape branches are shape-agnostic — lean on them for d2/d3. |
| Two branches double-count (correlated rankings) | RRF is robust to this, but worth checking rank-correlation between B1 and B3 in the ablation table; if >0.9, drop weights. |
| Submission budget burned on noise | 100/day cap → only submit after local MRR moves; C gate-keeps. |

## Verification (how we know it works)

1. **Harness sanity:** hand-construct a 3-query toy with known matches; MRR harness returns exactly the expected number. (C, H1)
2. **Baseline reproduction:** run `slice_clip_baseline.py` end-to-end, submit, confirm local-vs-Kaggle MRR delta < 0.02. (H2; calibrates the harness against the truth.)
3. **Per-branch local MRR on held-out dataset1 split** (40-query proxy for L1) and on the **synthetic-deformation proxy** (proxy for L2): each branch logs its number to a shared table. No branch ships to RRF until its number beats random (1/50 ≈ 0.02).
4. **Fusion sanity:** RRF score must be ≥ max of its components on the held-out split; if not, weights or input rankings are broken.
5. **Stage-2 sanity:** re-rank on identical input ranking returns identical ranking (idempotence on already-correct top-1); re-rank on a corrupted top-10 (shuffled) recovers most of the top-1s on the L1 proxy.
6. **Final submission:** dry-run the CSV writer on val, check 377 rows, no duplicate query IDs, gallery IDs are subsets of the correct same-dataset same-split gallery.

## What we are **not** doing (deliberately deferred)

- Custom 3D foundation model pretraining from scratch (use frozen weights only).
- C-MIR ColBERT late-interaction (P2 stretch; only if Stage-2 SynthMorph is finished and re-rank still has headroom).
- FLAIR↔T1 generalization (`docs/RESEARCH_PLAN.md` §11) — out of scope for 24h.
- A second learned encoder family — one (B3) is enough; diversity comes from B1/B2/B4, not from a second deep model.
