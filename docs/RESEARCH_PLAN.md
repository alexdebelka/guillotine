# Contrast-Agnostic Brain MRI Cross-Modal Retrieval — Research Plan

**Hackathon:** EHL Paris 2026 — Cross-modal Content-based Retrieval for 3D Medical Images (Inria / Paris Brain Institute / PRAIRIE)
**Task:** For each query **contrast-enhanced T1 (ceT1)** volume, rank a gallery of **T2** volumes so the same-subject T2 ranks first.
**Metric:** Mean Reciprocal Rank (MRR), computed per dataset and **macro-averaged** over dataset1/2/3.
**Future extension:** same machinery for FLAIR↔T1 to reduce missing-modality data.

---

## 1. The key reframing (read this first)

This is **not** "find a similar tumor." It is **same-subject re-identification across a modality (contrast) gap and a geometry gap.** The signal that identifies a subject is their **individual anatomy** — ventricular and sulcal morphology, skull/scalp shape, midline, and the specific location/shape of their tumor and edema. All of that is present in both ceT1 and T2; only the *intensity mapping* and the *geometry* differ.

So the whole problem decomposes into building a representation that is **invariant to the three nuisance factors** the organizers deliberately introduced, while staying **sensitive to subject identity**:

| Level | Dataset | Nuisance to be invariant to | What stays constant |
|------|---------|------------------------------|---------------------|
| 1 | dataset1 (registered) | **Contrast** (ceT1 vs T2 intensity inversion) | Geometry is shared → pure modality gap |
| 2 | dataset2 (synthetic deform.) | Contrast **+ independent rigid + non-linear deformation** | Topology / relative anatomy |
| 3 | dataset3 (pre→intra-op, other hospital) | Contrast + deformation + **missing/shifted tissue + scanner domain shift** | Coarse subject anatomy only |

**Strategic consequence of the macro-average:** a method that is *merely decent on all three* beats one that is *excellent on dataset1 and random on 2 & 3*. Dataset1 has only 40 (val) / 100 (test) candidates and is registered — it is the easy 1/3. **Generalization and robustness, not dataset1 accuracy, win this challenge.** Budget effort accordingly.

---

## 2. What the baseline does, and why it leaves points on the table

`slice_clip_baseline.py` = a CLIP-style **dual encoder** (separate query/target tiny 2D CNNs), trained with in-batch contrastive loss on **only the 350 registered dataset1 pairs**, using **3 axial middle slices** (z = 0.35/0.50/0.65), 96×96, ranking by cosine similarity.

Weaknesses to beat:
1. **Discards 3D** — 3 axial slices throw away most subject-identifying anatomy and are fragile to the L2 rotations.
2. **Two separate encoders** don't *structurally* close the modality gap; they just hope the projection aligns from 350 pairs.
3. **Trains on registered geometry only** → no reason to be deformation- or resection-robust → likely near-random on dataset2/3.
4. **No use of unlabeled val/test pools, no foundation priors, no test-time geometry.**
5. **Tiny capacity, 350 pairs** → high variance.

---

## 2b. Consensus synthesis (50-paper review) — external validation

A Consensus search over the primary research question ([thread, 50 results analyzed](https://consensus.app/search/contrast-agnostic-cross-modal-mri-retrieval/AsLlVIvST5iA2eQerrsWTg/)) concludes the task is enabled by **three pillars**:
1. **Modality-invariant representation learning** → our branches B1 (MIND) and B3 (learned).
2. **Cross-modal synthesis *or harmonization*** → our matching-by-synthesis (B3); *add intensity harmonization as an explicit pre-step/branch.*
3. **Contrast-robust spatial normalization** → relevant to deformation handling and Stage-2 registration.

Key strategic takeaway: **"direct retrieval evidence is still limited."** This exact problem (cross-contrast, same-subject, 3D retrieval) is under-explored — there is no entrenched SOTA to beat. The strongest supporting evidence is contrastive **fingerprinting** (DeepBrainPrint [5]), **metadata-aligned contrastive learning** (MR-CLIP [2]), and **disentanglement** methods. → Differentiation comes from *robustness engineering* (fusion + label-free re-rank across the 3 levels), not from matching a known benchmark.

## 3. Core idea — a contrast-agnostic latent space, built two ways and fused

Two complementary families. Build **both**; they fail in different places, so fusing their rankings is the robustness play.

**Family A — Training-free, invariant-by-construction (works on all 3 levels day 1).**
Representations that are contrast-invariant *by design*, needing zero labels — so they transfer to dataset2/3 for free.
- **MIND / MIND-SSC self-similarity descriptors** [4]: encode each voxel by its *local self-similarity pattern*, which is (largely) independent of the intensity-to-tissue mapping. This is the classic modality-agnostic structural representation and underlies modern contrast-invariant registration [1].
- **SynthSeg-style contrast-agnostic segmentation** → turn each volume into a label/parcellation map; identity then lives in **shape and relative geometry**, fully contrast-free.

**Family B — Learned contrast-agnostic embedding (the organizer's own recipe).**
The challenge PI, **Reuben Dorent**, just published *exactly* the relevant method: **CrossKEY**, a **3D cross-modal keypoint descriptor learned by "matching-by-synthesis" + supervised contrastive training** with rotation invariance, probabilistic keypoint detection, and curriculum-based triplet loss with hard-negative mining [9]. That is a very strong hint about the intended creative direction. **Code, data and model weights are public: https://github.com/morozovdd/CrossKEY** — clone it on hour 0 and adapt from MR–US to ceT1–T2 (the synthesis step changes; the contrastive-descriptor machinery transfers directly). The related **Synth-by-Reg** [14] shows the same synthesis+contrastive trick for nonlinear inter-modality alignment.

---

## 4. Candidate approaches, ranked by 24h payoff/effort

> Build in this order; each produces a submittable system, so you always have a fallback.

**A1 — Modality-invariant descriptor + correlation (no training).** Compute MIND-SSC features for every volume; retrieve by descriptor similarity (and, on registered dataset1, normalized cross-correlation / MIND distance directly). *Payoff:* immediate non-trivial score on **all three** datasets. *Effort:* low. Refs [1][4].

**A2 — Foundation-model frozen embeddings (zero/few-shot).** Use a pretrained 3D brain-MRI foundation model as a frozen feature extractor and rank by cosine; optionally fit a tiny projection head on the 350 pairs. Candidates: **BrainIAC** (Nat. Neurosci. 2026) [8], **3D-Neuro-SimCLR** (public weights) [7], **BrainFound** (slice-based, multi-contrast) [6], or **M3Ret** zero-shot retrieval encoder [10]. *Payoff:* high, leverages huge pretraining → robust to domain shift (L3). *Effort:* low–medium (mostly plumbing). Reuse your **MedGemma latent harness** for extraction + sanity UMAP.

**A3 — Shared encoder via matching-by-synthesis + synthetic contrast/deformation augmentation.** One 3D encoder (not two). During training, aggressively **synthesize contrast variations** (à la SynthMorph/SynthSeg generative augmentation [3]) and **apply random non-linear deformations independently** to the two views, then a **supervised contrastive loss** pulls same-subject ceT1/T2 together. This directly *manufactures* the L1 (contrast) and L2 (deformation) invariances the test sets demand. *Payoff:* highest for L1/L2, helps L3. *Effort:* medium. Refs [1][3][9][12][13].

**A4 — Anatomy-as-fingerprint (shape/geometry features).** From contrast-agnostic segmentation, derive subject descriptors: ventricle/structure volumes, shape (your SSM pipeline), tumor-mask geometry, relational/topological features that survive deformation. *Payoff:* orthogonal signal, strong on L2/L3. *Effort:* medium. Reuse `modeling/SSM`.

**A5 — Keypoint / local-descriptor set matching (deformation- & partial-anatomy-robust).** Detect 3D keypoints, attach learned cross-modal rotation-invariant descriptors [9], match by mutual-nearest-neighbor / optimal transport, score by inlier count. *Payoff:* best for L2 deformation and L3 missing tissue (global embeddings break there). *Effort:* high. Refs [9] + universal matchers MatchAnything [12] / MINIMA [13].

**A6 — Test-time re-ranking of the shortlist.** Shortlist top-k by any embedding above, then **re-rank** with a stronger, label-free score. Two options, ideally both:
  - *Geometry:* fast **contrast-invariant registration** (SynthMorph [3]) between query and each candidate → score the residual similarity / **MIND distance** [15] / segmentation overlap. Big lift on L1/L2.
  - *Late interaction:* **C-MIR** [16], a ColBERT-style contextualized late-interaction re-ranker proven on 3D medical retrieval — compares *sets* of local features instead of one global vector, and localizes the ROI without pre-segmentation. Robust where global embeddings blur (L2/L3).
  *Payoff:* large MRR lift (turns "in top-5" into "rank 1"). *Effort:* medium; needs no target labels.

---

## 5. Recommended architecture (the bet)

A **two-stage, multi-branch, rank-fused** system — designed so the macro-average is high because *every* dataset is covered by at least one branch that is invariant to its nuisance factor.

```
                 ┌─ Branch B1: MIND-SSC descriptor sim         (no training; all levels)
   ceT1 query ─┐ ├─ Branch B2: foundation-model embedding cos  (pretrained; robust to L3 domain shift)
               ├─┤ ├─ Branch B3: shared encoder, synth-contrast (learned; best L1/L2)
   T2 gallery ─┘ └─ Branch B4: anatomy/shape fingerprint        (geometry; L2/L3)
                          │
                 Reciprocal-Rank Fusion  →  top-k per query
                          │
        Stage 2: SynthMorph registration + seg-overlap re-rank of top-k
                          │
                   Final ranking → submission.csv
```

- **Stage 1 (recall):** each branch produces a full ranking; combine with **Reciprocal Rank Fusion** (parameter-free, robust to scale differences between branches). RRF is the single highest-leverage, lowest-risk trick here.
- **Stage 2 (precision):** re-rank only the top-k (e.g. k=10) with a geometry-aware score that needs no labels — this is where dataset1/2 MRR jumps toward 1.0.
- **Per-dataset routing (optional):** because datasets are scored separately, you may tune branch weights per dataset (e.g. lean on B1/B2/B4 for dataset3, lean on B3+Stage2 for dataset1).

---

## 6. Reuse from `/Desktop/INTERNSHIP`

| Asset | Path | Repurpose for |
|------|------|----------------|
| **MedGemma latent harness** (embedding contract `latents.csv`, t-SNE/UMAP plotter, **residualize-on-covariates** for cross-subject alignment) | `medgemma/` | Foundation-model embedding extraction (A2), latent-space sanity viz, and the "cross-patient alignment" trick → subtract population/contrast-mean components so identity dominates |
| **SwinUNETR 3D encoder** | `modeling/oa_progression/models/seg_swinunetr.py`, `modeling/Swin/` | Backbone for the shared 3D encoder (A3), init from MONAI pretrained weights |
| **SSM + Registrations** | `modeling/SSM/Model`, `modeling/SSM/Registrations` | Shape fingerprint (A4) + registration re-ranking (A6) |
| **MONAI pipeline / Jean Zay deploy** | baseline + `jean_zay_deploy/` | Data caching, GPU runs, mixed precision |

The internship's central pattern — *extract a per-item latent → analyze/align in latent space (residualize) → downstream task* — is **exactly** the contract this retrieval task needs. The `--residualize covariates` idea maps cleanly onto "remove contrast/scanner nuisance so subject identity is what remains."

---

## 7. Evaluation harness (do this before any modeling)

Val/test labels are **hidden** and Kaggle allows only **100 submissions/day**, so you must score locally.
- **Make a local split from the 350 labeled dataset1 pairs** (e.g. 300 train / 50 held-out gallery+query) and implement MRR yourself. This proxies **L1**.
- **Synthesize an L2 proxy**: apply independent random rigid + non-linear deformations to the held-out pairs and re-measure MRR. This lets you tune deformation-invariance *without* spending submissions.
- **L3 has no local proxy** (no labels, different hospital) → rely on label-free branches (B1/B2/B4/Stage2) and validate qualitatively (UMAP, nearest-neighbor sanity).
- Submit to Kaggle only to confirm the local→leaderboard correlation and at milestones.

---

## 8. 24-hour timeline

| Hours | Goal | Deliverable |
|------|------|-------------|
| 0–2 | Data loading (MONAI, reuse baseline), **local MRR harness + self-made dataset1 split + synthetic-deformation L2 proxy** | trustworthy offline metric |
| 2–4 | **B1 (MIND-SSC)** + **B2 (frozen foundation embedding)** → first **RRF** submission | non-trivial score on all 3 datasets |
| 4–10 | **B3 shared encoder**: synthetic contrast + independent deformation augmentation + supervised contrastive (SwinUNETR backbone) | learned contrast-agnostic embedding |
| 10–14 | **Stage-2 re-rank** (SynthMorph residual + seg overlap) on top-k | big MRR lift on L1/L2 |
| 14–18 | **B4 shape fingerprint** + tune **RRF / per-dataset weights** | full 4-branch ensemble |
| 18–22 | Full submissions, ablations (per-branch per-dataset MRR, aug on/off, re-rank on/off) | results table |
| 22–24 | Freeze best, write up, slides, repro script | final submission + presentation |

---

## 9. Why this is creative *and* likely to win

- It treats the problem as **invariance engineering against three named nuisance factors**, and manufactures each invariance explicitly (synthesis for contrast, deformation aug + keypoints/shape for geometry, foundation pretraining for domain shift).
- It **mirrors the organizer's own published approach** (matching-by-synthesis + contrastive descriptor [9]) — a strong signal of the intended solution space.
- **Rank fusion across heterogeneous branches** is the cheapest robustness mechanism and is directly optimal for a macro-averaged-across-heterogeneous-datasets metric.
- The **training-free + foundation branches need no target labels**, so they carry dataset2/3 where the baseline collapses.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Hidden labels → can't measure L2/L3 | Synthetic-deformation local proxy; label-free branches; UMAP sanity checks |
| 100 submissions/day | Trust local MRR; submit only at milestones |
| 3D compute cost | Patch/downsample, fp16, MONAI PersistentDataset cache, SwinUNETR |
| 350 pairs → overfit | Heavy synthetic aug, foundation priors, freeze backbone + train projection head |
| Variable volume shapes (esp. L2/L3) | Resample to common spacing (1 mm given), pad/crop, keypoint matching is shape-agnostic |
| Domain shift L3 | Lean on B1/B2/B4/Stage-2 (no target training needed) |

---

## 11. Future extension: FLAIR↔T1 (missing-modality reduction)

The same shared **contrast-agnostic latent space + matching-by-synthesis** generalizes to *any-contrast-to-any-contrast*: train the encoder with FLAIR, T1, T2, ceT1 all synthesized from the same generative augmentation, so all contrasts land in one space. Then a missing FLAIR can be retrieved (or its embedding imputed) from an available T1. This is the natural paper-worthy generalization of the hackathon system.

---

## References

[1] [Modality-Agnostic Structural Image Representation Learning for Deformable Multi-Modality Medical Image Registration](https://consensus.app/papers/details/0d0ff0e8ec13599ebc35c2770f7c456f/?utm_source=claude_code) (Mok et al., 2024, CVPR) — DNS + anatomy-aware contrastive learning for contrast-invariant structural descriptors (DSIR).
[2] [MR-CLIP: Efficient Metadata-Guided Learning of MRI Contrast Representations](https://consensus.app/papers/details/828985bb5ca95a4bb5a73f994b8b2cd9/?utm_source=claude_code) (Avci et al., 2025) — contrast-aware / anatomy-invariant reps, demonstrated on cross-modal retrieval; code public.
[3] [SynthMorph: Learning Contrast-Invariant Registration Without Acquired Images](https://consensus.app/papers/details/dd95597e50335ccfa3a984d8303c80f9/?utm_source=claude_code) (Hoffmann et al., 2020, IEEE TMI) — generative synthesis → contrast-agnostic networks; basis for Stage-2 re-ranking and synthetic-contrast augmentation.
[4] [Cross-modality sub-image retrieval using contrastive multimodal image representations](https://consensus.app/papers/details/edda1dafc50e55bfa80de9f222a077b4/?utm_source=claude_code) (Breznik et al., 2022, Scientific Reports) — cross-modality CBIR; importance of invariance/equivariance in the representation.
[5] [DeepBrainPrint: A Contrastive Framework for Brain MRI Re-Identification](https://consensus.app/papers/details/92213052a80e56dea69885d802ada391/?utm_source=claude_code) (Puglisi et al., 2023) — semi-self-supervised contrastive brain-MRI retrieval, explicit transforms for contrast/age/progression robustness. Closest published analogue to this task.
[6] [Towards Generalisable Foundation Models for 3D Brain MRI (BrainFound)](https://consensus.app/papers/details/0eaa17958ecc5b7fa1cd3ca3176d05d6/?utm_source=claude_code) (Mazher et al., 2025, arXiv) — slice-based SSL, supports T1/T2/FLAIR multimodal input.
[7] [Building a General SimCLR Self-Supervised Foundation Model … 3D Brain MRI](https://consensus.app/papers/details/01acdd664efc57fd87c11b4bb1bed527/?utm_source=claude_code) (Kaczmarek et al., 2025, ICCVW) — public 3D brain-MRI SSL weights (`3D-Neuro-SimCLR`), strong in low-data / OOD.
[8] [A generalizable foundation model for analysis of human brain MRI (BrainIAC)](https://consensus.app/papers/details/5e3b568bd10c57a2aab0ff6c78dcd474/?utm_source=claude_code) (Tak et al., 2026, Nature Neuroscience) — SSL on 48,965 MRIs; strong few-shot / OOD embeddings.
[9] [A 3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration (CrossKEY)](https://consensus.app/papers/details/67ea6b9c65ce50eeb8030f056690b3b0/?utm_source=claude_code) (Morozov, **Dorent**, Haouchine, 2025) — matching-by-synthesis + supervised contrastive + rotation-invariant keypoint descriptors; curriculum triplet loss + dynamic hard-negative mining. **Co-authored by the challenge PI; the most direct blueprint. Code/weights: https://github.com/morozovdd/CrossKEY** ([arXiv/HF mirror](https://hf.co/papers/2507.18551)).
[10] [M3Ret: Zero-shot Multimodal Medical Image Retrieval via Self-Supervision](https://hf.co/papers/2509.01360) (Liu et al., 2025) — unified encoder, SOTA zero-shot image-to-image retrieval and cross-modal alignment.
[11] [Single-subject Multi-contrast MRI Super-resolution via Implicit Neural Representations](https://hf.co/papers/2303.15065) (McGinnis et al., 2023) — INR exchanges anatomical info across contrasts of one subject; relevant to any-contrast latent.
[12] [MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training](https://hf.co/papers/2501.07556) (He et al., 2025) — synthetic cross-modal pretraining for generalizable matching.
[13] [MINIMA: Modality Invariant Image Matching](https://hf.co/papers/2412.19412) (Jiang et al., 2024) — generative data engine for modality-invariant matching features.
[14] [Synth-by-Reg (SbR): Contrastive learning for synthesis-based registration of paired images](https://consensus.app/papers/details/3a61ef7db6405cbfa6289defe5cc5d47/?utm_source=claude_code) (Casamitjana et al., 2021, SASHIMI/MICCAI) — synthesis+contrastive turns inter-modality into easier intra-modality matching; code public.
[15] [MIND: Modality Independent Neighbourhood Descriptor for multi-modal deformable registration](https://consensus.app/papers/details/59ba79621f4c5cbd9acae45de397d169/?utm_source=claude_code) (Heinrich et al., 2012, Medical Image Analysis, 729 citations) — the canonical contrast-invariant self-similarity descriptor; basis for the training-free branch B1 and the re-rank distance.
[16] [Content-Based 3D Image Retrieval and a ColBERT-Inspired Re-ranking (C-MIR)](https://consensus.app/papers/details/350cbf7ae2ad57b681923fd6f9b41186/?utm_source=claude_code) (Khun Jush et al., 2025, J. Imaging Informatics in Medicine) — volumetric late-interaction re-ranking; no pre-segmentation; localizes ROI. Basis for the late-interaction re-rank option in Stage 2.

*Consensus, PubMed, arXiv and Hugging Face were queried; full Consensus result lists (20 per query) are linked at the top of each search.*
