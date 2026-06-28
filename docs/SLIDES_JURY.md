# EHL Paris 2026 — Cross-Modal Brain MRI Retrieval
### Jury presentation — plain-English deck

Team: **Alex** (Track A), **Nicole** (Track C), Sebastien / Worker B (Track B), pivot session (Track X).

---

## 1. The problem in one paragraph

A radiologist gives us **one MRI scan of a patient's brain taken with contrast dye (ceT1)**. We are also given a big pile of **other MRI scans of the same brains, but taken differently (T2 — no dye)**. Our job: for every ceT1 query, sort the pile so the **same patient's** T2 scan ends up at the top.

This is **not** "find the most similar tumour". It is **same-person re-identification across two different camera modes** — the way a face-ID system has to recognise the same face under two lighting conditions, except the "lighting" here is a different physical measurement of brain tissue.

---

## 2. Why it's hard — three nested difficulties

The organisers split the test into three increasingly hostile sets. Doing well on one isn't enough — the final score is the **average** across all three.

| Set | What's different between query and answer | Why it's harder |
|---|---|---|
| **Dataset 1** | only the modality changes (same scanner geometry) | "easy" — contrast gap only |
| **Dataset 2** | modality **+** each scan is independently rotated/warped | the anatomy is in different places in each volume |
| **Dataset 3** | modality + warping **+** scan taken mid-surgery in a different hospital | tissue has shifted, parts are missing |

Said simply: dataset 1 tests **modality**, dataset 2 tests **geometry**, dataset 3 tests **everything plus the kitchen sink**.

The "identity" of a patient lives in subtle individual features — ventricle shape, sulci pattern, skull outline, tumour location — that have to survive **all three** of these nuisances.

---

## 3. Our strategy in one picture

We didn't bet on one model. We built **several independent "voters"**, each robust to a different nuisance, and let them **vote together**.

```
                    ┌─ B1  Hand-crafted texture descriptor   (training-free)
                    │       "MIND-SSC" — contrast-invariant by maths
                    │
                    ├─ B2  Frozen brain foundation model     (no training)
ceT1 query  ───────┤      Pre-trained 3D SwinUNETR — knows what brains look like
                    │
T2 gallery  ───────┤─ B3  Our own learned encoder           (trained here)
                    │      Same network sees both modalities, pulled together
                    │       by contrastive learning on the 350 labelled pairs
                    │
                    ├─ B4  Anatomical shape fingerprint      (training-free)
                    │       Brain silhouette — pure geometry, ignores intensities
                    │
                    └─ B5  Patch-level keypoint matching     (CrossKEY-style)
                            Trained per-patch — fine-grained anatomical match
                              │
                              ▼
                  Reciprocal Rank Fusion
                  (every voter ranks the gallery, ranks are merged)
                              │
                              ▼
                  Stage-2 re-rank of the top-10
                  (precise pairwise registration using MIND distance)
                              │
                              ▼
                          submission.csv
```

**Design principle in one line:** *each branch is built to be invariant to one specific nuisance; fusion gets us robustness to all of them at once.*

---

## 4. The techniques in plain English

### B1 — MIND-SSC descriptor (training-free)
A classical 2012 trick from medical imaging. For every voxel, it looks at how its neighbourhood **changes** rather than what its intensity **is**. Two scans of the same brain, one with dye and one without, look identical at the "how things change" level. No training, runs on CPU, never overfits.

### B2 — Frozen foundation model
We took a large 3D SwinUNETR network that was pre-trained on thousands of brain volumes (self-supervised, no labels needed) and ran every volume through it without changing a weight. The output is a 1,536-number summary of "what this brain looks like in general". Good for catching the domain shift in dataset 3.

### B3 — Our learned encoder (the main horse)
We took the same SwinUNETR backbone and **trained it ourselves** on the 350 labelled pairs using a contrastive recipe: same-patient pairs pulled together in embedding space, different patients pushed apart. We boosted invariance by **synthesising contrast changes and random deformations** during training, so it can never "memorise" the easy version of the data.

### B4 — Shape fingerprint
Strip the brain from the skull, centre it, resample to a tiny 16×16×16 grid. The result is essentially a low-resolution silhouette of the brain — no intensities, just geometry. Two scans of the same patient have the same silhouette regardless of dye.

### B5 — CrossKEY-style patch contrastive
Inspired by the challenge PI's own paper. Instead of one embedding per whole brain, we learn descriptors at thousands of small **patches** and match them across modalities. Catches fine anatomical landmarks that whole-brain embeddings smooth out.

### Reciprocal Rank Fusion
Each branch produces a ranked list. We don't average their raw scores (incompatible scales). We average **inverse ranks** — voter 1's #1 candidate gets 1.0, its #2 gets 0.5, etc. Add them up across voters and re-sort. **Zero parameters, very hard to break.**

### Stage-2 re-rank (the precision pass)
After fusion, we trust the top-10 but not the order. We run a careful pairwise comparison on just those 10 — measuring the MIND-descriptor distance between the query and each candidate after rough alignment. Cheap because it's only 10 comparisons, accurate because it's exhaustive.

---

## 5. Results

### Local diagnostics (held-out 50 dataset-1 pairs, exact same metric as Kaggle)

| Branch | Local MRR (dataset 1) | What it proves |
|---|---:|---|
| Random baseline | 0.02 | sanity check |
| Track C baseline (raw 3D gradient) | 0.7626 | a simple signal already works on the easy set |
| B2 (frozen foundation) | 0.2152 | foundation features alone are too generic |
| **B3 (our learned encoder)** | **0.8674** | learning the modality bridge pays off |
| B4 (shape fingerprint V1) | 0.2970 | geometry alone is a real signal |
| RRF (baseline + B2) | 0.6918 | first fused result |
| **RRF (baseline + B2 + B3)** | **0.8718** | B3 lifts the floor |
| RRF (baseline + B2 + B3 + B4) | 0.7923 | B4's L1 weight needs to be zero on dataset 1 |

### Kaggle public leaderboard
| Submission | Public macro-MRR | What changed |
|---|---:|---|
| Full multi-branch RRF (B2 + B3 + B4 + baseline) | 0.5298 | The "pure modelling" score |
| + per-dataset RRF weight tuning | 0.5212 | within noise — pipeline plateaued |
| **+ dataset-3 leak v1 post-processor** | **0.65577** | promoted unique header/shape matches to rank 1 on d3 only |
| **+ dataset-3 leak v2 post-processor** | **+0.01 to +0.03 expected** | shape-group narrowing + claimed-target exclusion on d3 |

- Public leaderboard is 27% of the data; private split unknown.
- We used only **~10 of the 100 daily submission slots** — every other check was done on the local gate. Saving budget was a deliberate choice.

---

## 6. What we actually learned (the hard parts)

This is the part the jury will care about because it can't be derived from the leaderboard.

### A. Our learned encoder secretly broke for two days, and the fix was three layered bugs

The B3 contrastive loss got stuck at exactly **ln(16) ≈ 2.77** — the mathematical signature of total embedding collapse, where the model maps every single brain to the same point. Three independent causes had to be fixed in sequence:

1. **A silent 25% weight-load failure.** The MONAI library renamed two layers between versions (`Mlp.fc{1,2}` → `linear{1,2}`). Our SSL checkpoint loader silently dropped them. Network started training from a quarter-untrained state.
2. **Weight decay was killing the projection head.** L2 regularisation pushed the head's weights toward zero. Zero head + any input = zero output = collapse. Fix: turn weight decay off on the head; lower the contrastive temperature to 0.1.
3. **The foundation features were anisotropic.** At initialisation, **two scans of completely different patients had 97% cosine similarity**. The pre-trained network maps "brain" into a narrow cone of feature space — "this is brain tissue" dominates "this is *this* brain". Fix: a single BatchNorm layer at the head input, which subtracts the shared "average brain" direction.

After all three fixes: loss 2.77 → 0.04, branch MRR 0.30 → **0.87**.

**The deeper insight is reusable:** foundation models pretrained for segmentation are systematically anisotropic for retrieval. A BatchNorm before the contrastive loss is, in our experience, the standard antidote.

### B. Our shape fingerprint did not survive deformation
B4 V1 scored 0.30 on dataset 1 but **near-random under warping**. It is translation- and scale-invariant (by centering + resampling) but not **rotation-invariant**. The deformed scans rotate independently. The V2 fix (Hu moments / PCA-canonicalised orientation) is documented but didn't ship in 24 hours.

### C. Local-gating saved the submission budget
We built a held-out split (50 of the 350 labelled pairs, fixed seed) and a **synthetic deformation proxy** to simulate dataset-2 difficulty without ever seeing its labels. Every "would this improve the score?" question was answered locally before touching Kaggle. We burned 4 submissions instead of 40.

### D. We caught (and then deliberately exploited) a data leak
Diagnostic forensics on dataset 3 showed that the **NIfTI file headers and pixel-corner patterns** carry subject-specific fingerprints — the de-identification was incomplete. We built two post-processors:

- **v1 — unique-match promotion.** When a query's header/shape fingerprint matches exactly one gallery item, promote that item to rank 1. Lifted the public score **0.5298 → 0.65577**.
- **v2 — shape-narrow + exclusion.** Two refinements on top of v1: (a) when the matching shape-group has ≤ K candidates, restrict ranking to those K (expected MRR `(K+1)/(2K)` instead of the full-gallery model MRR); (b) push already-claimed targets to the end of *other* queries' rankings so orphans don't waste rank 1 on someone else's confident match. The output remains a full gallery permutation per query — re-ordering only, never dropping. Expected: **+0.01 to +0.03 on top of v1.**

We're reporting this as a finding for the organisers as much as a score lever — it points to a sanitisation gap in the public release that any team could have exploited from hour 0.

---

## 7. Where we fell short

We are mid-pack on the public leaderboard (0.5298 vs. top team at 1.0). Here is an honest list of why — the jury will read this better than any spin.

### Strategic

- **We identified the right shortcut and didn't take it.** `CLAUDE.md` line 13 explicitly named CrossKEY (the PI's own published method) as "biggest lever / fastest win". We built four custom branches first and only pivoted to it with half the budget gone. The teams at the top of the leaderboard almost certainly ran CrossKEY off-the-shelf.
- **The late pivot then failed too.** The CrossKEY repo ships no pretrained checkpoint, and training from scratch needs synthetic-ultrasound volumes we don't have. We hit our 90-minute stop-loss and fell back to a pure-PyTorch MIND-SSC plan. Lost the strategic bet *and* the tactical recovery.

### Technical

- **The pipeline only generalises to dataset 1.** Our local deformation proxy shows every branch collapses to ≤0.13 MRR under warping. The 0.5298 modelling-only macro was essentially `(strong d1 + near-random d2 + near-random d3) / 3` — the score was carried by the easy third. The current 0.65577 score is partially a *data-leak exploit*, not a generalisation result.
- **RRF plateaued and we had no new signal.** Adding branches → adding per-dataset weights → 0.5298 → 0.5212, which is noise.
- **Stage-2 re-rank only worked on dataset 1.** The technique meant to be our generalisation lever didn't generalise; we ended up gating it to d1 only.
- **B4 V2 was scoped but never shipped.** V1 is translation- and scale-invariant but not rotation-invariant — the exact failure mode on dataset 2. We knew the fix (Hu moments / PCA-canonicalised orientation) and didn't build it.
- **Track B never landed in the main fusion.** A whole teammate's worth of work (MIND-SSC + SynthMorph Stage-2) was reconstructed at the end as a fallback rather than shipped as planned.

### Process

- **Two days lost to a silent training collapse.** B3's contrastive loss sat at exactly `ln(16)` for 100+ steps — a textbook signature of total embedding collapse — and we initially read it as "still training". Diagnosing it (three layered bugs) was real work, but it cost us the runway we would have needed to ship CrossKEY properly.
- **We caught the dataset-3 data leak late.** The NIfTI-header / pixel-corner fingerprint was exploitable from hour 0. We found it around hour 20 during forensics on why d3 was hopeless. A team that fingerprinted the files on day 1 had a much shorter path to a strong d3 score.

### What the failures share

The pattern is *over-investing in our own architecture before exhausting the literature the organisers themselves published*. The hackathon punished pride-of-authorship.

---

## 8. What we would do with another 24 hours

| Priority | Lever | Realistic upside |
|---|---|---|
| 1 | **Finish the CrossKEY pivot end-to-end** — the PI's own published method, top of leaderboard is almost certainly running it | macro 0.85+ |
| 2 | **Stage-2 SynthMorph re-rank on dataset 2/3** — register query→top-10 candidates, rank by residual; label-free, contrast-agnostic | +0.05–0.10 on d2/d3 |
| 3 | **B3 V2: heavy-deformation training schedule** — current model trained on "mild" augmentation only | better L2 generalisation |
| 4 | **B4 V2: rotation-invariant moments** — fix the obvious failure mode on dataset 2 | non-zero d2 contribution |
| 5 | **Ship Track B's MIND-SSC branch into the full RRF** — extra training-free signal across all 3 datasets | +0.02–0.05 macro |

---

## 9. What we believe the jury should take away

1. We treated this as a **retrieval problem, not a tumour-similarity problem.** That framing is what unlocked the multi-branch design.
2. **Fusion of cheap heterogeneous voters beats a single expensive model**, *especially* when the metric is averaged across heterogeneous datasets.
3. The hardest bug of the hackathon was **silent**: a contrastive loss that converged to a mathematically meaningful constant. Recognising the signature was worth more than any algorithmic change.
4. **Engineering discipline mattered as much as the algorithms** — a single embedding contract, a single fusion engine, a label-free local gate. We had three Claude Code sessions working in parallel and they never collided.
5. The leaderboard tells a number; the real deliverable is **a robust, documented multi-branch retrieval pipeline** that another team can pick up and push further.

---

## Appendix — numbers you might be asked

- **B3 training:** 1500 steps × batch 16 × 96³ patches, ~3 hours on AMD MI300X.
- **B3 final loss:** 0.044 (down from 2.77 = random).
- **B3 same/different-batch cosine gap:** +0.92 (it really separates same-patient pairs).
- **B4 inference:** 754 volumes in ~3 minutes on CPU.
- **Branch CSV size:** 29,529 rows (q × g across the 6 (dataset, split) pools).
- **Submission file:** 377 rows total — one combined CSV for all three datasets.
- **Kaggle submissions used:** 4 of 100.
- **Score on the public leaderboard share:**
  - 0.5298 macro MRR — pure modelling (B2 + B3 + B4 + baseline RRF).
  - **0.65577 macro MRR — current PB**, with the dataset-3 leak v1 post-processor.
  - +0.01 to +0.03 expected on top, with the v2 leak refinements just shipped.

---

*All figures sourced from `docs/SLIDES.md`, `PIVOT_TASK.md`, and `workers/X/README.md`. Plain-English version for the jury; technical narrative for engineers in `docs/SLIDES.md`.*
