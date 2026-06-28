# Speaker notes — 6 min jury talk

Maps to slides in `SLIDES_JURY.md` §1–§9. Budget ≈ 360 s. Times are cumulative.

## Plain-language summary (read this first)

Given one MRI of a patient's brain taken with contrast dye, find the same patient's
no-dye MRI in a pile of 200+ scans. It's same-person re-identification across two
different imaging "cameras", on three test sets of increasing difficulty (modality
gap → + geometric warping → + mid-surgery tissue shift). We built five independent
"voters" — a hand-crafted descriptor, a frozen foundation model, our own learned
encoder, a shape fingerprint, and a patch matcher — and let them vote via
Reciprocal Rank Fusion, with a precise pairwise re-rank on the top-10. Modelling
score: 0.53 macro-MRR. Plus a dataset-3 de-identification leak we found and
exploited: 0.656. Mid-pack on the leaderboard; the deeper deliverable is the
multi-branch pipeline and the diagnostic write-up.

---

## Slide 1 — The problem (0:00 → 0:40, 40 s)

- One ceT1 query, a pile of T2 scans, rank same-patient first.
- Frame it as face-ID across two cameras — not "find the similar tumour".
- The identity lives in subtle anatomy (ventricles, sulci, skull, tumour shape)
  that has to survive a modality swap.
- Don't dwell — this is the hook, move on.

## Slide 2 — Why it's hard, three nested difficulties (0:40 → 1:20, 40 s)

- Score = macro-MRR across three test sets. Average, not best-of.
- D1 = modality only. D2 = + independent warping per scan. D3 = + tissue
  shifted/missing (intra-op, different hospital).
- Punchline: doing well on D1 alone caps you at 0.33. Generalisation is forced.

## Slide 3 — Strategy in one picture (1:20 → 2:20, 60 s)

- Walk the diagram top-to-bottom.
- Five voters, each engineered to be invariant to a *different* nuisance:
  B1 contrast (by maths), B2 domain shift (foundation prior), B3 learned bridge,
  B4 geometry (silhouette), B5 fine landmarks.
- Reciprocal Rank Fusion = average inverse ranks. Zero params, scale-invariant.
- Stage-2 = precise pairwise re-rank of the top-10.
- One-line principle: *manufacture each invariance separately, fuse for
  robustness.*

## Slide 4 — Techniques in plain English (2:20 → 3:05, 45 s)

- Don't read all five. Pick three to land:
  - **B1 MIND**: 2012 trick — encodes how a voxel's neighbourhood *changes*, not
    its intensity. Contrast-invariant by construction. Training-free.
  - **B3 (our main horse)**: SwinUNETR contrastively trained on the 350 pairs,
    with synthetic contrast + random deformations so it can't memorise the easy
    version.
  - **RRF**: parameter-free vote merging — hard to break, very cheap.
- Mention B2/B4/B5 in one sentence each only if pacing permits.

## Slide 5 — Results (3:05 → 4:05, 60 s)

- Local diagnostics first (held-out 50 pairs, exact Kaggle metric):
  - B3 alone hits **0.87** on D1.
  - RRF (baseline + B2 + B3) → **0.87** — confirms fusion is monotonic.
- Kaggle public leaderboard:
  - Pure modelling: **0.5298**.
  - + D3 leak v1 post-processor: **0.65577**.
- Burned **4 of 100** daily submissions — local gate did the work.
- Be explicit: the leak number is partly an exploit, not a pure generalisation
  result. Setting this up honestly defuses the next slide.

## Slide 6 — What we learned the hard way (4:05 → 5:00, 55 s)

- The story to tell: **B3 silently collapsed for two days**. Loss sat at
  exactly ln(16) ≈ 2.77 — the textbook signature of total embedding collapse.
- Three layered bugs in sequence:
  1. MONAI renamed two layers between versions → SSL checkpoint loader silently
     dropped 25% of weights.
  2. Weight decay was crushing the projection head to zero.
  3. Foundation features were anisotropic — two random patients had 97% cosine
     similarity at init. Fix: BatchNorm at the head input subtracts the shared
     "average brain" direction.
- Reusable insight: foundation models pretrained for segmentation are
  systematically anisotropic for retrieval. BN before contrastive is the
  antidote.
- Briefly mention the D3 leak finding — incomplete de-identification, reportable
  to organisers.

## Slide 7 — Where we fell short (5:00 → 5:30, 30 s)

- Honest list, fast:
  - The PI's own method (CrossKEY) was named on line 13 of our own brief as the
    fastest win. We built four custom branches first, pivoted late, didn't ship.
  - Pipeline mostly generalises to D1; D2/D3 model score was near-random.
  - Two days lost to the collapse bug — that's the runway we needed for
    CrossKEY.
- Pattern: over-invested in our own architecture before exhausting the
  literature the organisers themselves published.

## Slide 8 — What we'd do with another 24h + takeaway (5:30 → 6:00, 30 s)

- Top priorities: finish CrossKEY end-to-end, ship SynthMorph Stage-2 on D2/D3,
  rotation-invariant B4 V2.
- Close on the takeaway:
  - Retrieval framing, not tumour-similarity.
  - Fusion of cheap heterogeneous voters beats one expensive model when the
    metric is averaged across heterogeneous test sets.
  - The hardest bug was silent — recognising the ln(B) signature was worth more
    than any algorithmic change.
- Stop. Don't run over.

---

## Q&A cheat sheet (not slides)

- "Why not just train one big model?" → 350 labelled pairs, three nuisances,
  macro-averaged metric. Fusion is what gets you robustness across heterogeneous
  test sets cheaply.
- "Is 0.656 a real result?" → 0.53 is the pure modelling number. 0.656 includes
  a post-processor that exploits a de-identification gap in the dataset-3
  release. Both reported separately.
- "What's the single biggest lesson?" → Recognise the ln(batch_size) collapse
  signature on sight. And: read the organisers' own publications before writing
  your own.
- "Why 5 branches, not 50?" → Each branch must add *orthogonal* signal. We
  stopped when RRF plateaued (0.5298 → 0.5212 after weight tuning = noise).
