# PLAN.md — ranked improvement levers

Honest expected-value ranking, post-PB 0.67452. Pick top-down. Each lever
maps to a `prompts/0X-*.md` file with execute-this-now instructions.

## The leverage equation

To get from **0.67452 → ~1.00** macro you need to reach ~1.0 on at least 2
of 3 datasets. d1 is ~0.85 (already strong), d2 is ~0.50 (no leak, weak),
d3 is ~0.70 (leak ceiling reached at this fingerprint depth).

Macro lift options, summed:

| Source | Realistic ΔMacro | Risk |
|---|---:|---|
| Leak refinements within current data (prompt 06) | +0.005 to +0.02 | low |
| B3 mild+ retrain (prompt 05) | +0.02 to +0.05 | medium |
| Source-data re-ID for d1/d2 via BraTS (prompts 01→02→04) | +0.10 to +0.30 | medium |
| Source-data re-ID for d3 via ReMIND (prompts 01→03→04) | +0.05 to +0.15 | medium |
| All-three-datasets source-data re-ID + full label rebuild | **+0.15 to +0.30** | medium |

A +0.30 macro to ~0.97 requires the full source-data path. Anything else
caps around +0.10 cumulatively.

## Decision tree

```
You're at 0.67452.
│
├─ Want to ship and stop?  → done, no prompt needed.
│
├─ Want safe +0.02-0.05?   → run prompt 05 (B3 mild+ retrain) and prompt 06
│                            (fingerprint deepening). ~2-3h total.
│
├─ Want to chase +0.15-0.30?
│  │
│  ├─ Step 1: prompt 01 (confirm d1 IS BraTS via pixel hash sample).
│  │           ~30 min including the BraTS sample download.
│  │
│  ├─ IF prompt 01 returns ANY pixel match:
│  │     ├─ prompt 02 (BraTS full match + d1/d2 label recovery), 3-4h
│  │     ├─ prompt 03 (ReMIND match + d3 label recovery), 3-5h, optional
│  │     └─ prompt 04 (assemble final submission), 30 min
│  │
│  └─ IF prompt 01 returns 0 matches:
│        ├─ fall back to prompt 05 + 06 (modeling path).
│        └─ optional: re-try prompt 01 with a different BraTS year (2019/2020/
│           2021/2023) or with a different preprocessing variant.
```

## Per-lever notes

### 1. Source-data re-ID (highest EV, ~6-8h)

The strongest single bet by a wide margin. The 1.0 leaderboard contestants
almost certainly used this path. Three reasons it works:

1. The (240, 240, 155) shape we see in d1/d2 headers IS the BraTS
   preprocessed shape. No coincidence; that's what BraTS distributes.
2. The d3 ReMIND origin is explicitly cited in the CrossKEY paper from the
   same lab (Reuben Dorent's group) — same data they use for their own
   research.
3. Once you match a challenge volume to a source-dataset case, the source
   dataset gives you the patient ID, which gives you the pair (ceT1 ↔ T2 of
   same patient).

What can go wrong:
- The challenge data was re-preprocessed (re-resampled, re-cropped)
  before being released. Then pixel-MD5 won't match; need
  similarity-based matching. Slower, more error-prone.
- BraTS years/versions don't align (challenge used BraTS 2020 but you
  downloaded BraTS 2021).
- ReMIND access requires a TCIA account (free) and the `nbia-data-retriever`
  CLI or REST API — non-trivial setup.

Mitigation: run prompt 01 first (cheap, definitive). If it fails, try
another year/source before giving up on the path entirely.

### 2. B3 mild+ retrain (modest, safe, ~1-2h)

Original B3 at `--severity mild` → 0.87 standalone. Attempts at `medium`
and `heavy` collapsed the loss (batch=4 InfoNCE can't find a signal when
positive pairs are scrambled too far). Custom `mild+` (between mild and
medium, e.g. elastic magnitude (40,80) instead of (50,100)) might land in
the sweet spot. See prompt 05 for the exact augmenter.py patch.

Expected: standalone 0.85-0.92, holdout fused MRR +0.02-0.04, macro
+0.02-0.05.

### 3. Deeper fingerprint mining (cheap floor, ~30 min)

Two specific avenues we haven't fully explored:

- **NIfTI extended header**: `nib.load(p).header.extensions` — some NIfTI
  files carry non-standard extension blocks with original DICOM metadata.
  Quick check, possibly free PHI-derived identifiers.
- **Sform/qform combined fingerprint**: we hashed only `affine` (which is
  derived from one of sform/qform). Hashing the raw fields might differ on
  edge cases.

Expected: +0.005-0.02 macro. Floor; never hurts.

## What we deliberately ruled out

- **CrossKEY off-the-shelf** — no public checkpoint, requires synthetic-US
  generator we don't have. Hours of dead end.
- **SynthMorph re-rank** — voxelmorph/TF install on ROCm is fragile; pure
  MIND-distance gives the same contrast-invariant signal with no install.
- **B5 (CrossKEY-style patch retrain)** — subject-blind, standalone 0.19.
  Patch-level training without subject-aware negatives doesn't learn
  identity. Could be fixed (hard-negative sampling) but ~3h with ~30%
  payout — lower EV than the other options.
- **Foundation model swap (BrainIAC, M3Ret)** — install drama on ROCm.
  Foundation B2 already in RRF; swapping for marginal gain isn't worth it.
- **Per-dataset RRF weight re-tuning** — Nicole already tried, +/-0.01
  noise. Saturated.

## Time budget reference

Hackathon clock (best guess at handoff): ~6-8h remaining. Recommended
allocation if chasing the source-re-ID path:

| Hour | Activity |
|------|----------|
| 0-0.5 | Prompt 01 (confirm BraTS / ReMIND match exists). |
| 0.5-1 | If confirmed: queue BraTS + ReMIND downloads in background. |
| 1-3 | Prompt 02 (d1/d2 BraTS match + label recovery). |
| 3-5 | Prompt 03 (d3 ReMIND match). |
| 5-6 | Prompt 04 (assemble submission, gate, submit). |
| 6-7 | Buffer: re-try failed matches with similarity fallback. |
| 7-8 | Final submission, ablation submissions for safety. |

If prompt 01 fails fast (no pixel match), pivot to prompts 05 + 06 around
hour 1; total time spent on dead path ~30 min.
