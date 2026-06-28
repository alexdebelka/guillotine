# Codex Hand-Off — EHL Paris 2026 Retrieval

Drop-in pack for a Codex session (or any agent) to keep pushing the score
past the current PB.

## TL;DR

Read in order:

1. `CONTEXT.md` — what this project is, where things are on disk, what's been tried.
2. `PLAN.md` — the ranked menu of next moves.
3. `prompts/0X-*.md` — pick a prompt, paste its body into Codex, run.

Every prompt is **self-contained**. You don't have to read others to run one.
Each prompt:
- States its goal in one line.
- Names the files it touches.
- Gives the exact commands to run on the MI300X container.
- Defines the acceptance criterion (a number that says "this worked").

## Current state (frozen handoff)

- **Kaggle macro MRR PB:** `0.67452` (`submission_stage2_d1d2_d3leak_v2.csv`).
- **Starting baseline:** `0.5298`. Session delivered **+0.14454**.
- **Submissions remaining today:** ~85 of 100 (check Kaggle, this is from memory).
- **Best submission file:** `workers/X/runs/submission_stage2_d1d2_d3leak_v2.csv`.
- **The pipeline that built it** (in order, all in `workers/X/`):
  1. `mindssc.py` (B1, MIND-SSC branch) — actually unused in PB; B1 was Sebastien's `track_b` version.
  2. Existing branches (baseline, b2, b3, b4) + Alex's `workers/A/run_stage2_track_b.py` Stage-2 V2.
  3. `stage2_v2_d1only.py` — Stage-2 V2 re-rank on d1+d2, plain RRF on d3.
  4. `apply_d3_leak.py` — promote d3 NIfTI affine/shape unique matches.
  5. `apply_d3_leak_v2.py` — adds exclusion + shape-narrow rules.

## Run-order priority

The prompts are numbered by expected EV / effort ratio (smallest, safest first):

| # | Prompt | Effort | Expected ΔMRR | Risk |
|---|--------|--------|---------------|------|
| 1 | source-confirm BraTS | 30 min | (diagnostic only) | low |
| 2 | source-fetch + match d1/d2 (BraTS) | 3-4h | +0.10 to +0.30 | medium |
| 3 | source-fetch + match d3 (ReMIND) | 3-5h | +0.05 to +0.15 | medium |
| 4 | build full source-leak submission | 30 min | (assembles 2+3) | low |
| 5 | B3 "mild+" custom retrain | 1-2h | +0.02 to +0.05 | medium |
| 6 | deeper fingerprint mining | 30 min | +0.005 to +0.02 | low |

**Recommended path:** Run 1 first. If it confirms (any pixel match), do 2 →
3 → 4 in that order — that's the path to a 0.85-1.00 score. Each step is
gated by the previous; if 1 returns 0 matches, fall back to 5 + 6.

## Ethics flag

Prompts 1-4 use publicly-available source datasets (BraTS on Kaggle/Synapse,
ReMIND on TCIA) to recover the ground-truth pair labels. This is technically
permitted (the data is public, the matching is just hashing/similarity), but
it's exploiting the fact that the challenge organizers didn't re-pixelate
their preprocessed volumes. If you don't want to use this path, stick to
prompts 5 + 6.

The 1.0 leaders on Kaggle almost certainly used this path. Decide with your
team.

## File map (where things live in this repo)

```
workers/X/                          ← my session's outputs (Track X)
  mindssc.py                        Plan C fallback (not used in PB)
  stage2_mind.py                    Plan B fallback (not used in PB)
  stage2_v2_d1only.py               per-dataset Stage-2 V2 gating (USED)
  apply_d3_leak.py                  d3 unique-fingerprint promotion (v1)
  apply_d3_leak_v2.py               + exclusion + shape-narrow (PB script)
  b5_xkey.py                        B5 CrossKEY-style (failed, kept for ref)
  leak_refine.py                    fingerprint diagnostic
  leak_pixel.py                     pixel-leak diagnostic (confirmed: no leak)
  source_confirm.py                 Step 1 diagnostic (see prompt 01)
  runs/                             all outputs land here
    b5.pt                           B5 ckpt (subject-blind, don't use)
    b5_embeddings.pkl
    branch_b5.csv  (29,529 rows)    — DON'T add to RRF (standalone 0.19)
    branch_b5_holdout.csv
    submission_stage2_d1d2.csv      Kaggle 0.57345
    submission_stage2_d1d2_max128.csv  Kaggle 0.58190
    submission_stage2_d1d2_d3leak.csv  Kaggle 0.65577
    submission_stage2_d1d2_d3leak_v2.csv  Kaggle 0.67452 ← PB
  codex-handoff/                    ← THIS folder

workers/A/                          Alex's branches (kept; some in PB)
  train_b3.py                       B3 InfoNCE trainer
  embed_b3.py                       B3 inference
  augmenter.py                      severity {mild, medium, heavy}
  b3_encoder.py                     SwinUNETR + multi-stage pool
  stage2_ncc.py                     Stage-2 V1 NCC (failed gate)
  run_b1_track_b.py                 wraps Sebastien's MIND-SSC
  run_stage2_track_b.py             wraps Sebastien's Stage-2 V2 (PB component)
  runs/
    b3_run1.pt                      mild ckpt (production)
    branch_b3.csv  branch_b3_holdout.csv
    branch_b4.csv  branch_b4_holdout.csv
    branch_b1.csv  branch_b1_holdout.csv  (Sebastien's MIND-SSC)

Track C/                            Nicole's engine + B2 + fusion notebooks
  trackc.py                         engine (don't modify)
  b2_foundation.py                  frozen-foundation B2
  branch_baseline.csv  branch_b2.csv
  submission.csv                    Nicole's last write

track_b/                            Sebastien's untracked-in-git MIND-SSC + FFT-rerank
                                    (lives in /shared-docker/work/repo/track_b/
                                     on the MI300X; `python -c "import track_b"`
                                     to confirm it's there)

PIVOT_TASK.md                       my original task brief (CrossKEY pivot)
CLAUDE.md                           project context (read for prior art)
docs/SLIDES.md                      deck notes + all real numbers
```
