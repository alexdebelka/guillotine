# PIVOT TASK — fresh-session handoff

Paste this whole file (or `cat PIVOT_TASK.md`) at the start of a new Claude Code session pointed at this repo. It is self-contained: nothing from the conversation that produced it carries over.

---

## TL;DR
- 24h hackathon, ~half consumed. Current Kaggle macro MRR = **0.5298**. Leaderboard top is 1.0; we are mid-pack.
- Multi-branch RRF (B2 + B3 + B4 + baseline) is plateaued. Per-dataset weight tuning didn't move the public score (`0.5298 → 0.5212`, within noise).
- Pivot to **CrossKEY** — the PI's own published method, the most likely route to a top-tier score in the remaining time. Repo: https://github.com/morozovdd/CrossKEY
- Owner: this session. Output: `branch_crosskey.csv` in Track C's existing branch contract.
- Hard gate: only submit to Kaggle if the local MRR (on the seed=0 first-50 dataset1 split) materially beats `0.5298 + 0.02`.

---

## Project overview (read once)

See `CLAUDE.md` and `docs/SLIDES.md` for the full picture. Compact version:

**Task:** ceT1 → T2 brain MRI same-subject retrieval. For each query ceT1 volume, rank a gallery of T2 volumes so the same subject is rank 1.

**Three datasets, increasing nuisance:**
| Dataset | Nuisance | Labels |
|---|---|---|
| dataset1 | contrast only (registered grid) | 350 train pairs + 40 val + 100 test |
| dataset2 | + independent rigid + nonlinear deformation per image | 40 val + 100 test |
| dataset3 | + structural change (preop→intra-op) + domain shift | 20 val + 77 test |

**Metric:** MRR macro-averaged over the 3 datasets. **Submission budget:** 100/day. **Submission file:** 377 rows total (one row per query, gallery IDs space-separated).

**Team layout:**
- **Track A — Alex** — learned encoder (B3, InfoNCE on SwinUNETR) + shape fingerprint (B4). Code in `workers/A/`.
- **Track C — Nicole** — eval harness (`trackc.py`), frozen-foundation B2 (`b2_foundation.py`), RRF fusion, submission writer. Code in `Track C/` and `workers/C/`.
- **Track B** — was supposed to be MIND-SSC + Stage-2 re-rank. **Empty.** A `branch_b.csv` of unknown provenance landed in Nicole's folder.
- **This session (you)** — owns the CrossKEY pivot.

**Compute:** MI300X (192 GB vRAM) on JupyterHub container at `/shared-docker/`. Data at `/shared-docker/data/`. Repo at `/shared-docker/work/repo/` synced with `https://github.com/alexdebelka/guillotine.git` (branch `main`).

---

## Where we are right now

**Branches shipped (CSV at `/shared-docker/work/repo/workers/A/runs/` or `Track C/`):**
| Branch | What | Local d1 MRR (seed=0 first-50) |
|---|---|---|
| baseline | gradient-magnitude on 48³ downsample | 0.7626 |
| b2 | frozen SwinUNETR SSL features, avg-pool | 0.2152 |
| b3 | SwinUNETR + InfoNCE + BatchNorm projection head | **0.8674** |
| b4 | centered brain mask resampled to 16³ | 0.2970 |
| b | unknown teammate branch in Nicole's folder | unknown |
| **fused (RRF, per-dataset weights)** | Kaggle macro | **0.5298** |

**The B3 collapse story** (in case it's useful background): InfoNCE was stuck exactly at `ln(B)` (embeddings collapsing to a point) due to three compounding issues — MONAI 1.6 renamed `Mlp.fc{1,2}`→`linear{1,2}` so 25% of SSL weights weren't loading; weight-decay was pushing the projection head to zero; SSL features were anisotropic (~97% cosine sim across DIFFERENT brains at init). Fix: SSL loader patch + `wd=0` + temp 0.1 + **BatchNorm at the head input** (subtracts the "common brain" direction). After fixes: loss 2.77 → 0.04, B3 standalone 0.87. Full story in `docs/SLIDES.md` §5.

**Held-out for local gates:** `trackc.make_local_split(seed=0)` takes the FIRST 50 of a shuffled `train_pairs.csv`. Use this exactly. Existing held-out CSVs in `workers/A/runs/branch_b{3,4}_holdout.csv`.

---

## Why we're pivoting

- Kaggle macro 0.5298 with full RRF; tuned 0.5212 (~noise).
- Top of leaderboard at 1.0 with 4 submissions — almost certainly **CrossKEY off-the-shelf**.
- CLAUDE.md line 13 already flagged CrossKEY as "biggest lever / fastest win." We skipped it. The pivot is correcting that.
- Our current branches do not generalize to dataset2 (deformation) or dataset3 (domain shift) — the local L2 deformation proxy showed every branch we have collapses to ≤0.13 MRR under deformation.

---

## Pivot deliverable

**Output:** `/shared-docker/work/repo/workers/X/runs/branch_crosskey.csv` (X = your track folder; create `workers/X/` and put outputs in `runs/`).

**Format (exact):** CSV with columns `query_id, target_id, score`. Higher score = more similar. ONE row per (query, target) WITHIN each of the 6 (dataset, split) pools — never across pools. Same format Nicole already consumes; see `workers/A/runs/branch_b3.csv` for a working example (29,529 rows total).

**Hand-off:** Add the path to Nicole's notebook (`Track C/run_fuse_submit.ipynb`):
```python
branch_crosskey = pd.read_csv('/shared-docker/work/repo/workers/X/runs/branch_crosskey.csv')
# add to the fuse_branches({...}) dict and to the weights dict for each dataset
```

**Gate before submitting to Kaggle:** produce a held-out version too (`branch_crosskey_holdout.csv`) on the same seed=0 first-50 d1 pairs. Pattern in `workers/A/embed_holdout.py`. Nicole's gate cell already takes `pd.read_csv(...) → scores_to_rankings(...)`. If `rrf([all branches including crosskey])` on the held-out doesn't clear our current 0.6918 fusion by ≥0.02, don't burn a submission.

---

## CrossKEY specifics

- **Repo:** https://github.com/morozovdd/CrossKEY
- **Authors:** Morozov et al. (same lab as the challenge PI Reuben Dorent). Paper: "Cross-Modal Keypoint Matching" — designed for MR→US registration but ports to MR→MR cross-contrast.
- **Approach:** Per-voxel keypoint descriptors learned via synthetic contrast generation + supervised contrastive on matched keypoints. Optionally MIND-SSC as the geometric descriptor baseline.
- **Inputs:** 3D volumes, usually preprocessed. Need to adapt to our 1mm RAS NIfTI.
- **Outputs:** dense descriptor maps; rank gallery candidates by mean descriptor cosine sim (or top-k keypoint match score).

**Concrete first-30-minutes plan:**
1. `cd /shared-docker/work/repo && git pull` (sync the latest state).
2. `cd /shared-docker && git clone https://github.com/morozovdd/CrossKEY`
3. Read CrossKEY's README. Check `requirements.txt` for ROCm compatibility (PyTorch version, any CUDA-only ops).
4. If they ship pretrained weights — download. If not, see if their training script can finetune from MONAI SSL `model_swinvit.pt` (already at `/shared-docker/work/weights/`).
5. Run their inference script on ONE volume to verify the stack runs at all.
6. Wrap their model into a function `crosskey_embed(volume_path) → np.ndarray`. Call it the same way Alex's `workers/A/embed_b3.py` is structured.
7. Process all 754 val+test volumes across the 6 pools → `branch_crosskey.csv`.
8. Process the 100 held-out volumes → `branch_crosskey_holdout.csv`.
9. Gate, decide, hand to Nicole.

**Stop-loss criteria:** if after 90 minutes you cannot get a single CrossKEY inference to succeed on the MI300X, abort and fall back to **Plan B** below. The hackathon time budget is harder than getting CrossKEY working.

---

## Plan B — Stage-2 SynthMorph re-rank (if CrossKEY blocks)

Lower ceiling than CrossKEY but more reliable. Works on top of our existing RRF output.

- **Idea:** take RRF's top-10 per query; register query→candidate with SynthMorph (contrast-invariant) and rank by inverse registration residual.
- **Why it works:** same-subject pairs register cleanly (low residual) even under deformation/domain-shift, while different-subject pairs don't.
- **Repo:** SynthMorph ships in `voxelmorph` (`pip install voxelmorph`) + pretrained weights from https://github.com/voxelmorph/voxelmorph/blob/dev/scripts/torch/synthmorph_register.py.
- **Path:** for each query → top-10 candidates from `fused_rankings` → register each → rank by residual. Write `branch_stage2_rerank.csv` with refined scores. Plug into RRF at high weight, or replace.

Estimated time: 2–3h.

---

## Plan C — pure MIND-SSC + per-region matching (if both above block)

Classical training-free contrast-invariant descriptor. Open-source PyTorch impls in ~50 lines.

- Compute MIND-SSC at every voxel → 12-channel descriptor map.
- Aggregate by SynthSeg parcels (or by a fixed 8³ grid if SynthSeg isn't installed) → fixed-length per-volume vector.
- Cosine sim ranks gallery.
- Estimated time: 1.5h. Expected MRR: 0.4–0.6 on d1, decent fall-off on d2/d3.

---

## Files to know

```
CLAUDE.md                              full project context
docs/SLIDES.md                         deck notes + all real numbers
docs/RESEARCH_PLAN.md                  the original strategy (multi-branch + RRF + Stage-2)
docs/ROADMAP.md                        the 3-person hour-by-hour plan
docs/REFERENCES.md                     annotated paper list (includes MIND, CrossKEY, SynthMorph)
papers/                                PDFs of the cited methods

workers/A/                             Alex (B3 + B4) — read these as integration templates
  b3_encoder.py                        SwinUNETR backbone, multi-stage pool, SSL loader
  train_b3.py                          InfoNCE loop (BN, wd=0, temp 0.1)
  embed_b3.py                          val/test embedding extraction
  embed_holdout.py                     held-out split CSV generator (matches Nicole's seed=0 first-50)
  pkl_to_branch.py                     {id:vec} pkl → Nicole's branch CSV format
  b4_shape.py                          V1 shape fingerprint (centered mask 16³)
  runs/                                outputs: b3_run1.pt, *.pkl, *.csv

Track C/                               Nicole — engine, B2, fusion driver
  trackc.py                            engine: split, MRR, RRF, submission writer
  b2_foundation.py                     frozen B2 SwinUNETR
  run_fuse_submit.ipynb                the submission driver — read its cells to understand the
                                       fuse_branches(...) call you'll plug into
  branch_baseline.csv  branch_b2.csv   29,529-row example CSVs in the contract format

workers/C/NICOLE_TASK.md               Nicole's hand-off (B3+B4 wiring)
workers/B/WORKER_B_TASK.md             original Track B brief (MIND-SSC + Stage-2)
PIVOT_TASK.md                          this file
```

---

## Cross-cutting rules

- **Never modify** `trackc.py`, the submission writer, or another track's branch CSVs.
- **Local-MRR gate every Kaggle submission.** We have ~95 slots left today.
- **Branch CSVs are within-pool only.** Do not emit (q from d1) ranked against (t from d3) — it tanks the score silently.
- **B3 SSL weight loader trick:** if you reuse `b3_encoder.py`'s `load_ssl_weights`, it already handles MONAI 1.6's `Mlp.fc{1,2}` → `linear{1,2}` rename. Don't undo it.
- **Coordination:** push commits to `main` directly; team is on Slack. Big edits to Nicole's notebook should ping her first (or write your additions as new cells, not edits to hers).
- **Don't reuse** `~/Desktop/INTERNSHIP` venvs. Repo's working venv is what JupyterHub already has installed.

---

## First message to send back to the user
After you read CLAUDE.md, this file, and `docs/SLIDES.md`, summarize in 3 lines:
1. What you're going to try (CrossKEY vs Plan B vs Plan C).
2. Your stop-loss criterion in minutes.
3. The single concrete first command you're going to run.

Then start.
