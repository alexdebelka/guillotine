# Prompt 05 — B3 "mild+" custom-severity retrain

**Goal.** Retrain B3 (the SwinUNETR InfoNCE branch) with augmentation
severity between `mild` (current production) and `medium` (collapses). If
it lands a stronger holdout MRR than mild, swap in the new branch and
re-fuse. Expected ΔMacro: **+0.02 to +0.05**. 1-2h.

---

## Paste this entire block into Codex

```
You are working in /shared-docker/work/repo. Read:
  - workers/X/codex-handoff/CONTEXT.md       (project background)
  - workers/A/augmenter.py                   (current severity configs)
  - workers/A/train_b3.py                    (B3 trainer)
  - workers/A/embed_b3.py, embed_holdout.py  (inference)

Background: B3 at --severity mild trained to 0.87 standalone holdout MRR
(production). Attempts at medium (elastic mag 50-100 on 96^3 volumes) and
heavy (mag 80-150) both collapsed -- the InfoNCE loss at batch=4 can't
find a signal when independent augmentations of a positive pair land too
far apart in input space. Try the intermediate: "mild+" with mag (40, 80).

Goal: produce workers/A/runs/b3_mildplus.pt + branch_b3_mildplus.csv with
holdout MRR > 0.87.

Steps:

1. cd /shared-docker/work/repo && git pull

2. Patch workers/A/augmenter.py to add a "mild+" severity entry. INSERT
   between mild and medium:

     "mild+": dict(hist_p=0.7, gamma=(0.7, 1.4), bias_p=0.2, bias_coeff=(0.0, 0.18),
                   affine_p=0.7, rot=0.08, trans=4, scale=0.04,
                   elastic_p=0.4, sigma=(5, 8), mag=(40, 80)),

   And update the train_b3.py --severity choices to include "mild+":
     ap.add_argument("--severity", default="medium",
                     choices=["mild", "mild+", "medium", "heavy"])

3. Back up the mild B3 artifacts so retrain doesn't overwrite:
     cp workers/A/runs/b3_run1.pt           workers/A/runs/b3_mild.pt
     cp workers/A/runs/branch_b3.csv        workers/A/runs/branch_b3_mild.csv
     cp workers/A/runs/branch_b3_holdout.csv workers/A/runs/branch_b3_mild_holdout.csv

4. Train. ~2-3h. Watch the first 50 steps -- if loss isn't dropping below
   1.0 and head delta isn't climbing above +0.3, kill immediately and tune
   down to mag (35, 70) or revert to mild.

     python workers/A/train_b3.py \
       --pairs-csv /shared-docker/data/dataset1/train_pairs.csv \
       --severity mild+ \
       --steps 1500 \
       --out workers/A/runs/b3_mildplus.pt

   Healthy log signature (from earlier successful B3 mild run):
     step  50  loss 0.47  head delta +0.3-0.6
     step 100  loss 0.18  head delta +0.6+
     step 200  loss 0.10  head delta +0.8+
     final: loss ~0.04-0.10, head delta +0.85+

   Collapsed log signature (kill these):
     loss stuck at 1.0-1.5
     head delta oscillating around 0 (e.g. -0.07 to +0.10)
     diag and offdiag both ~0.90+

5. Embed full pools + holdout. Note that embed_holdout.py overwrites the
   default branch_b3_holdout.csv -- we backed it up in step 3, so this
   is safe.

     python workers/A/embed_b3.py \
       --ckpt workers/A/runs/b3_mildplus.pt \
       --out workers/A/runs/b3_mildplus_embeddings.pkl
     python workers/A/pkl_to_branch.py \
       --in workers/A/runs/b3_mildplus_embeddings.pkl \
       --out workers/A/runs/branch_b3_mildplus.csv
     python workers/A/embed_holdout.py \
       --branch b3 \
       --b3-ckpt workers/A/runs/b3_mildplus.pt
     mv workers/A/runs/branch_b3_holdout.csv workers/A/runs/branch_b3_mildplus_holdout.csv
     cp workers/A/runs/branch_b3_mild_holdout.csv workers/A/runs/branch_b3_holdout.csv

6. Compare standalone holdout MRR:
     python -c "
     import sys; sys.path.insert(0, 'Track C')
     from trackc import scores_to_rankings, mrr, make_local_split
     import pandas as pd
     pairs = pd.read_csv('/shared-docker/data/dataset1/train_pairs.csv')
     _, hold, gt = make_local_split(pairs)
     gt = {str(k): str(v) for k, v in gt.items()}
     mild = scores_to_rankings(pd.read_csv('workers/A/runs/branch_b3_mild_holdout.csv'))
     mp   = scores_to_rankings(pd.read_csv('workers/A/runs/branch_b3_mildplus_holdout.csv'))
     print(f'B3 mild  standalone holdout MRR = {mrr(mild, gt):.4f}')
     print(f'B3 mild+ standalone holdout MRR = {mrr(mp,   gt):.4f}')
     "

7. Decision:

   - mild+ MRR > mild MRR by ≥ +0.02: replace b3 in the production RRF.
     Build a new submission by re-running the Stage-2 V2 pipeline with
     branch_b3_mildplus.csv instead of branch_b3.csv, then apply the d3
     leak v2 post-processor.

     python workers/X/stage2_v2_d1only.py \
       --branch "Track C/branch_baseline.csv" \
       --branch "Track C/branch_b2.csv" \
       --branch workers/A/runs/branch_b3_mildplus.csv \
       --branch workers/A/runs/branch_b4.csv \
       --max-side 128 \
       --out workers/X/runs/submission_mildplus.csv

     python workers/X/apply_d3_leak_v2.py \
       --in  workers/X/runs/submission_mildplus.csv \
       --out workers/X/runs/submission_mildplus_d3v2.csv

     Upload submission_mildplus_d3v2.csv to Kaggle. Expected: +0.02-0.05
     above the current PB 0.67452.

   - mild+ MRR ≤ mild MRR: revert. Keep mild as production. Pure ablation,
     no submission burned. Try mild++ (mag 50-90) if you have time, or
     give up on this lever.

8. Commit (whether or not mild+ won):
     git add workers/A/augmenter.py workers/A/runs/b3_mildplus.pt \
             workers/A/runs/branch_b3_mildplus*.csv \
             workers/A/runs/b3_mildplus_embeddings.pkl
     git commit -m "Track A: B3 mild+ severity retrain (mag 40-80)"
     git push origin main

Acceptance criterion: either (a) b3_mildplus has higher holdout MRR than
b3_mild AND a Kaggle score better than 0.67452, or (b) clear ablation
showing mild+ doesn't help. Either way, no regression to production.
```

---

## Files this prompt creates

- Patch to `workers/A/augmenter.py` (one new severity entry)
- Patch to `workers/A/train_b3.py` (one CLI choice update)
- `workers/A/runs/b3_mildplus.pt` (~50 MB)
- `workers/A/runs/b3_mildplus_embeddings.pkl`
- `workers/A/runs/branch_b3_mildplus.csv`
- `workers/A/runs/branch_b3_mildplus_holdout.csv`
- `workers/X/runs/submission_mildplus*.csv` if the gate passes

## Expected runtime

2-3h (training dominates).

## What success looks like

`B3 mild+ standalone holdout MRR > 0.87` AND new Kaggle score > 0.67452.

## What failure looks like

Training collapses (loss stuck at ln(4)=1.39, head delta oscillating around
0). Action: tune `mag` down to (35, 70) and retry once. If still collapses,
abandon the mild+ approach and try prompt 06 instead.
