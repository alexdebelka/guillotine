# Track X — Plan B (MIND-distance Stage-2 re-rank) + Plan C (MIND-SSC branch)

CrossKEY pivot was aborted: the PI's repo ships no pretrained checkpoint and
training requires per-brain synthetic ultrasound volumes we don't have. Pivot
of the pivot: parallel Plan B + Plan C, both built on the same MIND descriptor
(Heinrich et al. 2012 — provably contrast-invariant under affine intensity
transforms).

## Relationship to Sebastien's `track_b` (Alex's wrappers in workers/A/)

`workers/A/run_b1_track_b.py` and `workers/A/run_stage2_track_b.py` import a
`track_b/` package that is NOT in git (untracked on the MI300X — Sebastien
created it but never pushed). If `track_b/` exists at the repo root on the box,
prefer those — Stage-2 V2 uses FFT phase correlation + MIND, strictly better
than my pure MIND-distance fallback.

**If `track_b/` is missing** (fresh container, `python -c "import track_b"`
raises), use this directory: `mindssc.py` and `stage2_mind.py` are pure-PyTorch
and have zero external module dependencies beyond `torch + scipy + nibabel`.

Quick check on the MI300X:
```bash
python -c "import track_b" 2>&1 && echo "USE workers/A/run_*_track_b.py" \
                              || echo "USE workers/X/{mindssc,stage2_mind}.py"
```

## Run order on MI300X

```bash
cd /shared-docker/work/repo
git pull

# 0. self-test (no data needed, ~2 s)
python workers/X/mindssc.py --smoke

# 1. Plan C — MIND-SSC branch (holdout first, then full)
python workers/X/mindssc.py --mode holdout              # 2-3 min CPU, < 30s GPU
python workers/X/mindssc.py --mode full                 # 7 min CPU, ~1 min GPU
#   -> workers/X/runs/branch_mindssc.csv          (29,529 rows)
#   -> workers/X/runs/branch_mindssc_holdout.csv  ( 2,500 rows)

# 2. Local-MRR gate Plan C (Nicole's gate; from any python with trackc):
python -c "
import sys; sys.path.insert(0, 'Track C')
from trackc import scores_to_rankings, rrf, mrr, make_local_split
import pandas as pd

pairs = pd.read_csv('data/dataset1/train_pairs.csv')
_, hold, gt = make_local_split(pairs)
gt = {str(k): str(v) for k, v in gt.items()}

branches = {
    'baseline': 'workers/A/runs/branch_baseline_holdout.csv',  # or whatever Alex has
    'b2':       'Track C/branch_b2_holdout.csv',               # if exists
    'b3':       'workers/A/runs/branch_b3_holdout.csv',
    'b4':       'workers/A/runs/branch_b4_holdout.csv',
    'mindssc':  'workers/X/runs/branch_mindssc_holdout.csv',
}
ranks = []
for name, p in branches.items():
    try:
        ranks.append(scores_to_rankings(pd.read_csv(p)))
        print(f'  loaded {name}')
    except FileNotFoundError:
        print(f'  skip   {name} (no holdout csv)')
print(f'fused MRR with MIND-SSC = {mrr(rrf(ranks), gt):.4f}')
"
# If this clears the current 0.6918 fused-holdout baseline by >= 0.02 -> add to
# Nicole's run_fuse_submit.ipynb and submit.

# 3. Plan B — MIND-distance stage-2 re-rank (depends on existing branch CSVs)
#    Holdout gate first:
python workers/X/stage2_mind.py --gate \
  --branch workers/A/runs/branch_b3_holdout.csv \
  --branch workers/A/runs/branch_b4_holdout.csv \
  --branch workers/X/runs/branch_mindssc_holdout.csv \
  --topk 10
#    -> prints "fused MRR ... refined MRR ... delta ..." and PASS/FAIL.
#    If GATE PASSED, run the submission build:
python workers/X/stage2_mind.py \
  --branch workers/A/runs/branch_b3.csv \
  --branch workers/A/runs/branch_b4.csv \
  --branch workers/X/runs/branch_mindssc.csv \
  --topk 10 \
  --out workers/X/runs/submission_stage2_mind.csv
#    -> ONE complete submission CSV (377 rows). This REPLACES Nicole's submission
#       for this attempt; do not RRF it on top — it already consumes the branches.
```

## What each branch contributes

| Branch | Per-volume vector? | Stage 2? | Contrast-invariant? | New signal? |
|---|---|---|---|---|
| `mindssc.py` | yes (3072-d, cosine) | no | yes (affine intensity) | yes — adds to RRF dict |
| `stage2_mind.py` | no | yes (top-k re-rank) | yes (same MIND math) | refines existing RRF top-k |

Plan C is the safer add (just a new branch in the dict, can't hurt the fusion
much if it underperforms — RRF damps weak signals). Plan B is higher-leverage
when same-subject pairs reliably rank in top-10 of the existing fused output
(true on dataset1 today; the bet is it's also true enough on dataset2/3).

## What's NOT here

- **CrossKEY**: dead, see `PIVOT_TASK.md` and the message that pivoted away.
  Repo ships no pretrained ckpt; needs synthetic US per brain we don't have.
- **SynthMorph**: skipped. MIND-distance gives us the same contrast-invariant
  re-rank without the voxelmorph/TF/ROCm install gauntlet. If we end up wanting
  a true registration residual after this lands, the right next move is to wrap
  `stage2_mind.py:mind_score` with a SynthMorph warp step — identical CLI.

## Contract notes

- All CSVs match `workers/A/runs/branch_b3.csv` exactly: columns
  `query_id, target_id, score` (string ids, higher score = more similar), one
  row per (q, t) **within each (dataset, split) pool**. Cross-pool combinations
  are never emitted.
- The submission CSV from `stage2_mind.py --out ...` matches the official
  format used by `trackc.write_submission` (`query_id, target_id_ranking`,
  space-separated full ranking).
- Held-out split: `np.random.default_rng(0).permutation(len(train_pairs))[:50]`
  — identical to `trackc.make_local_split(pairs, n_holdout=50, seed=0)`.

## Open risks (still untested)

1. MIND descriptor at 96^3 may pool away too much for dataset2/3
   (geometry+structural-shift). Mitigation: bump RESIZE to 128^3 in
   `mindssc.py` — 4x memory, still trivial on MI300X.
2. MIND-distance assumes both volumes have the same shape AFTER our 96^3
   resize. The resize is unconditional in `load_resize_zscore`, so this holds,
   but if a NIfTI fails to load the run aborts loudly — not silently.
3. Local-MRR gate is dataset1-only. dataset2/3 generalization is the bet.
   If gate passes by >0.02 on dataset1 alone, that's necessary but not
   sufficient — burn one submission to verify on Kaggle, no more.
