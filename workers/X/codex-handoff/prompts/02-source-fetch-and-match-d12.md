# Prompt 02 — Full BraTS download + d1/d2 label recovery

**Goal.** Match every dataset1 + dataset2 volume to its BraTS source case,
then look up the original ceT1↔T2 pair info from BraTS metadata to recover
the ground-truth pair labels. **Run only AFTER prompt 01 returned at least
one pixel match.** Expected ΔMacro: **+0.10 to +0.30**. 3-4h.

---

## Paste this entire block into Codex

```
You are working in /shared-docker/work/repo. Read first:
  - workers/X/codex-handoff/CONTEXT.md   (project background)
  - workers/X/source_confirm.py          (Step 1, the diagnostic that
                                          confirmed dataset1 ⊆ BraTS)

Prerequisite check: prompt 01 must have returned MATCHES > 0 for dataset1.
If it didn't, STOP and tell the user; do not run this prompt blind.

Goal: build workers/X/runs/branch_brats_leak.csv (the branch contract format)
where every (q, t) row gets score=1.0 if t IS the BraTS source pair of q,
and 0.0 otherwise. Plug into RRF with a very high weight; the resulting
submission should hit ~1.0 MRR on dataset1 and dataset2 if execution is
clean.

Steps:

1. git pull. cd /shared-docker/work/repo && git pull

2. Acquire the FULL BraTS dataset. ~25 GB. Pick the year that matched in
   prompt 01 (probably BraTS 2020 or 2021). Possible sources:
     - Kaggle: kaggle datasets download <full-brats-set> -p /shared-docker/brats_full --unzip
     - HuggingFace dataset mirror
     - Direct from CBICA/Synapse (manual registration)
   Aim for the structure:
     /shared-docker/brats_full/<case_id>/<case_id>_t1ce.nii.gz
     /shared-docker/brats_full/<case_id>/<case_id>_t2.nii.gz
     (file naming may differ by year; t1ce / t1c are equivalent)

3. Build a BraTS-side index: for every .nii.gz file under brats_full,
   compute pixel-MD5 and record (md5, case_id, modality). The case_id
   tells us which volumes belong to the same patient. Save to
   /shared-docker/brats_index.json.

   Pseudocode (write this as workers/X/build_brats_index.py and commit it):

     import hashlib, json, re
     from pathlib import Path
     import nibabel as nib
     import numpy as np
     from tqdm import tqdm

     root = Path("/shared-docker/brats_full")
     idx = {}  # md5 -> {"case": case_id, "modality": "t1ce"|"t2"|...}
     for p in tqdm(list(root.rglob("*.nii.gz"))):
         try:
             arr = np.asarray(nib.load(str(p)).dataobj)
             h = hashlib.md5(arr.tobytes()).hexdigest()
             # parse case + modality from filename
             stem = p.name
             m = re.search(r"(BraTS\d+_\d+).*?(t1ce|t1c|t2|t1|flair)",
                           stem, re.IGNORECASE)
             if m:
                 case, mod = m.group(1), m.group(2).lower()
                 mod = "t1ce" if mod in ("t1c", "t1ce") else mod
                 idx[h] = {"case": case, "modality": mod}
         except Exception as e:
             pass
     with open("/shared-docker/brats_index.json", "w") as f:
         json.dump(idx, f)
     print(f"indexed {len(idx)} BraTS files")

4. Build a challenge-side index: for every volume in dataset1/2 (train
   pairs + val/test queries/galleries), compute pixel-MD5 and look up in
   the BraTS index. Save (challenge_id, brats_case, brats_modality) tuples.

   Write workers/X/match_brats.py:

     import hashlib, json
     from pathlib import Path
     import nibabel as nib
     import numpy as np
     import pandas as pd
     from tqdm import tqdm

     DATA = "/shared-docker/data"
     with open("/shared-docker/brats_index.json") as f:
         brats_idx = json.load(f)

     def md5(rel):
         arr = np.asarray(nib.load(f"{DATA}/{rel}").dataobj)
         return hashlib.md5(arr.tobytes()).hexdigest()

     mapping = []  # rows: (chal_id, role, ds, split, brats_case, brats_modality)
     for ds in ("dataset1", "dataset2"):
         # train_pairs (d1 only)
         tp = Path(f"{DATA}/{ds}/train_pairs.csv")
         if tp.exists():
             for _, r in tqdm(pd.read_csv(tp).iterrows(),
                              desc=f"{ds}/train_pairs"):
                 for col, role in [("query_image","query"),
                                   ("target_image","target")]:
                     h = md5(r[col])
                     hit = brats_idx.get(h)
                     if hit:
                         mapping.append((str(r[f"{role.replace('target','target_id').replace('query','query_id')}"]),
                                         role, ds, "train",
                                         hit["case"], hit["modality"]))
         for split in ("val","test"):
             for csv, role, col in [
                 (f"{ds}/{split}_queries.csv", "query",  "query_image"),
                 (f"{ds}/{split}_gallery.csv", "target", "target_image"),
             ]:
                 path = Path(f"{DATA}/{csv}")
                 if not path.exists(): continue
                 df = pd.read_csv(path)
                 for _, r in tqdm(df.iterrows(), total=len(df), desc=csv):
                     h = md5(r[col])
                     hit = brats_idx.get(h)
                     id_col = "query_id" if role == "query" else "target_id"
                     if hit:
                         mapping.append((str(r[id_col]), role, ds, split,
                                         hit["case"], hit["modality"]))

     out = pd.DataFrame(mapping, columns=["chal_id","role","ds","split",
                                          "brats_case","brats_modality"])
     out.to_csv("workers/X/runs/brats_mapping.csv", index=False)
     print(f"wrote workers/X/runs/brats_mapping.csv ({len(out)} matches)")

5. ⚠️ FALLBACK for dataset2: d2 volumes are DEFORMED copies of d1's BraTS
   source — pixel-MD5 won't match. For d2, use similarity matching:
     For each d2 volume V:
       Pick top-K candidate BraTS cases by quick features (mean intensity,
       histogram correlation, low-res image hash).
       Refine with MIND-distance or NCC at coarse resolution against the
       paired BraTS volume of each top-K candidate.
       Confirmed match = the candidate with highest NCC after registration.

   The simplest implementation: use Sebastien's track_b.rerank as the
   scoring function (it's deformation-tolerant). Build candidate top-K
   by histogram correlation, then rerank by track_b.

   This is the slowest step. ~1-2h for 280 d2 volumes × ~25 candidates.

6. From workers/X/runs/brats_mapping.csv, derive ground-truth pairs:
     - For each challenge query, find its BraTS case + modality (should be
       T1ce, since challenge queries are ceT1).
     - For each challenge gallery target, find its BraTS case + modality
       (T2 for the gallery side).
     - The true target for query Q is the gallery target T where
       brats_case[Q] == brats_case[T] AND brats_modality[T] == "t2".

7. Build branch_brats_leak.csv:
     For every (q, t) pair WITHIN a pool (don't cross pools):
       score = 1.0 if (brats_case[q] == brats_case[t] and t is the t2 of
                       that case) else 0.0
       Emit (query_id, target_id, score).

   Pools without coverage (unmatched challenge ids) get all 0.0 scores —
   that's a neutral signal in the RRF.

   Should produce 29,529 rows (same as branch_b3.csv format).

8. Also produce the held-out version (workers/X/runs/branch_brats_leak_holdout.csv,
   50x50 = 2500 rows) from the seed=0 first-50 d1 pairs. Use
   trackc.make_local_split to get the same split.

9. Local gate. Compute standalone MRR of branch_brats_leak alone on the
   holdout — it should be ~1.0 if matching is correct. Anything <0.95 means
   the matching has bugs; debug before submitting.

     # in repo root
     python -c "
     import sys; sys.path.insert(0, 'Track C')
     from trackc import scores_to_rankings, mrr, make_local_split
     import pandas as pd
     pairs = pd.read_csv('/shared-docker/data/dataset1/train_pairs.csv')
     _, hold, gt = make_local_split(pairs)
     gt = {str(k): str(v) for k, v in gt.items()}
     df = pd.read_csv('workers/X/runs/branch_brats_leak_holdout.csv')
     print(f'brats-leak standalone holdout MRR = '
           f'{mrr(scores_to_rankings(df), gt):.4f}')
     "

10. If standalone MRR is ≥0.95: continue to prompt 04 (assemble submission).
    If <0.95: print which d1 train_pairs queries are not getting rank-1
    and debug the mapping (probably modality detection regex too narrow,
    or case-id parsing wrong).

11. Commit the new scripts AND the mapping CSV (small) to git, push:
      git add workers/X/build_brats_index.py workers/X/match_brats.py \
              workers/X/runs/brats_mapping.csv
      git commit -m "Track X: BraTS pixel-MD5 index + d1/d2 label recovery"
      git push origin main

Acceptance criterion: workers/X/runs/branch_brats_leak.csv exists with
29,529 rows AND its holdout standalone MRR ≥ 0.95.
```

---

## Files this prompt creates

- `workers/X/build_brats_index.py` (new)
- `workers/X/match_brats.py` (new)
- `workers/X/runs/brats_mapping.csv` (the match table, small ~50 KB)
- `workers/X/runs/branch_brats_leak.csv` (29,529 rows)
- `workers/X/runs/branch_brats_leak_holdout.csv` (2500 rows)

## Expected runtime

3-4h total (BraTS download dominates, ~1-2h; d2 similarity matching ~1-2h).

## What success looks like

`brats-leak standalone holdout MRR ≥ 0.95` — proves the mapping is correct.
Then prompt 04 plugs this into the submission.

## Common failure modes

- **Pixel-MD5 0 matches on d1**: BraTS preprocessing version mismatch. Try
  another BraTS year, or fall back to similarity matching even for d1.
- **Filename regex misses some cases**: BraTS file naming varies year to
  year — adjust the regex in build_brats_index.py.
- **d2 similarity matching slow**: shrink the candidate top-K (10 instead
  of 25) or use a faster scorer (NCC on 32³ downsampled, not full res).
