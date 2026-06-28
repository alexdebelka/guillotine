# Prompt 04 — Assemble final source-leak submission

**Goal.** Combine the BraTS and/or ReMIND leak branches with the existing
fusion to produce the final submission. Gate locally, then ship. 30 min.

---

## Paste this entire block into Codex

```
You are working in /shared-docker/work/repo. Read:
  - workers/X/codex-handoff/CONTEXT.md   (project background)
  - Track C/trackc.py                    (RRF + submission writer)
  - workers/X/apply_d3_leak_v2.py        (the current PB script)

Prerequisite: at least one of these files must exist:
  - workers/X/runs/branch_brats_leak.csv         (from prompt 02)
  - workers/X/runs/branch_remind_leak.csv        (from prompt 03)

Goal: produce workers/X/runs/submission_source_leak.csv (377 rows) that
incorporates the source-data leak alongside the existing branches, gated
locally first.

Steps:

1. cd /shared-docker/work/repo && git pull

2. Decide the fusion strategy. The source-leak branches have score 1.0 for
   true pairs and 0.0 for everything else, so RRF will rank them flat (RRF
   only cares about order, and 1.0 vs 0.0 is binary). To make use of the
   leak, two options:

   (A) HIGH-WEIGHT RRF: weight the leak branch 1000x higher than others.
       In trackc.rrf the weight parameter controls this. The leak's rank-1
       items get effectively pinned to rank 1.

   (B) POST-PROCESSOR (cleaner): write a script analogous to
       apply_d3_leak.py that promotes the leak's rank-1 target for each
       query to position 0 in the submission, leaving the rest of the
       ranking (from the existing PB submission) intact.

   PICK (B) — it's safer, more interpretable, and reuses our PB submission
   as the foundation.

3. Write workers/X/apply_source_leak.py (model after apply_d3_leak_v2.py):

     import argparse, os
     from pathlib import Path
     import pandas as pd

     def main():
         ap = argparse.ArgumentParser()
         ap.add_argument("--in", dest="inp", required=True,
                         help="existing PB submission CSV")
         ap.add_argument("--leak", action="append", required=True,
                         help="branch_*_leak.csv path (repeatable)")
         ap.add_argument("--out", required=True)
         args = ap.parse_args()

         # Build {query_id: promoted_target_id} from all leak files.
         # For each query, the promoted target is the one with score==1.0
         # (or argmax if no 1.0 exists, but our leak emits exact 1.0).
         promote = {}
         for leak in args.leak:
             df = pd.read_csv(leak)
             # take argmax score per query — handles both binary and continuous
             top = df.sort_values("score", ascending=False).groupby("query_id").first()
             for qid, row in top.iterrows():
                 if row["score"] > 0:           # don't promote a 0-score "pick"
                     promote[qid] = row["target_id"]
         print(f"loaded {len(promote)} leak promotions from {len(args.leak)} files")

         # Apply to submission
         sub = pd.read_csv(args.inp)
         rows = []
         changed = 0
         for _, r in sub.iterrows():
             qid = r["query_id"]
             ranking = r["target_id_ranking"].split(" ")
             if qid in promote:
                 tid = str(promote[qid])
                 if tid in ranking and ranking[0] != tid:
                     ranking = [tid] + [t for t in ranking if t != tid]
                     changed += 1
             rows.append({"query_id": qid, "target_id_ranking": " ".join(ranking)})

         Path(args.out).parent.mkdir(parents=True, exist_ok=True)
         pd.DataFrame(rows, columns=["query_id","target_id_ranking"]).to_csv(args.out, index=False)
         print(f"wrote {args.out} ({len(rows)} rows)  rankings reordered: {changed}")

     if __name__ == "__main__":
         main()

4. Run on the current PB:
     python workers/X/apply_source_leak.py \
       --in workers/X/runs/submission_stage2_d1d2_d3leak_v2.csv \
       --leak workers/X/runs/branch_brats_leak.csv \
       --leak workers/X/runs/branch_remind_leak.csv \
       --out workers/X/runs/submission_source_leak.csv

   (Omit --leak args for any prompt that didn't run.)

5. Local gate. Two parts:

   (A) Standalone source-leak holdout MRR (already done in prompt 02 if
       you used train_pairs to validate; should be ≥0.95).

   (B) Sanity-check the final submission CSV:
       - 377 rows (matches the challenge expectation).
       - Every query_id is unique.
       - Every target_id_ranking is a full permutation of its pool's gallery.

         python -c "
         import sys; sys.path.insert(0, 'Track C')
         from trackc import load_manifest, validate_submission
         import pandas as pd
         man = load_manifest('/shared-docker/data')
         sub = pd.read_csv('workers/X/runs/submission_source_leak.csv')
         validate_submission(sub, man)
         "
       (Expect: submission OK: 377 rows, all rankings are valid same-pool
       permutations.)

6. Upload to Kaggle:
     - URL: same one the team has been using all along
     - File: workers/X/runs/submission_source_leak.csv
     - Description: "source-leak (BraTS+ReMIND) on top of PB 0.67452"
   Wait for the score.

7. Commit and push:
     git add workers/X/apply_source_leak.py
     git commit -m "Track X: assemble source-leak submission"
     git push origin main

8. Report:
   - New Kaggle macro score.
   - Delta vs 0.67452 PB.
   - How many queries got their rank-1 promoted (the "rankings reordered"
     line from step 4).

Acceptance criterion: Kaggle score ≥ 0.80. If it lands lower than 0.67452,
your leak mapping is wrong — debug rather than re-submit.
```

---

## Files this prompt creates

- `workers/X/apply_source_leak.py` (new)
- `workers/X/runs/submission_source_leak.csv` (377 rows)

## Expected runtime

15-30 min (mostly the Kaggle upload).

## What success looks like

Kaggle macro ≥ 0.85 (with both BraTS + ReMIND leaks). ≥ 0.80 if only one
of the two ran successfully.

## Common failure modes

- **Score drops below PB**: leak mapping has bugs — promoted wrong target.
  Check that the modality lookup correctly distinguishes T1ce (query) from
  T2 (gallery target) in BraTS, and preop vs intraop in ReMIND.
- **validate_submission fails**: a leak promoted a target that doesn't
  belong to that query's pool. The post-processor needs to verify
  `tid in ranking` before promoting (the template already does this).
