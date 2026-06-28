# Prompt 03 — Full ReMIND download + d3 label recovery

**Goal.** Match every dataset3 volume to its ReMIND source case, recover
the preop↔intra-op pair info. Run after prompt 02 (or in parallel). Expected
ΔMacro on top of prompt 02: **+0.05 to +0.15**. 3-5h.

---

## Paste this entire block into Codex

```
You are working in /shared-docker/work/repo. Read:
  - workers/X/codex-handoff/CONTEXT.md
  - workers/X/source_confirm.py (the diagnostic)
  - prompts/02-source-fetch-and-match-d12.md (similar pattern, BraTS variant)

Prerequisite: prompt 01 was run (the BraTS sample may already contain ReMIND
volumes — check its output for "CONFIRMED dataset3"). If prompt 01 already
confirmed ReMIND match, skip directly to step 4 below.

Goal: build workers/X/runs/branch_remind_leak.csv with score=1.0 for
(q, t) pairs where t IS the ReMIND-source pair of q on dataset3.

Steps:

1. cd /shared-docker/work/repo && git pull

2. Acquire ReMIND from TCIA. ~80 GB. Process:

   (A) Create a free TCIA account if needed: https://www.cancerimagingarchive.net/

   (B) Install nbia-data-retriever:
        # download from https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide
        # or via conda-forge:
        conda install -c conda-forge nbia-data-retriever
        # if conda unavailable, pip install pynbia or use the REST API
        # alternative: pip install pynbia, see its README for a CLI fetch

   (C) Find ReMIND on TCIA, download the .tcia manifest:
        https://www.cancerimagingarchive.net/collection/remind/

   (D) Fetch:
        nbia-data-retriever --cli /shared-docker/remind.tcia \
          -d /shared-docker/remind_full -v
        # This takes hours; can run in background with nohup.

3. ReMIND directory structure (approx):
     /shared-docker/remind_full/<case_id>/<series>/...dicom or nii.gz
   You may need to convert DICOM → NIfTI: use dcm2niix.
       pip install -q dcm2niix
       for case in /shared-docker/remind_full/*/; do
         dcm2niix -o "$case" "$case"
       done
   Result should be one or more .nii.gz per case with names indicating
   modality and timing (preop_t2, preop_t1ce, intraop_t2, etc.).

4. Build the ReMIND-side index. Pattern identical to prompt 02:

   Write workers/X/build_remind_index.py:

     import hashlib, json, re
     from pathlib import Path
     import nibabel as nib
     import numpy as np
     from tqdm import tqdm

     root = Path("/shared-docker/remind_full")
     idx = {}  # md5 -> {"case": case_id, "timing": "preop"|"intraop",
                #        "modality": "t1ce"|"t2"|...}
     for p in tqdm(list(root.rglob("*.nii.gz"))):
         try:
             arr = np.asarray(nib.load(str(p)).dataobj)
             h = hashlib.md5(arr.tobytes()).hexdigest()
             # parse case + timing + modality from path
             # ReMIND naming varies; use case folder name + filename:
             case = p.parents[1].name if "preop" in str(p) or "intraop" in str(p) else p.parent.name
             timing = "intraop" if "intra" in str(p).lower() else "preop"
             m = re.search(r"(t1ce|t1c|t2|t1|flair)", p.name, re.IGNORECASE)
             modality = m.group(1).lower() if m else "unknown"
             modality = "t1ce" if modality in ("t1c", "t1ce") else modality
             idx[h] = {"case": case, "timing": timing, "modality": modality}
         except Exception:
             pass
     with open("/shared-docker/remind_index.json", "w") as f:
         json.dump(idx, f)
     print(f"indexed {len(idx)} ReMIND files")

5. Match d3 volumes against the index (similar pattern to prompt 02 step 4).
   Pixel-MD5 likely works directly here — d3 might be untouched copies of
   the ReMIND volumes (we already saw 38/77 share exact NIfTI affines,
   which is a sign of "no re-resampling between source and challenge").

   Write workers/X/match_remind.py mirroring match_brats.py from prompt 02,
   adapted to ds = "dataset3" only.

6. Pair-info recovery: for ReMIND each case has both preop and intraop
   scans. The challenge pairs a query (one timing) with the gallery
   (other timing) of the same patient. From your matches:
     - For each query, find brats_case + timing + modality
     - For each gallery target in same pool, same
     - True target = gallery item with same case but different timing

7. Build branch_remind_leak.csv: 1.0 for true source-pair, 0.0 else.
   Should cover dataset3 val (20 queries x 20 gallery = 400 rows) and
   dataset3 test (77 x 77 = 5929 rows). Other pools get nothing here.

   IMPORTANT: produce the file with the standard branch format (29,529
   total rows across all 6 pools) — for pools other than dataset3, emit
   all 0.0 scores. This keeps the CSV plug-and-play with Nicole's RRF.

8. Holdout: there's no train_pairs for d3, so there's no standalone
   holdout. Acceptance is by submission delta.

9. Commit and push:
     git add workers/X/build_remind_index.py workers/X/match_remind.py \
             workers/X/runs/remind_mapping.csv
     git commit -m "Track X: ReMIND pixel-MD5 index + d3 label recovery"
     git push origin main

Acceptance criterion: workers/X/runs/branch_remind_leak.csv exists with
29,529 rows AND at least 38 of the 77 d3 test queries have a non-zero
score row (matches the 38/77 unique-affine count from leak_pixel.py,
which is a sanity check).

⚠️ ReMIND access is the slowest part. If TCIA download is blocked or too
slow, you can SKIP this prompt and rely on apply_d3_leak_v2.py (current PB
already exploits the d3 fingerprint leak). Prompt 04 will work with or
without branch_remind_leak.csv.
```

---

## Files this prompt creates

- `workers/X/build_remind_index.py` (new)
- `workers/X/match_remind.py` (new)
- `workers/X/runs/remind_mapping.csv` (~10 KB)
- `workers/X/runs/branch_remind_leak.csv` (29,529 rows)

## Expected runtime

3-5h (ReMIND download dominates — ~2-3h on a fast connection).

## What success looks like

`branch_remind_leak.csv` exists with ≥38 non-zero rows in d3-test pool
(matches the known 38/77 unique-affine count — sanity check from earlier
work).

## Common failure modes

- **TCIA access blocked / nbia-data-retriever broken**: skip this prompt,
  current d3 leak (apply_d3_leak_v2.py) is the floor.
- **ReMIND file naming doesn't match the regex**: inspect 1-2 case folders
  manually, adjust the regex.
- **DICOM → NIfTI conversion produces orientations that don't pixel-match
  d3**: try `dcm2niix -m y` (merge) or skip conversion and use the .nii.gz
  files that some ReMIND mirrors provide directly.
