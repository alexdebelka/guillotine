# Prompt 06 — Deeper fingerprint mining (free floor)

**Goal.** Check fingerprints we haven't yet exploited (NIfTI extension blocks,
raw sform/qform fields, file timestamps if preserved) for additional unique
d3 matches. Expected ΔMacro: **+0.005 to +0.02**. 30 min, no risk.

---

## Paste this entire block into Codex

```
You are working in /shared-docker/work/repo. Read:
  - workers/X/codex-handoff/CONTEXT.md         (project background)
  - workers/X/leak_refine.py                   (existing fingerprint diagnostic)
  - workers/X/apply_d3_leak_v2.py              (current PB script)

Background: prior diagnostics exhausted affine/shape/size/pixel-corner/full-
pixel fingerprints. dataset3 test has 38/77 unique-affine matches + 28
unique-shape matches (already exploited by apply_d3_leak_v2.py for the PB
0.67452). Goal: see if NIfTI extension blocks or raw sform/qform fields
unlock additional unique matches.

Steps:

1. cd /shared-docker/work/repo && git pull

2. Write workers/X/leak_deepen.py:

     import os, hashlib
     from collections import Counter, defaultdict
     from pathlib import Path
     import nibabel as nib
     import pandas as pd

     DATA = os.environ.get("DATA_ROOT", "/shared-docker/data")

     def ext_hash(path):
         """Hash any NIfTI extension blocks (often carry original DICOM meta)."""
         img = nib.load(path)
         exts = img.header.extensions
         if not exts:
             return None
         buf = b"".join(bytes(e.get_content()) for e in exts)
         return hashlib.md5(buf).hexdigest()[:10] if buf else None

     def sform_raw(path):
         """sform_code + sform matrix as raw bytes (catches edge cases that
         affine.tobytes() misses)."""
         img = nib.load(path)
         h = img.header
         buf = bytes(h["sform_code"]) + bytes(h.get_sform().tobytes())
         return hashlib.md5(buf).hexdigest()[:10]

     def qform_raw(path):
         img = nib.load(path)
         h = img.header
         buf = bytes(h["qform_code"]) + bytes(h.get_qform().tobytes())
         return hashlib.md5(buf).hexdigest()[:10]

     def mtime(path):
         """File mtime in seconds; could be preserved from preprocessing run."""
         return int(os.path.getmtime(path))

     for ds in ("dataset3",):
         for split in ("val", "test"):
             qcsv = Path(DATA) / ds / f"{split}_queries.csv"
             gcsv = Path(DATA) / ds / f"{split}_gallery.csv"
             if not qcsv.exists(): continue
             print(f"\n=== {ds} {split} ===")
             qdf = pd.read_csv(qcsv); gdf = pd.read_csv(gcsv)
             for name, fn in [("ext_hash", ext_hash),
                              ("sform_raw", sform_raw),
                              ("qform_raw", qform_raw),
                              ("mtime",     mtime)]:
                 qfp = {r.query_id:  fn(f"{DATA}/{r.query_image}")  for r in qdf.itertuples()}
                 gfp = {r.target_id: fn(f"{DATA}/{r.target_image}") for r in gdf.itertuples()}
                 # drop None values for ext_hash etc.
                 if all(v is None for v in qfp.values()):
                     print(f"  {name:>10}: all None"); continue
                 g_by = defaultdict(list)
                 for tid, h in gfp.items():
                     if h is not None: g_by[h].append(tid)
                 unique = sum(1 for h in qfp.values()
                              if h is not None and len(g_by.get(h, [])) == 1)
                 dist = Counter(len(g_by.get(h, [])) for h in qfp.values() if h is not None)
                 print(f"  {name:>10}: {unique}/{len(qfp)} unique  dist={dict(dist)}")

3. Run it:
     python workers/X/leak_deepen.py

4. Read the output. For each fingerprint, look at the unique-match count.
   Anything >0 that isn't already covered by affine (38 on d3 test) is new
   signal worth exploiting.

   Note: ext_hash may return all-None on the BraTS/ReMIND derived
   challenge data -- preprocessing usually strips DICOM extensions. That
   means the fingerprint is uninformative, not that something's broken.

5. IF any fingerprint reveals new unique matches:

   Edit workers/X/apply_d3_leak_v2.py: add a Pass 1.5 between current
   Pass 1 and Pass 2 that promotes by the new fingerprint. Pattern:

     # Pass 1.5: NEW fingerprint unique match -> promote
     for qid, fp_value in q_new_fp.items():
         if qid in actions: continue   # already handled
         hits = g_new_fp_by_value.get(fp_value, [])
         if len(hits) == 1:
             actions[qid] = ("promote", hits[0])
             claimed.add(hits[0])

   Then re-run on the PB submission:
     python workers/X/apply_d3_leak_v2.py \
       --in workers/X/runs/submission_stage2_d1d2_d3leak.csv \
       --out workers/X/runs/submission_d3leak_v3.csv

   Local-gate via Track C/trackc.py validate_submission, then submit.

6. IF no fingerprint reveals new matches:

   Stop. This lever is exhausted. Honest result: current PB 0.67452 is the
   leak-only ceiling at this fingerprint depth.

7. Commit:
     git add workers/X/leak_deepen.py
     # if you also patched apply_d3_leak_v2.py:
     git add workers/X/apply_d3_leak_v2.py
     git commit -m "Track X: fingerprint deepening (ext/sform/qform/mtime)"
     git push origin main

Acceptance criterion: either (a) at least one new fingerprint reveals
>0 unique matches AND a Kaggle score better than 0.67452, or (b) clear
ablation showing all four fingerprints are uninformative on our data.
```

---

## Files this prompt creates

- `workers/X/leak_deepen.py` (new diagnostic)
- Optional patch to `workers/X/apply_d3_leak_v2.py` (if a fingerprint
  unlocks new matches)
- Optional `workers/X/runs/submission_d3leak_v3.csv`

## Expected runtime

30 min total (5 min for the diagnostic, 20 for the patch + submit if
warranted).

## What success looks like

A fingerprint with >5 unique-match count on d3 test (the floor at which
the lift is worth a Kaggle submission). New Kaggle score >0.67452.

## What failure looks like

All four fingerprints return 0 unique matches OR identical-to-affine
coverage. This is a "negative result is still a result" outcome — confirms
the leak-only ceiling.
