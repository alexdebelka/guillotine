# Prompt 01 — Confirm dataset1 IS BraTS (Step 1, diagnostic only)

**Goal.** Prove or disprove that dataset1's volumes were sourced from BraTS
by checking pixel-MD5 collisions against a small downloaded BraTS sample.
This is the gate for the entire source-re-ID path (prompts 02-04). 30 min,
diagnostic only — no submission produced.

---

## Paste this entire block into Codex

```
You are working in /shared-docker/work/repo (a git repo synced with
https://github.com/alexdebelka/guillotine.git, branch main). Run all
commands inside the container's default Python environment. DO NOT create
a new venv.

Read first:
  - workers/X/codex-handoff/CONTEXT.md  (project background, file map)
  - workers/X/source_confirm.py         (the diagnostic script, already in repo)

Goal: confirm whether dataset1's volumes are pixel-identical to BraTS source
volumes. This is a yes/no diagnostic; no submission CSV is produced.

Steps:

1. git pull to make sure the repo is current:
     cd /shared-docker/work/repo && git pull

2. Acquire ~10-20 BraTS cases (~600 MB total). Try these in order, stop at
   the first that works:

   (A) Kaggle (fastest if ~/.kaggle/kaggle.json exists with valid creds):
       pip install -q kaggle
       mkdir -p /shared-docker/brats_sample
       kaggle datasets download -d awsaf49/brats20-dataset-training-validation \
         -p /shared-docker/brats_sample --unzip
       # If quota error, try a different Kaggle BraTS dataset; many exist.

   (B) Hugging Face (no auth needed for public datasets):
       pip install -q huggingface_hub
       # Search HF for "BraTS" datasets and pick one with raw .nii.gz files.
       # Example pattern (verify the exact repo id is current):
       huggingface-cli download --repo-type dataset \
         <some-public-brats-mirror> \
         --local-dir /shared-docker/brats_sample

   (C) Direct from MICCAI / Synapse (requires manual account creation;
       skip if it requires more than 10 min of friction).

   Verify download succeeded:
       find /shared-docker/brats_sample -name '*.nii.gz' | head -5
       find /shared-docker/brats_sample -name '*.nii.gz' | wc -l
   You want at least 10-20 .nii.gz files.

3. Run the diagnostic:
     python workers/X/source_confirm.py --candidate-dir /shared-docker/brats_sample

4. Interpret the output:

   Look for the line "MATCHES: N / M hashed" for each dataset.

   - MATCHES > 0 on dataset1: CONFIRMED. d1 IS pixel-identical to BraTS.
     Proceed to prompt 02 (full BraTS download + label recovery).

   - MATCHES = 0 on dataset1:
     * Either the wrong source (try BraTS 2019, 2021, 2023 separately)
     * OR the data was re-resampled before being released by the
       challenge (then similarity matching is needed instead — see
       prompt 02's fallback section).
     * Try one more BraTS year/source. If still 0, fall back to prompts
       05 + 06 (modeling path) instead.

   - MATCHES > 0 on dataset3: bonus — the sample may also contain ReMIND
     volumes, in which case prompt 03 starts halfway done.

5. Report back to the user with:
   - The exact MATCHES count per dataset.
   - The first 3 matched paths (already printed by the script).
   - Your recommendation: proceed to prompt 02, or fall back to 05+06.

Do NOT submit anything to Kaggle in this prompt. This is diagnostic only.

Acceptance criterion: a yes/no answer to "is dataset1 pixel-identical to
BraTS?" with the matching count and file paths as evidence.
```

---

## Files this prompt touches

- READ: `workers/X/codex-handoff/CONTEXT.md`, `workers/X/source_confirm.py`
- DOWNLOADS: `/shared-docker/brats_sample/` (~600 MB)
- WRITES: nothing into the repo

## Expected runtime

15-30 min total (download dominates).

## What success looks like

```
MATCHES: 5 / 200 hashed         <-- d1 has 5 collisions with BraTS sample
>>> CONFIRMED dataset1 <-> candidate source. Re-ID is on the table.
```

Even 1 match is sufficient — it proves the path. (If 0/200, the sample is
the wrong source or wrong year; try another mirror.)

## What failure looks like

```
MATCHES: 0 / 200 hashed
    No collisions for dataset1.
```

Action: try another BraTS year/mirror, OR fall back to prompts 05 + 06.
