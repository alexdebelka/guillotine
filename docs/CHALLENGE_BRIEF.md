# Challenge Brief — EHL Paris 2026 Cross-Modal Medical Image Retrieval

Distilled from the official repo README and the organizers' presentation (R. Dorent & N. Stellwag, 27 June 2026).

## One-liner
Given a **3D ceT1 (contrast-enhanced T1)** brain MRI query, rank a gallery of **T2** volumes so the **same patient's T2** ranks first. Cross-modal, content-based, 3D.

## Clinical motivation (from the deck)
- Brain-tumor surgery balances **maximal resection** vs **minimal functional damage**.
- Medical data is multimodal (structural MRI T1/ceT1/T2/FLAIR, fMRI, intra-op ultrasound, omics, histopathology); interpretation relies on experience.
- **Content-based retrieval** = automatically surface similar prior cases (outcomes, survival, deficits) to inform a new case.
- The hard part is the **"modality gap"**: a prior case in T2 vs a new case in ceT1 look very different in intensity even for the same anatomy.

## Exact task & evaluation
- Query = ceT1, Gallery = T2. For each query, output a full ranking of the same-pool gallery.
- **Metric:** MRR per dataset, then **macro-average**: `score = (MRR_d1 + MRR_d2 + MRR_d3)/3`. RR = 1/rank of the true match; 0 if omitted.
- Three difficulty "levels" = three datasets:
  - **Level 1 / dataset1 — "perfectly aligned":** ceT1 & T2 registered to a common grid. 350 labeled train pairs. Only nuisance = contrast.
  - **Level 2 / dataset2 — "non-linear deformations":** independent rigid + nonlinear deformation per image; no shared geometry. No labels.
  - **Level 3 / dataset3 — "before/after surgery + different hospital":** preop ceT1 → intra-op T2; tissue shifted/missing/resected; scanner domain shift. No labels.

## Data facts
- Format: NIfTI `.nii.gz`, RAS, 1.0×1.0×1.0 mm. **No** intensity normalization, histogram matching, skull-stripping, deformable registration, or cropping in the release.
- **Do not assume a fixed shape;** matching query/target may differ in shape (esp. d2/d3).
- Counts: d1 350 train / 40 val / 100 test; d2 40 val / 100 test; d3 20 val / 77 test. Template = 377 rows.

## Submission
- One combined CSV: `query_id,target_id_ranking` (space-separated, full gallery length, most→least likely).
- Rank only within same dataset & split. Partial submissions allowed (omitted datasets score 0).
- **100 submissions/team/day.**

## Provided baseline
- `slice_clip_baseline.py` — MONAI + PyTorch, dual 2D-CNN CLIP on 3 axial slices, trained on dataset1 pairs only. Intentionally weak; demonstrates the data format + a valid submission.

## Links
- Repo: https://github.com/NicoStellwag/ehl-paris-2026-medical-retrieval
- Kaggle: https://www.kaggle.com/t/b33ec3e76c3d4e16a6b56852470b3ebf
- PI's open-source method (CrossKEY): https://github.com/morozovdd/CrossKEY
