# Annotated References

Grouped by how we use them. ⭐ = highest priority / directly reusable.

## A. Contrast-invariant representations (training-free or generative) — Branch B1, augmentation, re-rank
- ⭐ **MIND — Modality Independent Neighbourhood Descriptor** (Heinrich et al., 2012, *Medical Image Analysis*, 729 cit.) — canonical contrast-invariant self-similarity descriptor; robust to non-functional intensity relations, noise, bias fields. Basis for the **zero-training branch (B1)** and the **re-rank distance**. https://consensus.app/papers/details/59ba79621f4c5cbd9acae45de397d169/
- ⭐ **SynthMorph — Contrast-Invariant Registration Without Acquired Images** (Hoffmann et al., 2020, *IEEE TMI*, 265 cit.) — generative synthesis → contrast-agnostic networks. Use for **synthetic-contrast augmentation** (B3) and **Stage-2 registration re-rank**. https://consensus.app/papers/details/dd95597e50335ccfa3a984d8303c80f9/
- **Modality-Agnostic Structural Representation Learning (DSIR)** (Mok et al., 2024, *CVPR*) — deep self-similarity + anatomy-aware contrastive; contrast-invariant descriptors without labels/pre-alignment. https://consensus.app/papers/details/0d0ff0e8ec13599ebc35c2770f7c456f/
- **MR-CLIP** (Avci et al., 2025) — metadata-guided contrast-aware → anatomy-invariant reps; shown on cross-modal retrieval; code public. https://consensus.app/papers/details/828985bb5ca95a4bb5a73f994b8b2cd9/

## B. Matching-by-synthesis + contrastive (the organizer's recipe) — Branch B3 ⭐
- ⭐⭐ **CrossKEY — 3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration** (Morozov, **Dorent**, Haouchine, 2025) — matching-by-synthesis + supervised contrastive + rotation-invariant keypoints + curriculum triplet loss with hard-negative mining. **Co-authored by the challenge PI. Open source — start here.** Code/weights: https://github.com/morozovdd/CrossKEY · paper: https://consensus.app/papers/details/67ea6b9c65ce50eeb8030f056690b3b0/
- ⭐ **Synth-by-Reg (SbR)** (Casamitjana et al., 2021, SASHIMI/MICCAI) — synthesis+contrastive converts inter-modality into easier intra-modality matching; code public. https://consensus.app/papers/details/3a61ef7db6405cbfa6289defe5cc5d47/
- **MatchAnything** (He et al., 2025) — synthetic cross-modal pretraining for generalizable matching. https://hf.co/papers/2501.07556
- **MINIMA — Modality Invariant Image Matching** (Jiang et al., 2024) — generative data engine for modality-invariant features. https://hf.co/papers/2412.19412

## C. Subject re-identification / "brain fingerprinting" — framing + Branch design
- ⭐ **DeepBrainPrint** (Puglisi et al., 2023) — semi-self-supervised contrastive brain-MRI **re-identification/retrieval**; explicit transforms for contrast/age/progression robustness. Closest published analogue to this exact task. https://consensus.app/papers/details/92213052a80e56dea69885d802ada391/

## D. Foundation models for 3D brain MRI (frozen embeddings) — Branch B2
- ⭐ **BrainIAC** (Tak et al., 2026, *Nature Neuroscience*) — SSL on 48,965 MRIs; strong few-shot/OOD embeddings. https://consensus.app/papers/details/5e3b568bd10c57a2aab0ff6c78dcd474/
- ⭐ **3D-Neuro-SimCLR** (Kaczmarek et al., 2025, ICCVW) — public 3D brain-MRI SSL weights; strong low-data/OOD. https://consensus.app/papers/details/01acdd664efc57fd87c11b4bb1bed527/
- **BrainFound** (Mazher et al., 2025) — slice-based SSL, T1/T2/FLAIR multimodal input. https://consensus.app/papers/details/0eaa17958ecc5b7fa1cd3ca3176d05d6/
- **M3Ret** (Liu et al., 2025) — unified encoder, SOTA **zero-shot image-to-image retrieval** + cross-modal alignment. https://hf.co/papers/2509.01360

## E. Retrieval fusion & re-ranking — Stage 2
- ⭐ **C-MIR — ColBERT-inspired re-ranking for 3D medical retrieval** (Khun Jush et al., 2025, *J. Imaging Informatics in Medicine*) — volumetric late-interaction re-rank; no pre-segmentation; localizes ROI. https://consensus.app/papers/details/350cbf7ae2ad57b681923fd6f9b41186/
- **Nonlinear fusion of manifold rankings in CBIR** (Dao et al., 2024) — rank-fusion for medical retrieval. https://consensus.app/papers/details/625237bda157538fbf80ebc0bf63ba20/

## F. Adjacent / future (FLAIR↔T1 extension)
- **Single-subject Multi-contrast MRI Super-resolution via INR** (McGinnis et al., 2023) — INR exchanges anatomy across a subject's contrasts; relevant to any-contrast latent. https://hf.co/papers/2303.15065

---
*Searched via Consensus, PubMed, arXiv, Hugging Face (June 2026). Verify exact venues/years from the linked pages before citing in a paper.*
