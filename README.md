# EHL Paris 2026 — Contrast-Agnostic Brain MRI Cross-Modal Retrieval

24h hackathon project. Retrieve the **same subject's T2** volume given a **ceT1** query (cross-modal, 3D brain MRI). Scored by **MRR macro-averaged over 3 datasets** of increasing difficulty (registered → deformed → post-surgical/cross-hospital).

## Start here
1. **`CLAUDE.md`** — full agent/team onboarding: task, data, baseline, approach, build order, gotchas.
2. **`docs/CHALLENGE_BRIEF.md`** — the official challenge facts, distilled.
3. **`docs/RESEARCH_PLAN.md`** — the strategy: multi-branch invariant embeddings → Reciprocal Rank Fusion → label-free re-rank. Timeline, ablations, risks.
4. **`docs/REFERENCES.md`** — annotated literature (what each paper is for).
5. **`docs/CONSENSUS_QUERIES.md`** — search prompts for Consensus.

## The one thing to remember
This is **same-subject re-identification across a contrast gap and a geometry gap**, not tumor similarity. Because the score is averaged over 3 heterogeneous datasets, **generalization/robustness wins, not dataset1 accuracy.**

## Fastest strong start
Clone the PI's open-source method and adapt it: https://github.com/morozovdd/CrossKEY
Build a **local MRR harness** from the 350 labeled pairs first (Kaggle = 100 submissions/day, hidden labels).

## Key links
- Challenge repo: https://github.com/NicoStellwag/ehl-paris-2026-medical-retrieval
- Kaggle: https://www.kaggle.com/t/b33ec3e76c3d4e16a6b56852470b3ebf

*Status: planning complete, implementation not started.*
