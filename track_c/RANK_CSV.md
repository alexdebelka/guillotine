# Branch ranking contract (target this and RRF absorbs you with zero rework)

Every retrieval branch — B1 (MIND), B2 (foundation), B3, B4, Stage-2 re-rank — emits **one CSV**.
Track C's `fuse_branches(...)` reads them, does per-dataset RRF, and writes the final `submission.csv`.

## File

- Name: `branch_<name>.csv`  (e.g. `branch_b1.csv`, `branch_b3.csv`). `<name>` is your branch key in the fusion dict.
- Location: `/shared-docker/Nicole/` (next to `trackc.py`).

## Columns — exactly these three, this order

| column | type | meaning |
|---|---|---|
| `query_id`  | str   | the query id, **verbatim** from the manifest (`q_...`). Do not alter. |
| `target_id` | str   | a gallery candidate id, **verbatim** (`g_...`). |
| `score`     | float | similarity, **higher = more similar**. Any real scale — RRF uses rank, not magnitude, so it's scale-free. |

No header renaming, no extra columns, no index column (`to_csv(path, index=False)`).

## Coverage — every query scored against its **own pool's** gallery only

A query is only ever compared to the gallery of the *same* (dataset, split) pool. Never cross pools.
Your file must contain **one row per (query, candidate) pair** across all 6 pools:

| dataset | split | queries × gallery | rows |
|---|---|---|---|
| dataset1 | val  | 40 × 40   | 1 600 |
| dataset1 | test | 100 × 100 | 10 000 |
| dataset2 | val  | 40 × 40   | 1 600 |
| dataset2 | test | 100 × 100 | 10 000 |
| dataset3 | val  | 20 × 20   | 400 |
| dataset3 | test | 77 × 77   | 5 929 |
| **total** | | | **29 529** |

If your CSV isn't 29 529 rows, a pool is missing — RRF will drop those queries. (You don't add dataset/split
columns; Track C recovers each query's pool from the manifest.)

## Emitting it from embeddings (the easy path)

If your branch produces a vector per id, you don't build the CSV by hand:

```python
import trackc, pandas as pd
manifest = trackc.load_manifest('/shared-docker/data')

rows = []
for key, pool in manifest.items():
    if key == 'train_pairs':
        continue
    qE = {r['query_id']:  my_embed(r['query_image'])  for _, r in pool.queries.iterrows()}
    gE = {r['target_id']: my_embed(r['target_image']) for _, r in pool.gallery.iterrows()}
    rows.append(trackc.embeddings_to_scores_df(qE, gE))   # -> [query_id, target_id, score]
pd.concat(rows, ignore_index=True).to_csv('branch_<name>.csv', index=False)
```

If your branch is a **re-ranker** (already has scores, not embeddings), just write the three columns directly.

## How it gets fused (Track C runs this — shown so you can self-check)

```python
import pandas as pd, trackc
branches = {
    'baseline': pd.read_csv('branch_baseline.csv'),
    'b2':       pd.read_csv('branch_b2.csv'),
    'b1':       pd.read_csv('branch_b1.csv'),   # <- your file drops in here, nothing else changes
}
weights = {  # per-dataset RRF weights, tuned on local proxies
    'dataset1': {'baseline': 1.0, 'b2': 0.0, 'b1': ...},
    'dataset2': {'baseline': 0.3, 'b2': 1.0, 'b1': ...},
    'dataset3': {'baseline': 0.3, 'b2': 1.0, 'b1': ...},
}
pool_rankings = trackc.fuse_branches(branches, manifest, weights_by_dataset=weights)
sub = trackc.write_submission(pool_rankings, manifest, 'submission.csv')
trackc.validate_submission(sub, manifest)
```

## Self-check before you hand off (3 asserts)

```python
df = pd.read_csv('branch_<name>.csv')
assert list(df.columns) == ['query_id', 'target_id', 'score']
assert len(df) == 29529
assert df['score'].notna().all()
```

Pass these three and your branch is guaranteed to fuse. That's the whole contract.
