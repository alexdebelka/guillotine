# Per-dataset RRF weight tuning — cells to paste into Nicole's notebook

Goal: instead of guessing weights, pick the per-dataset combo that maximizes **local macro MRR** before burning a Kaggle submission.

Local proxies:
- **L1 (d1)** = held-out 50 dataset1 pairs, no deformation. Direct proxy for dataset1.
- **L2 (d2/d3 proxy)** = same queries, but targets deformed via `trackc.deform_volume`. Independent random rigid + nonlinear deformation per target = exactly the gap d2 vs d1 has.

We grid-search per-dataset weights against `0.5*L1 + 0.5*L2` (or pure L2 for d3 since it has the most domain shift). Pick the winner per dataset.

---

## Cell 1 — Compute B3 + B4 holdout embeddings (L1 + L2 versions)

Paste **after** Nicole's existing L2 cell (the one that builds `gb_l2`, `g2_l2`, `rb2`, `rq2`).

```python
# Hijack Nicole's L2 setup (qb, q2, gb, g2 are L1 query/gallery embeddings already in memory).
# Add B3 + B4 versions — L1 (no deform) for q/g, L2 (deform gallery only) for g.

import sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/shared-docker/work/repo/workers/A')
from b3_encoder import build_encoder, INPUT_SIZE as B3_INPUT
from train_b3 import ProjHead, pool_features, get_pooled_dim
from viz_aug import load_and_resize as b3_load_resize
from b4_shape import shape_fingerprint

# --- B3 embedder (uses trained ckpt) ---
b3_model, b3_dev = build_encoder()
b3_pooled = get_pooled_dim(b3_model, b3_dev)
b3_head = ProjHead(in_dim=b3_pooled).to(b3_dev)
b3_ckpt = torch.load('/shared-docker/work/repo/workers/A/runs/b3_run1.pt',
                     map_location=b3_dev, weights_only=False)
b3_model.swinViT.load_state_dict(b3_ckpt['swinViT'])
b3_head.load_state_dict(b3_ckpt['head'])
b3_model.eval(); b3_head.eval()

@torch.no_grad()
def _b3_l1(rel_path):
    vol = b3_load_resize(rel_path)  # already z-scored + resized to 96^3
    x = torch.from_numpy(vol)[None, None].float().to(b3_dev)
    out = b3_model.swinViT(x)
    vec = pool_features(out)
    z = b3_head(vec)
    z = F.normalize(z, dim=1)
    return z.flatten().cpu().numpy().astype('float32')

@torch.no_grad()
def _b3_l2(rel_path):
    # L2: load, resize, apply Nicole's deform_volume, then encode
    vol = b3_load_resize(rel_path)
    vol_def = deform_volume(vol, rng=rng)
    x = torch.from_numpy(vol_def)[None, None].float().to(b3_dev)
    out = b3_model.swinViT(x)
    vec = pool_features(out)
    z = b3_head(vec)
    z = F.normalize(z, dim=1)
    return z.flatten().cpu().numpy().astype('float32')

def _b4_l1(rel_path):
    return shape_fingerprint(f'{DATA_ROOT}/{rel_path}')

def _b4_l2(rel_path):
    # Manually replicate the load+threshold+deform+crop+resample path because the
    # original shape_fingerprint reads from disk; here we deform after threshold.
    import nibabel as nib
    from scipy.ndimage import zoom as _zoom
    img = nib.load(f'{DATA_ROOT}/{rel_path}')
    vol = img.get_fdata().astype('float32')
    mask = (vol > vol.mean() * 0.1).astype('float32')
    mask_def = deform_volume(mask, rng=rng)
    mask_def = (mask_def > 0.5).astype('float32')
    if mask_def.sum() < 100:
        return np.zeros(4096, dtype='float32')
    coords = np.argwhere(mask_def)
    mins, maxs = coords.min(0), coords.max(0) + 1
    crop = mask_def[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    factors = [16/s for s in crop.shape]
    cube = _zoom(crop, factors, order=1)
    v = cube.flatten()
    return (v / (np.linalg.norm(v) + 1e-8)).astype('float32')

# L1: queries + gallery, no deformation
q3 = embed_pool(hold_df, 'query_id',  'query_image', _b3_l1)
g3 = embed_pool(hold_df, 'target_id', 'target_image', _b3_l1)
q4 = embed_pool(hold_df, 'query_id',  'query_image', _b4_l1)
g4 = embed_pool(hold_df, 'target_id', 'target_image', _b4_l1)

# L2: same queries (clean), gallery deformed
g3_l2 = embed_pool(hold_df, 'target_id', 'target_image', _b3_l2)
g4_l2 = embed_pool(hold_df, 'target_id', 'target_image', _b4_l2)

r_b3   = rank_by_embeddings(q3, g3);    r_b3_l2 = rank_by_embeddings(q3, g3_l2)
r_b4   = rank_by_embeddings(q4, g4);    r_b4_l2 = rank_by_embeddings(q4, g4_l2)

print(f"B3 L1 MRR = {mrr(r_b3, gt):.4f}  | B3 L2 MRR = {mrr(r_b3_l2, gt):.4f}")
print(f"B4 L1 MRR = {mrr(r_b4, gt):.4f}  | B4 L2 MRR = {mrr(r_b4_l2, gt):.4f}")
```

**Expected**: B3 L1 ≈ 0.87 (matches Alex's number), B3 L2 should be lower but hopefully > 0.4. B4 L1 ≈ 0.30, B4 L2 ≈ ?? (the rotation in deform_volume should kill the mask shape — likely <0.10).

---

## Cell 2 — Grid search per-dataset weights

```python
# Branches available at holdout level: r_base, r_b2 (L1+L2 versions from Nicole's earlier cells),
# r_b3, r_b3_l2, r_b4, r_b4_l2
# rb1 = baseline L1, rb2 = baseline L2 (from Nicole's existing cell)
# rq1 = b2 L1,       rq2 = b2 L2

# We score per dataset:
#  d1 weights against L1 MRR  (registered, no deformation)
#  d2 weights against 0.7*L2 + 0.3*L1  (mostly deformation)
#  d3 weights against L2 MRR  (heavily deformed + domain shift)

import itertools

def score_combo(rankings_l1, rankings_l2, weights, target='l1'):
    fused_l1 = rrf(rankings_l1, weights=weights)
    fused_l2 = rrf(rankings_l2, weights=weights)
    if target == 'l1':  return mrr(fused_l1, gt)
    if target == 'l2':  return mrr(fused_l2, gt)
    return 0.7 * mrr(fused_l2, gt) + 0.3 * mrr(fused_l1, gt)  # d2 mix

branches_l1 = [rb1, rq1, r_b3, r_b4]                     # baseline, b2, b3, b4 (L1)
branches_l2 = [rb2, rq2, r_b3_l2, r_b4_l2]               # same in L2
names = ['baseline', 'b2', 'b3', 'b4']

# coarse grid — each weight in {0, 0.5, 1, 2}. 4^4 = 256 combos per dataset, ~1 sec total.
grid = [0.0, 0.5, 1.0, 2.0]

def search(target):
    best = (-1, None)
    for w in itertools.product(grid, repeat=4):
        if sum(w) == 0: continue  # skip all-zero
        s = score_combo(branches_l1, branches_l2, list(w), target=target)
        if s > best[0]:
            best = (s, w)
    return best

s1, w1 = search('l1');   print(f"d1 best (L1):       MRR={s1:.4f}  weights={dict(zip(names,w1))}")
s2, w2 = search('mix');  print(f"d2 best (0.7L2+0.3L1): MRR={s2:.4f}  weights={dict(zip(names,w2))}")
s3, w3 = search('l2');   print(f"d3 best (L2):       MRR={s3:.4f}  weights={dict(zip(names,w3))}")
print(f"\nlocal macro MRR estimate = {(s1+s2+s3)/3:.4f}")
```

---

## Cell 3 — Plug winning weights into the submission writer

```python
weights = {
    'dataset1': dict(zip(['baseline','b2','b3','b4'], w1)),
    'dataset2': dict(zip(['baseline','b2','b3','b4'], w2)),
    'dataset3': dict(zip(['baseline','b2','b3','b4'], w3)),
}
# add a 'b' entry if you still want Nicole's old branch:
# for d in weights: weights[d]['b'] = 1.0 if d != 'dataset1' else 0.0

pool_rankings = fuse_branches(
    {'baseline': branch_baseline, 'b2': branch_b2, 'b3': branch_b3, 'b4': branch_b4},
    manifest, weights_by_dataset=weights)

sub = write_submission(pool_rankings, manifest, path='submission.csv')
validate_submission(sub, manifest)
print(f"wrote submission.csv with tuned per-dataset weights")
print(f"local macro estimate: {(s1+s2+s3)/3:.4f}  (Kaggle delta will tell us how accurate the L2 proxy is)")
```

---

## Decision rule
- `local macro estimate > previous Kaggle score (0.5298) + 0.02` → **submit**.
- Otherwise, the grid was too coarse or the L2 proxy is misleading. Don't submit; either:
  - Refine the grid around the winners (e.g., `[0.5, 1.0, 1.5, 2.0, 3.0]` per dataset).
  - Add `'b': branch_b` back into the mix.
  - Hard-cap `b4` at 0 for d1 (we know it's noise on registered data).

When you submit, also note the **per-dataset weights** that won — that's the data we'll use to decide whether B4 V2 (rotation-invariant) is worth building. If `b4` got weight 0 on d2 and d3, V1 is useless and we focus elsewhere.
