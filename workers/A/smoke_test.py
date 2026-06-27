"""
Smoke test for Task #3 — build SwinUNETR, load MONAI SSL weights, forward pass on ROCm.
Pass criteria printed inline. Exits non-zero if any step fails.
"""
import sys
import time
import numpy as np
import torch
from b3_encoder import build_encoder, load_ssl_weights, embed, INPUT_SIZE

WEIGHTS = "/shared-docker/work/weights/model_swinvit.pt"

print(f"torch: {torch.__version__}  cuda.is_available: {torch.cuda.is_available()}")
print(f"hip: {getattr(torch.version, 'hip', None)}")

t0 = time.time()
model, device = build_encoder()
print(f"built SwinUNETR on {device} in {time.time()-t0:.1f}s  "
      f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

t0 = time.time()
matched = load_ssl_weights(model, WEIGHTS)
print(f"loaded weights in {time.time()-t0:.1f}s")

# fake volume: random tensor at the SSL input size
vol = np.random.randn(*INPUT_SIZE).astype(np.float32)
t0 = time.time()
v = embed(model, device, vol)
print(f"forward pass + embed in {time.time()-t0:.2f}s  shape: {v.shape}  norm: {np.linalg.norm(v):.4f}")
assert v.ndim == 1 and v.dtype == np.float32, "embedding shape/dtype wrong"
assert abs(np.linalg.norm(v) - 1.0) < 1e-3, "embedding not unit-norm"

# bigger fake batch — check the GPU isn't pathologically slow
B = 8
batch = torch.from_numpy(np.random.randn(B, 1, *INPUT_SIZE).astype(np.float32)).to(device)
torch.cuda.synchronize() if device == "cuda" else None
t0 = time.time()
with torch.no_grad():
    out = model.swinViT(batch)
torch.cuda.synchronize() if device == "cuda" else None
print(f"batch={B} forward in {time.time()-t0:.2f}s  feature stages: {len(out) if isinstance(out, (list, tuple)) else 1}")

print("OK")
