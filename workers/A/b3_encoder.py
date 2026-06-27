"""
Track A — B3 backbone wrapper.

Thin wrapper around MONAI's SwinUNETR. Provides:
  - build_encoder(device) -> (model, device)
  - load_ssl_weights(model, path) -> int (num matched tensors)
  - embed(model, device, volume) -> np.ndarray of shape (C,), unit-norm float32

Embedding contract matches Track C's `rank_by_embeddings`:
  dict[id_str -> np.ndarray (C,) unit-norm float32]
so the output drops into RRF without conversion.

ponytail: thin wrapper, no class hierarchy. monai SwinUNETR is the model.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

INPUT_SIZE = (96, 96, 96)   # matches MONAI SSL pretraining patch size


def build_encoder(device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # out_channels is unused for embedding; we only forward through model.swinViT.
    model = SwinUNETR(
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_checkpoint=False,
    ).to(device).eval()
    return model, device


def load_ssl_weights(model: SwinUNETR, ckpt_path: str) -> int:
    """Load MONAI SwinUNETR SSL weights into model.swinViT. Returns matched-tensor count."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # strip common prefixes
    cleaned = {}
    for k, v in state.items():
        nk = k
        for pre in ("module.", "swinViT.", "backbone.swinViT.", "backbone."):
            if nk.startswith(pre):
                nk = nk[len(pre):]
        # MONAI 1.6 renamed Mlp.fc{1,2} -> linear{1,2}
        nk = nk.replace("mlp.fc1.", "mlp.linear1.").replace("mlp.fc2.", "mlp.linear2.")
        cleaned[nk] = v
    msg = model.swinViT.load_state_dict(cleaned, strict=False)
    matched = len([k for k in model.swinViT.state_dict() if k not in msg.missing_keys])
    print(f"SSL load: matched {matched}/{len(model.swinViT.state_dict())} swinViT tensors "
          f"(missing {len(msg.missing_keys)}, unexpected {len(msg.unexpected_keys)})")
    return matched


@torch.no_grad()
def embed(model: SwinUNETR, device: str, volume: np.ndarray) -> np.ndarray:
    """volume: (D,H,W) float32 already preprocessed to INPUT_SIZE. Returns unit-norm float32 (C,)."""
    x = torch.from_numpy(volume)[None, None].float().to(device)
    out = model.swinViT(x)
    feat = out[-1] if isinstance(out, (list, tuple)) else out  # deepest stage
    vec = F.adaptive_avg_pool3d(feat, 1).flatten().cpu().numpy()
    return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)
