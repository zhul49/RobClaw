# shared DINOv3 helpers: load the backbone, run it on one image, normalize.
# used by the tracker and any demos that want raw features.

import os

import numpy as np
import torch
from torch.nn.functional import interpolate


DEFAULT_DINOV3_REPO = os.path.expanduser("~/Code/dinov3")
DEFAULT_DINOV3_WEIGHTS = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
DEFAULT_PATCH_SIZE = 16


def load_dinov3(repo=DEFAULT_DINOV3_REPO, weights=DEFAULT_DINOV3_WEIGHTS, device="cuda"):
    # load DINOv3 ViT-S/16 from a local clone of the repo + a local .pth.
    model = torch.hub.load(repo, "dinov3_vits16", source="local", weights=weights)
    return model.eval().to(device)


@torch.inference_mode()
def per_pixel_features(backbone, rgb_uint8, patch_size=DEFAULT_PATCH_SIZE, device="cuda"):
    # rgb_uint8: (H, W, 3) uint8. returns (H, W, D) float32 features on `device`.
    # H and W must be divisible by patch_size (caller's job to crop/resize).
    H, W, _ = rgb_uint8.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        f"image size {H}x{W} not divisible by patch_size {patch_size}"
    )

    # (H, W, 3) uint8 -> (1, 3, H, W) float on device, scaled to [0, 1].
    rgb = (
        torch.from_numpy(rgb_uint8.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    # one D-dim vector per patch.
    with torch.amp.autocast("cuda"):
        feats = backbone.forward_features(rgb)["x_norm_patchtokens"]   # (1, P_h*P_w, D)

    # reshape the flat patch list into a 2D grid, then bilinear-upsample
    # to one vector per pixel.
    P_h, P_w = H // patch_size, W // patch_size
    D = feats.shape[-1]
    feats = feats.reshape(1, P_h, P_w, D).permute(0, 3, 1, 2).float()  # (1, D, P_h, P_w)
    feats = interpolate(feats, size=(H, W), mode="bilinear")           # (1, D, H, W)
    feats = feats.squeeze(0).permute(1, 2, 0).contiguous()             # (H, W, D)
    return feats


def normalize(t, eps=1e-8):
    # L2-normalize the last dim. eps avoids div-by-zero on all-zero vectors.
    return t / (t.norm(dim=-1, keepdim=True) + eps)
