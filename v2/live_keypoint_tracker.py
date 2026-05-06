"""
v2/live_keypoint_tracker.py — DINO-feature-based live keypoint tracking.

Replaces the body-snap crutch (`sim.data.body_xpos[body_id]`, which only
works in sim because it reads simulator ground truth) with perception
that would actually work on a real robot.

Two-step API:

    register(rgb, points_3d, keypoints_world):
        At task start, for each 3D keypoint:
          1. project to a pixel in this RGB
          2. run DINO on the RGB to get per-pixel features
          3. sample the feature at that pixel — the "anchor"
        Stored as a [N, 384] tensor of anchor features on the GPU.

    update(rgb, points_3d) -> (N, 3):
        Each subsequent call:
          1. run DINO on the new RGB
          2. for each anchor, compute cosine similarity to every pixel
             feature in the new image
          3. take argmax → the pixel that most resembles the anchor
          4. return that pixel's world-frame 3D position from points_3d

This is how ReKep's `env.register_keypoints` / `get_keypoint_positions`
work conceptually. Same DINO backbone as `KeypointProposer`.
"""
import os
import re
import numpy as np
import torch
from torch.nn.functional import interpolate as F_interpolate


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(p):
    """Mirrors keypoint_proposal._resolve_path so the same DINOv3
    DINOV3_REPO/DINOV3_WEIGHTS env-var pattern works here too."""
    p = re.sub(r"\$\{([^:}]+):-([^}]*)\}",
               lambda m: os.environ.get(m.group(1), m.group(2)), p)
    p = os.path.expandvars(p)
    if not os.path.isabs(p):
        p = os.path.join(PROJECT_ROOT, p)
    return p


class LiveKeypointTracker:
    def __init__(self, config):
        self.device = torch.device(config.get('device', 'cuda'))

        # Same backbone-selection logic as KeypointProposer. 'auto' picks
        # DINOv3 if its files are present, else falls back to DINOv2 (which
        # torch.hub fetches automatically).
        backbone = config.get('backbone', 'auto')
        if backbone == 'auto':
            try:
                repo = _resolve_path(config.get('dinov3_repo', ''))
                weights = _resolve_path(config.get('dinov3_weights', ''))
            except Exception:
                repo = weights = ''
            if repo and weights and os.path.isdir(repo) and os.path.isfile(weights):
                backbone = 'dinov3'
            else:
                backbone = 'dinov2'

        if backbone == 'dinov3':
            self.backbone = torch.hub.load(
                _resolve_path(config['dinov3_repo']),
                'dinov3_vits16',
                source='local',
                weights=_resolve_path(config['dinov3_weights']),
            ).eval().to(self.device)
            self.patch_size = 16
            print("[live_keypoint_tracker] DINOv3 ViT-S/16")
        else:
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14'
            ).eval().to(self.device)
            self.patch_size = 14
            print("[live_keypoint_tracker] DINOv2 ViT-S/14")

        # [N, D] feature anchors set by register(); D=384 for ViT-S.
        # L2-normalized so cosine similarity is just a dot product.
        self.anchor_features = None
        self.last_pixels = None  # [N, 2] last-known pixels (for windowed search)

    # ------------------------------------------------------------------
    def _compute_features(self, rgb_uint8):
        """Run DINO on an RGB image; return per-pixel features [H, W, D] on GPU.

        Same pipeline as KeypointProposer._get_features:
          - resize to a multiple of patch_size (DINO requires it)
          - normalize to [0, 1]
          - forward_features → patch tokens [1, ph*pw, D]
          - bilinear upsample patch grid → per-pixel features [H, W, D]
        """
        import cv2
        H, W = rgb_uint8.shape[:2]
        ph = H // self.patch_size
        pw = W // self.patch_size
        new_H = ph * self.patch_size
        new_W = pw * self.patch_size
        if (new_H, new_W) != (H, W):
            rgb_uint8 = cv2.resize(rgb_uint8, (new_W, new_H))
        rgb_f = rgb_uint8.astype(np.float32) / 255.0
        img = torch.from_numpy(rgb_f).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.backbone.forward_features(img)
        patch_feats = out['x_norm_patchtokens']  # [1, ph*pw, D]
        D = patch_feats.shape[-1]
        patch_grid = patch_feats.reshape(1, ph, pw, D).permute(0, 3, 1, 2)  # [1, D, ph, pw]
        feats = F_interpolate(
            patch_grid, size=(new_H, new_W), mode='bilinear'
        ).permute(0, 2, 3, 1).squeeze(0)  # [H, W, D]
        return feats

    # ------------------------------------------------------------------
    def register(self, rgb_uint8, world_keypoints, world_to_pixel_fn):
        """Sample DINO features at each keypoint's pixel location.

        Args:
            rgb_uint8         : (H, W, 3) uint8 RGB of the initial scene.
            world_keypoints   : (N, 3) keypoint XYZ in world frame.
            world_to_pixel_fn : callable(world_xyz) -> (col, row) or None
                                if the point projects outside the image.

        After this call, self.anchor_features holds N L2-normalized
        feature vectors — one per keypoint — that update() will match
        against in subsequent frames.
        """
        feats = self._compute_features(rgb_uint8)
        H, W, D = feats.shape
        anchors = []
        last_pixels = []
        n_oob = 0
        for kp in world_keypoints:
            px = world_to_pixel_fn(kp)
            if px is None or not (0 <= px[0] < W and 0 <= px[1] < H):
                anchors.append(torch.zeros(D, device=self.device))
                last_pixels.append((-1, -1))
                n_oob += 1
            else:
                col, row = int(px[0]), int(px[1])
                anchors.append(feats[row, col])
                last_pixels.append((col, row))
        stacked = torch.stack(anchors, dim=0)  # [N, D]
        # L2-normalize so dot product == cosine similarity in update()
        stacked = stacked / (stacked.norm(dim=-1, keepdim=True) + 1e-9)
        self.anchor_features = stacked
        self.last_pixels = np.array(last_pixels, dtype=np.int32)
        print(f"[live_keypoint_tracker] registered {len(world_keypoints)} keypoints "
              f"({n_oob} out-of-frame)")

    # ------------------------------------------------------------------
    def update(self, rgb_uint8, points_3d, search_radius_px=None):
        """Track each registered anchor in a new RGB frame.

        Args:
            rgb_uint8        : (H, W, 3) uint8 RGB for the current frame.
            points_3d        : (H, W, 3) world-frame 3D position per pixel
                               (typically from depth + camera intrinsics).
            search_radius_px : if set, only search within this pixel-radius
                               of the last known position. Helps when the
                               scene has multiple visually-similar regions
                               (e.g. the alphabet soup vs tomato sauce
                               cans). None = global argmax.

        Returns:
            (N, 3) numpy array of updated keypoint positions in world
            frame. Anchors that registered out-of-frame are returned as
            zeros — caller should keep their previous values.
        """
        if self.anchor_features is None:
            raise RuntimeError("LiveKeypointTracker.update before register")

        feats = self._compute_features(rgb_uint8)
        H, W, D = feats.shape
        N = self.anchor_features.shape[0]
        # Normalize new feature map for cosine sim
        feats_norm = feats / (feats.norm(dim=-1, keepdim=True) + 1e-9)

        new_positions = np.zeros((N, 3), dtype=np.float64)
        new_pixels = np.zeros((N, 2), dtype=np.int32)
        for i in range(N):
            if not self.anchor_features[i].any():
                # was out-of-frame at register time — leave at zeros
                new_pixels[i] = (-1, -1)
                continue

            anchor = self.anchor_features[i]  # [D]

            if search_radius_px is not None and self.last_pixels[i, 0] >= 0:
                # Windowed search around last known position
                last_col, last_row = self.last_pixels[i]
                r = int(search_radius_px)
                r0 = max(0, last_row - r)
                r1 = min(H, last_row + r + 1)
                c0 = max(0, last_col - r)
                c1 = min(W, last_col + r + 1)
                window = feats_norm[r0:r1, c0:c1]  # [h, w, D]
                sims = window @ anchor  # [h, w]
                flat_idx = sims.argmax().item()
                local_h = sims.shape[0]
                local_w = sims.shape[1]
                row = r0 + flat_idx // local_w
                col = c0 + flat_idx % local_w
            else:
                # Global argmax over the whole frame
                feats_flat = feats_norm.reshape(-1, D)  # [H*W, D]
                sims = feats_flat @ anchor  # [H*W]
                idx = sims.argmax().item()
                row = idx // W
                col = idx % W

            new_pixels[i] = (col, row)
            new_positions[i] = points_3d[row, col]

        self.last_pixels = new_pixels
        return new_positions
