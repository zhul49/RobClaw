# paper-faithful keypoint proposer (ReKep appendix A.5).
#
# pipeline for a single RGB-D camera:
#   1. DINOv3 per-pixel features.
#   2. SAM-everything masks (one per object). caller supplies these as a
#      label image — typically from gsam2_service /predict_automatic_masks.
#   3. for each mask:
#        f_mask  = features inside the mask
#        f_pca   = PCA(f_mask, 3)
#        labels  = KMeans(f_pca, k=5).fit_predict
#        for each cluster:
#            pixel = median-ish pixel of that cluster (snapped to a real member)
#            3D    = unproject(pixel, depth, K)
#        keypoints += [(3D, mask_id) for each cluster]
#   4. drop keypoints outside the workspace box.
#   5. MeanShift dedupe (8 cm bandwidth, paper) merges nearby keypoints.
#
# differences from v2/sim/keypoint_proposal.py (the LIBERO-sim version):
#   - k-means on PCA features ONLY (no xyz). xyz biases clusters toward
#     spatial separation, which is fine for sim depth but bad for real depth.
#   - cluster rep is the MEDIAN pixel (robust to mask boundary noise),
#     not the cluster-center-nearest member.
#   - MeanShift bandwidth is 8 cm (paper) instead of 6 cm.

import os

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans, MeanShift
from torch.nn.functional import interpolate

# we use sklearn k-means instead of kmeans_pytorch because the GPU version
# has no iter cap and prints a tqdm bar per call. with ~50 masks per scene
# that's minutes per proposal. sklearn at k=5 converges in <0.2s.
_KMEANS_MAX_ITER = 100
_KMEANS_N_INIT = 4
_KMEANS_SUBSAMPLE = 20000  # fit on at most this many points; predict on all


# ---------------- backbone helpers ----------------

def _patch_features(backbone, rgb_uint8, patch_size, device):
    # rgb_uint8: (H, W, 3) uint8. returns (H, W, D) float32 features on device.
    # H and W must already be multiples of patch_size.
    H, W, _ = rgb_uint8.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        f"image {H}x{W} not divisible by patch_size {patch_size}"
    )
    rgb = (
        torch.from_numpy(rgb_uint8.astype(np.float32) / 255.0)
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )
    with torch.amp.autocast("cuda"), torch.inference_mode():
        feats = backbone.forward_features(rgb)["x_norm_patchtokens"]   # (1, P_h*P_w, D)
    P_h, P_w = H // patch_size, W // patch_size
    D = feats.shape[-1]
    feats = feats.reshape(1, P_h, P_w, D).permute(0, 3, 1, 2).float()  # (1, D, P_h, P_w)
    feats = interpolate(feats, size=(H, W), mode="bilinear")           # (1, D, H, W)
    return feats.squeeze(0).permute(1, 2, 0).contiguous()              # (H, W, D)


def _round_to_patch(H, W, patch_size):
    # round image dims DOWN to the nearest multiple of patch_size.
    return (H // patch_size) * patch_size, (W // patch_size) * patch_size


# ---------------- geometry helpers ----------------

def _unproject(uv, depth_u16, fx, fy, cx, cy, depth_scale=1e-3):
    # uv = (row, col). returns 3D (x, y, z) in camera frame, or None if no depth.
    r, c = int(uv[0]), int(uv[1])
    z = float(depth_u16[r, c]) * depth_scale
    if z <= 0:
        return None
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def _transform(T, p):
    # T is 4x4, p is (3,). returns (3,) transformed.
    h = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (T @ h)[:3].astype(np.float32)


# ---------------- cluster helpers ----------------

def _pca3(X):
    # X is (N, D) torch tensor. returns (N, 3). no centering — paper doesn't
    # specify it and v2/sim/keypoint_proposal.py also skips it.
    X = X.double()
    _, _, v = torch.pca_lowrank(X, center=False)
    return (X @ v[:, :3]).float()


def _kmeans_labels(X, k, device):
    # X is (N, 3) torch tensor or numpy. returns (N,) numpy int labels in [0, k).
    # big masks get subsampled before fit so background masks don't tank runtime;
    # labels are then predicted for all N points.
    X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    N = X_np.shape[0]
    k_eff = min(k, N)
    if k_eff < 2:
        return np.zeros(N, dtype=np.int32)
    if N > _KMEANS_SUBSAMPLE:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, _KMEANS_SUBSAMPLE, replace=False)
        km = KMeans(n_clusters=k_eff, n_init=_KMEANS_N_INIT,
                    max_iter=_KMEANS_MAX_ITER, random_state=0).fit(X_np[idx])
        return km.predict(X_np).astype(np.int32)
    km = KMeans(n_clusters=k_eff, n_init=_KMEANS_N_INIT,
                max_iter=_KMEANS_MAX_ITER, random_state=0)
    return km.fit_predict(X_np).astype(np.int32)


# ---------------- main class ----------------

class PaperKeypointProposer:
    # hand it a pre-loaded DINOv3 backbone, then call .propose() with a single
    # RGB-D frame, a SAM-everything multi-instance label image, and intrinsics.
    # output keypoints are in WORLD frame if T_world_cam is passed, else in
    # the camera frame.

    def __init__(self, backbone, *, k=5, mean_shift_bandwidth_m=0.08,
                 max_mask_ratio=0.9, min_mask_pixels=100,
                 patch_size=16, device="cuda"):
        self.backbone = backbone
        self.k = int(k)
        self.bandwidth = float(mean_shift_bandwidth_m)
        self.max_mask_ratio = float(max_mask_ratio)
        self.min_mask_pixels = int(min_mask_pixels)
        self.patch_size = int(patch_size)
        self.device = device

        # n_jobs=1 avoids a joblib+CUDA-context conflict that bites if MeanShift
        # runs in the same process that has DINO on GPU.
        self._mean_shift = MeanShift(
            bandwidth=self.bandwidth, bin_seeding=True, n_jobs=1,
        )

    @torch.inference_mode()
    def propose(self, rgb, depth, masks_label_image, intrinsics,
                T_world_cam=None, bounds_min=None, bounds_max=None,
                depth_scale=1e-3, verbose=True):
        # rgb               : (H, W, 3) uint8
        # depth             : (H, W) uint16, units = depth_scale meters / unit
        # masks_label_image : (H, W) int — 0=background, 1..N=mask ids
        # intrinsics        : dict {fx, fy, cx, cy}. assumes depth is aligned to color.
        # T_world_cam       : 4x4 cam->world transform. None = output in cam frame.
        # bounds_min/max    : (3,) — workspace bounding box in OUTPUT frame. None = no filter.
        #
        # returns: keypoints_3d (N,3), rigid_group_ids (N,), pixels (N,2), annotated_rgb (H,W,3).

        # round image to a patch-aligned size, run features, then resize features
        # back to the original resolution so they line up with depth and masks.
        H_orig, W_orig, _ = rgb.shape
        Hp, Wp = _round_to_patch(H_orig, W_orig, self.patch_size)
        rgb_p = cv2.resize(rgb, (Wp, Hp)) if (Hp, Wp) != (H_orig, W_orig) else rgb

        feats = _patch_features(self.backbone, rgb_p, self.patch_size, self.device)
        if (Hp, Wp) != (H_orig, W_orig):
            feats = interpolate(
                feats.permute(2, 0, 1).unsqueeze(0),
                size=(H_orig, W_orig), mode="bilinear",
            ).squeeze(0).permute(1, 2, 0).contiguous()

        if verbose:
            print(f"  features: {tuple(feats.shape)} on {self.device}")

        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])

        masks = np.asarray(masks_label_image)
        mask_ids = sorted(int(uid) for uid in np.unique(masks) if int(uid) != 0)
        total_pixels = H_orig * W_orig

        candidate_kp_world = []
        candidate_pixels = []
        candidate_rigid_ids = []

        skipped_too_big = 0
        skipped_too_small = 0
        skipped_no_depth = 0

        for mid in mask_ids:
            binary = (masks == mid)
            n_pix = int(binary.sum())

            # skip tiny masks (noise) and whole-scene masks (table/wall).
            if n_pix < self.min_mask_pixels:
                skipped_too_small += 1
                continue
            if n_pix / total_pixels > self.max_mask_ratio:
                skipped_too_big += 1
                continue

            pixel_rc = np.argwhere(binary).astype(np.int32)         # (N_j, 2)
            f_mask = feats[binary]                                   # (N_j, D)

            # PCA -> 3 dims (paper), then k-means k=5 on the PCA features ALONE.
            f_pca = _pca3(f_mask)
            labels = _kmeans_labels(f_pca, k=self.k, device=self.device)

            for c in range(self.k):
                cluster_pix = pixel_rc[labels == c]
                if len(cluster_pix) == 0:
                    continue

                # paper says "median centroids". but per-axis median can land
                # OUTSIDE the cluster (and thus outside the mask) for non-convex
                # shapes — depth there reads background and the 3D point goes
                # to mid-air. snap to the closest actual cluster member to
                # guarantee the pixel is inside the mask.
                med_target = np.median(cluster_pix, axis=0)
                d = np.linalg.norm(cluster_pix - med_target, axis=1)
                med = cluster_pix[int(np.argmin(d))]

                p_cam = _unproject(med, depth, fx, fy, cx, cy, depth_scale)
                if p_cam is None:
                    skipped_no_depth += 1
                    continue
                p_out = _transform(T_world_cam, p_cam) if T_world_cam is not None else p_cam

                candidate_kp_world.append(p_out)
                candidate_pixels.append(med)
                candidate_rigid_ids.append(mid)

        candidate_kp_world = np.asarray(candidate_kp_world, dtype=np.float32).reshape(-1, 3)
        candidate_pixels = np.asarray(candidate_pixels, dtype=np.int32).reshape(-1, 2)
        candidate_rigid_ids = np.asarray(candidate_rigid_ids, dtype=np.int32)
        if verbose:
            print(f"  masks: {len(mask_ids)} considered, "
                  f"{skipped_too_big} too big, {skipped_too_small} too small")
            print(f"  candidates before bounds: {len(candidate_kp_world)} "
                  f"(skipped_no_depth={skipped_no_depth})")

        # bounds filter (paper step).
        if bounds_min is not None and bounds_max is not None and len(candidate_kp_world) > 0:
            bmin = np.asarray(bounds_min, dtype=np.float32)
            bmax = np.asarray(bounds_max, dtype=np.float32)
            inside = np.all((candidate_kp_world >= bmin) & (candidate_kp_world <= bmax), axis=1)
            candidate_kp_world = candidate_kp_world[inside]
            candidate_pixels = candidate_pixels[inside]
            candidate_rigid_ids = candidate_rigid_ids[inside]
            if verbose:
                print(f"  candidates after bounds: {len(candidate_kp_world)}")

        # MeanShift dedupe in 3D (paper bandwidth = 8 cm).
        if len(candidate_kp_world) > 1:
            self._mean_shift.fit(candidate_kp_world)
            centers = self._mean_shift.cluster_centers_
            keep_idx = []
            for c in centers:
                d = np.linalg.norm(candidate_kp_world - c, axis=1)
                keep_idx.append(int(np.argmin(d)))
            keep_idx = sorted(set(keep_idx))
            candidate_kp_world = candidate_kp_world[keep_idx]
            candidate_pixels = candidate_pixels[keep_idx]
            candidate_rigid_ids = candidate_rigid_ids[keep_idx]
            if verbose:
                print(f"  candidates after MeanShift {self.bandwidth*100:.0f}cm: "
                      f"{len(candidate_kp_world)}")

        # sort by pixel position (row first, then col) so the numbering shown
        # to a VLM is stable across runs.
        if len(candidate_pixels) > 0:
            order = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
            candidate_kp_world = candidate_kp_world[order]
            candidate_pixels = candidate_pixels[order]
            candidate_rigid_ids = candidate_rigid_ids[order]

        annotated = self._annotate(rgb, candidate_pixels)
        return candidate_kp_world, candidate_rigid_ids, candidate_pixels, annotated

    @staticmethod
    def _annotate(rgb, pixels):
        # draw numbered labels next to each keypoint pixel. flip the label
        # to the other side if the box would go off the image edge.
        out = rgb.copy()
        H, W = out.shape[:2]
        for i, (r, c) in enumerate(pixels):
            r, c = int(r), int(c)
            if not (0 <= r < H and 0 <= c < W):
                continue
            label = str(i)
            box_w = 24 + 8 * (len(label) - 1)
            box_h = 24
            ox, oy = 22, 22
            lx = c + ox if c + ox + box_w // 2 < W - 2 else c - ox
            ly = r + oy if r + oy + box_h // 2 < H - 2 else r - oy

            # connecting line + keypoint dot.
            cv2.line(out, (c, r), (lx, ly), (0, 0, 0), 1)
            cv2.circle(out, (c, r), 4, (255, 255, 255), -1)
            cv2.circle(out, (c, r), 4, (0, 0, 0), 1)

            # label box: white fill, black outline.
            cv2.rectangle(out, (lx - box_w // 2, ly - box_h // 2),
                          (lx + box_w // 2, ly + box_h // 2), (255, 255, 255), -1)
            cv2.rectangle(out, (lx - box_w // 2, ly - box_h // 2),
                          (lx + box_w // 2, ly + box_h // 2), (0, 0, 0), 2)

            # number text, blue.
            cv2.putText(out, label, (lx - 6 * len(label), ly + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return out
