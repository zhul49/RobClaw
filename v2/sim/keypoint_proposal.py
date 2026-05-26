# picks ~5 keypoints per object mask using DINO features + 3D position.
# pipeline: backbone features -> PCA -> k-means in (features + xyz) -> one
# keypoint per cluster -> bounds filter -> merge close ones.

import os
import re
import numpy as np
import torch
import cv2
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from v2.common.utils import filter_points_by_bounds
from sklearn.cluster import MeanShift

# PROJECT_ROOT is the v2/ directory. relative paths in configs (e.g.
# "weights/dinov3.pth", "third_party/dinov3") resolve against it.
# (this file lives in v2/sim/, so we go up one level from __file__.)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(p):
    # expand ${VAR:-default} style placeholders and $VAR env vars,
    # then make the path absolute against PROJECT_ROOT.
    p = re.sub(r"\$\{([^:}]+):-([^}]*)\}", lambda m: os.environ.get(m.group(1), m.group(2)), p)
    p = os.path.expandvars(p)
    if not os.path.isabs(p):
        p = os.path.join(PROJECT_ROOT, p)
    return p


class KeypointProposer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])

        # pick the DINO backbone. 'auto' tries DINOv3 first (better, but needs
        # the repo + weights present locally), then falls back to DINOv2.
        backbone = self.config.get('backbone', 'auto')
        if backbone == 'auto':
            try:
                repo = _resolve_path(self.config.get('dinov3_repo', ''))
                weights = _resolve_path(self.config.get('dinov3_weights', ''))
            except Exception:
                repo = weights = ''
            if repo and weights and os.path.isdir(repo) and os.path.isfile(weights):
                backbone = 'dinov3'
            else:
                print("DINOv3 files not found — using DINOv2.")
                backbone = 'dinov2'

        if backbone == 'dinov3':
            dinov3_repo = _resolve_path(self.config['dinov3_repo'])
            dinov3_weights = _resolve_path(self.config['dinov3_weights'])
            if not os.path.isdir(dinov3_repo):
                raise FileNotFoundError(
                    f"DINOv3 repo not found at {dinov3_repo}. Clone "
                    f"https://github.com/facebookresearch/dinov3 there, or set "
                    f"DINOV3_REPO to its location."
                )
            if not os.path.isfile(dinov3_weights):
                raise FileNotFoundError(
                    f"DINOv3 weights not found at {dinov3_weights}. Request "
                    f"access at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ "
                    f"and place the .pth there, or set DINOV3_WEIGHTS to its path."
                )
            self.backbone = torch.hub.load(
                dinov3_repo,
                'dinov3_vits16',
                source='local',
                weights=dinov3_weights,
            ).eval().to(self.device)
            self.patch_size = 16
            print(f"KeypointProposer using DINOv3 ViT-S/16 (patch_size={self.patch_size})")
        elif backbone == 'dinov2':
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14'
            ).eval().to(self.device)
            self.patch_size = 14
            print(f"KeypointProposer using DINOv2 ViT-S/14 (patch_size={self.patch_size})")
        else:
            raise ValueError(f"unknown backbone: {backbone}")

        # workspace bounds — anything outside this box gets dropped later.
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])

        # MeanShift is used at the very end to merge keypoints that ended up
        # very close in 3D (one physical spot, two clusters chose it).
        self.mean_shift = MeanShift(
            bandwidth=self.config['min_dist_bt_keypoints'],
            bin_seeding=True,
            n_jobs=32,
        )

        # fix all RNGs so kmeans/MeanShift give the same answer across runs.
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

    def get_keypoints(self, rgb, points, masks):
        # main entry point. rgb (H,W,3) uint8, points (H,W,3) world xyz per pixel,
        # masks (H,W) int label image (one int per object, 0 = background).

        # 1. resize rgb to be a multiple of patch_size and split masks per object.
        transformed_rgb, rgb, points, masks, shape_info = self._preprocess(rgb, points, masks)

        # 2. backbone features, then bilinear-upsample so we get one vector per pixel.
        features_flat = self._get_features(transformed_rgb, shape_info)

        # 3. for each mask, run k-means in (PCA-of-features + xyz) and pick one
        # keypoint per cluster.
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids = (
            self._cluster_features(points, features_flat, masks)
        )

        # 4. drop anything outside the workspace box (depth glitches, walls, etc.)
        print(
            f"  before bounds filter: {len(candidate_keypoints)} keypoints across "
            f"rigid_groups {sorted(set(candidate_rigid_group_ids.tolist()))}"
        )
        within_space = filter_points_by_bounds(
            candidate_keypoints, self.bounds_min, self.bounds_max, strict=True,
        )
        print(
            f"  after bounds filter:  {within_space.sum()} keypoints across "
            f"rigid_groups {sorted(set(candidate_rigid_group_ids[within_space].tolist()))}"
        )
        candidate_keypoints = candidate_keypoints[within_space]
        candidate_pixels = candidate_pixels[within_space]
        candidate_rigid_group_ids = candidate_rigid_group_ids[within_space]

        # 5. merge keypoints that landed too close in 3D.
        print(
            f"  before merge: {len(candidate_keypoints)} across "
            f"rigid_groups {sorted(set(candidate_rigid_group_ids.tolist()))}"
        )
        merged_indices = self._merge_clusters(candidate_keypoints)
        print(
            f"  after merge:  {len(merged_indices)} across "
            f"rigid_groups {sorted(set(candidate_rigid_group_ids[merged_indices].tolist()))}"
        )
        candidate_keypoints = candidate_keypoints[merged_indices]
        candidate_pixels = candidate_pixels[merged_indices]
        candidate_rigid_group_ids = candidate_rigid_group_ids[merged_indices]

        # sort by pixel position (row first, then column) so the numbering
        # GPT-4o sees is stable across runs.
        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_pixels = candidate_pixels[sort_idx]
        candidate_rigid_group_ids = candidate_rigid_group_ids[sort_idx]

        # print the final keypoint inventory for easy cross-check.
        for kp_idx in range(len(candidate_pixels)):
            px = candidate_pixels[kp_idx]
            rg = candidate_rigid_group_ids[kp_idx]
            print(f"  keypoint {kp_idx}: pixel=({px[0]}, {px[1]}), rigid_group={rg}")

        # 6. draw numbered labels on the rgb image (for the VLM and for humans).
        projected = self._project_keypoints_to_img(
            rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat,
        )
        return candidate_keypoints, projected

    def _preprocess(self, rgb, points, masks):
        # split the label image into one boolean mask per object id.
        masks = [masks == uid for uid in np.unique(masks)]

        # resize rgb so its height and width are exact multiples of patch_size.
        H, W, _ = rgb.shape
        patch_h = int(H // self.patch_size)
        patch_w = int(W // self.patch_size)
        new_H = patch_h * self.patch_size
        new_W = patch_w * self.patch_size
        transformed_rgb = cv2.resize(rgb, (new_W, new_H))
        transformed_rgb = transformed_rgb.astype(np.float32) / 255.0

        shape_info = {
            'img_h': H,
            'img_w': W,
            'patch_h': patch_h,
            'patch_w': patch_w,
        }
        return transformed_rgb, rgb, points, masks, shape_info

    def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids,
                                  masks, features_flat):
        # draw numbered boxes next to each keypoint, with a line from the
        # box to the pixel. flip the label to the other side if the box
        # would go off the image edge.
        projected = rgb.copy()
        H, W = projected.shape[:2]
        for keypoint_count, pixel in enumerate(candidate_pixels):
            kp_y, kp_x = int(pixel[0]), int(pixel[1])

            text = f"{keypoint_count}"
            text_length = len(text)
            box_w = 24 + 8 * (text_length - 1)
            box_h = 24
            offset_x, offset_y = 22, 22

            label_cx = kp_x + offset_x
            label_cy = kp_y + offset_y
            if label_cx + box_w // 2 > W - 2:
                label_cx = kp_x - offset_x
            if label_cy + box_h // 2 > H - 2:
                label_cy = kp_y - offset_y

            # connecting line + keypoint dot.
            cv2.line(projected, (kp_x, kp_y), (label_cx, label_cy), (0, 0, 0), 1)
            cv2.circle(projected, (kp_x, kp_y), 4, (255, 255, 255), -1)
            cv2.circle(projected, (kp_x, kp_y), 4, (0, 0, 0), 1)

            # label box: white fill, black outline.
            top_left = (label_cx - box_w // 2, label_cy - box_h // 2)
            bot_right = (label_cx + box_w // 2, label_cy + box_h // 2)
            cv2.rectangle(projected, top_left, bot_right, (255, 255, 255), -1)
            cv2.rectangle(projected, top_left, bot_right, (0, 0, 0), 2)

            # number text, blue.
            org = (label_cx - 6 * text_length, label_cy + 6)
            cv2.putText(projected, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)
        return projected

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_features(self, transformed_rgb, shape_info):
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']
        patch_h = shape_info['patch_h']
        patch_w = shape_info['patch_w']

        # (H, W, 3) -> (1, 3, H, W) on device.
        img_tensors = (
            torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        assert img_tensors.shape[1] == 3, "expected 3-channel rgb image"

        # raw patch features: one D-dim vector per patch in a grid.
        features_dict = self.backbone.forward_features(img_tensors)
        raw_feature_grid = features_dict['x_norm_patchtokens']            # (1, patch_h*patch_w, D)
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w, -1)

        # bilinear-upsample the patch grid back to the full image resolution
        # so we get one feature vector per pixel.
        interpolated = interpolate(
            raw_feature_grid.permute(0, 3, 1, 2),  # (1, D, patch_h, patch_w)
            size=(img_h, img_w),
            mode='bilinear',
        ).permute(0, 2, 3, 1).squeeze(0)           # (H, W, D)

        # flatten to (H*W, D).
        features_flat = interpolated.reshape(-1, interpolated.shape[-1])
        return features_flat

    def _cluster_features(self, points, features_flat, masks):
        # for each per-object binary mask, run k-means on (PCA features + xyz)
        # and pick the pixel closest to each cluster center.
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []

        for rigid_group_id, binary_mask in enumerate(masks):
            # skip masks that cover most of the image — almost always table/wall.
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue

            # features + 3D positions of pixels inside this mask.
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            feature_points = points[binary_mask]

            # PCA: take top 3 principal components, then normalize to [0, 1].
            obj_features_flat = obj_features_flat.double()
            _, _, v = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            features_pca = (
                (features_pca - features_pca.min(0)[0]) /
                (features_pca.max(0)[0] - features_pca.min(0)[0])
            )

            # also normalize the 3D positions to [0, 1] so xyz and features
            # are on comparable scales for k-means.
            feature_points_torch = torch.tensor(
                feature_points, dtype=features_pca.dtype, device=features_pca.device,
            )
            feature_points_torch = (
                (feature_points_torch - feature_points_torch.min(0)[0]) /
                (feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            )

            # final clustering features = [pca_feat_1, pca_feat_2, pca_feat_3, x, y, z]
            X = torch.cat([features_pca, feature_points_torch], dim=-1)

            cluster_ids_x, cluster_centers = kmeans(
                X=X,
                num_clusters=self.config['num_candidates_per_mask'],
                distance='euclidean',
                device=self.device,
            )
            cluster_centers = cluster_centers.to(self.device)

            # for each cluster, pick the pixel whose PCA-feature is closest to the center.
            for cluster_id in range(self.config['num_candidates_per_mask']):
                cluster_center = cluster_centers[cluster_id][:3]   # only the feature half
                member_idx = cluster_ids_x == cluster_id
                member_points = feature_points[member_idx]
                member_pixels = feature_pixels[member_idx]
                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                closest_idx = torch.argmin(dist)
                candidate_keypoints.append(member_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)
        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids

    def _merge_clusters(self, candidate_keypoints):
        # MeanShift on xyz collapses keypoints that ended up very close.
        # for each surviving cluster center, keep the original keypoint nearest to it.
        self.mean_shift.fit(candidate_keypoints)
        cluster_centers = self.mean_shift.cluster_centers_
        merged_indices = []
        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices
