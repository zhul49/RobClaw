# multi-camera CoTracker3 point tracker.
#
# public API (matches MultiCamDinoTracker so the same demo plumbing works):
#   register(rgb_by_cam, depth_by_cam, keypoints_3d_base) -> list[int]
#   track(rgb_by_cam, depth_by_cam)                       -> (positions (N,3), n_inliers (N,), mean_sim (N,))
#   project_into(cam_id, kp_base)                         -> (row, col) or None
#
# what's different from the DINO tracker:
#   - CoTracker3 is a transformer that tracks 2D queries across a sliding
#     window of step*2=16 frames. it doesn't use 3D directly.
#   - to get 3D output we look up depth at the tracked pixel in each camera,
#     unproject to camera frame, transform to base frame, then average across
#     cameras (weighted by CoTracker's visibility score).
#   - the model needs window_len=16 frames buffered before the first forward.
#     while warming up, track() holds the registered position and reports
#     n_inliers=0 so the caller knows.
#   - after warmup, the model forwards once every step=8 frames. between
#     forwards the last tracked position is held, so track() output updates
#     at the forward cadence, not every call.

import os

import numpy as np
import torch

# re-export so callers that grab load_extrinsic_yaml from here keep working.
from v2.common.extrinsics import load_extrinsic_yaml  # noqa: F401


VIS_THRESHOLD_DEFAULT = 0.5


def _project_to_pixel(intr, Tcb, p_base):
    # base-frame point -> pixel (u, v). returns None if the point is behind the camera.
    p = Tcb @ np.array([float(p_base[0]), float(p_base[1]), float(p_base[2]), 1.0])
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if z <= 0:
        return None
    u = x * intr["fx"] / z + intr["cx"]
    v = y * intr["fy"] / z + intr["cy"]
    return float(u), float(v)


def _unproject_pixel(intr, u, v, z):
    # (u, v, z_meters) -> camera-frame 3D point.
    x = (u - intr["cx"]) * z / intr["fx"]
    y = (v - intr["cy"]) * z / intr["fy"]
    return np.array([x, y, z], dtype=np.float64)


class CoTracker3MultiCam:
    # internals:
    #   - one CoTrackerOnlinePredictor per camera (see note below).
    #   - per cam: rolling frame buffer of size window_len + matching depth buffer.
    #     latest depth is used for 2D->3D lookup.
    #   - per cam: queries tensor (1, N, 3) on device, frame_idx=0, x=u, y=v.
    #   - output 2D pixel = tracks[:, -1] from the most recent forward; updates
    #     on the step=8 cadence.
    #   - per kp per cam: depth -> 3D camera frame -> base frame. cams whose
    #     visibility >= vis_threshold contribute; per-kp 3D = mean of contributors.
    #     if all cams are below threshold, hold last position (n_inliers=0).
    #   - uniform filter (window 10) over the fused 3D output. paper default.

    def __init__(
        self,
        cams,                          # {cam_id: {fx, fy, cx, cy, T_base_cam}}
        smooth_window=10,
        depth_scale=1e-3,
        vis_threshold=VIS_THRESHOLD_DEFAULT,
        device="cuda",
    ):
        if not cams:
            raise ValueError("CoTracker3MultiCam requires at least one camera")
        self.cams = cams
        self.cam_order = list(cams.keys())
        self.smooth_window = int(smooth_window)
        self.depth_scale = float(depth_scale)
        self.vis_threshold = float(vis_threshold)
        self.device = device

        # one CoTrackerOnlinePredictor per camera. running B=2 (both cams in
        # one batch) hits a `.view()` non-contiguity bug in cotracker3_online's
        # forward. running two B=1 predictors sequentially avoids that.
        # memory cost is ~800MB total, fine on a modern card.
        self.models = {}
        for cid in self.cam_order:
            self.models[cid] = (
                torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
                .to(device).eval()
            )
        any_m = next(iter(self.models.values()))
        self.window_len = int(any_m.model.window_len)
        self.step = int(any_m.step)

        # cache the inverse extrinsic so projection doesn't re-invert per call.
        self._T_cam_base = {cid: np.linalg.inv(c["T_base_cam"])
                            for cid, c in cams.items()}
        self._reset_state()

    def _reset_state(self):
        self.frame_bufs = {cid: [] for cid in self.cam_order}
        self.depth_bufs = {cid: [] for cid in self.cam_order}
        # per-cam (1, N, 3) query tensor on device — each cam has its own predictor.
        self.queries_per_cam = {}
        self.initialized = False
        self.frame_count = 0
        # per-cam latest 2D tracks and visibility from the most recent forward.
        self.latest_tracks = {}   # cam_id -> (N, 2) np
        self.latest_vis = {}      # cam_id -> (N,)  np float in [0, 1]
        self.last_positions = None
        self.history = []

    # ---------- public API ----------

    def register(self, rgb_by_cam, depth_by_cam, keypoints_3d_base):
        # project the 3D keypoints into each camera's image plane to seed
        # CoTracker queries. returns a list of int (one per kp): 1 if the kp
        # projected into FOV cleanly, else raises (we currently don't support
        # silently dropping out-of-FOV kps — would break the per-cam alignment).
        # the rgb_by_cam / depth_by_cam args are intentionally not used here;
        # the first call to track() will start filling the frame buffer.
        del rgb_by_cam, depth_by_cam

        kp = np.asarray(keypoints_3d_base, dtype=np.float32).reshape(-1, 3)
        N = kp.shape[0]
        self._reset_state()

        ok_per_kp = [1] * N
        for cam_id in self.cam_order:
            intr = self.cams[cam_id]
            Tcb = self._T_cam_base[cam_id]
            q = np.zeros((1, N, 3), dtype=np.float32)
            for i, k in enumerate(kp):
                uv = _project_to_pixel(intr, Tcb, k)
                if uv is None:
                    raise ValueError(
                        f"keypoint {i} ({k.tolist()}) projects behind camera {cam_id}; "
                        f"CoTracker can't seed an out-of-FOV query."
                    )
                q[0, i] = (0.0, uv[0], uv[1])
            self.queries_per_cam[cam_id] = (
                torch.from_numpy(q).float().to(self.device).contiguous()
            )
            self.latest_tracks[cam_id] = q[0, :, 1:].astype(np.float32).copy()
            self.latest_vis[cam_id] = np.ones((N,), dtype=np.float32)

        self.last_positions = kp.copy()
        return ok_per_kp

    def track(self, rgb_by_cam, depth_by_cam):
        # add a frame, possibly run a forward, return the fused 3D position per kp.
        assert self.queries_per_cam, "call register() first"

        # 1. push the new frame into each cam's ring buffer.
        for cam_id in self.cam_order:
            if cam_id not in rgb_by_cam or cam_id not in depth_by_cam:
                raise RuntimeError(f"missing {cam_id} in track() input")
            self.frame_bufs[cam_id].append(rgb_by_cam[cam_id])
            self.depth_bufs[cam_id].append(depth_by_cam[cam_id])
            if len(self.frame_bufs[cam_id]) > self.window_len:
                self.frame_bufs[cam_id].pop(0)
                self.depth_bufs[cam_id].pop(0)
        self.frame_count += 1

        # 2. decide whether to forward.
        #    first forward fires when the buffer fills (frame_count == window_len).
        #    after that, fire every `step` frames so windows slide cleanly.
        do_forward = False
        is_first = False
        if not self.initialized and self.frame_count >= self.window_len:
            do_forward = True
            is_first = True
        elif self.initialized and \
             ((self.frame_count - self.window_len) % self.step == 0) and \
             (self.frame_count > self.window_len):
            do_forward = True
        if do_forward:
            self._forward(is_first=is_first)

        # 3. fuse latest 2D tracks across cams into one 3D position per kp,
        #    using the most recent depth maps.
        any_cam = self.cam_order[0]
        N = int(self.queries_per_cam[any_cam].shape[1])
        positions = np.zeros((N, 3), dtype=np.float32)
        n_inliers = np.zeros(N, dtype=np.int32)
        mean_sim = np.zeros(N, dtype=np.float32)

        for i in range(N):
            contribs = []
            vis_score_sum = 0.0
            n_vis = 0
            for cam_id in self.cam_order:
                vis_score = float(self.latest_vis[cam_id][i])
                if vis_score < self.vis_threshold:
                    continue
                u, v = self.latest_tracks[cam_id][i]
                col, row = int(round(u)), int(round(v))
                depth = self.depth_bufs[cam_id][-1]
                H, W = depth.shape
                if not (0 <= row < H and 0 <= col < W):
                    continue
                z = float(depth[row, col]) * self.depth_scale
                if z <= 0.0:
                    continue
                intr = self.cams[cam_id]
                p_cam = _unproject_pixel(intr, float(u), float(v), z)
                p_base = (intr["T_base_cam"] @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]
                contribs.append(p_base.astype(np.float32))
                vis_score_sum += vis_score
                n_vis += 1

            if contribs:
                positions[i] = np.mean(np.stack(contribs), axis=0).astype(np.float32)
                n_inliers[i] = n_vis
                mean_sim[i] = vis_score_sum / max(1, n_vis)
            else:
                # nothing visible enough -> hold last fused position.
                positions[i] = self.last_positions[i]
                n_inliers[i] = 0
                mean_sim[i] = 0.0

        # 4. uniform filter over the last `smooth_window` raw 3D positions.
        self.history.append(positions)
        if len(self.history) > self.smooth_window:
            self.history = self.history[-self.smooth_window:]
        smoothed = np.mean(np.stack(self.history), axis=0).astype(np.float32)
        self.last_positions = smoothed
        return smoothed, n_inliers, mean_sim

    def project_into(self, cam_id, kp_base):
        # same convention as MultiCamDinoTracker (used by the demo overlay).
        # returns (row, col) — note the swap from (u, v) = (col, row).
        intr = self.cams[cam_id]
        Tcb = self._T_cam_base[cam_id]
        uv = _project_to_pixel(intr, Tcb, kp_base)
        if uv is None:
            return None
        return int(round(uv[1])), int(round(uv[0]))

    # ---------- internals ----------

    def _forward(self, is_first):
        # one CoTracker3 forward per camera (B=1 each) over the latest window.
        # on is_first the model returns nothing useful — we just init internal
        # state. after that it returns (tracks (1, T, N, 2), vis (1, T, N, 1)
        # or (1, T, N)) and we cache the LAST frame's per-cam tracks + vis
        # for downstream 2D->3D fusion.
        T = self.window_len
        for cam_id in self.cam_order:
            buf = self.frame_bufs[cam_id]
            assert len(buf) == T, f"{cam_id} buffer len {len(buf)} != {T}"

            stack = np.stack(buf, axis=0).astype(np.float32)        # (T, H, W, 3)
            stack = np.transpose(stack, (0, 3, 1, 2))               # (T, 3, H, W)
            video = (
                torch.from_numpy(stack).unsqueeze(0).to(self.device).contiguous()
            )                                                        # (1, T, 3, H, W)

            model = self.models[cam_id]
            with torch.inference_mode():
                if is_first:
                    model(video_chunk=video, is_first_step=True,
                          queries=self.queries_per_cam[cam_id])
                    continue
                tracks, vis = model(video_chunk=video)
                last_tracks = tracks[0, -1].detach().cpu().numpy().astype(np.float32)
                vis_last = vis[0, -1]
                if vis_last.ndim == 2:                              # shape (N, 1) on some checkpoints
                    vis_last = vis_last[:, 0]
                vis_arr = vis_last.detach().cpu().numpy().astype(np.float32)
                self.latest_tracks[cam_id] = last_tracks
                self.latest_vis[cam_id] = vis_arr

        if is_first:
            self.initialized = True
