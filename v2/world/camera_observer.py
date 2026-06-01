# Converts a FramePairer step() dict into a batched cuRobo CameraObservation.
# FramePairer (in v2/perception/frame_pairer.py) already owns the ZMQ socket,
# the multipart-skip, and the time-sync; we just translate its output.
#
# pair: {"front_left": (rgb_np, depth_np_u16), "front_right": (...), "t": ...}
# cams: dict from extrinsics.load_extrinsic_yaml — provides intrinsics + T_base_cam.

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from curobo.types import CameraObservation, Pose

DEPTH_SCALE = 1e-3  # uint16 mm -> meters, matches proposer/tracker.


def _pose_from_T(T_base_cam, device):
    R = T_base_cam[:3, :3]
    t = T_base_cam[:3, 3]
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return (
        torch.tensor([t], dtype=torch.float32, device=device),
        torch.tensor([[w, x, y, z]], dtype=torch.float32, device=device),
    )


def _intrinsics_K(cam, device):
    return torch.tensor(
        [[cam["fx"], 0.0, cam["cx"]],
         [0.0, cam["fy"], cam["cy"]],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )


class PairToCuroboObs:
    """Stateless converter: precomputes per-cam pose/intrinsics tensors once."""

    def __init__(self, cams, cam_order=("front_left", "front_right"), device="cuda:0"):
        self.cam_order = list(cam_order)
        self.device = device
        self._pos = {}
        self._quat = {}
        self._K = {}
        for cid in self.cam_order:
            pos, quat = _pose_from_T(cams[cid]["T_base_cam"], device)
            self._pos[cid] = pos
            self._quat[cid] = quat
            self._K[cid] = _intrinsics_K(cams[cid], device)

    def num_cameras(self):
        return len(self.cam_order)

    def convert(self, pair):
        """pair: dict from FramePairer.step(). Returns batched CameraObservation."""
        rgbs, depths = [], []
        for cid in self.cam_order:
            rgb_np, depth_u16 = pair[cid]
            rgbs.append(torch.from_numpy(np.ascontiguousarray(rgb_np)).to(self.device))
            depth_m = depth_u16.astype(np.float32) * DEPTH_SCALE
            depths.append(torch.from_numpy(depth_m).to(self.device))

        return CameraObservation(
            depth_image=torch.stack(depths),
            rgb_image=torch.stack(rgbs),
            pose=Pose(
                position=torch.cat([self._pos[cid] for cid in self.cam_order]),
                quaternion=torch.cat([self._quat[cid] for cid in self.cam_order]),
            ),
            intrinsics=torch.stack([self._K[cid] for cid in self.cam_order]),
        )
