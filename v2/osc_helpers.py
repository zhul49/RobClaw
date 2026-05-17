"""Pure conversion helpers for LIBERO's OSC_POSE controller. No env deps.
Smoke tests: v2_dev/tests/test_osc_helpers.py."""

import numpy as np
from scipy.spatial.transform import Rotation as R


# Probed live from OSC_POSE config — not guessed.
OSC_POS_CLIP = 0.05         # m / tick at 20 Hz
OSC_ROT_CLIP = 0.5          # rad axis-angle / tick


def axisangle_from_quats(q_from_xyzw, q_to_xyzw):
    """World-frame axis-angle rotating q_from to q_to.
    World-frame = q_delta * q_from (matches robosuite OSC_POSE with control_delta=True)."""
    r_from = R.from_quat(np.asarray(q_from_xyzw, dtype=np.float64))
    r_to = R.from_quat(np.asarray(q_to_xyzw, dtype=np.float64))
    return (r_to * r_from.inv()).as_rotvec()


def pose_to_osc_action(target_xyzw, current_xyzw, gripper_cmd):
    """7-D OSC delta action in [-1, 1]. Clipping is silent; streaming converges
    by re-deriving delta each tick from the live EE pose."""
    target = np.asarray(target_xyzw, dtype=np.float64)
    current = np.asarray(current_xyzw, dtype=np.float64)

    delta_pos = target[:3] - current[:3]
    delta_rot_aa = axisangle_from_quats(current[3:], target[3:])

    action = np.zeros(7, dtype=np.float64)
    action[0:3] = np.clip(delta_pos / OSC_POS_CLIP, -1.0, 1.0)
    action[3:6] = np.clip(delta_rot_aa / OSC_ROT_CLIP, -1.0, 1.0)
    action[6] = float(gripper_cmd)
    return action
