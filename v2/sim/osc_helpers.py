# helpers for converting (target pose, current pose) into the 7-element action
# that robosuite's OSC_POSE controller expects. pure math, no robot.

import numpy as np
from scipy.spatial.transform import Rotation as R


# per-tick clip ranges baked into the OSC_POSE controller config.
# anything bigger gets clipped to these.
OSC_POS_CLIP = 0.05   # meters per tick at 20 Hz
OSC_ROT_CLIP = 0.5    # radians of axis-angle per tick


def axisangle_from_quats(q_from_xyzw, q_to_xyzw):
    # rotation that takes q_from to q_to, returned as an axis-angle 3-vector.
    # world frame, so q_delta * q_from (matches OSC_POSE with control_delta=True).
    r_from = R.from_quat(np.asarray(q_from_xyzw, dtype=np.float64))
    r_to = R.from_quat(np.asarray(q_to_xyzw, dtype=np.float64))
    return (r_to * r_from.inv()).as_rotvec()


def pose_to_osc_action(target_xyzw, current_xyzw, gripper_cmd):
    # build the 7-element action: [dx, dy, dz, rx, ry, rz, gripper], all in [-1, 1].
    # if the delta is bigger than one tick can do we clip silently; the next tick
    # will reduce the remaining gap because we recompute from the new live pose.
    target = np.asarray(target_xyzw, dtype=np.float64)
    current = np.asarray(current_xyzw, dtype=np.float64)

    delta_pos = target[:3] - current[:3]
    delta_rot = axisangle_from_quats(current[3:], target[3:])

    action = np.zeros(7, dtype=np.float64)
    action[0:3] = np.clip(delta_pos / OSC_POS_CLIP, -1.0, 1.0)
    action[3:6] = np.clip(delta_rot / OSC_ROT_CLIP, -1.0, 1.0)
    action[6] = float(gripper_cmd)
    return action
