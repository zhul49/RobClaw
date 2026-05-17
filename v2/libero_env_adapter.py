"""LIBEROReKepEnv — bridges LIBERO's OffScreenRenderEnv to the ReKepOGEnv
interface main3 expects. Same surface as upstream ReKepOGEnv plus
set_grasping/clear_grasping for grasp bookkeeping.

Gotchas worth knowing:
  - SDF is built once in register_keypoints. get_sdf_voxels' voxel_size arg is ignored.
  - get_collision_points returns None when nothing is held. path_solver2 has no
    None guard — callers must substitute a placeholder. Intentional: silent
    fallback would hide whether collision data is present.
  - is_grasping is bookkeeping only (tracks set/clear_grasping calls), not
    physics — physical slip won't update it.
"""

import os
import numpy as np
import cv2

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

import transform_utils as T
from build_sdf import build_sdf_for_env
from osc_helpers import pose_to_osc_action


LIBERO_BOUNDS_MIN = np.array([-0.40, -0.50, 0.005])
LIBERO_BOUNDS_MAX = np.array([ 0.70,  0.50, 0.60])

SDF_VOXEL_SIZE_BUILD = 0.02

OSC_CONTROL_HZ = 20
SETTLE_POS_TOL = 0.005       # 5 mm
SETTLE_MAX_TICKS = 40

OPEN_GRIPPER_TICKS = 40

VIDEO_FRAME_STRIDE = 2
VIDEO_FPS = 10               # OSC_CONTROL_HZ / VIDEO_FRAME_STRIDE
VIDEO_RES = 480

N_COLLISION_POINTS = 60


class LIBEROReKepEnv:
    """Drop-in replacement for ReKepOGEnv from main.py's perspective."""

    def __init__(self, suite_name, task_idx, video_path=None, verbose=False):
        self.suite_name = suite_name
        self.task_idx = task_idx
        self.verbose = verbose

        bench = benchmark.get_benchmark_dict()[suite_name]()
        bddl = bench.get_task_bddl_file_path(task_idx)
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=VIDEO_RES,
            camera_widths=VIDEO_RES,
            camera_depths=False,
        )
        # robosuite replaces MjSim on every reset, so any sim handle cached
        # here would dangle. Re-acquired in reset().
        self.sim = None
        self.GRIP_SITE_ID = None
        self.ROBOT_BASE_BODY_ID = None
        self.qpos_addr = None

        if video_path is None:
            video_path = os.path.expanduser(
                f"~/libero_keypoint_project/outputs/v2_{suite_name}_{task_idx}/main2_task.mp4"
            )
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        self._video_path = video_path
        self._video_writer = None

        self._step_counter = 0
        self._gripper_state = -1.0           # open at boot
        self._grasped_indices = []
        self._reset_joint_pos = None
        self._world2robot_homo = None

        self._init_keypoint_positions = None
        self._kp_to_body = None

        self._sdf_full = None
        self._sdf_per_body = None

        self._tick_log = []
        self._log_stage = 0
        self._finger_qpos_addrs = None
        # Cached in register_keypoints once the kp→body map exists. NaN
        # placeholder in tick records before then.
        self._can_body_id = None
        self._basket_body_id = None

        self.disturbance_seq = None

        # main3 reads reset_joint_pos / world2robot_homo at __init__ time
        # (to build IK), before any caller calls reset(). Bootstrap reset
        # here so those properties are readable on construction.
        self.reset()

    def reset(self):
        self.env.reset()

        self.sim = self.env.env.sim
        self.GRIP_SITE_ID = self.sim.model.site_name2id("gripper0_grip_site")
        self.ROBOT_BASE_BODY_ID = self.sim.model.body_name2id("robot0_base")
        qa = self.sim.model.get_joint_qpos_addr("robot0_joint1")
        if isinstance(qa, tuple):
            qa = qa[0]
        self.qpos_addr = qa

        arm_qpos = self.sim.data.qpos[self.qpos_addr:self.qpos_addr + 7].copy()
        self._reset_joint_pos = np.append(arm_qpos, 0.0)

        # Finger joint ranges: joint1 [0, 0.04], joint2 [-0.04, 0].
        # Open ≈ (0.04, -0.04), closed ≈ (0, 0).
        fa1 = self.sim.model.get_joint_qpos_addr("gripper0_finger_joint1")
        fa2 = self.sim.model.get_joint_qpos_addr("gripper0_finger_joint2")
        if isinstance(fa1, tuple): fa1 = fa1[0]
        if isinstance(fa2, tuple): fa2 = fa2[0]
        self._finger_qpos_addrs = (fa1, fa2)

        self._tick_log = []
        self._can_body_id = None
        self._basket_body_id = None

        R_wb = self.sim.data.body_xmat[self.ROBOT_BASE_BODY_ID].reshape(3, 3).copy()
        p_wb = self.sim.data.body_xpos[self.ROBOT_BASE_BODY_ID].copy()
        H = np.eye(4)
        H[:3, :3] = R_wb.T
        H[:3, 3] = -R_wb.T @ p_wb
        self._world2robot_homo = H

        self._step_counter = 0
        self._gripper_state = -1.0
        self._grasped_indices = []

        if self._video_writer is not None:
            self._video_writer.release()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(
            self._video_path, fourcc, VIDEO_FPS, (VIDEO_RES, VIDEO_RES)
        )

    def register_keypoints(self, init_keypoint_positions):
        self._init_keypoint_positions = np.asarray(
            init_keypoint_positions, dtype=np.float64
        )
        self._kp_to_body = self._autodetect_keypoint_to_body_all(
            self._init_keypoint_positions
        )

        if self.verbose:
            print(f"[adapter] registered {len(self._init_keypoint_positions)} keypoints")
            for kp_idx, body in self._kp_to_body.items():
                print(f"  kp{kp_idx} → {body}")

        cache = build_sdf_for_env(
            self.env, self.suite_name, self.task_idx,
            bounds_min=LIBERO_BOUNDS_MIN,
            bounds_max=LIBERO_BOUNDS_MAX,
            voxel_size=SDF_VOXEL_SIZE_BUILD,
        )
        self._sdf_full = cache["sdf_full"]
        self._sdf_per_body = cache["sdf_per_body"]

        # kp0 = held target, kp4 = a basket keypoint (kps 4–7 share the
        # basket body under nearest-body autodetection).
        can_body = self._kp_to_body.get(0)
        basket_body = self._kp_to_body.get(4)
        self._can_body_id = (
            self.sim.model.body_name2id(can_body) if can_body else None
        )
        self._basket_body_id = (
            self.sim.model.body_name2id(basket_body) if basket_body else None
        )

    def get_keypoint_positions(self):
        assert self._kp_to_body is not None, "call register_keypoints first"
        n = len(self._init_keypoint_positions)
        out = np.zeros((n, 3), dtype=np.float64)
        for kp_idx in range(n):
            body_name = self._kp_to_body[kp_idx]
            bid = self.sim.model.body_name2id(body_name)
            out[kp_idx] = self.sim.data.body_xpos[bid]
        return out

    def get_ee_pos(self):
        return self.sim.data.site_xpos[self.GRIP_SITE_ID].copy()

    def get_ee_pose(self):
        """[xyz + xyzw quat] in world frame."""
        pos = self.sim.data.site_xpos[self.GRIP_SITE_ID].copy()
        rot = self.sim.data.site_xmat[self.GRIP_SITE_ID].reshape(3, 3).copy()
        return np.concatenate([pos, T.mat2quat(rot)])

    def get_arm_joint_postions(self):
        """8-vec (7 arm + 1 gripper pad) to match reset_joint_pos shape.
        subgoal_solver's reset regularizer subtracts these slot-wise — keep
        them aligned or it crashes. ('postions' misspelling preserved to
        match main.py.)"""
        arm = self.sim.data.qpos[self.qpos_addr:self.qpos_addr + 7].copy()
        return np.append(arm, 0.0)

    def get_sdf_voxels(self, voxel_size, about_to_grasp_kp=None):
        """voxel_size IGNORED. Returns SDF with the appropriate target body
        excluded: held body if grasping, otherwise the about-to-grasp body
        (so solvers don't detour around the grasp target during approach)."""
        assert self._sdf_full is not None, "call register_keypoints first"
        if self._grasped_indices:
            held_kp = self._grasped_indices[0]
            held_body = self._kp_to_body.get(held_kp)
            if held_body in self._sdf_per_body:
                return self._sdf_per_body[held_body]
        if about_to_grasp_kp is not None:
            target_body = self._kp_to_body.get(about_to_grasp_kp)
            if target_body in self._sdf_per_body:
                return self._sdf_per_body[target_body]
        return self._sdf_full

    def get_collision_points(self):
        """None if nothing held; else N_COLLISION_POINTS uniform samples in
        the held body's AABB (world frame). path_solver2 will crash on None
        — callers must guard."""
        if not self._grasped_indices:
            return None
        held_kp = self._grasped_indices[0]
        held_body = self._kp_to_body.get(held_kp)
        if held_body is None:
            return None
        return self._sample_body_aabb_points(held_body, N_COLLISION_POINTS)

    def execute_action(self, action_8d, precise=False):
        """action_8d = [xyz, qx, qy, qz, qw, gripper_cmd] in world frame.
        precise=False: 1 OSC tick. precise=True: stream until <5 mm or 40 ticks."""
        target_xyzw = np.asarray(action_8d[:7], dtype=np.float64)
        gripper_cmd = float(action_8d[7])

        if not precise:
            self._osc_tick(target_xyzw, gripper_cmd)
            return

        for _ in range(SETTLE_MAX_TICKS):
            self._osc_tick(target_xyzw, gripper_cmd)
            err = float(np.linalg.norm(self.get_ee_pos() - target_xyzw[:3]))
            if err < SETTLE_POS_TOL:
                return

    def save_video(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        return self._video_path

    def open_gripper(self):
        for _ in range(OPEN_GRIPPER_TICKS):
            current = self.get_ee_pose()
            self._osc_tick(current, -1.0)

    def get_gripper_null_action(self):
        """Last commanded gripper value — keeps gripper continuity across waypoints."""
        return self._gripper_state

    def get_gripper_close_action(self):
        return +1.0

    def get_object_by_keypoint(self, i):
        """Returns the body name for keypoint i. The name IS the object handle."""
        assert self._kp_to_body is not None, "call register_keypoints first"
        return self._kp_to_body[i]

    def is_grasping(self, candidate_obj=None):
        """Bookkeeping only — NOT a physics check (won't fire on physical slip)."""
        if candidate_obj is None:
            return bool(self._grasped_indices)
        for kp in self._grasped_indices:
            if self._kp_to_body.get(kp) == candidate_obj:
                return True
        return False

    @property
    def reset_joint_pos(self):
        assert self._reset_joint_pos is not None, "call reset() first"
        return self._reset_joint_pos

    @property
    def world2robot_homo(self):
        assert self._world2robot_homo is not None, "call reset() first"
        return self._world2robot_homo

    def set_grasping(self, keypoint_idx):
        if keypoint_idx not in self._grasped_indices:
            self._grasped_indices.append(keypoint_idx)

    def clear_grasping(self, keypoint_idx):
        if keypoint_idx in self._grasped_indices:
            self._grasped_indices.remove(keypoint_idx)

    def get_finger_qpos(self):
        """(finger_joint1, finger_joint2). Open ≈ (+0.04, -0.04); closed ≈ (0, 0)."""
        assert self._finger_qpos_addrs is not None, "call reset() first"
        a1, a2 = self._finger_qpos_addrs
        return np.array([self.sim.data.qpos[a1], self.sim.data.qpos[a2]], dtype=np.float64)

    def mark_event(self, event_str):
        """Attach an event label to the most recently logged tick."""
        if self._tick_log:
            self._tick_log[-1]["event"] = event_str

    def set_log_stage(self, stage):
        self._log_stage = int(stage)

    def _osc_tick(self, target_xyzw, gripper_cmd):
        current = self.get_ee_pose()
        action_7d = pose_to_osc_action(target_xyzw, current, gripper_cmd)
        self.env.step(action_7d)
        self._gripper_state = float(gripper_cmd)
        self._step_counter += 1

        if self.disturbance_seq is not None:
            try:
                next(self.disturbance_seq)
            except StopIteration:
                pass

        if self._video_writer is not None and (self._step_counter % VIDEO_FRAME_STRIDE == 0):
            frame = self.sim.render(
                camera_name="agentview", width=VIDEO_RES, height=VIDEO_RES
            )[::-1]
            self._video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        nan3 = np.full(3, np.nan, dtype=np.float64)
        can_xpos = (
            self.sim.data.body_xpos[self._can_body_id].copy()
            if self._can_body_id is not None else nan3
        )
        basket_xpos = (
            self.sim.data.body_xpos[self._basket_body_id].copy()
            if self._basket_body_id is not None else nan3
        )
        self._tick_log.append({
            "tick":         self._step_counter,
            "stage":        int(self._log_stage),
            "target_pose":  np.asarray(target_xyzw, dtype=np.float64).copy(),
            "actual_pose":  self.get_ee_pose(),
            "gripper_cmd":  float(gripper_cmd),
            "finger_qpos":  self.get_finger_qpos(),
            "can_xpos":     can_xpos,
            "basket_xpos":  basket_xpos,
            "event":        None,
        })

    def _autodetect_keypoint_to_body_all(self, init_keypoints):
        """Nearest-body lookup. Body candidates: anything ending in '_1_main'
        (LIBERO's convention for primary object bodies)."""
        candidates = [
            self.sim.model.body_id2name(i)
            for i in range(self.sim.model.nbody)
            if self.sim.model.body_id2name(i)
            and self.sim.model.body_id2name(i).endswith("_1_main")
        ]
        mapping = {}
        for kp_idx, kp_pos in enumerate(init_keypoints):
            best_name, best_dist = None, np.inf
            for body_name in candidates:
                bid = self.sim.model.body_name2id(body_name)
                d = float(np.linalg.norm(self.sim.data.body_xpos[bid] - kp_pos))
                if d < best_dist:
                    best_name, best_dist = body_name, d
            mapping[kp_idx] = best_name
        return mapping

    def _sample_body_aabb_points(self, body_name, n_points):
        """Sample uniformly inside the union of geom AABBs of body_name.
        Treats each geom as a box; exact for LIBERO's box collision proxies,
        slightly conservative for visual meshes."""
        bid = self.sim.model.body_name2id(body_name)
        g_start = self.sim.model.body_geomadr[bid]
        g_num = self.sim.model.body_geomnum[bid]
        if g_num == 0:
            return None
        aabb_min_local = np.full(3, np.inf)
        aabb_max_local = np.full(3, -np.inf)
        for gid in range(g_start, g_start + g_num):
            gsize = self.sim.model.geom_size[gid]
            gpos = self.sim.model.geom_pos[gid]
            aabb_min_local = np.minimum(aabb_min_local, gpos - gsize)
            aabb_max_local = np.maximum(aabb_max_local, gpos + gsize)
        samples_local = np.random.uniform(
            aabb_min_local, aabb_max_local, size=(n_points, 3)
        )
        body_xpos = self.sim.data.body_xpos[bid]
        body_xmat = self.sim.data.body_xmat[bid].reshape(3, 3)
        return (body_xmat @ samples_local.T).T + body_xpos
