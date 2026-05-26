# wraps LIBERO's OffScreenRenderEnv so main3 sees the same interface ReKepOGEnv
# offers in the upstream rekep repo, with two extra methods (set_grasping /
# clear_grasping) for grasp bookkeeping.
#
# things to know:
#   - SDF is built once in register_keypoints. get_sdf_voxels' voxel_size arg
#     is ignored (it's just there to match the upstream signature).
#   - get_collision_points returns None when nothing is held. path_solver2 has
#     no None guard — callers must substitute a placeholder.
#   - is_grasping is bookkeeping only — it tracks set/clear calls, not physics,
#     so a physical slip won't update it.

import os
import numpy as np
import cv2

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from v2.common import transform_utils as T
from v2.sim.build_sdf import build_sdf_for_env
from v2.sim.osc_helpers import pose_to_osc_action


# workspace cube the SDF is built over (LIBERO scenes fit comfortably inside).
LIBERO_BOUNDS_MIN = np.array([-0.40, -0.50, 0.005])
LIBERO_BOUNDS_MAX = np.array([ 0.70,  0.50, 0.60])

# 2 cm SDF voxels — coarse enough to be fast, fine enough for collision queries.
SDF_VOXEL_SIZE_BUILD = 0.02

# OSC controller runs at 20 Hz. "precise" execution streams until error
# drops below 5 mm or we run out the budget.
OSC_CONTROL_HZ = 20
SETTLE_POS_TOL = 0.005
SETTLE_MAX_TICKS = 40

# how long to hold "gripper open" when opening (40 ticks @ 20 Hz = 2 s).
OPEN_GRIPPER_TICKS = 40

# video output: stride 2 means we save every other rendered frame.
# fps = 20/2 = 10 → playback duration matches wall-clock duration.
VIDEO_FRAME_STRIDE = 2
VIDEO_FPS = 10
VIDEO_RES = 480

# how many sample points to use as a stand-in for the held object's volume.
N_COLLISION_POINTS = 60


class LIBEROReKepEnv:
    # drop-in replacement for ReKepOGEnv from main.py's perspective.

    def __init__(self, suite_name, task_idx, video_path=None, verbose=False):
        self.suite_name = suite_name
        self.task_idx = task_idx
        self.verbose = verbose

        # build the LIBERO offscreen renderer for this task.
        bench = benchmark.get_benchmark_dict()[suite_name]()
        bddl = bench.get_task_bddl_file_path(task_idx)
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=VIDEO_RES,
            camera_widths=VIDEO_RES,
            camera_depths=False,
        )

        # robosuite swaps out MjSim on every reset, so any sim handle cached
        # here would dangle. we re-acquire these in reset().
        self.sim = None
        self.GRIP_SITE_ID = None
        self.ROBOT_BASE_BODY_ID = None
        self.qpos_addr = None

        # default video output path under outputs/v2_<suite>_<task>/.
        if video_path is None:
            video_path = os.path.expanduser(
                f"~/libero_keypoint_project/outputs/v2_{suite_name}_{task_idx}/main2_task.mp4"
            )
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        self._video_path = video_path
        self._video_writer = None

        # runtime state.
        self._step_counter = 0
        self._gripper_state = -1.0     # -1 = open at boot
        self._grasped_indices = []
        self._reset_joint_pos = None
        self._world2robot_homo = None

        # keypoint registration state (filled in by register_keypoints).
        self._init_keypoint_positions = None
        self._kp_to_body = None

        # SDFs (filled in by register_keypoints).
        self._sdf_full = None
        self._sdf_per_body = None

        # per-tick log used for plots / debugging.
        self._tick_log = []
        self._log_stage = 0
        self._finger_qpos_addrs = None

        # convenience cache for the two scene bodies we care about in
        # libero_object tasks (the can and the basket). filled in by
        # register_keypoints once we know the kp→body mapping.
        self._can_body_id = None
        self._basket_body_id = None

        # test-time disturbance generator (None outside tests).
        self.disturbance_seq = None

        # main3 reads reset_joint_pos and world2robot_homo before it ever
        # calls reset(), so bootstrap a reset here so those properties exist.
        self.reset()

    def reset(self):
        self.env.reset()

        # rebind everything that depends on the (now new) MjSim instance.
        self.sim = self.env.env.sim
        self.GRIP_SITE_ID = self.sim.model.site_name2id("gripper0_grip_site")
        self.ROBOT_BASE_BODY_ID = self.sim.model.body_name2id("robot0_base")

        # joint1's qpos address; the next 6 slots are joint2..joint7.
        qa = self.sim.model.get_joint_qpos_addr("robot0_joint1")
        if isinstance(qa, tuple):
            qa = qa[0]
        self.qpos_addr = qa

        # 7 arm joint angles + 1 placeholder for the gripper.
        arm_qpos = self.sim.data.qpos[self.qpos_addr:self.qpos_addr + 7].copy()
        self._reset_joint_pos = np.append(arm_qpos, 0.0)

        # finger joints: joint1 ∈ [0, 0.04], joint2 ∈ [-0.04, 0].
        # open ≈ (0.04, -0.04), closed ≈ (0, 0).
        fa1 = self.sim.model.get_joint_qpos_addr("gripper0_finger_joint1")
        fa2 = self.sim.model.get_joint_qpos_addr("gripper0_finger_joint2")
        if isinstance(fa1, tuple): fa1 = fa1[0]
        if isinstance(fa2, tuple): fa2 = fa2[0]
        self._finger_qpos_addrs = (fa1, fa2)

        self._tick_log = []
        self._can_body_id = None
        self._basket_body_id = None

        # world -> robot-base homogeneous transform (inverse of base pose in world).
        R_wb = self.sim.data.body_xmat[self.ROBOT_BASE_BODY_ID].reshape(3, 3).copy()
        p_wb = self.sim.data.body_xpos[self.ROBOT_BASE_BODY_ID].copy()
        H = np.eye(4)
        H[:3, :3] = R_wb.T
        H[:3, 3] = -R_wb.T @ p_wb
        self._world2robot_homo = H

        self._step_counter = 0
        self._gripper_state = -1.0
        self._grasped_indices = []

        # start a fresh mp4 file.
        if self._video_writer is not None:
            self._video_writer.release()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(
            self._video_path, fourcc, VIDEO_FPS, (VIDEO_RES, VIDEO_RES),
        )

    def register_keypoints(self, init_keypoint_positions):
        # called once after keypoints are proposed. records initial positions,
        # figures out which body each keypoint belongs to (nearest-body lookup),
        # and builds the SDF.
        self._init_keypoint_positions = np.asarray(
            init_keypoint_positions, dtype=np.float64,
        )
        self._kp_to_body = self._autodetect_keypoint_to_body_all(
            self._init_keypoint_positions
        )

        if self.verbose:
            print(f"[adapter] registered {len(self._init_keypoint_positions)} keypoints")
            for kp_idx, body in self._kp_to_body.items():
                print(f"  kp{kp_idx} → {body}")

        # build a full-scene SDF + one-per-body SDFs (with one body removed).
        cache = build_sdf_for_env(
            self.env, self.suite_name, self.task_idx,
            bounds_min=LIBERO_BOUNDS_MIN,
            bounds_max=LIBERO_BOUNDS_MAX,
            voxel_size=SDF_VOXEL_SIZE_BUILD,
        )
        self._sdf_full = cache["sdf_full"]
        self._sdf_per_body = cache["sdf_per_body"]

        # convention used elsewhere: kp0 is the held target, kps 4-7 are the
        # basket (all map to the basket body via nearest-body autodetect).
        can_body = self._kp_to_body.get(0)
        basket_body = self._kp_to_body.get(4)
        self._can_body_id = (
            self.sim.model.body_name2id(can_body) if can_body else None
        )
        self._basket_body_id = (
            self.sim.model.body_name2id(basket_body) if basket_body else None
        )

    def get_keypoint_positions(self):
        # returns current world positions of all registered keypoints by
        # reading the body each keypoint was bound to.
        assert self._kp_to_body is not None, "call register_keypoints first"
        n = len(self._init_keypoint_positions)
        out = np.zeros((n, 3), dtype=np.float64)
        for kp_idx in range(n):
            body_name = self._kp_to_body[kp_idx]
            bid = self.sim.model.body_name2id(body_name)
            out[kp_idx] = self.sim.data.body_xpos[bid]
        return out

    def get_ee_pos(self):
        # world-frame end-effector position.
        return self.sim.data.site_xpos[self.GRIP_SITE_ID].copy()

    def get_ee_pose(self):
        # 7-vec [x, y, z, qx, qy, qz, qw] in world frame.
        pos = self.sim.data.site_xpos[self.GRIP_SITE_ID].copy()
        rot = self.sim.data.site_xmat[self.GRIP_SITE_ID].reshape(3, 3).copy()
        return np.concatenate([pos, T.mat2quat(rot)])

    def get_arm_joint_postions(self):
        # 8-vec (7 arm + 1 gripper pad) so the shape matches reset_joint_pos.
        # ('postions' misspelling preserved to match main.py's call site.)
        arm = self.sim.data.qpos[self.qpos_addr:self.qpos_addr + 7].copy()
        return np.append(arm, 0.0)

    def get_sdf_voxels(self, voxel_size, about_to_grasp_kp=None):
        # voxel_size is ignored — we built the SDF at SDF_VOXEL_SIZE_BUILD already.
        # if we're currently holding something, return the SDF with that body removed.
        # if we're about to grasp something, also remove that body so the solver
        # doesn't detour around its own grasp target.
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
        # represents the held object's volume as N points sampled in its
        # axis-aligned bounding box. returns None if nothing held — callers
        # must guard, path_solver2 will crash on None.
        if not self._grasped_indices:
            return None
        held_kp = self._grasped_indices[0]
        held_body = self._kp_to_body.get(held_kp)
        if held_body is None:
            return None
        return self._sample_body_aabb_points(held_body, N_COLLISION_POINTS)

    def execute_action(self, action_8d, precise=False):
        # action_8d = [x, y, z, qx, qy, qz, qw, gripper] in world frame.
        # precise=False: 1 OSC tick.
        # precise=True:  stream OSC ticks until error < 5 mm or 40 ticks pass.
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
        # close the writer and return the path of the saved mp4.
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        return self._video_path

    def open_gripper(self):
        # hold "open" for OPEN_GRIPPER_TICKS while staying at the current pose.
        for _ in range(OPEN_GRIPPER_TICKS):
            current = self.get_ee_pose()
            self._osc_tick(current, -1.0)

    def get_gripper_null_action(self):
        # last commanded gripper value — keeps the gripper state continuous
        # when stitching consecutive waypoints together.
        return self._gripper_state

    def get_gripper_close_action(self):
        return +1.0

    def get_object_by_keypoint(self, i):
        # body name acts as the object handle elsewhere in main3.
        assert self._kp_to_body is not None, "call register_keypoints first"
        return self._kp_to_body[i]

    def is_grasping(self, candidate_obj=None):
        # bookkeeping only — not a physics check, won't fire on physical slip.
        if candidate_obj is None:
            return bool(self._grasped_indices)
        for kp in self._grasped_indices:
            if self._kp_to_body.get(kp) == candidate_obj:
                return True
        return False

    def get_grasped_keypoints(self):
        return list(self._grasped_indices)

    def get_object_pose(self, keypoint_idx):
        # 3D world position of the body bound to this keypoint, or None.
        if self._kp_to_body is None:
            return None
        body_name = self._kp_to_body.get(keypoint_idx)
        if body_name is None:
            return None
        bid = self.sim.model.body_name2id(body_name)
        return self.sim.data.body_xpos[bid].copy()

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
        # (finger_joint1, finger_joint2). open ≈ (+0.04, -0.04), closed ≈ (0, 0).
        assert self._finger_qpos_addrs is not None, "call reset() first"
        a1, a2 = self._finger_qpos_addrs
        return np.array([self.sim.data.qpos[a1], self.sim.data.qpos[a2]], dtype=np.float64)

    def mark_event(self, event_str):
        # attach a label to the most recent tick log entry.
        if self._tick_log:
            self._tick_log[-1]["event"] = event_str

    def set_log_stage(self, stage):
        self._log_stage = int(stage)

    def _osc_tick(self, target_xyzw, gripper_cmd):
        # one tick: convert (target, current) into an OSC action, step the env,
        # advance the disturbance generator, record a video frame, log the tick.
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

        # write a video frame every VIDEO_FRAME_STRIDE ticks.
        if self._video_writer is not None and (self._step_counter % VIDEO_FRAME_STRIDE == 0):
            frame = self.sim.render(
                camera_name="agentview", width=VIDEO_RES, height=VIDEO_RES,
            )[::-1]
            self._video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # record this tick. NaNs for can/basket if those bodies aren't bound yet.
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
        # for each keypoint, find the body whose world position is closest to
        # the keypoint and bind them. body candidates are anything ending in
        # "_1_main" (LIBERO's naming for primary object bodies).
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
        # uniformly sample n_points inside the body's axis-aligned bounding box
        # (union of its geom AABBs), then transform from body-local to world.
        # exact for LIBERO's box collision proxies; slightly conservative for meshes.
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
            aabb_min_local, aabb_max_local, size=(n_points, 3),
        )
        body_xpos = self.sim.data.body_xpos[bid]
        body_xmat = self.sim.data.body_xmat[bid].reshape(3, 3)
        return (body_xmat @ samples_local.T).T + body_xpos
