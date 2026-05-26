# FrankaEnvAdapter — drop-in for LIBEROReKepEnv against the laptop_bridge over ZMQ.
#
# Phase 3 scope: state subscriber, camera subscriber (raw cache only), action
# republisher, and the state/control methods main3 needs at runtime.
# Phase 5 stubs: perception methods (register_keypoints, get_keypoint_positions,
# get_sdf_voxels, get_collision_points, get_object_pose, save_video) raise
# NotImplementedError until tracker/SDF/GSAM2 land on sheep.
#
# Quirks copied from the LIBERO adapter for main3 compatibility:
#   - reset_joint_pos is an 8-vec (7 arm + 1 gripper pad).
#   - get_arm_joint_postions misspelling preserved.
#   - get_collision_points returns None when nothing held; path_solver2 crash
#     unless callers substitute.
#   - is_grasping is bookkeeping only until Phase 7 swaps in a width-based check.

import threading
import time

import msgpack
import numpy as np
import zmq


LAPTOP_IP = "10.105.240.232"
ROBOT_STATE_SUB_PORT = 5570
CAMERA_FRAMES_SUB_PORT = 5571
ACTION_TARGET_PUB_PORT = 5572

REPUBLISH_HZ = 20.0                  # keeps laptop bridge watchdog asleep (max_age_ms=200, factor=2)
DEFAULT_MAX_AGE_MS = 200

# OSC tracking on the real arm is soft (~4 mm/s vertical with Kp=150 in
# osc-pose-controller.yml). LIBERO's 5 mm tolerance is unrealistic here;
# 1 cm is the operational floor. Tune from end-to-end runs.
SETTLE_POS_TOL = 0.010
SETTLE_TIMEOUT_S = 15.0
SETTLE_POLL_PERIOD_S = 0.05

# Deoxys gripper open is ~0.08 m total width. Per-finger qpos in LIBERO is
# (+0.04 open, 0 closed); mirror that here.
GRIPPER_OPEN_WIDTH = 0.08

# Static home pose (joint vector), matches the calibration script the user ran
# on station. The 8th slot is the gripper pad expected by subgoal_solver's
# reset regularizer.
REAL_HOME_Q7 = np.array(
    [0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78], dtype=np.float64
)


class _StateSubscriber(threading.Thread):
    def __init__(self, laptop_ip):
        super().__init__(daemon=True)
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.connect(f"tcp://{laptop_ip}:{ROBOT_STATE_SUB_PORT}")
        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()

    def run(self):
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        while not self._stop.is_set():
            ev = dict(poller.poll(timeout=100))
            if self.sub in ev:
                try:
                    msg = msgpack.unpackb(self.sub.recv())
                    with self._lock:
                        self._latest = msg
                except Exception as e:
                    print(f"[franka_adapter] bad state msg: {e}")

    def get(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop.set()
        self.join(timeout=1.0)
        self.sub.close(linger=0)


class _CameraSubscriber(threading.Thread):
    # Stores latest (header, jpeg_bytes, depth_bytes) per cam_id. Decoding
    # is deferred to consumers so the SUB loop stays cheap.
    def __init__(self, laptop_ip):
        super().__init__(daemon=True)
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.set_hwm(20)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.connect(f"tcp://{laptop_ip}:{CAMERA_FRAMES_SUB_PORT}")
        self._lock = threading.Lock()
        self._frames = {}
        self._stop = threading.Event()

    def run(self):
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        while not self._stop.is_set():
            ev = dict(poller.poll(timeout=100))
            if self.sub in ev:
                try:
                    header_bytes, jpg_bytes, depth_bytes = self.sub.recv_multipart()
                    header = msgpack.unpackb(header_bytes)
                    with self._lock:
                        self._frames[header["cam_id"]] = (header, jpg_bytes, depth_bytes)
                except Exception as e:
                    print(f"[franka_adapter] bad frame msg: {e}")

    def get(self, cam_id):
        with self._lock:
            return self._frames.get(cam_id)

    def cam_ids(self):
        with self._lock:
            return list(self._frames.keys())

    def stop(self):
        self._stop.set()
        self.join(timeout=1.0)
        self.sub.close(linger=0)


class _ActionRepublisher(threading.Thread):
    # Holds the latest commanded target and republishes at REPUBLISH_HZ so the
    # bridge's staleness check stays fresh between main3's slower replan ticks.
    def __init__(self, laptop_ip):
        super().__init__(daemon=True)
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f"tcp://{laptop_ip}:{ACTION_TARGET_PUB_PORT}")
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._stop = threading.Event()

    def set_target(self, target_pose7, gripper_cmd, precise=False,
                   max_age_ms=DEFAULT_MAX_AGE_MS):
        with self._lock:
            self._latest = {
                "target_pose": [float(x) for x in target_pose7],
                "gripper_cmd": int(gripper_cmd),
                "precise":     bool(precise),
                "max_age_ms":  int(max_age_ms),
            }

    def clear_target(self):
        with self._lock:
            self._latest = None

    def run(self):
        period = 1.0 / REPUBLISH_HZ
        next_t = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                payload = self._latest
            if payload is not None:
                self._seq += 1
                msg = {"seq": self._seq, "t": time.monotonic(), **payload}
                self.pub.send(msgpack.packb(msg))
            next_t += period
            slack = next_t - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.monotonic()

    def stop(self):
        self._stop.set()
        self.join(timeout=1.0)
        self.pub.close(linger=0)


class FrankaEnvAdapter:
    """Drop-in for LIBEROReKepEnv on the real Franka via laptop_bridge ZMQ.
    Phase 3: state + action wired. Phase 5: perception methods become real."""

    def __init__(self, laptop_ip=LAPTOP_IP, verbose=False, wait_first_state_s=10.0):
        self.verbose = verbose
        self._state_sub = _StateSubscriber(laptop_ip)
        self._cam_sub = _CameraSubscriber(laptop_ip)
        self._action_pub = _ActionRepublisher(laptop_ip)
        self._state_sub.start()
        self._cam_sub.start()
        self._action_pub.start()

        if verbose:
            print(f"[franka_adapter] connecting to laptop @ {laptop_ip} "
                  f"(state {ROBOT_STATE_SUB_PORT}, cam {CAMERA_FRAMES_SUB_PORT}, "
                  f"action {ACTION_TARGET_PUB_PORT})")
        self._wait_first_state(wait_first_state_s)

        self._grasped_indices = []
        self._gripper_state = -1.0
        self._init_keypoint_positions = None
        self._log_stage = 0
        self._tick_log = []

        # main3 reads this attribute (set in LIBERO adapter for test-time
        # disturbances). Keep the name; real-world disturbances would be a
        # physical event, not a generator.
        self.disturbance_seq = None

    def _wait_first_state(self, timeout_s):
        t0 = time.monotonic()
        while self._state_sub.get() is None:
            if time.monotonic() - t0 > timeout_s:
                raise RuntimeError(
                    f"No state from laptop_bridge after {timeout_s:.1f}s. "
                    f"Is franka_bridge.py running on the laptop, and is "
                    f"auto_arm.sh up on station?"
                )
            time.sleep(0.1)

    def close(self):
        self._action_pub.clear_target()
        self._state_sub.stop()
        self._cam_sub.stop()
        self._action_pub.stop()

    # ---------- robot state ----------
    def get_ee_pose(self):
        s = self._state_sub.get()
        return np.asarray(s["eef_pose"], dtype=np.float64)

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_arm_joint_postions(self):
        s = self._state_sub.get()
        return np.append(np.asarray(s["q"], dtype=np.float64), 0.0)

    def get_finger_qpos(self):
        # Bridge sends gripper_width (m). LIBERO convention: per-finger qpos
        # +/-(width/2). finger_force is always 0 from the bridge.
        s = self._state_sub.get()
        w = float(s["gripper_width"])
        a = w / 2.0
        return np.array([a, -a], dtype=np.float64)

    def get_gripper_null_action(self):
        return float(self._gripper_state)

    def get_gripper_close_action(self):
        return +1.0

    @property
    def reset_joint_pos(self):
        return np.append(REAL_HOME_Q7, 0.0)

    @property
    def world2robot_homo(self):
        # Real planning happens in robot base frame; world == base.
        return np.eye(4)

    # ---------- control ----------
    def execute_action(self, action_8d, precise=False):
        """action_8d = [xyz, qx,qy,qz,qw, gripper_cmd]. Real-arm semantics:
        publish target into the republisher (laptop bridge consumes at 20Hz).
        precise=True polls live EE pose until within SETTLE_POS_TOL or timeout."""
        arr = np.asarray(action_8d, dtype=np.float64)
        target_pose = arr[:7]
        raw_g = float(arr[7])
        # Bridge expects -1 / 0 / +1; treat near-zero as no-op.
        if abs(raw_g) < 1e-3:
            gripper_cmd = 0
        else:
            gripper_cmd = 1 if raw_g > 0 else -1
        self._gripper_state = raw_g

        self._action_pub.set_target(target_pose, gripper_cmd, precise=precise)

        if not precise:
            return

        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while time.monotonic() < deadline:
            err = float(np.linalg.norm(self.get_ee_pos() - target_pose[:3]))
            if err < SETTLE_POS_TOL:
                return
            time.sleep(SETTLE_POLL_PERIOD_S)
        if self.verbose:
            err = float(np.linalg.norm(self.get_ee_pos() - target_pose[:3]))
            print(f"[franka_adapter] precise timeout: {err*1000:.1f} mm "
                  f"after {SETTLE_TIMEOUT_S:.0f} s")

    def open_gripper(self):
        current = self.get_ee_pose()
        self._action_pub.set_target(current, gripper_cmd=-1, precise=False)
        self._gripper_state = -1.0
        # Gripper transition latency on real Deoxys is ~0.5 s.
        time.sleep(0.5)

    # ---------- camera frames (raw cache; decoding is consumer's job) ----------
    def get_raw_frame(self, cam_id):
        """Returns (header_dict, jpeg_bytes, depth_bytes) or None.
        Decode RGB with cv2.imdecode, depth with np.frombuffer(..., dtype=uint16)."""
        return self._cam_sub.get(cam_id)

    def active_cam_ids(self):
        return self._cam_sub.cam_ids()

    # ---------- grasp bookkeeping ----------
    def is_grasping(self, candidate_obj=None):
        # Phase 7 will replace with gripper_width threshold + duration check.
        if candidate_obj is None:
            return bool(self._grasped_indices)
        return False

    def set_grasping(self, keypoint_idx):
        if keypoint_idx not in self._grasped_indices:
            self._grasped_indices.append(keypoint_idx)

    def clear_grasping(self, keypoint_idx):
        if keypoint_idx in self._grasped_indices:
            self._grasped_indices.remove(keypoint_idx)

    def get_grasped_keypoints(self):
        return list(self._grasped_indices)

    # ---------- logging ----------
    def mark_event(self, event_str):
        if self._tick_log:
            self._tick_log[-1]["event"] = event_str

    def set_log_stage(self, stage):
        self._log_stage = int(stage)

    # ---------- perception (Phase 5) ----------
    def register_keypoints(self, init_keypoint_positions):
        self._init_keypoint_positions = np.asarray(
            init_keypoint_positions, dtype=np.float64
        )

    def get_keypoint_positions(self):
        raise NotImplementedError(
            "Phase 5: keypoint tracker on sheep not yet wired up."
        )

    def get_object_by_keypoint(self, i):
        raise NotImplementedError("Phase 5: label-by-keypoint not yet wired up.")

    def get_object_pose(self, keypoint_idx):
        raise NotImplementedError("Phase 5: object pose via tracker not wired up.")

    def get_sdf_voxels(self, voxel_size, about_to_grasp_kp=None):
        raise NotImplementedError("Phase 4 stub or Phase 5 real SDF.")

    def get_collision_points(self):
        # main3 expects None when nothing held; preserve that semantics.
        return None

    def save_video(self):
        raise NotImplementedError("Phase 5: stream recorder.")
