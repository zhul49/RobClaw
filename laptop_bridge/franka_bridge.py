# Deoxys client + state PUB + action SUB + watchdog.
import os
import signal
import sys
import threading
import time

import msgpack
import numpy as np
import zmq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from laptop_bridge import config

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils


def _open_robot():
    # has_gripper=False: we manage the gripper manually so control() doesn't
    # spam gripper_control() every tick. automatic_gripper_reset=False keeps
    # preprocess() from twitching the gripper on every controller switch.
    return FrankaInterface(
        os.path.join(config_root, config.DEOXYS_INTERFACE_CFG),
        use_visualizer=False,
        has_gripper=False,
        automatic_gripper_reset=False,
    )


def _eef_quat_and_pos(robot):
    quat, pos = robot.last_eef_quat_and_pos
    if quat is None or pos is None:
        return None, None
    return np.asarray(quat).flatten(), np.asarray(pos).flatten()


def _gripper_state(robot):
    width = robot.last_gripper_q
    width_f = float(width) if width is not None else 0.0
    last_a = getattr(robot, "last_gripper_action", 0.0)
    state = -1 if last_a < 0.0 else (1 if last_a > 0.0 else 0)
    return width_f, 0.0, state


class ActionSubscriber(threading.Thread):
    # Receives action_target msgs; stores latest only (latest-wins).
    def __init__(self):
        super().__init__(daemon=True)
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.set_hwm(10)
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.bind(f"{config.BIND_ADDR}:{config.ACTION_TARGET_SUB_PORT}")
        self.lock = threading.Lock()
        self.latest = None
        self.stop_flag = threading.Event()

    def run(self):
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        while not self.stop_flag.is_set():
            ev = dict(poller.poll(timeout=100))
            if self.sub not in ev:
                continue
            try:
                msg = msgpack.unpackb(self.sub.recv())
                msg["_recv_t"] = time.monotonic()
                with self.lock:
                    self.latest = msg
            except Exception as e:
                print(f"[franka_bridge] bad action msg: {e}")

    def get_latest(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=1.0)
        self.sub.close(linger=0)


class ControlLoop(threading.Thread):
    # Sole caller of robot_interface.control(). 20 Hz.
    def __init__(self, robot, action_sub):
        super().__init__(daemon=True)
        self.robot = robot
        self.action_sub = action_sub
        self.osc_cfg = YamlConfig(os.path.join(config_root, config.DEOXYS_OSC_POSE_CFG)).as_easydict()
        self.joint_cfg = YamlConfig(os.path.join(config_root, config.DEOXYS_JOINT_POS_CFG)).as_easydict()
        self.stop_flag = threading.Event()
        self.last_applied_gripper = 0  # -1 open, 0 unknown, +1 close
        self.was_stale = True
        self.last_warn_t = 0.0

    def _compute_osc_action(self, target_pose):
        target_pos = np.asarray(target_pose[:3], dtype=np.float64)
        target_quat = np.asarray(target_pose[3:7], dtype=np.float64)
        current_pose = self.robot.last_eef_pose
        current_pos = current_pose[:3, 3:].flatten()
        current_quat = transform_utils.mat2quat(current_pose[:3, :3])
        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        action_pos = (target_pos - current_pos) * config.POS_GAIN
        quat_diff = transform_utils.quat_distance(target_quat, current_quat)
        action_aa = transform_utils.quat2axisangle(quat_diff) * config.ROT_GAIN
        action_pos = np.clip(action_pos, -config.POS_CLIP, config.POS_CLIP)
        action_aa = np.clip(action_aa, -config.ROT_CLIP, config.ROT_CLIP)
        # Gripper slot ignored (has_gripper=False); managed via _maybe_apply_gripper.
        return action_pos.tolist() + action_aa.tolist() + [0.0]

    def _hold_action(self):
        q = self.robot.last_q
        if q is None:
            return None
        return q.tolist() + [0.0]

    def _maybe_apply_gripper(self, gripper_cmd):
        if gripper_cmd == 0 or gripper_cmd == self.last_applied_gripper:
            return
        self.robot.gripper_control(-1.0 if gripper_cmd < 0 else 1.0)
        self.last_applied_gripper = gripper_cmd
        print(f"[franka_bridge] gripper -> {'open' if gripper_cmd < 0 else 'close'}")

    def run(self):
        period = 1.0 / config.CONTROL_HZ
        next_t = time.monotonic()
        tick = 0
        while not self.stop_flag.is_set():
            tick += 1
            msg = self.action_sub.get_latest()
            stale = True
            if msg is not None:
                max_age_ms = int(msg.get("max_age_ms", config.ACTION_DEFAULT_MAX_AGE_MS))
                age_ms = (time.monotonic() - msg["_recv_t"]) * 1000.0
                stale = age_ms > max_age_ms * config.ACTION_STALE_FACTOR

            if stale:
                hold = self._hold_action()
                if hold is not None:
                    self.robot.control(
                        controller_type="JOINT_POSITION",
                        action=hold,
                        controller_cfg=self.joint_cfg,
                    )
                if not self.was_stale:
                    print(f"[franka_bridge] HOLD enter (reason={'no_msg' if msg is None else f'age={age_ms:.0f}ms'})")
                    self.last_warn_t = time.monotonic()
                elif time.monotonic() - self.last_warn_t > 1.0:
                    reason = "no_msg_yet" if msg is None else f"age_ms={age_ms:.0f}"
                    print(f"[franka_bridge] WATCHDOG hold ({reason})")
                    self.last_warn_t = time.monotonic()
                self.was_stale = True
            else:
                action = self._compute_osc_action(msg["target_pose"])
                self.robot.control(
                    controller_type="OSC_POSE",
                    action=action,
                    controller_cfg=self.osc_cfg,
                )
                self._maybe_apply_gripper(int(msg.get("gripper_cmd", 0)))
                if self.was_stale:
                    print(f"[franka_bridge] OSC enter; action={np.round(action, 4).tolist()}")
                elif tick % 20 == 0:
                    cp = self.robot.last_eef_pose[:3, 3:].flatten()
                    tp = np.asarray(msg['target_pose'][:3])
                    print(f"[franka_bridge] OSC tick={tick} delta_xyz={np.round(tp-cp, 4).tolist()} action={np.round(action[:6], 3).tolist()}")
                self.was_stale = False

            next_t += period
            slack = next_t - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.monotonic()

    def stop(self):
        self.stop_flag.set()


class StatePublisher(threading.Thread):
    def __init__(self, robot):
        super().__init__(daemon=True)
        self.robot = robot
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.set_hwm(50)
        self.pub.bind(f"{config.BIND_ADDR}:{config.ROBOT_STATE_PUB_PORT}")
        self.stop_flag = threading.Event()
        self.seq = 0

    def run(self):
        period = 1.0 / config.STATE_PUB_HZ
        next_t = time.monotonic()
        while not self.stop_flag.is_set():
            q = self.robot.last_q
            quat, pos = _eef_quat_and_pos(self.robot)
            if q is None or quat is None:
                time.sleep(0.01)
                continue
            width, finger_force, gstate = _gripper_state(self.robot)
            self.seq += 1
            payload = {
                "seq": self.seq,
                "t": time.monotonic(),
                "q": np.asarray(q, dtype=np.float64).tolist(),
                "eef_pose": pos.tolist() + quat.tolist(),
                "gripper_width": width,
                "finger_force": finger_force,
                "gripper_state": gstate,
            }
            self.pub.send(msgpack.packb(payload, use_single_float=False))
            next_t += period
            slack = next_t - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.monotonic()

    def stop(self):
        self.stop_flag.set()
        self.pub.close(linger=0)


def main():
    print("[franka_bridge] opening Deoxys interface...")
    robot = _open_robot()
    print("[franka_bridge] waiting for first state...")
    t0 = time.monotonic()
    while robot.state_buffer_size == 0:
        if time.monotonic() - t0 > 10.0:
            print("[franka_bridge] FAIL: no state from franka-interface in 10s")
            print("[franka_bridge] is auto_arm.sh running on station? is PC.IP correct?")
            sys.exit(1)
        time.sleep(0.1)
    print(f"[franka_bridge] first state received. q={robot.last_q}")

    pub = StatePublisher(robot)
    asub = ActionSubscriber()
    ctrl = ControlLoop(robot, asub)

    def _shutdown(signum, frame):
        print("[franka_bridge] shutting down")
        ctrl.stop()
        asub.stop()
        pub.stop()
        try:
            robot.close()
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    pub.start()
    asub.start()
    ctrl.start()
    print(f"[franka_bridge] state PUB on tcp://*:{config.ROBOT_STATE_PUB_PORT} @ {config.STATE_PUB_HZ} Hz")
    print(f"[franka_bridge] action SUB on tcp://*:{config.ACTION_TARGET_SUB_PORT}")
    print(f"[franka_bridge] control loop @ {config.CONTROL_HZ} Hz (watchdog hold via JOINT_POSITION)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
