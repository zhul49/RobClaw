# Motion smoke. Requires franka_bridge.py running AND e-stop within reach.
#
# Steps:
#   1. Read current state from robot_state PUB.
#   2. Publish ONE action_target = current pose. Watchdog should NOT fire.
#      Arm should hold still.
#   3. Publish a slightly-shifted target (z += 2 cm). Arm should track.
#   4. Go silent. After ~400ms, bridge should log WATCHDOG hold.
#   5. Return to original pose, then exit.
import argparse
import os
import sys
import time

import msgpack
import numpy as np
import zmq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from laptop_bridge import config

DEFAULT_SHIFT_M = 0.02   # 2 cm
DEFAULT_SETTLE_S = 2.0
DEFAULT_STEP2_S = 3.0
DEFAULT_STEP4_S = 3.0


def read_state(state_sub, poller, timeout_s=2.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        ev = dict(poller.poll(timeout=200))
        if state_sub in ev:
            return msgpack.unpackb(state_sub.recv())
    return None


def publish_target_burst(pub, target_pose, gripper_cmd, seq_start, duration_s, hz=20):
    period = 1.0 / hz
    seq = seq_start
    t0 = time.monotonic()
    next_t = t0
    while time.monotonic() - t0 < duration_s:
        seq += 1
        msg = {
            "seq": seq,
            "t": time.monotonic(),
            "target_pose": list(target_pose),
            "gripper_cmd": int(gripper_cmd),
            "precise": False,
            "max_age_ms": config.ACTION_DEFAULT_MAX_AGE_MS,
        }
        pub.send(msgpack.packb(msg))
        next_t += period
        slack = next_t - time.monotonic()
        if slack > 0:
            time.sleep(slack)
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift-m", type=float, default=DEFAULT_SHIFT_M, help="z shift in meters")
    ap.add_argument("--step1-s", type=float, default=DEFAULT_SETTLE_S, help="step 1 hold duration")
    ap.add_argument("--step2-s", type=float, default=DEFAULT_STEP2_S, help="step 2 rise duration")
    ap.add_argument("--step4-s", type=float, default=DEFAULT_STEP4_S, help="step 4 return duration")
    args = ap.parse_args()
    shift_m = args.shift_m

    ctx = zmq.Context.instance()
    state_sub = ctx.socket(zmq.SUB)
    state_sub.setsockopt(zmq.CONFLATE, 1)
    state_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    state_sub.connect(f"{config.LOCAL_ADDR}:{config.ROBOT_STATE_PUB_PORT}")
    poller = zmq.Poller()
    poller.register(state_sub, zmq.POLLIN)

    action_pub = ctx.socket(zmq.PUB)
    action_pub.connect(f"{config.LOCAL_ADDR}:{config.ACTION_TARGET_SUB_PORT}")
    time.sleep(0.5)  # let PUB/SUB handshake

    print("[smoke04] reading current robot state…")
    s = read_state(state_sub, poller, timeout_s=5.0)
    if s is None:
        print("[smoke04] FAIL: no state from franka_bridge. Is it running?")
        return
    home_pose = list(s["eef_pose"])
    print(f"[smoke04] home pose = {np.round(home_pose, 4)}")

    print(f"[smoke04] STEP 1: publish target = home for {args.step1_s}s (expect: no motion, no watchdog)")
    seq = publish_target_burst(action_pub, home_pose, gripper_cmd=0, seq_start=0, duration_s=args.step1_s)
    s_after = read_state(state_sub, poller)
    delta = np.linalg.norm(np.array(s_after["eef_pose"][:3]) - np.array(home_pose[:3]))
    print(f"[smoke04]   pos drift after step1: {delta*1000:.1f} mm  (should be < 5 mm)")

    shifted = list(home_pose)
    shifted[2] += shift_m
    print(f"[smoke04] STEP 2: publish target = home + z={shift_m*1000:.0f}mm for {args.step2_s}s (expect: arm rises)")
    seq = publish_target_burst(action_pub, shifted, gripper_cmd=0, seq_start=seq, duration_s=args.step2_s)
    s_top = read_state(state_sub, poller)
    rise = s_top["eef_pose"][2] - home_pose[2]
    print(f"[smoke04]   z rise: {rise*1000:.1f} mm  (target {shift_m*1000:.0f} mm)")

    print("[smoke04] STEP 3: go silent for 1.5s (expect: WATCHDOG hold logs in bridge)")
    time.sleep(1.5)

    print(f"[smoke04] STEP 4: publish target = home for {args.step4_s}s (expect: arm returns)")
    seq = publish_target_burst(action_pub, home_pose, gripper_cmd=0, seq_start=seq, duration_s=args.step4_s)
    s_back = read_state(state_sub, poller)
    ret = np.linalg.norm(np.array(s_back["eef_pose"][:3]) - np.array(home_pose[:3]))
    print(f"[smoke04]   pos error after return: {ret*1000:.1f} mm  (should be small)")

    print("[smoke04] DONE. Check bridge log for WATCHDOG line during step 3.")
    state_sub.close(linger=0)
    action_pub.close(linger=0)


if __name__ == "__main__":
    main()
