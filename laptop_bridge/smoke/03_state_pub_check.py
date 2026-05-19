# Subscribe to franka_bridge state PUB, sample ~50Hz for 10s, sanity-check.
# Run AFTER franka_bridge.py is running:
#   python laptop_bridge/smoke/03_state_pub_check.py
import os
import sys
import time

import msgpack
import numpy as np
import zmq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from laptop_bridge import config

DURATION_S = 10.0


def main():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"{config.LOCAL_ADDR}:{config.ROBOT_STATE_PUB_PORT}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    msgs = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < DURATION_S:
        ev = dict(poller.poll(timeout=200))
        if sub not in ev:
            continue
        payload = msgpack.unpackb(sub.recv())
        msgs.append(payload)

    if not msgs:
        print("[smoke03] FAIL: no messages received")
        return
    rate = len(msgs) / DURATION_S
    q = np.array(msgs[-1]["q"])
    eef = np.array(msgs[-1]["eef_pose"])
    qnorm = float(np.linalg.norm(np.diff([m["q"] for m in msgs], axis=0)))
    print(f"[smoke03] received {len(msgs)} msgs in {DURATION_S:.1f}s ({rate:.1f} Hz)")
    print(f"[smoke03] last q       = {np.round(q, 4)}")
    print(f"[smoke03] last eef pose = {np.round(eef, 4)}  (xyz + xyzw)")
    print(f"[smoke03] last gripper width = {msgs[-1]['gripper_width']:.4f} m")
    print(f"[smoke03] q sample-to-sample L2 sum = {qnorm:.4f} (≈ 0 if arm still)")
    if rate < 40:
        print(f"[smoke03] WARN: rate {rate:.1f} Hz is below 40 Hz target")
    else:
        print("[smoke03] OK")
    sub.close(linger=0)


if __name__ == "__main__":
    main()
