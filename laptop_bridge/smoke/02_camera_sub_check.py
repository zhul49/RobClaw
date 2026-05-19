# Subscribe to camera_bridge locally, decode one frame per cam_id, save to /tmp.
# Run in second terminal AFTER camera_bridge.py / smoke 01 is running:
#   python laptop_bridge/smoke/02_camera_sub_check.py
import os
import sys
import time

import cv2
import msgpack
import numpy as np
import zmq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from laptop_bridge import config

TIMEOUT_S = 10.0
WANTED = set(k for k, v in config.CAMERA_SERIALS.items() if v)


def main():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"{config.LOCAL_ADDR}:{config.CAMERA_FRAMES_PUB_PORT}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    seen = {}
    t0 = time.monotonic()
    while WANTED - seen.keys() and time.monotonic() - t0 < TIMEOUT_S:
        ev = dict(poller.poll(timeout=500))
        if sub not in ev:
            continue
        header_bytes, jpg_bytes, depth_bytes = sub.recv_multipart()
        h = msgpack.unpackb(header_bytes)
        cam_id = h["cam_id"]
        if cam_id in seen:
            continue
        rgb = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        depth = np.frombuffer(depth_bytes, np.uint16).reshape(h["height"], h["width"])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"/tmp/cam_{cam_id}_rgb.png", bgr)
        depth_vis = (np.clip(depth.astype(np.float32) / 4000.0, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(f"/tmp/cam_{cam_id}_depth.png", depth_vis)
        seen[cam_id] = (h, depth.shape, depth.dtype)
        print(f"[smoke02] got {cam_id}: shape={depth.shape} d_min={depth.min()} d_max={depth.max()}")

    missing = WANTED - seen.keys()
    if missing:
        print(f"[smoke02] FAIL: never received {missing}")
    else:
        print("[smoke02] OK — all cams received. /tmp/cam_<id>_rgb.png + _depth.png")
        print("[smoke02] EYEBALL the rgb files. If physical mapping is wrong,")
        print("[smoke02] swap the serials in laptop_bridge/config.py CAMERA_SERIALS.")
    sub.close(linger=0)


if __name__ == "__main__":
    main()
