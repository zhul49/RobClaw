# subscribes to the laptop bridge's camera_frames stream and emits matched
# (front_left, front_right) pairs that are within max_dt_s of each other.
# used by run_keypoint_tracker.py and v2/world/run_world_smoke.py so both can
# share one camera-IO implementation.

import cv2
import msgpack
import numpy as np
import zmq

from v2.common.franka_env_adapter import LAPTOP_IP, CAMERA_FRAMES_SUB_PORT


class FramePairer:
    # the bridge publishes ~30 msgs/s. if our consumer eats slower (e.g. 5 Hz),
    # the SUB buffer piles up and we'd track stale frames. so step() drains
    # the buffer first (non-blocking) and keeps only the newest per cam;
    # the small HWM keeps the queue tight between calls.
    # set_hwm + setsockopt_string MUST run before connect().

    def __init__(self, ctx, max_dt_s=0.10, hwm=4):
        self.sub = ctx.socket(zmq.SUB)
        self.sub.set_hwm(hwm)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.connect(f"tcp://{LAPTOP_IP}:{CAMERA_FRAMES_SUB_PORT}")
        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)

        # latest[cam_id] = (t_header, rgb, depth)
        self.latest = {}
        # last emitted pair's timestamp; refuse to re-emit the same pair twice.
        self.last_emit_t = -np.inf
        self.max_dt = max_dt_s

        # bookkeeping for caller diagnostics.
        self.latest_msg_t = None       # header time of the most recent message we saw
        self.drained_total = 0         # how many messages we've drained from the buffer

    def _ingest(self, hb, jb, db):
        # decode one multipart message: (header bytes, jpeg bytes, depth bytes).
        h = msgpack.unpackb(hb)
        cam_id = h["cam_id"]
        self.latest_msg_t = h["t"]
        if cam_id not in ("front_left", "front_right"):
            return
        bgr = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.frombuffer(db, np.uint16).reshape(h["height"], h["width"])
        self.latest[cam_id] = (h["t"], rgb, depth)

    def _maybe_emit(self):
        # do we have both cams within max_dt_s, and is this a fresher pair
        # than the last one we emitted?
        if "front_left" in self.latest and "front_right" in self.latest:
            tl, rl, dl = self.latest["front_left"]
            tr, rr, dr = self.latest["front_right"]
            if abs(tl - tr) <= self.max_dt and max(tl, tr) > self.last_emit_t:
                self.last_emit_t = max(tl, tr)
                return {
                    "front_left":  (rl, dl),
                    "front_right": (rr, dr),
                    "t": max(tl, tr),
                }
        return None

    def step(self, timeout_ms=300):
        # drain all buffered messages without blocking; keep only the latest
        # per cam. if nothing was buffered, wait briefly for a fresh message.
        # the bridge sometimes emits 1-part or 5-part multipart messages
        # instead of 3-part — skip those rather than crashing.
        drained = 0
        while True:
            try:
                parts = self.sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) == 3:
                self._ingest(*parts)
            drained += 1

        if drained == 0:
            ev = dict(self.poller.poll(timeout=timeout_ms))
            if self.sub in ev:
                parts = self.sub.recv_multipart()
                if len(parts) == 3:
                    self._ingest(*parts)

        self.drained_total += drained
        return self._maybe_emit()

    def close(self):
        self.sub.close(linger=0)
