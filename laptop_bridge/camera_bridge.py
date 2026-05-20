# 3x RealSense D435i -> ZMQ PUB on CAMERA_FRAMES_PUB_PORT. 15 Hz per cam.
import os
import signal
import sys
import threading
import time

import cv2
import msgpack
import numpy as np
import pyrealsense2 as rs
import zmq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from laptop_bridge import config


class CameraStream:
    def __init__(self, cam_id: str, serial: str):
        self.cam_id = cam_id
        self.serial = serial
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, config.CAMERA_WIDTH, config.CAMERA_HEIGHT, rs.format.bgr8, config.CAMERA_HZ)
        cfg.enable_stream(rs.stream.depth, config.CAMERA_WIDTH, config.CAMERA_HEIGHT, rs.format.z16, config.CAMERA_HZ)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.seq = 0

    def grab(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=250)
        except RuntimeError:
            return None
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            return None
        bgr = np.asanyarray(color.get_data())
        d = np.asanyarray(depth.get_data())  # uint16, little-endian on x86
        return bgr, d

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class CameraBridge:
    def __init__(self):
        self.streams = []
        for cam_id, serial in config.CAMERA_SERIALS.items():
            if serial is None:
                print(f"[camera_bridge] {cam_id}: no serial assigned, skipping")
                continue
            try:
                s = CameraStream(cam_id, serial)
                self.streams.append(s)
                print(f"[camera_bridge] {cam_id} ({serial}) started")
            except Exception as e:
                print(f"[camera_bridge] {cam_id} ({serial}) FAILED: {e}")

        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.set_hwm(10)
        self.pub.bind(f"{config.BIND_ADDR}:{config.CAMERA_FRAMES_PUB_PORT}")

        self.stop_flag = threading.Event()
        self.threads = []

    def _run_stream(self, s: CameraStream):
        period = 1.0 / config.CAMERA_HZ
        next_t = time.monotonic()
        while not self.stop_flag.is_set():
            res = s.grab()
            if res is None:
                continue
            bgr, depth = res
            ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, config.CAMERA_JPEG_QUALITY])
            if not ok:
                continue
            s.seq += 1
            header = {
                "seq": s.seq,
                "t": time.monotonic(),
                "cam_id": s.cam_id,
                "width": config.CAMERA_WIDTH,
                "height": config.CAMERA_HEIGHT,
            }
            self.pub.send_multipart([msgpack.packb(header), jpg.tobytes(), depth.tobytes()])
            next_t += period
            slack = next_t - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.monotonic()

    def start(self):
        for s in self.streams:
            t = threading.Thread(target=self._run_stream, args=(s,), daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        self.stop_flag.set()
        for t in self.threads:
            t.join(timeout=3.0)
        for s in self.streams:
            s.stop()
        self.pub.close(linger=0)


def main():
    bridge = CameraBridge()
    if not bridge.streams:
        print("[camera_bridge] no streams opened; exiting")
        sys.exit(1)

    def _shutdown(signum, frame):
        print("[camera_bridge] shutting down")
        bridge.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    bridge.start()
    print(f"[camera_bridge] publishing on tcp://*:{config.CAMERA_FRAMES_PUB_PORT}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
