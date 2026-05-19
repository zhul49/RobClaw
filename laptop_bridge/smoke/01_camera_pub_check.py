# Open all cameras, grab N frames from each, report rates.
# Run: python laptop_bridge/smoke/01_camera_pub_check.py
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from laptop_bridge.camera_bridge import CameraBridge

N_FRAMES_PER_CAM = 30

def main():
    bridge = CameraBridge()
    if not bridge.streams:
        print("FAIL: no streams opened")
        return
    bridge.start()
    print(f"[smoke01] running for {N_FRAMES_PER_CAM/15.0:.1f}s")
    time.sleep(N_FRAMES_PER_CAM / 15.0)
    for s in bridge.streams:
        print(f"[smoke01] {s.cam_id} ({s.serial}) seq={s.seq}")
    bridge.stop()

if __name__ == "__main__":
    main()
