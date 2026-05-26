# Shared camera-extrinsics loader. Imported by both the paper keypoint
# proposer and the CoTracker3 tracker so neither depends on the other.

import numpy as np
import yaml


def load_extrinsic_yaml(path):
    """Returns {cam_id: {fx, fy, cx, cy, width, height, T_base_cam (4x4)}}.
    Cameras without T_base_cam (e.g. wrist) are omitted from the result."""
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    cams = {}
    for cam_id, cfg in d["cameras"].items():
        intr = cfg["intrinsics"]
        Tbc = cfg.get("T_base_cam")
        if Tbc is None:
            continue
        cams[cam_id] = {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
            "width": int(intr.get("width", 0)),
            "height": int(intr.get("height", 0)),
            "T_base_cam": np.asarray(Tbc["matrix"], dtype=np.float64).reshape(4, 4),
        }
    return cams
