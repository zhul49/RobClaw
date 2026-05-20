"""Compute camera->franka_base extrinsics from an AprilTag on the table.

Tag pose in the franka base frame is known from the physical setup:
  X = +0.5175 m  (in front of the base)
  Y =  0.0    m  (in line with the base)
  Z = +0.02   m  (table surface, 2 cm above the base mounting plane)

The tag is assumed to lie flat on the table, face up, with its own axes
aligned to the franka base axes (tag +x along base +x, tag +y along base +y,
tag +z up). Override with --tag-rpy if the tag is rotated about base +z.

For each requested camera we:
  1) Open the RealSense color stream and read the factory color intrinsics.
  2) Average a few frames to reduce noise.
  3) Detect the AprilTag with cv2.aruco (default family: 36h11).
  4) solvePnP to get T_cam_tag.
  5) Compose: T_base_cam = T_base_tag @ inv(T_cam_tag).
  6) Write results to laptop_bridge/extrinsics.yaml (translation + quaternion +
     4x4 matrix) and print to stdout.

Usage:
  python3 laptop_bridge/calibrate_extrinsics.py \
      --cameras front_left front_right \
      --tag-size 0.06 \
      --tag-id 0

Run with --preview to also save annotated PNGs next to extrinsics.yaml so you
can sanity-check the detection.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from laptop_bridge import config


APRILTAG_DICTS = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_quat_xyzw(R):
    # Shepperd's method, returns (x, y, z, w).
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def grab_intrinsics_and_image(serial, n_frames=10, warmup=15):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(
        rs.stream.color,
        config.CAMERA_WIDTH,
        config.CAMERA_HEIGHT,
        rs.format.bgr8,
        config.CAMERA_HZ,
    )
    profile = pipeline.start(cfg)
    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        K = np.array(
            [[intr.fx, 0, intr.ppx],
             [0, intr.fy, intr.ppy],
             [0, 0, 1]],
            dtype=np.float64,
        )
        # RealSense color stream on D435i: distortion model is "none" /
        # Brown-Conrady with zero coeffs after factory rectification.
        D = np.array(intr.coeffs, dtype=np.float64).reshape(-1, 1)

        # Warm up auto-exposure.
        for _ in range(warmup):
            pipeline.wait_for_frames(timeout_ms=1000)

        # Average a few frames in float to denoise.
        acc = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.float64)
        got = 0
        for _ in range(n_frames):
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color = frames.get_color_frame()
            if not color:
                continue
            acc += np.asanyarray(color.get_data()).astype(np.float64)
            got += 1
        if got == 0:
            raise RuntimeError(f"no color frames from {serial}")
        bgr = (acc / got).astype(np.uint8)
        return K, D, bgr
    finally:
        pipeline.stop()


def detect_tag_pose(bgr, K, D, tag_size, tag_id, dict_name):
    aruco_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTS[dict_name])
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None, None

    ids = ids.flatten()
    if tag_id is not None and tag_id >= 0:
        if tag_id not in ids:
            return None, ids.tolist(), None
        idx = int(np.where(ids == tag_id)[0][0])
    else:
        idx = 0  # first detected tag

    img_pts = corners[idx].reshape(4, 2).astype(np.float64)
    # AprilTag corner order from cv2.aruco: top-left, top-right, bottom-right,
    # bottom-left (looking at the tag face). Tag frame: +x right, +y down,
    # +z out of the tag face.
    half = tag_size / 2.0
    obj_pts = np.array(
        [[-half,  half, 0.0],
         [ half,  half, 0.0],
         [ half, -half, 0.0],
         [-half, -half, 0.0]],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, ids.tolist(), None
    R_cam_tag, _ = cv2.Rodrigues(rvec)
    T_cam_tag = np.eye(4)
    T_cam_tag[:3, :3] = R_cam_tag
    T_cam_tag[:3, 3] = tvec.flatten()
    return T_cam_tag, ids.tolist(), (corners[idx], int(ids[idx]))


def annotate(bgr, K, D, detection, T_cam_tag, tag_size):
    out = bgr.copy()
    if detection is None:
        return out
    corners, tag_id = detection
    cv2.aruco.drawDetectedMarkers(out, [corners], np.array([[tag_id]]))
    if T_cam_tag is not None:
        rvec, _ = cv2.Rodrigues(T_cam_tag[:3, :3])
        tvec = T_cam_tag[:3, 3].reshape(3, 1)
        cv2.drawFrameAxes(out, K, D, rvec, tvec, tag_size * 0.75, 2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cameras",
        nargs="+",
        default=["front_left", "front_right"],
        help="Which entries from config.CAMERA_SERIALS to calibrate.",
    )
    p.add_argument("--tag-size", type=float, default=0.105,
                   help="Black-square edge length of the AprilTag in meters.")
    p.add_argument("--tag-id", type=int, default=0,
                   help="Specific tag ID to use (-1 = first detected).")
    p.add_argument("--tag-family", default="36h11", choices=list(APRILTAG_DICTS),
                   help="AprilTag family. Default 36h11.")
    p.add_argument("--tag-xyz", nargs=3, type=float, default=[0.5175, 0.0, 0.02],
                   help="Tag origin in the franka base frame (meters).")
    p.add_argument("--tag-rpy", nargs=3, type=float, default=[0.0, 0.0, -1.5707963267948966],
                   help="Tag orientation in base frame as roll/pitch/yaw (rad). "
                        "Default yaw=-pi/2 matches the physical tag pose on this rig "
                        "(printed top of tag points +base_x, away from the franka).")
    p.add_argument("--output", default=os.path.join(os.path.dirname(__file__),
                                                    "extrinsics.yaml"))
    p.add_argument("--preview", action="store_true",
                   help="Save annotated frame next to the output yaml.")
    p.add_argument("--frames", type=int, default=10,
                   help="Frames to average per camera.")
    args = p.parse_args()

    # T_base_tag from CLI / defaults.
    T_base_tag = np.eye(4)
    T_base_tag[:3, :3] = rpy_to_matrix(*args.tag_rpy)
    T_base_tag[:3, 3] = np.array(args.tag_xyz, dtype=np.float64)

    print(f"[calib] tag-in-base translation: {args.tag_xyz}  rpy: {args.tag_rpy}")
    print(f"[calib] tag size: {args.tag_size} m  family: {args.tag_family}  "
          f"id: {'any' if args.tag_id < 0 else args.tag_id}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "tag": {
            "family": args.tag_family,
            "id": None if args.tag_id < 0 else int(args.tag_id),
            "size_m": float(args.tag_size),
            "T_base_tag_xyz": [float(v) for v in args.tag_xyz],
            "T_base_tag_rpy": [float(v) for v in args.tag_rpy],
        },
        "cameras": {},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    any_failed = False
    for cam_id in args.cameras:
        if cam_id not in config.CAMERA_SERIALS:
            print(f"[calib] {cam_id}: not in config.CAMERA_SERIALS, skipping")
            any_failed = True
            continue
        serial = config.CAMERA_SERIALS[cam_id]
        if serial is None:
            print(f"[calib] {cam_id}: no serial assigned, skipping")
            any_failed = True
            continue

        print(f"\n[calib] === {cam_id} ({serial}) ===")
        try:
            K, D, bgr = grab_intrinsics_and_image(serial, n_frames=args.frames)
        except Exception as e:
            print(f"[calib] {cam_id}: failed to open / grab: {e}")
            any_failed = True
            continue

        tag_id_arg = None if args.tag_id < 0 else args.tag_id
        T_cam_tag, seen_ids, detection = detect_tag_pose(
            bgr, K, D, args.tag_size, tag_id_arg, args.tag_family
        )
        if T_cam_tag is None:
            print(f"[calib] {cam_id}: tag not found. seen ids = {seen_ids}")
            any_failed = True
            if args.preview:
                cv2.imwrite(os.path.join(out_dir, f"calib_{cam_id}_raw.png"), bgr)
            continue

        T_base_cam = T_base_tag @ invert_transform(T_cam_tag)
        t_bc = T_base_cam[:3, 3]
        q_bc = matrix_to_quat_xyzw(T_base_cam[:3, :3])
        # Distance from tag to camera, in camera frame (sanity check).
        dist_cam_tag = float(np.linalg.norm(T_cam_tag[:3, 3]))

        print(f"[calib] {cam_id}: detected tag id={detection[1]}  "
              f"cam->tag dist = {dist_cam_tag:.3f} m")
        print(f"[calib] {cam_id}: T_base_cam translation (m) = "
              f"[{t_bc[0]:+.4f}, {t_bc[1]:+.4f}, {t_bc[2]:+.4f}]")
        print(f"[calib] {cam_id}: T_base_cam quat (xyzw)     = "
              f"[{q_bc[0]:+.4f}, {q_bc[1]:+.4f}, {q_bc[2]:+.4f}, {q_bc[3]:+.4f}]")

        results["cameras"][cam_id] = {
            "serial": serial,
            "intrinsics": {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "width": int(config.CAMERA_WIDTH),
                "height": int(config.CAMERA_HEIGHT),
                "distortion": [float(v) for v in D.flatten()],
            },
            "T_base_cam": {
                "xyz": [float(v) for v in t_bc],
                "quat_xyzw": q_bc,
                "matrix": [[float(v) for v in row] for row in T_base_cam],
            },
            "T_cam_tag_translation": [float(v) for v in T_cam_tag[:3, 3]],
            "tag_distance_m": dist_cam_tag,
        }

        if args.preview:
            ann = annotate(bgr, K, D, detection, T_cam_tag, args.tag_size)
            png = os.path.join(out_dir, f"calib_{cam_id}.png")
            cv2.imwrite(png, ann)
            print(f"[calib] {cam_id}: wrote preview {png}")

    with open(args.output, "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    print(f"\n[calib] wrote {args.output}")

    sys.exit(0 if not any_failed and results["cameras"] else 1)


if __name__ == "__main__":
    main()
