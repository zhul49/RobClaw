# CoTracker3 multi-camera point tracker entry script.
#
# fuses front_left + front_right (extrinsically calibrated to the robot base)
# into one 3D position per keypoint. keypoints come from
# run_keypoint_proposer.py (its keypoints.npz). per frame, each cam's
# CoTracker3 online predictor tracks the 2D query points; we look up depth
# at each tracked pixel, unproject to base frame, and average across cams
# (weighted by CoTracker's visibility score).
#
# run:
#   cd ~/libero_keypoint_project
#   ~/miniconda3/envs/rekep_curobo/bin/python \
#       v2/perception/run_keypoint_tracker.py \
#       --keypoints path/to/keypoints.npz --duration 30 \
#       --out /tmp/cotracker3.mp4 --csv /tmp/cotracker3.csv

import argparse
import os
import signal
import sys
import time

import cv2
import numpy as np
import zmq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # sibling imports inside perception/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 'v2.X.Y' resolution

from v2.common.extrinsics import load_extrinsic_yaml
from v2.perception.keypoint_tracker import CoTracker3MultiCam
from v2.perception.frame_pairer import FramePairer

DEFAULT_EXTRINSIC = os.path.normpath(
    os.path.join(HERE, "..", "..", "laptop_bridge", "extrinsics.yaml")
)


def annotate(bgr, positions_3d_base, projector, mean_sim=None, n_inliers=None,
             color=(0, 255, 255)):
    # draw a white-on-black numbered dot at each keypoint's projected pixel.
    out = bgr.copy()
    for i, p in enumerate(positions_3d_base):
        px = projector(p)
        if px is None:
            continue
        r, c = px
        if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]):
            continue
        cv2.circle(out, (c, r), 5, (255, 255, 255), -1)
        cv2.circle(out, (c, r), 6, (0, 0, 0), 1)
        label_pos = (c + 8, r - 8)
        cv2.putText(out, str(i), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(i), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints", required=True,
                    help="Path to a keypoints.npz produced by "
                         "run_keypoint_proposer.py. Must contain "
                         "keypoints_world (N, 3) in base frame.")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--out", default="/tmp/cotracker3.mp4")
    ap.add_argument("--extrinsic", default=DEFAULT_EXTRINSIC,
                    help="Path to extrinsic.yaml (intrinsics + T_base_cam per camera).")
    ap.add_argument("--video-fps", type=float, default=15.0,
                    help="Output video fps. The mp4 is resampled from the "
                         "timestamped tracker frames so playback duration == "
                         "wall-clock capture duration.")
    ap.add_argument("--smooth", type=int, default=10,
                    help="Uniform-filter window (frames). Paper: 10.")
    ap.add_argument("--csv", default=None,
                    help="Optional CSV path. Per tracker frame, writes one row "
                         "per keypoint: t,kp,x,y,z,n_inliers,mean_sim. Read by "
                         "franka_smoke/08_tracker_drift.py to characterize 3D wobble.")
    args = ap.parse_args()

    print(f"[demo] loading extrinsics from {args.extrinsic}")
    cams = load_extrinsic_yaml(args.extrinsic)
    for cid, c in cams.items():
        print(f"  {cid}: fx={c['fx']:.1f} fy={c['fy']:.1f} "
              f"cx={c['cx']:.1f} cy={c['cy']:.1f} ({c['width']}x{c['height']})")
    if "front_left" not in cams or "front_right" not in cams:
        print("[demo] FAIL: need front_left and front_right in extrinsic.yaml")
        sys.exit(1)

    # load the pre-computed keypoints we want to track.
    print(f"[demo] loading pre-computed keypoints from {args.keypoints}")
    npz = np.load(args.keypoints, allow_pickle=False)
    kp_preloaded = np.asarray(npz["keypoints_world"], dtype=np.float32).reshape(-1, 3)
    print(f"  {len(kp_preloaded)} keypoints (base frame):")
    for i, p in enumerate(kp_preloaded):
        print(f"    kp {i}: xyz=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    print("[demo] loading CoTracker3 online…")
    tracker = CoTracker3MultiCam(cams=cams, smooth_window=args.smooth)
    print(f"  window_len={tracker.window_len}, step={tracker.step} "
          f"(warmup = window_len frames = {tracker.window_len} frames "
          f"≈ {tracker.window_len/20.0:.1f}s at 20 Hz)")

    ctx = zmq.Context.instance()
    pairer = FramePairer(ctx)
    print("[demo] waiting for first matched (front_left, front_right) pair…")

    # Ctrl+C handling: set a flag and break out of loops cleanly so the video
    # writer finalizes the mp4 instead of leaving it half-written.
    stop = {"flag": False}
    def _sigint_handler(*a):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sigint_handler)

    # buffer of (t_relative_to_tracking_start, canvas_bgr) frames.
    # registration canvas is stamped at t=0; tracking canvases at their
    # wall-clock offset. resampled to args.video_fps at the end so video
    # duration matches wall-clock capture duration.
    frames_buf = []
    H_im = W_im = 0
    t_wait = time.monotonic()

    # ---- registration: first matched pair, then register on the preloaded keypoints ----
    while not stop["flag"]:
        pair = pairer.step()
        if pair is None:
            if time.monotonic() - t_wait > 60.0:
                print("[demo] FAIL: no matched pair after 60s"); sys.exit(1)
            continue
        rgb_l, depth_l = pair["front_left"]
        rgb_r, depth_r = pair["front_right"]
        H_im, W_im = rgb_l.shape[:2]
        print(f"[demo] first pair ({W_im}x{H_im}); registering on pre-loaded keypoints…")
        kp_3d_base = kp_preloaded

        rgb_by_cam = {"front_left": rgb_l, "front_right": rgb_r}
        depth_by_cam = {"front_left": depth_l, "front_right": depth_r}
        n_within = tracker.register(rgb_by_cam, depth_by_cam, kp_3d_base)
        print(f"  registered {len(kp_3d_base)} kps (cotracker3 seeded with "
              f"per-cam 2D queries from base-frame projection); {n_within}")

        # frame 0 of the output video (annotated registration).
        bgr_l = cv2.cvtColor(rgb_l, cv2.COLOR_RGB2BGR)
        bgr_r = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR)
        vis_l = annotate(bgr_l, kp_3d_base, lambda p: tracker.project_into("front_left", p))
        vis_r = annotate(bgr_r, kp_3d_base, lambda p: tracker.project_into("front_right", p))
        canvas = np.hstack([vis_l, vis_r])
        cv2.putText(canvas, "front_left (registered)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "front_right (registered)", (W_im + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        frames_buf.append((0.0, canvas))
        break

    if not frames_buf:
        print("[demo] never registered; exiting"); sys.exit(1)

    # optional per-frame per-kp diagnostics CSV. write header + a t=0 row for
    # every keypoint so post-hoc summaries can reference the initial position.
    csv_f = None
    if args.csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        csv_f = open(args.csv, "w")
        csv_f.write("t,kp,x,y,z,n_inliers,mean_sim\n")
        for kp_i, p in enumerate(kp_3d_base):
            csv_f.write(f"0.0000,{kp_i},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},0,1.0000\n")
        print(f"[demo] CSV log -> {args.csv}")

    # ---- tracking ----
    print(f"[demo] tracking for {args.duration:.0f}s "
          f"(cotracker3, vis_threshold=0.5, smooth={args.smooth})…")
    t0 = time.monotonic()
    n_track = 0

    while time.monotonic() - t0 < args.duration and not stop["flag"]:
        pair = pairer.step()
        if pair is None:
            continue
        t_rel = time.monotonic() - t0
        rgb_l, depth_l = pair["front_left"]
        rgb_r, depth_r = pair["front_right"]

        rgb_by_cam = {"front_left": rgb_l, "front_right": rgb_r}
        depth_by_cam = {"front_left": depth_l, "front_right": depth_r}
        positions_base, n_inliers, mean_sim = tracker.track(rgb_by_cam, depth_by_cam)

        if csv_f is not None:
            for kp_i in range(len(positions_base)):
                x, y, z = positions_base[kp_i]
                csv_f.write(
                    f"{t_rel:.4f},{kp_i},{x:.6f},{y:.6f},{z:.6f},"
                    f"{int(n_inliers[kp_i])},{float(mean_sim[kp_i]):.4f}\n"
                )

        # build the annotated side-by-side canvas for this frame.
        bgr_l = cv2.cvtColor(rgb_l, cv2.COLOR_RGB2BGR)
        bgr_r = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR)
        vis_l = annotate(bgr_l, positions_base,
                         lambda p: tracker.project_into("front_left", p),
                         mean_sim=mean_sim, n_inliers=n_inliers)
        vis_r = annotate(bgr_r, positions_base,
                         lambda p: tracker.project_into("front_right", p),
                         mean_sim=mean_sim, n_inliers=n_inliers)
        canvas = np.hstack([vis_l, vis_r])
        cv2.putText(canvas, "front_left", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "front_right", (W_im + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # bottom overlay: stats line.
        bottom = (10, H_im - 12)
        info = (f"t={time.monotonic() - t0:5.1f}s f={n_track + 1}  "
                f"vis=[{mean_sim.min():.2f}..{mean_sim.max():.2f}]  "
                f"cams={int(n_inliers.sum())}/{len(n_inliers)*len(cams)}")
        cv2.putText(canvas, info, bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, info, bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        frames_buf.append((t_rel, canvas))
        n_track += 1

        # progress log every 10 tracker frames. drained_total grows if the
        # bridge is publishing faster than we consume — should stay near 0
        # with FramePairer's drain logic.
        if n_track % 10 == 0:
            elapsed = time.monotonic() - t0
            print(f"  t={elapsed:5.1f}s f={n_track} fps={n_track/elapsed:.2f} "
                  f"q=[{mean_sim.min():.2f}..{mean_sim.max():.2f}] "
                  f"inliers={int(n_inliers.sum())} "
                  f"drained={pairer.drained_total}")

    elapsed = time.monotonic() - t0
    fps_real = n_track / elapsed if elapsed > 0 else 0.0

    # resample by timestamp so video duration matches wall-clock capture.
    fps_out = float(args.video_fps)
    target_duration = max(elapsed, 1.0 / fps_out)
    n_out = max(1, int(round(target_duration * fps_out)))
    ts = np.array([t for t, _ in frames_buf], dtype=np.float64)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps_out, (W_im * 2, H_im))
    print(f"[demo] writing {n_out} frames at {fps_out:.1f} fps -> "
          f"{n_out / fps_out:.1f}s video "
          f"(captured {n_track} tracker frames in {elapsed:.1f}s @ {fps_real:.2f} fps)…")
    for i in range(n_out):
        t_target = i / fps_out
        j = int(np.argmin(np.abs(ts - t_target)))
        writer.write(frames_buf[j][1])
    writer.release()

    print(f"[demo] DONE  elapsed={elapsed:.1f}s tracked_frames={n_track} "
          f"capture_fps={fps_real:.2f}")
    print(f"[demo]   video: {args.out}  duration={n_out / fps_out:.1f}s")
    if csv_f is not None:
        csv_f.close()
        print(f"[demo]   csv:   {args.csv}")
    pairer.close()


if __name__ == "__main__":
    main()
