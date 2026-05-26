# one-shot demo of the paper-faithful keypoint proposer.
#
# grabs one matched (front_left, front_right) pair from the laptop bridge,
# picks one cam, asks GSAM2 for all masks via /predict_automatic_masks,
# runs PaperKeypointProposer, and dumps the annotated image + the 3D
# keypoints to disk.
#
# output (defaults to /tmp/paper_proposer/):
#   - keypoints.npz   keypoints_world (N,3), rigid_group_ids (N,), pixels (N,2)
#   - annotated.png   numbered overlay for VLM/visual inspection
#   - masks.png       SAM-everything multi-instance label image
#   - kp_on_masks.png keypoints drawn on top of their source masks (red = bug)
#
# run:
#   cd ~/libero_keypoint_project
#   ~/miniconda3/envs/rekep_curobo/bin/python \
#       v2/perception/run_keypoint_proposer.py --out-dir /tmp/paper_proposer

import argparse
import os
import sys
import time

import cv2
import numpy as np
import requests
import zmq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # sibling imports inside perception/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 'v2.X.Y' resolution

# point torch.hub at the local DINOv3 clone + weights before importing the loader.
os.environ.setdefault("DINOV3_REPO", os.path.expanduser("~/Code/dinov3"))
os.environ.setdefault(
    "DINOV3_WEIGHTS",
    os.path.expanduser(
        "~/.cache/torch/hub/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    ),
)

from v2.common.extrinsics import load_extrinsic_yaml
from v2.perception.dino_features import load_dinov3
from v2.perception.frame_pairer import FramePairer
from v2.perception.keypoint_proposer import PaperKeypointProposer

GSAM2_AUTO = "http://127.0.0.1:8765/predict_automatic_masks"
GSAM2_ALL = "http://127.0.0.1:8765/predict_all_masks"
GSAM2_HEALTH = "http://127.0.0.1:8765/health"
DEFAULT_EXTRINSIC = os.path.normpath(
    os.path.join(HERE, "..", "..", "laptop_bridge", "extrinsics.yaml")
)


def wait_gsam2(timeout_s=120.0):
    # poll GSAM2's /health endpoint until it answers OK or we time out.
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            r = requests.get(GSAM2_HEALTH, timeout=2)
            if r.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return False


def call_exclude_mask(rgb, prompt, box_threshold=0.30, text_threshold=0.25,
                      jpg_quality=90):
    # ask GSAM2 to detect things described by `prompt` (multi-class, dot-separated).
    # returns ((H, W) bool mask of detected pixels, n_detections).
    # used to subtract regions like "table. apriltag." before running the proposer.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    if not ok:
        return None
    r = requests.post(
        GSAM2_ALL,
        files={"image": ("frame.jpg", jpg.tobytes(), "image/jpeg")},
        data={"prompt": prompt,
              "box_threshold": box_threshold,
              "text_threshold": text_threshold},
        timeout=30.0,
    )
    r.raise_for_status()
    label = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    n_det = int(r.headers.get("X-Detections", "0"))
    return (label > 0), n_det


def apply_exclusion(label_img, exclude_mask, overlap_thr):
    # for each instance in label_img, zero it out if more than overlap_thr of
    # its pixels lie inside exclude_mask. returns (filtered, n_dropped, n_kept).
    out = label_img.copy()
    instances = sorted(set(int(u) for u in np.unique(out) if u != 0))
    n_dropped = 0
    for uid in instances:
        region = (out == uid)
        n = int(region.sum())
        if n == 0:
            continue
        n_in = int((region & exclude_mask).sum())
        if n_in / n > overlap_thr:
            out[region] = 0
            n_dropped += 1
    return out, n_dropped, len(instances) - n_dropped


def call_automatic_masks(rgb, points_per_side=32, min_area=100, jpg_quality=90):
    # ask GSAM2 to run SAM-everything over the image.
    # returns (label_img (H, W) int, n_detections). slow: 10-60s typical.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    if not ok:
        return None, None
    r = requests.post(
        GSAM2_AUTO,
        files={"image": ("frame.jpg", jpg.tobytes(), "image/jpeg")},
        data={"points_per_side": points_per_side,
              "min_mask_region_area": min_area},
        timeout=180.0,
    )
    r.raise_for_status()
    label_img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    n_det = int(r.headers.get("X-Detections", "0"))
    return label_img, n_det


def grab_one_pair(timeout_s=30.0):
    # wait for one matched (front_left, front_right) pair from the bridge.
    # returns (rgb_l, depth_l, rgb_r, depth_r). raises if it times out.
    ctx = zmq.Context.instance()
    pairer = FramePairer(ctx)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < timeout_s:
            pair = pairer.step()
            if pair is None:
                continue
            rgb_l, depth_l = pair["front_left"]
            rgb_r, depth_r = pair["front_right"]
            return rgb_l, depth_l, rgb_r, depth_r
        raise RuntimeError(
            f"No matched (front_left, front_right) pair within {timeout_s}s"
        )
    finally:
        pairer.close()


def render_masks_color(label_img):
    # color each instance in the label image with a random color (seeded so
    # reruns produce the same colors).
    H, W = label_img.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for uid in np.unique(label_img):
        if uid == 0:
            continue
        color = rng.integers(40, 255, size=3, dtype=np.uint8)
        out[label_img == uid] = color
    return out


def overlay_masks_on_rgb(rgb, label_img, alpha=0.45):
    # translucent mask overlay so the underlying object stays visible.
    color = render_masks_color(label_img)
    out = rgb.astype(np.float32).copy()
    mask_hit = (label_img > 0)
    out[mask_hit] = (1.0 - alpha) * out[mask_hit] + alpha * color[mask_hit]
    return out.astype(np.uint8)


def annotate_with_source_mask(rgb, label_img, pixels, rigid_ids):
    # rgb + translucent masks + numbered keypoints with their source mask
    # outlined in yellow. off-mask keypoints are drawn RED so the failure
    # mode (kp lands outside its mask -> reads background depth) is visible.
    base = overlay_masks_on_rgb(rgb, label_img, alpha=0.45)
    H, W = base.shape[:2]
    for i, ((r, c), rg) in enumerate(zip(pixels, rigid_ids)):
        r, c = int(r), int(c)
        if not (0 <= r < H and 0 <= c < W):
            continue

        # outline this keypoint's source mask in yellow.
        m = (label_img == int(rg)).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base, contours, -1, (255, 255, 0), 2)

        # green if kp is on the mask, red if it isn't.
        on_mask = m[r, c] > 0
        dot_color = (0, 255, 0) if on_mask else (0, 0, 255)
        cv2.circle(base, (c, r), 5, dot_color, -1)
        cv2.circle(base, (c, r), 6, (0, 0, 0), 1)

        label = str(i)
        cv2.putText(base, label, (c + 8, r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(base, label, (c + 8, r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="front_right",
                    choices=["front_left", "front_right"],
                    help="Which static cam to use for the proposal "
                         "(paper picks the one with the best holistic view).")
    ap.add_argument("--extrinsic", default=DEFAULT_EXTRINSIC)
    ap.add_argument("--out-dir", default="/tmp/paper_proposer")
    ap.add_argument("--points-per-side", type=int, default=48,
                    help="SAM-everything grid density. Lower = fewer masks, faster.")
    ap.add_argument("--min-mask-area", type=int, default=200,
                    help="Drop masks under this pixel area BEFORE clustering.")
    ap.add_argument("--max-mask-ratio", type=float, default=0.9,
                    help="Drop masks covering more than this fraction of the image (tables/walls).")
    ap.add_argument("--bandwidth-cm", type=float, default=4.0,
                    help="MeanShift bandwidth in cm (paper: 8 cm).")
    ap.add_argument("--k", type=int, default=5,
                    help="K-means clusters per mask (paper: 5).")
    ap.add_argument("--bounds-min", type=float, nargs=3, default=[0.20, -0.60, 0.01],
                    help="Workspace lower bound in base frame [x y z] (m).")
    ap.add_argument("--bounds-max", type=float, nargs=3, default=[0.75, 0.60, 0.30],
                    help="Workspace upper bound in base frame [x y z] (m).")
    ap.add_argument("--exclude",
                    default="",
                    help='GroundingDINO prompt of things to EXCLUDE. Auto masks '
                         'that mostly overlap these regions are dropped before '
                         'clustering. Empty string ("") disables exclusion.')
    ap.add_argument("--exclude-overlap", type=float, default=0.5,
                    help="Drop a SAM-everything mask if more than this fraction "
                         "of its pixels lie inside the exclude region.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[demo] loading extrinsics from {args.extrinsic}")
    cams = load_extrinsic_yaml(args.extrinsic)
    if args.cam not in cams:
        print(f"[demo] FAIL: {args.cam} not in extrinsics"); sys.exit(1)
    intr = cams[args.cam]
    Tbc = intr["T_base_cam"]
    print(f"[demo] using {args.cam}: fx={intr['fx']:.1f} cx={intr['cx']:.1f} "
          f"({intr['width']}x{intr['height']})")

    print("[demo] GSAM2 health…")
    if not wait_gsam2():
        print("[demo] FAIL: GSAM2 never came up"); sys.exit(1)

    print("[demo] grabbing one matched (front_left, front_right) pair…")
    rgb_l, depth_l, rgb_r, depth_r = grab_one_pair()
    rgb = rgb_l if args.cam == "front_left" else rgb_r
    depth = depth_l if args.cam == "front_left" else depth_r
    H, W = rgb.shape[:2]
    print(f"[demo] got pair ({W}x{H})")

    print(f"[demo] SAM-everything (points_per_side={args.points_per_side}, "
          f"min_area={args.min_mask_area})… this can take 10-60s")
    t0 = time.monotonic()
    label_img, n_det = call_automatic_masks(
        rgb, points_per_side=args.points_per_side, min_area=args.min_mask_area,
    )
    print(f"  {n_det} masks in {time.monotonic()-t0:.1f}s")
    cv2.imwrite(os.path.join(args.out_dir, "masks.png"), label_img)
    cv2.imwrite(os.path.join(args.out_dir, "masks_color.png"),
                cv2.cvtColor(render_masks_color(label_img), cv2.COLOR_RGB2BGR))

    # optional: subtract a GroundingDINO-detected region (e.g. "table. apriltag.")
    # before running the proposer.
    if args.exclude.strip():
        print(f"[demo] excluding via prompt: '{args.exclude}'")
        t0 = time.monotonic()
        exclude_mask, n_exc_det = call_exclude_mask(rgb, args.exclude)
        print(f"  GroundingDINO matched {n_exc_det} detection(s) in "
              f"{time.monotonic()-t0:.1f}s")
        cv2.imwrite(os.path.join(args.out_dir, "exclude_mask.png"),
                    exclude_mask.astype(np.uint8) * 255)
        label_img, n_dropped, n_kept = apply_exclusion(
            label_img, exclude_mask, args.exclude_overlap,
        )
        print(f"  dropped {n_dropped} masks; {n_kept} remain")
        cv2.imwrite(os.path.join(args.out_dir, "masks_after_exclude.png"),
                    cv2.cvtColor(render_masks_color(label_img), cv2.COLOR_RGB2BGR))

    print("[demo] loading DINOv3 ViT-S/16…")
    backbone = load_dinov3()

    print("[demo] proposing keypoints…")
    proposer = PaperKeypointProposer(
        backbone,
        k=args.k,
        mean_shift_bandwidth_m=args.bandwidth_cm / 100.0,
        max_mask_ratio=args.max_mask_ratio,
    )
    t0 = time.monotonic()
    kps_base, rigid_ids, pixels, annotated = proposer.propose(
        rgb=rgb, depth=depth, masks_label_image=label_img,
        intrinsics={"fx": intr["fx"], "fy": intr["fy"],
                    "cx": intr["cx"], "cy": intr["cy"]},
        T_world_cam=Tbc,
        bounds_min=args.bounds_min, bounds_max=args.bounds_max,
    )
    print(f"  {len(kps_base)} keypoints in {time.monotonic()-t0:.2f}s")
    for i, (p, rg, px) in enumerate(zip(kps_base, rigid_ids, pixels)):
        print(f"    kp {i}: xyz_base=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})  "
              f"pixel=({px[0]}, {px[1]})  rigid_group={rg}")

    # main result file.
    out_npz = os.path.join(args.out_dir, "keypoints.npz")
    np.savez(
        out_npz,
        keypoints_world=kps_base, rigid_group_ids=rigid_ids, pixels=pixels,
        cam=args.cam,
        intrinsics=np.array(
            [intr["fx"], intr["fy"], intr["cx"], intr["cy"]], dtype=np.float32,
        ),
        T_base_cam=Tbc.astype(np.float32),
    )

    # also save depth + the post-exclusion label image. if the proposer returns
    # 0 kps we need both to know whether bounds or depth-validity killed them.
    np.savez(
        os.path.join(args.out_dir, "debug.npz"),
        depth=depth, label_img=label_img,
        bounds_min=np.asarray(args.bounds_min, dtype=np.float32),
        bounds_max=np.asarray(args.bounds_max, dtype=np.float32),
    )
    cv2.imwrite(os.path.join(args.out_dir, "annotated.png"),
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.out_dir, "rgb_raw.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    debug_overlay = annotate_with_source_mask(rgb, label_img, pixels, rigid_ids)
    cv2.imwrite(os.path.join(args.out_dir, "kp_on_masks.png"),
                cv2.cvtColor(debug_overlay, cv2.COLOR_RGB2BGR))

    # quick consistency check: any keypoints that landed off their own mask?
    n_off = sum(
        1 for (r, c), rg in zip(pixels, rigid_ids)
        if not (0 <= r < label_img.shape[0] and 0 <= c < label_img.shape[1]
                and label_img[int(r), int(c)] == int(rg))
    )
    if n_off:
        print(f"[demo] WARN: {n_off}/{len(pixels)} keypoints landed off their "
              f"source mask (drawn red in kp_on_masks.png)")

    print(f"[demo] DONE")
    print(f"  keypoints:    {out_npz}")
    print(f"  annotated:    {os.path.join(args.out_dir, 'annotated.png')}")
    print(f"  masks:        {os.path.join(args.out_dir, 'masks_color.png')}")
    print(f"  kp_on_masks:  {os.path.join(args.out_dir, 'kp_on_masks.png')} "
          f"<- check this; red dots = off-mask bug")


if __name__ == "__main__":
    main()
