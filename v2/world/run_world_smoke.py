# Phase A smoke: build a TSDF/ESDF of the tabletop from the two front cameras
# and visualize live in viser at http://localhost:8080.
#
# Reuses FramePairer (v2/perception/frame_pairer.py) for camera I/O — same path
# the tracker uses, no new ZMQ socket.
#
# Endpoint: after --duration seconds, ESDF is computed, mesh.ply is dumped,
# the viser server stays up so the operator can drag the slice gizmo.

import argparse
import concurrent.futures
import os
import signal
import sys
import time

import numpy as np
import torch
import zmq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # sibling imports inside world/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 'v2.X.Y' resolution

# Hard-imports so failures surface early.
from v2.common.extrinsics import load_extrinsic_yaml
from v2.perception.frame_pairer import FramePairer
from v2.world.camera_observer import PairToCuroboObs
from v2.world.world_mapper import WorldMapper

DEFAULT_EXTRINSIC = os.path.expanduser(
    "~/libero_keypoint_project/laptop_bridge/extrinsics.yaml"
)
CAM_ORDER = ("front_left", "front_right")


def _setup_viser_extent(visualizer, mapper_cfg):
    """Draw a yellow wireframe of the configured TSDF volume."""
    bmin, bmax = mapper_cfg.get_grid_bounds()
    bmin = np.array(bmin, dtype=np.float32)
    bmax = np.array(bmax, dtype=np.float32)
    c = np.array([
        [bmin[0], bmin[1], bmin[2]],
        [bmax[0], bmin[1], bmin[2]],
        [bmax[0], bmax[1], bmin[2]],
        [bmin[0], bmax[1], bmin[2]],
        [bmin[0], bmin[1], bmax[2]],
        [bmax[0], bmin[1], bmax[2]],
        [bmax[0], bmax[1], bmax[2]],
        [bmin[0], bmax[1], bmax[2]],
    ], dtype=np.float32)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    lines = np.array([[c[i], c[j]] for i, j in edges], dtype=np.float32)
    visualizer._server.scene.add_line_segments(
        "/workspace_extent",
        points=lines,
        colors=np.array([255, 255, 0], dtype=np.uint8),
        line_width=3.0,
    )


def _publish_voxels(visualizer, mapper, voxel_size, max_points=100_000):
    voxels = mapper.extract_voxels(surface_only=False)
    if len(voxels) == 0:
        return
    centers = voxels.centers
    colors = voxels.colors_uint8()
    if len(centers) > max_points:
        stride = int(np.ceil(len(centers) / max_points))
        centers = centers[::stride]
        colors = colors[::stride]
    visualizer.add_point_cloud(
        pointcloud=centers.cpu().numpy(),
        colors=colors.cpu().numpy(),
        point_size=voxel_size,
        name="/reconstruction",
    )


def _publish_cameras(visualizer, cams, cam_order):
    """Draw small frame axes at each camera pose — sanity check for extrinsics."""
    from curobo.types import Pose
    from scipy.spatial.transform import Rotation
    for cid in cam_order:
        T = cams[cid]["T_base_cam"]
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        pose = Pose(
            position=torch.tensor([T[:3, 3]], dtype=torch.float32),
            quaternion=torch.tensor([[w, x, y, z]], dtype=torch.float32),
        )
        visualizer.add_frame(f"/cameras/{cid}", pose, scale=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extrinsic", default=DEFAULT_EXTRINSIC)
    ap.add_argument("--duration", type=float, default=10.0,
                    help="Seconds to integrate frames before computing ESDF.")
    ap.add_argument("--max-frames", type=int, default=200,
                    help="Hard cap on integrated pairs.")
    ap.add_argument("--out-dir", default="/tmp/world_smoke")
    ap.add_argument("--no-visualize", action="store_true",
                    help="Skip viser. Just dump mesh + voxel count.")
    ap.add_argument("--vis-refresh-every", type=int, default=2,
                    help="Republish the voxel point cloud every N integrated pairs.")
    ap.add_argument("--live", action="store_true",
                    help="Run continuously until Ctrl+C, map updates in real time. "
                         "Skips ESDF compute and mesh dump.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[smoke] loading extrinsics: {args.extrinsic}")
    cams = load_extrinsic_yaml(args.extrinsic)
    for cid in CAM_ORDER:
        if cid not in cams:
            print(f"[smoke] FAIL: {cid} not in extrinsics", file=sys.stderr)
            sys.exit(1)
    H = cams[CAM_ORDER[0]]["height"]
    W = cams[CAM_ORDER[0]]["width"]
    print(f"[smoke] cams={CAM_ORDER} {W}x{H}")

    print("[smoke] building mapper (0.5 cm voxels, 0.55x1.20x0.58 m workspace)…")
    mapper = WorldMapper(num_cameras=len(CAM_ORDER), image_height=H, image_width=W)
    print(f"[smoke] mapper memory: {mapper.memory_mb():.1f} MB")

    converter = PairToCuroboObs(cams, cam_order=CAM_ORDER)

    visualizer = None
    if not args.no_visualize:
        from curobo.viewer import ViserVisualizer
        visualizer = ViserVisualizer(connect_port=8080)
        print("[smoke] viser ready: http://localhost:8080 "
              "(SSH? use ssh -L 8080:localhost:8080)")
        _setup_viser_extent(visualizer, mapper.cfg)
        _publish_cameras(visualizer, cams, CAM_ORDER)

    ctx = zmq.Context.instance()
    pairer = FramePairer(ctx, max_dt_s=0.10, hwm=4)

    if args.live:
        print("[smoke] LIVE mode: integrating continuously, Ctrl+C to exit.")
    else:
        print(f"[smoke] integrating for up to {args.duration:.1f}s or {args.max_frames} pairs…")
    stop = False

    def _sigint(*_):
        nonlocal stop
        stop = True
        print("\n[smoke] SIGINT received, finalizing…")

    signal.signal(signal.SIGINT, _sigint)

    t_start = time.monotonic()
    n_integrated = 0
    last_log = t_start

    # publish viser updates from a background thread so the SSH tunnel doesn't
    # block the integration loop. if the previous publish hasn't finished, we
    # drop the new one (latest-wins) — better to skip than to queue stale frames.
    publish_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    publish_future = None
    publish_dropped = 0

    while not stop:
        if not args.live:
            if n_integrated >= args.max_frames:
                break
            if time.monotonic() - t_start > args.duration:
                break
        pair = pairer.step(timeout_ms=300)
        if pair is None:
            continue
        obs = converter.convert(pair)
        mapper.integrate(obs)
        n_integrated += 1

        if visualizer is not None and (n_integrated % args.vis_refresh_every == 0):
            if publish_future is None or publish_future.done():
                publish_future = publish_pool.submit(
                    _publish_voxels, visualizer, mapper, mapper.cfg.voxel_size,
                )
            else:
                publish_dropped += 1

        if time.monotonic() - last_log > 1.0:
            print(f"  pairs={n_integrated} drained={pairer.drained_total} "
                  f"vis_dropped={publish_dropped} t={time.monotonic()-t_start:.1f}s")
            last_log = time.monotonic()

    publish_pool.shutdown(wait=False, cancel_futures=True)

    print(f"[smoke] integrated {n_integrated} pairs in {time.monotonic()-t_start:.1f}s")
    pairer.close()

    if n_integrated == 0:
        print("[smoke] FAIL: no frames integrated. Is the laptop bridge running?",
              file=sys.stderr)
        sys.exit(2)

    # in live mode we already showed the map in real time; skip ESDF+mesh since
    # those are snapshot artifacts. in batch mode dump them like before.
    if not args.live:
        print("[smoke] computing ESDF…")
        voxel_grid = mapper.compute_esdf()
        if voxel_grid.feature_tensor is not None:
            print(f"  ESDF shape: {tuple(voxel_grid.feature_tensor.shape)}, "
                  f"voxel_size: {voxel_grid.voxel_size:.4f} m")

        if visualizer is not None:
            _publish_voxels(visualizer, mapper, mapper.cfg.voxel_size)

        mesh_path = os.path.join(args.out_dir, "mesh.ply")
        n_verts = mapper.save_mesh(mesh_path)
        if n_verts > 0:
            print(f"[smoke] saved mesh: {mesh_path} ({n_verts:,} vertices)")
        else:
            print("[smoke] WARN: empty mesh — depth integration produced nothing.")

    if visualizer is None:
        return

    print("[smoke] viser running. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
