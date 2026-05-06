"""
Phase 2 Step 1.2 — build a real SDF for the LIBERO scene.

Approach: LIBERO provides per-object box collision proxies (each "main"
body has 1 visual mesh + many boxes that together approximate the
object's shape). We use ONLY the boxes — no trimesh, no mesh-SDF — and
compute the signed distance to the nearest box at every voxel.

Output: dict of
    {
      "bounds_min": (3,) float,
      "bounds_max": (3,) float,
      "voxel_size": float,
      "shape": (nx, ny, nz),
      "sdf_full":         (nx, ny, nz)  # all obstacles
      "sdf_per_body": {                  # one SDF excluding each holdable body
          body_name: (nx, ny, nz),
          ...
      },
      "boxes_per_body": { body_name: list of box dicts }   # for diagnostics
    }

Step 1.4 (held-object exclusion) consumes `sdf_per_body` directly: when
holding `alphabet_soup_1_main`, use sdf_per_body["alphabet_soup_1_main"];
otherwise use sdf_full.

Cache strategy: pickle the dict to outputs/v2_<suite>_<task>/sdf_cache.pkl.
Key by suite/task. Recompute when bounds/voxel_size change.

Usage:
    from build_sdf import build_sdf_for_env
    cache = build_sdf_for_env(env, suite="libero_object", task=0)
"""
import os
import pickle
import time
import numpy as np


# Bodies whose geoms can be excluded from the SDF when held by the gripper.
# Currently just "alphabet_soup_1_main" (kp0 = the can to grasp). Add others
# if/when other tasks need them. Bodies NOT in this list are always treated
# as static obstacles.
HOLDABLE_BODIES = {"alphabet_soup_1_main"}

# Bodies to skip entirely from the SDF (robot, world fixtures).
SKIP_BODY_PREFIXES = ("robot0", "gripper0", "world", "table", "floor", "wall", "mount")

DEFAULT_BOUNDS_MIN = np.array([-0.40, -0.50, 0.00])
DEFAULT_BOUNDS_MAX = np.array([ 0.70,  0.50, 0.60])
DEFAULT_VOXEL_SIZE = 0.02


def _enumerate_obstacle_boxes(sim):
    """Walk every geom in the sim and return the obstacle box list grouped by body.

    We use only boxes — the meshes in LIBERO are visual-only (textured_vis),
    the boxes are the collision proxies. Skipping meshes is a 5-mesh
    optimization that drops the SDF compute from "minutes" to "seconds".
    """
    boxes_per_body = {}
    skipped_meshes = 0
    for i in range(sim.model.ngeom):
        body_id = int(sim.model.geom_bodyid[i])
        body_name = sim.model.body_id2name(body_id)
        if body_name is None:
            continue
        if any(body_name.startswith(p) for p in SKIP_BODY_PREFIXES):
            continue
        gtype = int(sim.model.geom_type[i])
        if gtype == 7:  # mesh — skip, boxes provide collision shape
            skipped_meshes += 1
            continue
        if gtype != 6:  # box — for now only boxes; if we hit cylinders/spheres later, extend
            print(f"[build_sdf] WARNING: skipping geom {i} '{sim.model.geom_id2name(i)}' "
                  f"(body '{body_name}', type={gtype}) — non-box not yet supported")
            continue
        boxes_per_body.setdefault(body_name, []).append({
            "geom_idx": i,
            "geom_name": sim.model.geom_id2name(i),
            "pos": sim.data.geom_xpos[i].copy(),
            "rotmat": sim.data.geom_xmat[i].reshape(3, 3).copy(),
            "half_size": sim.model.geom_size[i].copy(),
        })
    return boxes_per_body, skipped_meshes


def _box_sdf_grid(grid_xyz, box_pos, box_rotmat, half_size):
    """Vectorized signed-distance from a grid of points to one oriented box.

    Args:
        grid_xyz: (N, 3) world-frame points (we'll reshape outside).
        box_pos: (3,) box center in world frame.
        box_rotmat: (3, 3) world←box rotation; columns are box's local axes in world frame.
        half_size: (3,) box half-extents along its local axes.

    Returns:
        (N,) signed distance per point. Negative inside the box, zero on
        surface, positive outside.

    SDF formula (standard for an oriented box):
        p_local = R^T (p_world - box_pos)
        q = |p_local| - half_size
        sdf = ||max(q, 0)|| + min(max(q), 0)
    The first term handles "outside in some axes", the second handles "inside in all axes".
    """
    delta = grid_xyz - box_pos                              # (N, 3) world-frame offset
    p_local = delta @ box_rotmat                             # (N, 3) box-local; R^T·v = v·R
    q = np.abs(p_local) - half_size                          # (N, 3)
    outside_norm = np.linalg.norm(np.maximum(q, 0.0), axis=-1)  # (N,)
    inside_max = np.minimum(np.max(q, axis=-1), 0.0)            # (N,)
    return outside_norm + inside_max


def _compute_sdf_over_grid(grid_xyz, boxes_per_body, exclude_bodies=()):
    """Compute SDF over a grid by taking the min over all boxes (excluding any in `exclude_bodies`).

    Boxes are processed body-by-body and the per-cell minimum is updated
    incrementally — much more memory-efficient than stacking all per-box
    distance fields and reducing.
    """
    sdf = np.full(grid_xyz.shape[0], np.inf)
    n_boxes_used = 0
    for body_name, boxes in boxes_per_body.items():
        if body_name in exclude_bodies:
            continue
        for box in boxes:
            d = _box_sdf_grid(grid_xyz, box["pos"], box["rotmat"], box["half_size"])
            sdf = np.minimum(sdf, d)
            n_boxes_used += 1
    return sdf, n_boxes_used


def build_sdf_for_env(env, suite, task,
                       bounds_min=DEFAULT_BOUNDS_MIN,
                       bounds_max=DEFAULT_BOUNDS_MAX,
                       voxel_size=DEFAULT_VOXEL_SIZE,
                       cache_path=None,
                       force_rebuild=False):
    """Build the SDF for the current env, caching to disk.

    Returns the cache dict (see module docstring for shape). The dict has
    `sdf_full` (all obstacles) plus `sdf_per_body[body_name]` (excluding
    that body's geoms — used when the body is held by the gripper).
    """
    bounds_min = np.asarray(bounds_min, dtype=np.float64)
    bounds_max = np.asarray(bounds_max, dtype=np.float64)

    if cache_path is None:
        out_dir = os.path.expanduser(f"~/libero_keypoint_project/outputs/v2_{suite}_{task}")
        os.makedirs(out_dir, exist_ok=True)
        cache_path = os.path.join(out_dir, "sdf_cache.pkl")

    if not force_rebuild and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        # Validate cache parameters match request — recompute if bounds/voxel changed.
        if (np.allclose(cache["bounds_min"], bounds_min)
                and np.allclose(cache["bounds_max"], bounds_max)
                and abs(cache["voxel_size"] - voxel_size) < 1e-9):
            print(f"[build_sdf] using cached SDF from {cache_path}")
            return cache
        print(f"[build_sdf] cache parameters differ from request — rebuilding")

    sim = env.env.sim
    print(f"[build_sdf] enumerating obstacle geoms")
    boxes_per_body, skipped_meshes = _enumerate_obstacle_boxes(sim)
    n_boxes = sum(len(v) for v in boxes_per_body.values())
    print(f"[build_sdf]   {len(boxes_per_body)} bodies, {n_boxes} boxes "
          f"(skipped {skipped_meshes} mesh visuals)")

    # Build the voxel grid. nx/ny/nz are cell counts; cell centers are at
    # bounds_min + (i + 0.5) * voxel_size etc.
    nx = int(np.ceil((bounds_max[0] - bounds_min[0]) / voxel_size))
    ny = int(np.ceil((bounds_max[1] - bounds_min[1]) / voxel_size))
    nz = int(np.ceil((bounds_max[2] - bounds_min[2]) / voxel_size))
    print(f"[build_sdf]   grid {nx}x{ny}x{nz} = {nx*ny*nz} cells at {voxel_size*100:.1f} cm resolution")

    xs = bounds_min[0] + (np.arange(nx) + 0.5) * voxel_size
    ys = bounds_min[1] + (np.arange(ny) + 0.5) * voxel_size
    zs = bounds_min[2] + (np.arange(nz) + 0.5) * voxel_size
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_xyz = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    # Full SDF (everything is an obstacle)
    print(f"[build_sdf] computing sdf_full...")
    t0 = time.perf_counter()
    sdf_flat, n_used = _compute_sdf_over_grid(grid_xyz, boxes_per_body)
    sdf_full = sdf_flat.reshape(nx, ny, nz)
    print(f"[build_sdf]   sdf_full: {n_used} boxes evaluated, {time.perf_counter()-t0:.1f}s")

    # Per-holdable-body SDF: exclude that body's boxes
    sdf_per_body = {}
    for body_name in HOLDABLE_BODIES:
        if body_name not in boxes_per_body:
            print(f"[build_sdf]   WARNING: holdable body '{body_name}' not in scene; skipping")
            continue
        t0 = time.perf_counter()
        sdf_flat, n_used = _compute_sdf_over_grid(
            grid_xyz, boxes_per_body, exclude_bodies={body_name},
        )
        sdf_per_body[body_name] = sdf_flat.reshape(nx, ny, nz)
        print(f"[build_sdf]   sdf_excluding[{body_name}]: {n_used} boxes, "
              f"{time.perf_counter()-t0:.1f}s")

    cache = {
        "suite": suite,
        "task": task,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "voxel_size": voxel_size,
        "shape": (nx, ny, nz),
        "sdf_full": sdf_full,
        "sdf_per_body": sdf_per_body,
        "boxes_per_body": boxes_per_body,  # for visualization / debugging
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[build_sdf] saved cache: {cache_path} "
          f"({sum(arr.nbytes for arr in [sdf_full] + list(sdf_per_body.values())) / 1e6:.1f} MB)")
    return cache


# ---------------------------------------------------------------------------
# Standalone run: build the SDF and print a quick summary
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    import sys
    sys.path.insert(0, os.environ.get("LIBERO_PATH",
                                      os.path.expanduser("~/LIBERO")))
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()["libero_object"]()
    env = OffScreenRenderEnv(
        bddl_file_name=bench.get_task_bddl_file_path(0),
        camera_heights=128, camera_widths=128, camera_depths=False,
    )
    env.reset()
    cache = build_sdf_for_env(env, "libero_object", 0, force_rebuild=True)
    env.close()

    sdf = cache["sdf_full"]
    print(f"\n[summary]")
    print(f"  shape           : {sdf.shape}")
    print(f"  min / mean / max: {sdf.min()*100:.1f} cm / {sdf.mean()*100:.1f} cm / {sdf.max()*100:.1f} cm")
    print(f"  fraction inside (sdf<0): {(sdf < 0).mean()*100:.2f}%")
    print(f"  fraction near surface (|sdf|<2cm): {(np.abs(sdf) < 0.02).mean()*100:.2f}%")
    print(f"  per-body excluded SDFs: {list(cache['sdf_per_body'].keys())}")
