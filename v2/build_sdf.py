"""Build a voxel SDF for the LIBERO scene from per-body box collision proxies.
Output (per task): sdf_full + sdf_per_body[name] (the latter excludes a held body's geoms).
Cached to outputs/v2_<suite>_<task>/sdf_cache.pkl."""
import os
import pickle
import time
import numpy as np


HOLDABLE_BODIES = {"alphabet_soup_1_main", "milk_1_main"}
SKIP_BODY_PREFIXES = ("robot0", "gripper0", "world", "table", "floor", "wall", "mount")

DEFAULT_BOUNDS_MIN = np.array([-0.40, -0.50, 0.00])
DEFAULT_BOUNDS_MAX = np.array([ 0.70,  0.50, 0.60])
DEFAULT_VOXEL_SIZE = 0.02


def _enumerate_obstacle_boxes(sim):
    # Boxes are the collision proxies; meshes are visual-only. Skipping meshes
    # is what gets the SDF build down from minutes to seconds.
    boxes_per_body = {}
    skipped_meshes = 0
    for i in range(sim.model.ngeom):
        body_id = int(sim.model.geom_bodyid[i])
        body_name = sim.model.body_id2name(body_id)
        if body_name is None or any(body_name.startswith(p) for p in SKIP_BODY_PREFIXES):
            continue
        gtype = int(sim.model.geom_type[i])
        if gtype == 7:
            skipped_meshes += 1
            continue
        if gtype != 6:
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
    """Standard oriented-box SDF. Negative inside, positive outside."""
    p_local = (grid_xyz - box_pos) @ box_rotmat
    q = np.abs(p_local) - half_size
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)


def _compute_sdf_over_grid(grid_xyz, boxes_per_body, exclude_bodies=()):
    sdf = np.full(grid_xyz.shape[0], np.inf)
    n_boxes_used = 0
    for body_name, boxes in boxes_per_body.items():
        if body_name in exclude_bodies:
            continue
        for box in boxes:
            sdf = np.minimum(sdf, _box_sdf_grid(grid_xyz, box["pos"], box["rotmat"], box["half_size"]))
            n_boxes_used += 1
    return sdf, n_boxes_used


def build_sdf_for_env(env, suite, task,
                       bounds_min=DEFAULT_BOUNDS_MIN,
                       bounds_max=DEFAULT_BOUNDS_MAX,
                       voxel_size=DEFAULT_VOXEL_SIZE,
                       cache_path=None,
                       force_rebuild=False):
    bounds_min = np.asarray(bounds_min, dtype=np.float64)
    bounds_max = np.asarray(bounds_max, dtype=np.float64)

    if cache_path is None:
        out_dir = os.path.expanduser(f"~/libero_keypoint_project/outputs/v2_{suite}_{task}")
        os.makedirs(out_dir, exist_ok=True)
        cache_path = os.path.join(out_dir, "sdf_cache.pkl")

    if not force_rebuild and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
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

    nx = int(np.ceil((bounds_max[0] - bounds_min[0]) / voxel_size))
    ny = int(np.ceil((bounds_max[1] - bounds_min[1]) / voxel_size))
    nz = int(np.ceil((bounds_max[2] - bounds_min[2]) / voxel_size))
    print(f"[build_sdf]   grid {nx}x{ny}x{nz} = {nx*ny*nz} cells at {voxel_size*100:.1f} cm resolution")

    xs = bounds_min[0] + (np.arange(nx) + 0.5) * voxel_size
    ys = bounds_min[1] + (np.arange(ny) + 0.5) * voxel_size
    zs = bounds_min[2] + (np.arange(nz) + 0.5) * voxel_size
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_xyz = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    # Flip sign — calculate_collision_cost expects positive-inside SDF (upstream ReKep does the same).
    print(f"[build_sdf] computing sdf_full...")
    t0 = time.perf_counter()
    sdf_flat, n_used = _compute_sdf_over_grid(grid_xyz, boxes_per_body)
    sdf_full = (-sdf_flat).reshape(nx, ny, nz)
    print(f"[build_sdf]   sdf_full: {n_used} boxes evaluated, {time.perf_counter()-t0:.1f}s")

    sdf_per_body = {}
    for body_name in HOLDABLE_BODIES:
        if body_name not in boxes_per_body:
            print(f"[build_sdf]   WARNING: holdable body '{body_name}' not in scene; skipping")
            continue
        t0 = time.perf_counter()
        sdf_flat, n_used = _compute_sdf_over_grid(
            grid_xyz, boxes_per_body, exclude_bodies={body_name},
        )
        sdf_per_body[body_name] = (-sdf_flat).reshape(nx, ny, nz)
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
        "boxes_per_body": boxes_per_body,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[build_sdf] saved cache: {cache_path} "
          f"({sum(arr.nbytes for arr in [sdf_full] + list(sdf_per_body.values())) / 1e6:.1f} MB)")
    return cache


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    import sys
    sys.path.insert(0, os.environ.get("LIBERO_PATH", os.path.expanduser("~/LIBERO")))
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
