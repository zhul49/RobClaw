# wraps cuRobo's Mapper for the tabletop workspace.
# volume: x[0.20, 0.75], y[-0.60, 0.60], z[0.02, 0.60] in robot-base frame.
# voxel: 0.5 cm. (see project_real_world_collision_design memory.)
#
# Mapper.integrate() expects a batched CameraObservation (N cams). camera_observer
# builds that. compute_esdf() materializes a VoxelGrid for the planner.

import torch

from curobo.perception import FilterDepth, Mapper, MapperCfg


# workspace floor at z=0.02, just above the table top — table noise stays
# outside the workspace so it doesn't show in the map. arm can't go below
# this anyway (Franka is bolted to the table edge).
WORKSPACE_BOUNDS_MIN = (0.20, -0.60, 0.02)
WORKSPACE_BOUNDS_MAX = (0.75,  0.60, 0.60)
VOXEL_SIZE_M = 0.005

DEPTH_MIN_M = 0.10   # ignore depth closer than this (RealSense unreliable)
DEPTH_MAX_M = 2.50   # ignore beyond — both cams are ~1 m from the workspace


def _extent_and_center(bmin, bmax):
    extent = tuple(float(bmax[i] - bmin[i]) for i in range(3))
    center = tuple(float(0.5 * (bmax[i] + bmin[i])) for i in range(3))
    return extent, center


class WorldMapper:
    # construct once, integrate many CameraObservations, then call compute_esdf()
    # before handing the world to the planner.

    def __init__(self, num_cameras, image_height, image_width, device="cuda:0"):
        extent, center = _extent_and_center(WORKSPACE_BOUNDS_MIN, WORKSPACE_BOUNDS_MAX)
        self.cfg = MapperCfg(
            voxel_size=VOXEL_SIZE_M,
            extent_meters_xyz=extent,
            grid_center=torch.tensor(center, dtype=torch.float32),
            truncation_distance=VOXEL_SIZE_M * 3,   # smaller = less back-side inflation
            depth_maximum_distance=DEPTH_MAX_M,
            depth_minimum_distance=DEPTH_MIN_M,
            minimum_tsdf_weight=6.0,                # higher = fewer noise voxels
            decay_factor=0.90,                      # <1 = old observations fade; lower = faster
            frustum_decay_factor=0.90,              # same idea for in-frustum-but-unseen voxels
            num_cameras=num_cameras,
            image_height=image_height,
            image_width=image_width,
        )
        self.mapper = Mapper(self.cfg)
        self.depth_filter = FilterDepth(
            image_shape=(image_height, image_width),
            depth_minimum_distance=DEPTH_MIN_M,
            depth_maximum_distance=DEPTH_MAX_M,
            flying_pixel_threshold=0.2,
            bilateral_kernel_size=5,
        )
        self.device = device

    def memory_mb(self):
        return self.mapper.memory_usage_mb()

    def integrate(self, obs):
        # mutates obs.depth_image (filtered in-place before integration).
        obs.depth_image = torch.nan_to_num(obs.depth_image, nan=0.0)
        filtered, _ = self.depth_filter(obs.depth_image)
        obs.depth_image = filtered
        self.mapper.integrate(obs)

    def compute_esdf(self):
        return self.mapper.compute_esdf()

    def extract_voxels(self, surface_only=False):
        return self.mapper.extract_occupied_voxels(surface_only=surface_only)

    def save_mesh(self, path):
        mesh = self.mapper.extract_mesh(surface_only=False)
        if mesh.vertices is None or len(mesh.vertices) == 0:
            return 0
        mesh.save_as_mesh(path)
        return len(mesh.vertices)
