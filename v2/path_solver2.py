import os
import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.interpolate import RegularGridInterpolator
import copy
import time
import transform_utils as T
from utils import (
    farthest_point_sampling,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
    normalize_vars,
    unnormalize_vars,
    get_samples_jitted,
    calculate_collision_cost,
    path_length,
    transform_keypoints,
)

# Calibrated by the weight sweep. 100 is the smallest weight where all collision-
# cloud configs (1/8/27 pt) reliably detour AND the gripper-volume effect is visible.
COLLISION_WEIGHT = 100.0
PATH_LENGTH_WEIGHT = 4.0
# 200 matches subgoal_solver. Bigger = stricter path constraints, longer paths.
PATH_CONSTRAINT_WEIGHT = float(os.environ.get("REKEP_PATH_CONSTRAINT_WEIGHT", "200.0"))
ROTATION_BOUND = np.pi

# Cold-solve mode flags. Defaults: dual_annealing only, no local search.
#   REKEP_PATH_LOCAL_SEARCH=1   — SLSQP refinement inside dual_annealing. Correct
#                                 but ~10× slower (high-dim parameter space).
#   REKEP_PATH_NO_ANNEAL=1      — skip dual_annealing entirely. Linear-interp +
#                                 SLSQP gets stuck in collision basins. Not recommended.
#   REKEP_PATH_REFINE_SLSQP=1   — keep annealing exploration, then one SLSQP pass
#                                 to gradient-refine soft constraints to zero.
# REKEP_PATH_NO_ANNEAL takes precedence.
PATH_LOCAL_SEARCH = os.environ.get("REKEP_PATH_LOCAL_SEARCH", "0") == "1"
PATH_NO_ANNEAL = os.environ.get("REKEP_PATH_NO_ANNEAL", "0") == "1"
PATH_REFINE_SLSQP = os.environ.get("REKEP_PATH_REFINE_SLSQP", "0") == "1"


def objective(opt_vars,
                og_bounds,
                start_pose,
                end_pose,
                keypoints_centered,
                keypoint_movable_mask,
                path_constraints,
                sdf_func,
                collision_points_centered,
                opt_interpolate_pos_step_size,
                opt_interpolate_rot_step_size,
                return_debug_dict=False):

    debug_dict = {}
    debug_dict['num_control_points'] = len(opt_vars) // 6

    unnormalized_opt_vars = unnormalize_vars(opt_vars, og_bounds)
    control_points_euler = np.concatenate(
        [start_pose[None], unnormalized_opt_vars.reshape(-1, 6), end_pose[None]], axis=0
    )
    control_points_homo = T.convert_pose_euler2mat(control_points_euler)
    control_points_quat = T.convert_pose_mat2quat(control_points_homo)
    poses_quat, num_poses = get_samples_jitted(
        control_points_homo, control_points_quat,
        opt_interpolate_pos_step_size, opt_interpolate_rot_step_size,
    )
    poses_homo = T.convert_pose_quat2mat(poses_quat)
    debug_dict['num_poses'] = num_poses
    start_idx, end_idx = 1, num_poses - 1  # exclude start and goal

    cost = 0
    if collision_points_centered is not None:
        # 5 cm margin. Tried 2 cm → regressed the grasp.
        collision_cost = COLLISION_WEIGHT * calculate_collision_cost(
            poses_homo[start_idx:end_idx], sdf_func, collision_points_centered, 0.05
        )
        debug_dict['collision_cost'] = collision_cost
        cost += collision_cost

    pos_length, rot_length = path_length(poses_homo)
    path_length_cost = PATH_LENGTH_WEIGHT * (pos_length + rot_length)
    debug_dict['path_length_cost'] = path_length_cost
    cost += path_length_cost

    # Upstream had a reachability cost here. Dropped — OSC solves IK every
    # control tick, so planning-time IK validation is redundant as long as
    # bounds stay inside Franka's workspace.

    debug_dict['path_violation'] = None
    if path_constraints is not None and len(path_constraints) > 0:
        path_constraint_cost = 0
        path_violation = []
        for pose in poses_homo[start_idx:end_idx]:
            transformed_keypoints = transform_keypoints(pose, keypoints_centered, keypoint_movable_mask)
            for constraint in path_constraints:
                violation = constraint(transformed_keypoints[0], transformed_keypoints[1:])
                path_violation.append(violation)
                path_constraint_cost += np.clip(violation, 0, np.inf)
        path_constraint_cost = PATH_CONSTRAINT_WEIGHT * path_constraint_cost
        debug_dict['path_constraint_cost'] = path_constraint_cost
        debug_dict['path_violation'] = path_violation
        cost += path_constraint_cost

    debug_dict['total_cost'] = cost

    if return_debug_dict:
        return cost, debug_dict
    return cost


class PathSolver:
    """Sequence of intermediate poses between start and goal.
    Decision variables: the intermediate control points (start/goal fixed)."""

    def __init__(self, config, ik_solver, reset_joint_pos):
        # ik_solver / reset_joint_pos kept for API stability with v2/main.py
        # and scripts/execute.py; not used now that IK is OSC's job.
        self.config = config
        self.ik_solver = ik_solver
        self.reset_joint_pos = reset_joint_pos
        self.last_opt_result = None
        self._warmup()

    def _warmup(self):
        # In-bounds poses with gripper-down orientation. Upstream used
        # z=0.0/0.3 which is below LIBERO's table — fine for annealing-only
        # but causes SLSQP local refinement to thrash on the bounds boundary.
        bm = np.asarray(self.config["bounds_min"], dtype=np.float64)
        bM = np.asarray(self.config["bounds_max"], dtype=np.float64)
        gripper_down_xyzw = np.array([1.0, 0.0, 0.0, 0.0])
        start_pose = np.concatenate([[bm[0] + 0.10, 0.0, bm[2] + 0.30], gripper_down_xyzw])
        end_pose   = np.concatenate([[bM[0] - 0.10, 0.0, bm[2] + 0.30], gripper_down_xyzw])
        keypoints = np.zeros((1, 3))
        keypoint_movable_mask = np.array([False])
        sdf_voxels = np.full((10, 10, 10), -1.0)  # empty-space SDF
        collision_points = np.array([start_pose[:3]])
        self.solve(start_pose, end_pose, keypoints, keypoint_movable_mask,
                   [], sdf_voxels, collision_points, None, from_scratch=True)
        self.last_opt_result = None

    def _setup_sdf(self, sdf_voxels):
        # Cell-center grid; matches build_sdf.py.
        # fill_value=-1.0: outside-grid = empty. fill_value=0 caused phantom
        # collision cost in the half-voxel dead-zone at bounds_min/max.
        nx, ny, nz = sdf_voxels.shape
        bm = np.asarray(self.config['bounds_min'], dtype=np.float64)
        bM = np.asarray(self.config['bounds_max'], dtype=np.float64)
        vs = (bM - bm) / np.array([nx, ny, nz], dtype=np.float64)
        x = bm[0] + (np.arange(nx) + 0.5) * vs[0]
        y = bm[1] + (np.arange(ny) + 0.5) * vs[1]
        z = bm[2] + (np.arange(nz) + 0.5) * vs[2]
        return RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=-1.0)

    def _check_opt_result(self, opt_result, path_quat, debug_dict, og_bounds):
        # Accept opt_result if it only failed due to iteration cap.
        if (not opt_result.success and ('maximum' in opt_result.message.lower()
                or 'iteration' in opt_result.message.lower()
                or 'not necessarily' in opt_result.message.lower())):
            opt_result.success = True
        elif not opt_result.success:
            opt_result.message += '; invalid solution'
        if debug_dict['path_violation'] is not None:
            path_violation = np.array(debug_dict['path_violation'])
            opt_result.message += f'; path_violation: {path_violation}'
            if not all(v <= self.config['constraint_tolerance'] for v in path_violation):
                opt_result.success = False
                opt_result.message += '; path constraint not satisfied'
        return opt_result

    def _center_collision_points_and_keypoints(self, ee_pose, collision_points, keypoints, keypoint_movable_mask):
        ee_pose_homo = T.pose2mat([ee_pose[:3], T.euler2quat(ee_pose[3:])])
        centering_transform = np.linalg.inv(ee_pose_homo)
        collision_points_centered = np.dot(collision_points, centering_transform[:3, :3].T) + centering_transform[:3, 3]
        keypoints_centered = transform_keypoints(centering_transform, keypoints, keypoint_movable_mask)
        return collision_points_centered, keypoints_centered

    def solve(self,
            start_pose,
            end_pose,
            keypoints,
            keypoint_movable_mask,
            path_constraints,
            sdf_voxels,
            collision_points,
            initial_joint_pos,
            from_scratch=False):
        """start_pose / end_pose : [7] xyz+xyzw quat
        sdf_voxels : [H, W, D]
        collision_points : [N, 3] world-frame held-object samples
        initial_joint_pos : retained for API parity; unused (OSC owns IK)."""
        if collision_points is not None and collision_points.shape[0] > self.config['max_collision_points']:
            collision_points = farthest_point_sampling(collision_points, self.config['max_collision_points'])
        sdf_func = self._setup_sdf(sdf_voxels)

        num_control_points = get_linear_interpolation_steps(
            start_pose, end_pose,
            self.config['opt_pos_step_size'], self.config['opt_rot_step_size'],
        )
        num_control_points = np.clip(num_control_points, 3, 6)
        start_pose = np.concatenate([start_pose[:3], T.quat2euler(start_pose[3:])])
        end_pose = np.concatenate([end_pose[:3], T.quat2euler(end_pose[3:])])

        og_bounds = [(b_min, b_max) for b_min, b_max in zip(self.config['bounds_min'], self.config['bounds_max'])] + \
                    [(-ROTATION_BOUND, ROTATION_BOUND) for _ in range(3)]
        og_bounds *= (num_control_points - 2)
        og_bounds = np.array(og_bounds, dtype=np.float64)
        bounds = [(-1, 1)] * len(og_bounds)
        num_vars = len(bounds)

        # Warm-start from previous solve when available.
        if not from_scratch and self.last_opt_result is not None:
            init_sol = self.last_opt_result.x
            if len(init_sol) < num_vars:
                # Extend with noisy copies of the last control point.
                new_x0 = np.empty(num_vars)
                new_x0[:len(init_sol)] = init_sol
                for i in range(len(init_sol), num_vars, 6):
                    new_x0[i:i+6] = init_sol[-6:] + np.random.randn(6) * 0.01
                init_sol = new_x0
            else:
                init_sol = init_sol[-num_vars:]
        else:
            from_scratch = True
            interp_poses = linear_interpolate_poses(start_pose, end_pose, num_control_points)
            init_sol = interp_poses[1:-1].flatten()
            init_sol = normalize_vars(init_sol, og_bounds)

        for i, (b_min, b_max) in enumerate(bounds):
            init_sol[i] = np.clip(init_sol[i], b_min, b_max)

        collision_points_centered, keypoints_centered = self._center_collision_points_and_keypoints(
            start_pose, collision_points, keypoints, keypoint_movable_mask,
        )
        aux_args = (og_bounds, start_pose, end_pose,
                    keypoints_centered, keypoint_movable_mask, path_constraints,
                    sdf_func, collision_points_centered,
                    self.config['opt_interpolate_pos_step_size'],
                    self.config['opt_interpolate_rot_step_size'])

        start = time.time()
        if from_scratch and not PATH_NO_ANNEAL:
            opt_result = dual_annealing(
                func=objective,
                bounds=bounds,
                args=aux_args,
                maxfun=self.config['sampling_maxfun'],
                x0=init_sol,
                no_local_search=not PATH_LOCAL_SEARCH,
                minimizer_kwargs={
                    'method': 'SLSQP',
                    'options': self.config['minimizer_options'],
                },
            )
            if PATH_REFINE_SLSQP:
                # Gradient-refine soft constraints once we're in a collision-free basin.
                refine_result = minimize(
                    fun=objective, x0=opt_result.x, args=aux_args, bounds=bounds,
                    method='SLSQP', options=self.config['minimizer_options'],
                )
                # Take whichever has lower cost — SLSQP can leave the basin if gradient geometry is bad.
                if refine_result.fun < opt_result.fun:
                    opt_result = refine_result
        else:
            opt_result = minimize(
                fun=objective, x0=init_sol, args=aux_args, bounds=bounds,
                method='SLSQP', options=self.config['minimizer_options'],
            )
        solve_time = time.time() - start

        if isinstance(opt_result.message, list):
            opt_result.message = opt_result.message[0]
        _, debug_dict = objective(opt_result.x, *aux_args, return_debug_dict=True)
        debug_dict['sol'] = opt_result.x.reshape(-1, 6)
        debug_dict['solve_time'] = solve_time
        debug_dict['from_scratch'] = from_scratch
        debug_dict['type'] = 'path_solver'

        sol = unnormalize_vars(opt_result.x, og_bounds)
        poses_euler = np.concatenate([sol.reshape(-1, 6), end_pose[None]], axis=0)
        poses_quat = T.convert_pose_euler2quat(poses_euler)
        opt_result = self._check_opt_result(opt_result, poses_quat, debug_dict, og_bounds)
        # Capture msg/success AFTER _check_opt_result — it appends violation
        # info and can flip success → False. Upstream ReKep captures before
        # the check and silently hides constraint failures from callers.
        debug_dict['msg'] = opt_result.message
        debug_dict['success'] = opt_result.success
        # Cache unconditionally — `x` is a warm-start hint, not a satisfaction claim.
        self.last_opt_result = copy.deepcopy(opt_result)
        return poses_quat, debug_dict
