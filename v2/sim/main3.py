"""ReKep pick-and-place loop for LIBERO. IK is delegated to OSC at runtime;
planning-time IK costs are dropped. To revive cuRobo see v2_dev/legacy/franka_ik_curobo.py
+ the BYPASS_IK env-var dance (memory: project_main3_curobo_removal)."""

import json
import os
import time

import numpy as np
import torch

from v2.sim.libero_env_adapter import (
    LIBEROReKepEnv,
    LIBERO_BOUNDS_MIN,
    LIBERO_BOUNDS_MAX,
)
from v2.sim.subgoal_solver import SubgoalSolver
from v2.sim.path_solver2 import PathSolver
from v2.common import transform_utils as T
from v2.common.utils import (
    get_config,
    get_callable_grasping_cost_fn,
    get_linear_interpolation_steps,
    load_functions_from_txt,
    spline_interpolate_poses,
)


class Main3:
    # runs dual_annealing 3x to find the lowest-cost answer
    SUBGOAL_RESTARTS = 3
    # compute the first 5 dense actions then replan from where we are
    ACTION_STEPS_PER_ITER = 5
    # add a safety cap for if the stage replans too much times
    MAX_ITER_PER_STAGE = 50

    def __init__(self, suite_name, task_idx, verbose=False):
        global_config = get_config()

        # Patch workspace bounds into solver configs
        for section in ("main", "path_solver", "subgoal_solver"):
            global_config[section]["bounds_min"] = LIBERO_BOUNDS_MIN.tolist()
            global_config[section]["bounds_max"] = LIBERO_BOUNDS_MAX.tolist()

        self.global_config = global_config
        self.config = global_config["main"]
        self.bounds_min = np.array(self.config["bounds_min"])
        self.bounds_max = np.array(self.config["bounds_max"])

        # keeps the same seed so traj are able to repeat since dual_annealing is stochastic
        seed = self.config["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # builds the Libero env wrapper (MuJoCo, robosuite + OSC controller, SDF building)
        self.env = LIBEROReKepEnv(suite_name, task_idx, verbose=verbose)

        # construct the two solvers
        self.subgoal_solver = SubgoalSolver( # picks where the EE should land at the end of the stage
            global_config["subgoal_solver"],
            None,
            self.env.reset_joint_pos,
        )
        self.path_solver = PathSolver( # picks how to get there with a sequence of control points that minimizes path length and obstacle cost
            global_config["path_solver"],
            None,
            self.env.reset_joint_pos,
        )

        self.program_info = None
        self.constraint_fns = None
        self.keypoint_movable_mask = None

    # loads the task from run_test.py that is given
    def load_task(self, rekep_program_dir):
        metadata_path = os.path.join(rekep_program_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.program_info = json.load(f)

        # builds the autodect to make the keypoint to the nearest MJCF bodies
        # builds the SDF cache
        self.env.register_keypoints(self.program_info["init_keypoint_positions"])

        # loads the stage and constraints
        self.constraint_fns = {}
        for stage in range(1, self.program_info["num_stages"] + 1):
            stage_dict = {}
            for constraint_type in ("subgoal", "path"):
                load_path = os.path.join(
                    rekep_program_dir,
                    f"stage{stage}_{constraint_type}_constraints.txt",
                )
                
                # returns a callable python function using exec()
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)
                stage_dict[constraint_type] = (
                    load_functions_from_txt(load_path, get_grasping_cost_fn)
                    if os.path.exists(load_path)
                    else []
                )
            self.constraint_fns[stage] = stage_dict

        # make a checklist of what currently moves with the gripper, start with nothing on it except the gripper itself
        # the mask tells it which keypoint is glued to the EE to predict where the keypoint will be at the pose
        self.keypoint_movable_mask = np.zeros(
            self.program_info["num_keypoints"] + 1, dtype=bool
        )
        self.keypoint_movable_mask[0] = True

    # takes a snapshot of everything the solvers need to know about the world right now and is called every replan iteration
    def get_observation(self, stage=None):

        # live position of each perception keypoint of the nearest body in MuJoCo
        scene_keypoints = self.env.get_keypoint_positions()
        keypoints = np.concatenate(
            [[self.env.get_ee_pos()], scene_keypoints], axis=0
        )

        # grippers current pose in world frame
        curr_ee_pose = self.env.get_ee_pose()
        curr_joint_pos = self.env.get_arm_joint_postions()

        # looks at the grasp keypoints in the metadata
        about_to_grasp_kp = None
        if stage is not None and self.program_info is not None:
            kp = self.program_info["grasp_keypoints"][stage - 1]
            if kp != -1:
                about_to_grasp_kp = kp

        # returns the SDF exluding the held body
        sdf_voxels = self.env.get_sdf_voxels(
            self.config["sdf_voxel_size"],
            about_to_grasp_kp=about_to_grasp_kp,
        )

        # returns 60 random points of the held object and the solver uses these to check whether any points of the held object collides
        collision_points = self.env.get_collision_points()
        if collision_points is None:
            collision_points = np.array([curr_ee_pose[:3]])
        return {
            "keypoints": keypoints,
            "curr_ee_pose": curr_ee_pose,
            "curr_joint_pos": curr_joint_pos,
            "sdf_voxels": sdf_voxels,
            "collision_points": collision_points,
        }

    # checks for grasp or release at which stage
    def stage_role(self, stage):
        is_grasp = self.program_info["grasp_keypoints"][stage - 1] != -1
        is_release = self.program_info["release_keypoints"][stage - 1] != -1
        return is_grasp, is_release

    # picks where the EE should end at the end of a stage
    def solve_subgoal(self, stage, from_scratch=True):
        assert self.constraint_fns is not None, "call load_task() first"

        obs = self.get_observation(stage=stage)
        is_grasp, _ = self.stage_role(stage)
        sg_cons = self.constraint_fns[stage]["subgoal"]
        path_cons = self.constraint_fns[stage]["path"]

        # multi-restart solve 3 times 
        candidates = []
        for _ in range(self.SUBGOAL_RESTARTS):
            cand_pose, cand_dbg = self.subgoal_solver.solve(
                obs["curr_ee_pose"],
                obs["keypoints"],
                self.keypoint_movable_mask,
                sg_cons, path_cons,
                obs["sdf_voxels"], obs["collision_points"],
                is_grasp,
                obs["curr_joint_pos"],
                from_scratch=True,
            )
            candidates.append((cand_pose.copy(), cand_dbg))

        # picks the best candidate with the lowest total_cost
        best_i = min(
            range(self.SUBGOAL_RESTARTS),
            key=lambda i: candidates[i][1]["total_cost"],
        )
        subgoal_pose, debug_dict = candidates[best_i]
        debug_dict["stage"] = stage
        debug_dict["restart_chosen"] = best_i

        subgoal_pose_pre_backoff = subgoal_pose.copy()
        if is_grasp:
            # hover the grip site 5 cm above the can
            sg_homo = T.convert_pose_quat2mat(subgoal_pose)
            subgoal_pose[:3] += sg_homo[:3, :3] @ np.array(
                [0, 0, -self.config["grasp_depth"] / 2.0]
            )
        return subgoal_pose, subgoal_pose_pre_backoff, debug_dict

    # calls the subgoal solver to get the goal pose
    def solve_path(self, stage, from_scratch=True):
        assert self.constraint_fns is not None, "call load_task() first"

        subgoal_pose, _sg_pre, sg_debug = self.solve_subgoal(
            stage, from_scratch=from_scratch
        )

        obs = self.get_observation(stage=stage)
        path_constraints = self.constraint_fns[stage]["path"]
        raw_path, path_debug = self.path_solver.solve(
            obs["curr_ee_pose"], subgoal_pose,
            obs["keypoints"], self.keypoint_movable_mask,
            path_constraints,
            obs["sdf_voxels"], obs["collision_points"],
            obs["curr_joint_pos"],
            from_scratch=from_scratch,
        )
        path_debug["stage"] = stage
        path_debug["subgoal_debug"] = sg_debug
        return raw_path, subgoal_pose, path_debug

    # takes the path solver's 3-6 sparse waypoints turns them into a sequence of executable actions to fill the gaps
    def process_path(self, path):
        curr_ee_pose = self.env.get_ee_pose()
        full_control_points = np.concatenate(
            [curr_ee_pose.reshape(1, -1), path], axis=0
        )
        num_steps = get_linear_interpolation_steps(
            full_control_points[0], full_control_points[-1],
            self.config["interpolate_pos_step_size"],
            self.config["interpolate_rot_step_size"],
        )
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    # path solver has it in the hover pose at 5 cm above, this step lunges it down 5 cm for grasping
    def _execute_grasp_action(self, stage):
        pregrasp_pose = self.env.get_ee_pose()
        finger_pre = self.env.get_finger_qpos()
        can_pre = self.env.get_object_pose(0)

        # lunge along +Z by half grasp_depth to cancel the matching solve_subgoal
        # backoff so EE lands at the constraint optimum.
        R = T.quat2mat(pregrasp_pose[3:])
        lunge = R @ np.array([0, 0, self.config["grasp_depth"] / 2.0])
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += lunge

        # precise=True streams ticks for up to 40 frames or until the EE settles within 5 mm of the target
        self.env.execute_action(
            np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]]),
            precise=True,
        )

        grasp_kp = self.program_info["grasp_keypoints"][stage - 1]
        # bookkeeping
        self.env.set_grasping(grasp_kp)
        self.env.mark_event(f"stage{stage}_grasp")

        # after grasp, go up 5 cm and pulls 60 collision points from inside the object
        post_lift = grasp_pose.copy()
        post_lift[2] += 0.05

        self.env.execute_action(
            np.concatenate([post_lift, [self.env.get_gripper_close_action()]]),
            precise=True,
        )
        self.env.mark_event(f"stage{stage}_post_grasp_lift")

        finger_post = self.env.get_finger_qpos()
        can_post = self.env.get_object_pose(0)
        return {
            "finger_qpos_pre": finger_pre,
            "finger_qpos_post": finger_post,
            "can_pre_grasp": can_pre,
            "can_post_grasp": can_post,
            "rot_matrix": R,
        }

    # mirror of grasp action
    def _execute_release_action(self, stage):
        finger_pre = self.env.get_finger_qpos()
        self.env.open_gripper()
        release_kp = self.program_info["release_keypoints"][stage - 1]

        # SDF flips back to full and the movable mask would unflip slot 1
        self.env.clear_grasping(release_kp)
        self.env.mark_event(f"stage{stage}_release")
        finger_post = self.env.get_finger_qpos()
        return {
            "finger_qpos_pre": finger_pre,
            "finger_qpos_post": finger_post,
        }

    # synchronize the movable mask with the env's current grasp state.
    def _update_keypoint_movable_mask(self):
        grasped = self.env.get_grasped_keypoints()
        for i in range(1, len(self.keypoint_movable_mask)):
            self.keypoint_movable_mask[i] = ((i - 1) in grasped)

    # called everytime we begin or re-enter a stage
    def _update_stage(self, stage):
        assert self.constraint_fns is not None, "call load_task() first"

        # checks what stage this is from metadata
        is_grasp, is_release = self.stage_role(stage)

        # label the logs
        self.env.set_log_stage(stage)
        self.env.mark_event(f"stage{stage}_start")

        # open gripper during grasp
        if is_grasp:
            self.env.open_gripper()

        # update mask tracker
        self._update_keypoint_movable_mask()
        return is_grasp, is_release

    # single-stage open-loop run, kept for debugging. Production path is execute_task
    def execute_stage(self, stage):
        """Single-stage open-loop run, kept for debugging.
        Production path is execute_task."""
        is_grasp, is_release = self._update_stage(stage)
        raw_path, subgoal_pose, debug_dict = self.solve_path(stage, from_scratch=True)
        dense_actions = self.process_path(raw_path)

        ee_before_stream = self.env.get_ee_pose()
        for i, action in enumerate(dense_actions):
            is_last = (i == dense_actions.shape[0] - 1)
            self.env.execute_action(action, precise=is_last)
        ee_after_stream = self.env.get_ee_pose()
        tracking_error_m = float(np.linalg.norm(ee_after_stream[:3] - subgoal_pose[:3]))

        grasp_info = release_info = None
        if is_grasp:
            grasp_info = self._execute_grasp_action(stage)
        elif is_release:
            release_info = self._execute_release_action(stage)

        return {
            "stage": stage,
            "is_grasp_stage": is_grasp,
            "is_release_stage": is_release,
            "subgoal_pose": subgoal_pose,
            "raw_path": raw_path,
            "dense_actions": dense_actions,
            "ee_before_stream": ee_before_stream,
            "ee_after_stream": ee_after_stream,
            "tracking_error_m": tracking_error_m,
            "grasp_info": grasp_info,
            "release_info": release_info,
            "path_debug": debug_dict,
        }

    # Closed-loop main loop. Mirrors main.py:_execute's flat while-True + action_queue
    def execute_task(self):
        assert self.constraint_fns is not None

        self.stage = 1
        self.first_iter = True
        self.action_queue = []
        self.iter_count = 0
        self.stage_diag = []
        t_task_start = time.perf_counter()
        t_stage_start = t_task_start
        last_subgoal_pose = None
        last_dense_actions = None
        last_path_debug = None

        self._update_stage(self.stage)
        self.backtrack_log = []

        while True:
            
            # runs at the top of every loop iteration (only book keeping right now, no physics might not work)
            if self.stage > 1:

                # get the live state
                scene_kp = self.env.get_keypoint_positions()
                kp_check = np.concatenate([[self.env.get_ee_pos()], scene_kp], axis=0)

                # check if constraints are violated
                path_cons = self.constraint_fns[self.stage]["path"]

                # anything above 0.1 means this is broken and we backtrack
                tol = self.config["constraint_tolerance"]
                if any(c(kp_check[0], kp_check[1:]) > tol for c in path_cons):

                    # finds the highest earlier stage whos constraints all hold and walk backwards
                    new_stage = 1
                    for s in range(self.stage - 1, 0, -1):
                        sc = self.constraint_fns[s]["path"]
                        if len(sc) == 0 or all(c(kp_check[0], kp_check[1:]) <= tol for c in sc):
                            new_stage = s
                            break

                    # log and rewind
                    self.env.mark_event(f"backtrack_stage{self.stage}_to_{new_stage}")
                    print(f"[backtrack] stage {self.stage} -> {new_stage} "
                          f"(at iter {self.iter_count})")
                    self.backtrack_log.append({
                        "from_stage": self.stage,
                        "to_stage": new_stage,
                        "at_iter": self.iter_count,
                    })
                    self.stage = new_stage
                    self.iter_count = 0
                    self.first_iter = True
                    self.action_queue = []
                    t_stage_start = time.perf_counter()
                    self._update_stage(self.stage)
                    continue

            self.iter_count += 1
            if self.iter_count > self.MAX_ITER_PER_STAGE:
                raise RuntimeError(
                    f"stage {self.stage}: exceeded MAX_ITER_PER_STAGE "
                    f"({self.MAX_ITER_PER_STAGE})"
                )

            # solve subgoal + path and densify into actions
            raw_path, subgoal_pose, path_debug = self.solve_path(
                self.stage, from_scratch=self.first_iter
            )
            self.action_queue = list(self.process_path(raw_path))
            self.first_iter = False
            last_subgoal_pose = subgoal_pose
            last_dense_actions = np.array(self.action_queue)
            last_path_debug = path_debug

            # pop and run upto 5 actions
            n = min(self.ACTION_STEPS_PER_ITER, len(self.action_queue))
            for _ in range(n):
                action = self.action_queue.pop(0)
                will_be_empty = len(self.action_queue) == 0
                self.env.execute_action(action, precise=will_be_empty)

            # grasp/release and advance to next state
            if len(self.action_queue) == 0:
                is_grasp, is_release = self.stage_role(self.stage)
                grasp_info = release_info = None
                if is_grasp:
                    grasp_info = self._execute_grasp_action(self.stage)
                elif is_release:
                    release_info = self._execute_release_action(self.stage)

                ee_after = self.env.get_ee_pose()
                tracking_error_m = float(np.linalg.norm(ee_after[:3] - last_subgoal_pose[:3]))
                t_stage_end = time.perf_counter()
                self.stage_diag.append({
                    "stage": self.stage,
                    "is_grasp_stage": is_grasp,
                    "is_release_stage": is_release,
                    "iters": self.iter_count,
                    "wall_time_s": t_stage_end - t_stage_start,
                    "subgoal_pose": last_subgoal_pose,
                    "dense_actions_last": last_dense_actions,
                    "ee_after_stage": ee_after,
                    "tracking_error_m": tracking_error_m,
                    "grasp_info": grasp_info,
                    "release_info": release_info,
                    "path_debug_last": last_path_debug,
                })

                # advance or break
                if self.stage == self.program_info["num_stages"]:
                    break
                self.stage += 1
                self.first_iter = True
                self.action_queue = []
                self.iter_count = 0
                self._update_stage(self.stage)
                t_stage_start = time.perf_counter()

        return {
            "per_stage": self.stage_diag,
            "total_iters": sum(d["iters"] for d in self.stage_diag),
            "wall_time_s": time.perf_counter() - t_task_start,
            "backtracks": self.backtrack_log,
        }
