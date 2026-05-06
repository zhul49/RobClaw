import numpy as np
from typing import List, Optional


# same TCP-offset constants as franka_ik_curobo.py
DEFAULT_TCP_OFFSET_Z = 0.0965
DEFAULT_TCP_ROT_Z = np.pi / 2

PANDA_NEUTRAL_QPOS = np.array(
    [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4, 0.0]
)


class CuRoboMotionPlanner:

    def __init__(
        self,
        world2robot_homo: Optional[np.ndarray] = None,
        tcp_offset_z: float = DEFAULT_TCP_OFFSET_Z,
        tcp_rot_z: float = DEFAULT_TCP_ROT_Z,
        num_ik_seeds: int = 32,
        num_trajopt_seeds: int = 4,
        # curobo locks obstacle count at construction, might need to update
        collision_cache_obb: int = 128,
        obstacle_activation_distance: float = 0.02,
    ):
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        self._torch = torch
        self.world2robot_homo = (
            np.asarray(world2robot_homo, dtype=np.float64)
            if world2robot_homo is not None
            else np.eye(4)
        )
        self.tcp_offset_z = float(tcp_offset_z)
        self.tcp_rot_z = float(tcp_rot_z)

        # TCP frame to curobo panda_hand frame
        c, s = np.cos(self.tcp_rot_z), np.sin(self.tcp_rot_z)
        self._tcp_to_curobo_target = np.array([
            [c, -s, 0.0, 0.0],
            [s,  c, 0.0, 0.0],
            [0.0, 0.0, 1.0, -self.tcp_offset_z],
            [0.0, 0.0, 0.0, 1.0],
        ])

        cfg = MotionPlannerCfg.create(
            robot="franka.yml",
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
            use_cuda_graph=True,
            max_goalset=4,
            collision_cache={"obb": int(collision_cache_obb)},
            optimizer_collision_activation_distance=float(obstacle_activation_distance),
        )
        self.planner = MotionPlanner(cfg)
        self.planner.warmup(enable_graph=True, num_warmup_iterations=2)
        self.tool_link = self.planner.tool_frames[0] 

    @staticmethod
    def _matrix_to_wxyz(R):
        from scipy.spatial.transform import Rotation
        x, y, z, w = Rotation.from_matrix(R).as_quat()
        return w, x, y, z

    def _box_to_cuboid(self, name, pos_world, rotmat_world, half_size):
        # convert a MuJoCo-style box (world frame) into a curobo cuboid (robot base frame)

        from curobo.scene import Cuboid
        T_world_box = np.eye(4)
        T_world_box[:3, :3] = np.asarray(rotmat_world)
        T_world_box[:3, 3] = np.asarray(pos_world)
        T_robot_box = self.world2robot_homo @ T_world_box
        pos = T_robot_box[:3, 3]
        w, x, y, z = self._matrix_to_wxyz(T_robot_box[:3, :3])
        dims = (2.0 * np.asarray(half_size, dtype=np.float64)).tolist()
        return Cuboid(
            name=name,
            pose=[float(pos[0]), float(pos[1]), float(pos[2]),
                  float(w), float(x), float(y), float(z)],
            dims=dims,
        )

    def update_world(self, boxes_per_body, exclude_bodies=()):
        # push the obstacle scene to curobo

        from curobo.scene import Scene
        excluded = set(exclude_bodies or ())
        cuboids = []
        for body_name, boxes in boxes_per_body.items():
            if body_name in excluded:
                continue
            for box in boxes:
                cuboids.append(self._box_to_cuboid(
                    f"{body_name}__{box['geom_idx']}",
                    box["pos"],
                    box["rotmat"],
                    box["half_size"],
                ))
        self.planner.update_world(Scene(cuboid=cuboids))
        return len(cuboids)

    def plan(
        self,
        # where each joint currently is. LIBERO sends 8 numbers (7 arm + 1 finger) and curobo only plans the arm, so it'll use the first 7
        start_joint_pos: np.ndarray,
        # target gripper pose, in world frame
        end_pose_xyzw: np.ndarray,
        max_attempts: int = 5,
        n_waypoints: int = 100,
    ) -> List[np.ndarray]:
        torch = self._torch
        from curobo.types import JointState, Pose, GoalToolPose
        import transform_utils as T

        end_pose_xyzw = np.asarray(end_pose_xyzw, dtype=np.float64)
        # convert the target into curobo's coordinate system
        target_world_homo = T.pose2mat([end_pose_xyzw[:3], end_pose_xyzw[3:]])
        target_robot_tcp = self.world2robot_homo @ target_world_homo
        target_for_curobo = target_robot_tcp @ self._tcp_to_curobo_target

        target_pos = target_for_curobo[:3, 3].astype(np.float32)
        w, x, y, z = self._matrix_to_wxyz(target_for_curobo[:3, :3])

        # 2. build cuRobo Pose + GoalToolPose
        goal = Pose(
            position=torch.tensor(target_pos, device="cuda", dtype=torch.float32).unsqueeze(0),
            quaternion=torch.tensor([w, x, y, z], device="cuda", dtype=torch.float32).unsqueeze(0),
        )
        goal_tool = GoalToolPose.from_poses({self.tool_link: goal}, num_goalset=1)

        # strip off the gripper finger
        q7 = np.asarray(start_joint_pos[:7], dtype=np.float32)
        start = JointState.from_position(
            torch.tensor(q7, device="cuda", dtype=torch.float32).unsqueeze(0)
        )

        # plan_pose 
        result = self.planner.plan_pose(goal_tool, start, max_attempts=max_attempts)

        if result is None or not bool(result.success.item()):
            return []
        traj = result.interpolated_trajectory.position[0, 0, :, :7].detach().cpu().numpy()
        T_full = traj.shape[0]
        last_tstep = int(result.interpolated_last_tstep[0]) if result.interpolated_last_tstep is not None else T_full
        last_tstep = max(2, min(last_tstep, T_full))
        if n_waypoints >= last_tstep:
            return [traj[i] for i in range(last_tstep)]
        idxs = np.linspace(0, last_tstep - 1, n_waypoints).astype(int)
        return [traj[i] for i in idxs]

    def plan_grasp(
        self,
        start_joint_pos: np.ndarray,
        grasp_pose_xyzw: np.ndarray,
        approach_offset: float = 0.15,
        n_waypoints_approach: int = 80,
        n_waypoints_grasp: int = 30,
        n_settle_steps: int = 30,
        max_attempts: int = 5,
    ) -> List[np.ndarray]:
        torch = self._torch
        from curobo.types import JointState, Pose, GoalToolPose
        import transform_utils as T

        grasp_pose_xyzw = np.asarray(grasp_pose_xyzw, dtype=np.float64)
        target_world_homo = T.pose2mat([grasp_pose_xyzw[:3], grasp_pose_xyzw[3:]])
        target_robot_tcp = self.world2robot_homo @ target_world_homo
        target_for_curobo = target_robot_tcp @ self._tcp_to_curobo_target

        target_pos = target_for_curobo[:3, 3].astype(np.float32)
        w, x, y, z = self._matrix_to_wxyz(target_for_curobo[:3, :3])

        goal = Pose(
            position=torch.tensor(target_pos, device="cuda", dtype=torch.float32).unsqueeze(0),
            quaternion=torch.tensor([w, x, y, z], device="cuda", dtype=torch.float32).unsqueeze(0),
        )
        goal_tool = GoalToolPose.from_poses({self.tool_link: goal}, num_goalset=1)

        q7 = np.asarray(start_joint_pos[:7], dtype=np.float32)
        start = JointState.from_position(
            torch.tensor(q7, device="cuda", dtype=torch.float32).unsqueeze(0),
            joint_names=self.planner.joint_names,
        )

        result = self.planner.plan_grasp(
            grasp_poses=goal_tool,
            current_state=start,
            grasp_approach_axis="z",
            grasp_approach_offset=-float(approach_offset),
            grasp_approach_in_tool_frame=True,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=False,
        )

        if result is None or not bool(result.success.item()):
            return []

        # the approach
        appr = result.approach_interpolated_trajectory.position[0, 0, :, :7].detach().cpu().numpy()
        # 5000-frame video of joint angles for the approach, fter this line, appr is a 2D numpy array of shape (5000, 7)
        appr_last = int(result.approach_interpolated_last_tstep[0])
        # find where the real motion ends
        if appr_last < 2:
            return []
        appr_idxs = np.linspace(0, appr_last - 1, n_waypoints_approach).astype(int)
        appr_wps = [appr[i] for i in appr_idxs]

        if result.grasp_interpolated_trajectory is None:
            return appr_wps
        grasp = result.grasp_interpolated_trajectory.position[0, 0, :, :7].detach().cpu().numpy()
        grasp_last = int(result.grasp_interpolated_last_tstep[0])
        if grasp_last < 2:
            return appr_wps
        grasp_idxs = np.linspace(0, grasp_last - 1, n_waypoints_grasp).astype(int)
        grasp_wps = [grasp[i] for i in grasp_idxs]

        # Settle: hold at pre-grasp pose so the controller catches up before descent.
        pre_grasp_q = appr_wps[-1]
        settle_wps = [pre_grasp_q.copy() for _ in range(n_settle_steps)]

        return appr_wps + settle_wps + grasp_wps
