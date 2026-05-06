import numpy as np
from dataclasses import dataclass
from typing import Optional

# cuRobo's panda_hand link and LIBERO's gripper0_grip_site but in different mathematical frames
DEFAULT_TCP_OFFSET_Z = 0.0965
DEFAULT_TCP_ROT_Z = np.pi / 2

PANDA_NEUTRAL_QPOS = np.array(
    [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4, 0.0]
)


@dataclass
class IKResult:
    cspace_position: np.ndarray
    success: bool
    position_error: float
    num_descents: int


class FrankaIKSolver:

    def __init__(
        self,
        # Lula-only argument; ignored
        robot_description_path=None,
        # ignored — cuRobo uses franka.yml
        robot_urdf_path: Optional[str] = None,
        # ignored — cuRobo uses franka.yml
        eef_name: Optional[str] = None,
        # ignored
        reset_joint_pos: Optional[np.ndarray] = None,
        world2robot_homo: Optional[np.ndarray] = None,
        tcp_offset_z: float = DEFAULT_TCP_OFFSET_Z,
        tcp_rot_z: float = DEFAULT_TCP_ROT_Z,
        num_seeds: int = 16,
    ):
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

        self._torch = torch

        self.reset_joint_pos = (
            np.asarray(reset_joint_pos, dtype=np.float64)
            if reset_joint_pos is not None
            else PANDA_NEUTRAL_QPOS.copy()
        )
        self.world2robot_homo = (
            np.asarray(world2robot_homo, dtype=np.float64)
            if world2robot_homo is not None
            else np.eye(4)
        )
        self.tcp_offset_z = float(tcp_offset_z)
        self.tcp_rot_z = float(tcp_rot_z)

        # 4×4 transform converting world frame to robot base frame
        c, s = np.cos(self.tcp_rot_z), np.sin(self.tcp_rot_z)
        self._tcp_to_curobo_target = np.array([
            [c, -s, 0.0, 0.0],
            [s,  c, 0.0, 0.0],
            [0.0, 0.0, 1.0, -self.tcp_offset_z],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._curobo_target_to_tcp = np.linalg.inv(self._tcp_to_curobo_target)

        cfg = InverseKinematicsCfg.create(robot="franka.yml", num_seeds=num_seeds)
        self.ik = InverseKinematics(cfg)
        self.tool_link = self.ik.tool_frames[0]

    @staticmethod
    def _matrix_to_wxyz(R):
        """3x3 rotation matrix → (w, x, y, z) — cuRobo's quat order."""
        from scipy.spatial.transform import Rotation
        x, y, z, w = Rotation.from_matrix(R).as_quat()
        return w, x, y, z

    @staticmethod
    def _wxyz_to_matrix(wxyz):
        from scipy.spatial.transform import Rotation
        w, x, y, z = wxyz
        return Rotation.from_quat([x, y, z, w]).as_matrix()

    def solve(
        self,
        target_pose_homo: np.ndarray,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        position_weight: float = 1.0,
        orientation_weight: float = 0.05,
        max_iterations: int = 150,
        initial_joint_pos: Optional[np.ndarray] = None,
    ) -> IKResult:
        torch = self._torch
        from curobo.types import GoalToolPose, Pose

        # Step 1+2: world → robot-base → panda_hand frame.
        target_robot_tcp = self.world2robot_homo @ np.asarray(target_pose_homo, dtype=np.float64)
        target_for_curobo = target_robot_tcp @ self._tcp_to_curobo_target

        target_pos = target_for_curobo[:3, 3].astype(np.float32)
        w, x, y, z = self._matrix_to_wxyz(target_for_curobo[:3, :3])

        goal = Pose(
            position=torch.tensor(target_pos, device="cuda", dtype=torch.float32).unsqueeze(0),
            quaternion=torch.tensor([w, x, y, z], device="cuda", dtype=torch.float32).unsqueeze(0),
        )
        # spawns 16 seed configurations (random inital joint angles) and runs gradient descent on all 16 in parallel and picks the best
        result = self.ik.solve_pose(
            GoalToolPose.from_poses({self.tool_link: goal}, num_goalset=1)
        )

        success = bool(result.success.item())
        position_error = float(result.position_error.item())
        full_joints = result.js_solution.position.detach().cpu().numpy().squeeze()
        arm_joints = full_joints[:7]

        # SubgoalSolver uses num_descent as a soft cost signal, so higher num_descent = higher cost so the optimizer avoids hard-to-reach poses
        if success:
            scale = min(position_error / max(position_tolerance, 1e-6), 1.0)
            num_descents = int(5 + scale * (max_iterations // 2 - 5))
        else:
            num_descents = max_iterations

        initial = (
            self.reset_joint_pos
            if initial_joint_pos is None
            else np.asarray(initial_joint_pos, dtype=np.float64)
        )
        cspace = np.array(initial, dtype=np.float64).copy()
        n_arm = min(7, max(0, len(cspace) - 1))
        cspace[:n_arm] = arm_joints[:n_arm]

        return IKResult(
            cspace_position=cspace,
            success=success,
            position_error=position_error,
            num_descents=num_descents,
        )


    def forward(self, joint_pos: np.ndarray) -> np.ndarray:
        # FK, takes 7 joint angles, calls cuRobo's kinematics, computes link poses for every link of the robot
        torch = self._torch
        from curobo.types import JointState
        q7 = np.zeros(7, dtype=np.float32)
        n_arm = min(7, len(joint_pos))
        q7[:n_arm] = np.asarray(joint_pos[:n_arm], dtype=np.float32)
        q7_t = torch.tensor(q7, device="cuda", dtype=torch.float32).unsqueeze(0)
        js = JointState.from_position(q7_t)
        kin_state = self.ik.kinematics.compute_kinematics(js)
        tool_frames = kin_state.tool_poses.tool_frames
        link_idx = tool_frames.index(self.tool_link)
        ee_pos = kin_state.tool_poses.position[0, 0, link_idx].detach().cpu().numpy()
        ee_quat_wxyz = kin_state.tool_poses.quaternion[0, 0, link_idx].detach().cpu().numpy()
        panda_hand_pose = np.eye(4)
        panda_hand_pose[:3, 3] = ee_pos
        panda_hand_pose[:3, :3] = self._wxyz_to_matrix(ee_quat_wxyz)
        return panda_hand_pose @ self._curobo_target_to_tcp

class StubIKSolver:
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "StubIKSolver is not implemented in the cuRobo backend — "
            "use FrankaIKSolver directly."
        )
