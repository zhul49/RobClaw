import os
os.environ["MUJOCO_GL"] = "egl"

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

LIBERO_PATH = os.environ.get("LIBERO_PATH", os.path.expanduser("~/LIBERO"))

sys.path.insert(0, LIBERO_PATH)
sys.path.insert(0, HERE)

import argparse
import glob
import json
import time
import numpy as np
import cv2

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from franka_ik_curobo import FrankaIKSolver
from subgoal_solver import SubgoalSolver
from curobo_motion_planner import CuRoboMotionPlanner
from utils import get_config, load_functions_from_txt
from build_sdf import build_sdf_for_env, _enumerate_obstacle_boxes
import transform_utils as T


def _autodetect_keypoint_to_body(sim, init_keypoints, grasp_keypoints, release_keypoints):
    held_kps = sorted({k for k in (list(grasp_keypoints) + list(release_keypoints)) if k != -1})
    candidate_bodies = [
        sim.model.body_id2name(i)
        for i in range(sim.model.nbody)
        if sim.model.body_id2name(i) and sim.model.body_id2name(i).endswith("_1_main")
    ]
    if not candidate_bodies:
        return {}

    mapping = {}
    for kp in held_kps:
        kp_pos = init_keypoints[kp]
        best_name, best_dist = None, np.inf
        for body_name in candidate_bodies:
            bid = sim.model.body_name2id(body_name)
            d = float(np.linalg.norm(sim.data.body_xpos[bid] - kp_pos))
            if d < best_dist:
                best_name, best_dist = body_name, d
        mapping[kp] = best_name
        print(f"[keypoint_to_body] kp{kp} → {best_name}  (dist={best_dist*100:.1f}cm)")
    return mapping

_DIRECT_CACHE = {}

def _direct_init(env, kp_arm=500.0, kd_arm=100.0):
    sim = env.env.sim
    if id(sim) in _DIRECT_CACHE:
        return _DIRECT_CACHE[id(sim)]
    qpos_addr = sim.model.get_joint_qpos_addr("robot0_joint1")
    if isinstance(qpos_addr, tuple): qpos_addr = qpos_addr[0]
    qvel_addr = sim.model.get_joint_qvel_addr("robot0_joint1")
    if isinstance(qvel_addr, tuple): qvel_addr = qvel_addr[0]
    arm_act = list(range(7))
    grip_acts = [7, 8]
    for ai, joint_name in enumerate([f"robot0_joint{n}" for n in range(1, 8)]):
        i = arm_act[ai]
        sim.model.actuator_biastype[i] = 1
        sim.model.actuator_gainprm[i, :3] = [kp_arm, 0, 0]
        sim.model.actuator_biasprm[i, :3] = [0, -kp_arm, -kd_arm]
        jid = sim.model.joint_name2id(joint_name)
        sim.model.actuator_ctrlrange[i] = sim.model.jnt_range[jid]
    info = {"qpos_addr": qpos_addr, "qvel_addr": qvel_addr,
            "arm_act": arm_act, "grip_acts": grip_acts}
    _DIRECT_CACHE[id(sim)] = info
    return info

def make_movable_mask(grasped_indices, n_keypoints):
    mask = np.zeros(n_keypoints + 1, dtype=bool)
    mask[0] = True
    for kp in grasped_indices:
        mask[kp + 1] = True
    return mask

def _grasping_cost_stub(_kp_idx):
    return 0.0

parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_object")
parser.add_argument("--task", type=int, default=0)
parser.add_argument("--constraint-dir", default=None)
parser.add_argument("--sampling-maxfun-subgoal", type=int, default=1500)
parser.add_argument("--steps-per-wp", type=int, default=8,
                    help="Sim steps per trajectory waypoint. Lower = faster motion "
                         "but controller may not track. Higher = slower but more reliable.")
parser.add_argument("--settle-steps", type=int, default=1000,
                    help="Max sim steps to settle on the final trajectory waypoint. "
                         "Bumped from 400 — at 400, joints exited early at 0.005 rad "
                         "(now 0.001) and trajectory tracking left ~3 cm error at the EE.")
parser.add_argument("--grasp-steps", type=int, default=120,
                    help="Sim steps holding the gripper-close command after a grasp event.")
parser.add_argument("--release-steps", type=int, default=80)
parser.add_argument("--render-every", type=int, default=10,
                    help="Save a video frame every N sim steps.")
args = parser.parse_args()

PROJECT_OUTPUTS = os.environ.get("LIBERO_KP_OUTPUTS",
                                 os.path.join(PROJECT_ROOT, "outputs"))
OUTPUT_DIR = os.path.join(PROJECT_OUTPUTS, f"v2_{args.suite}_{args.task}")
RUN_TS = time.strftime("%Y%m%d_%H%M%S")
RUN_OUT = os.path.join(OUTPUT_DIR, "08_phase2_run", RUN_TS)
os.makedirs(RUN_OUT, exist_ok=True)

if args.constraint_dir is None:
    candidates = glob.glob(os.path.join(OUTPUT_DIR, "constraints", "*"))
    assert candidates, f"No constraint dirs in {OUTPUT_DIR}/constraints/"
    constraint_dir = max(candidates, key=os.path.getmtime)
else:
    constraint_dir = args.constraint_dir

print("=" * 70)
print(f"Phase 2 end-to-end: {args.suite}/{args.task}")
print(f"  constraints     : {os.path.basename(constraint_dir)}")
print(f"  exec            : {args.steps_per_wp} steps per wp / {args.render_every}-step render")
print("=" * 70)


with open(os.path.join(constraint_dir, "metadata.json")) as f:
    meta = json.load(f)
init_keypoints = np.asarray(meta["init_keypoint_positions"], dtype=np.float64)
n_keypoints = init_keypoints.shape[0]
num_stages = int(meta["num_stages"])
grasp_keypoints = meta["grasp_keypoints"]
release_keypoints = meta["release_keypoints"]
print(f"\n[metadata] {num_stages} stages  grasp={grasp_keypoints}  release={release_keypoints}")

bench = benchmark.get_benchmark_dict()[args.suite]()
env = OffScreenRenderEnv(
    bddl_file_name=bench.get_task_bddl_file_path(args.task),
    camera_heights=480, camera_widths=480, camera_depths=False,
)
env.reset()
sim = env.env.sim
info = _direct_init(env, kp_arm=1000.0, kd_arm=140.0)
qa = info["qpos_addr"]

GRIP_SITE_ID = sim.model.site_name2id("gripper0_grip_site")
KEYPOINT_TO_BODY = _autodetect_keypoint_to_body(
    sim, init_keypoints, grasp_keypoints, release_keypoints,
)
# CAN_BODY_ID is the body used for the slip-log diagnostic. Pick the
# first held body — typically there's only one in single-pick-and-place.
_first_held_body = next(iter(KEYPOINT_TO_BODY.values()), None) if KEYPOINT_TO_BODY else None
CAN_BODY_ID = sim.model.body_name2id(_first_held_body) if _first_held_body else None
BASKET_BODY_ID = sim.model.body_name2id("basket_1_main")

def live_ee_xyzw():
    pos = sim.data.site_xpos[GRIP_SITE_ID].copy()
    rot = sim.data.site_xmat[GRIP_SITE_ID].reshape(3, 3).copy()
    return np.concatenate([pos, T.mat2quat(rot)])

bid = sim.model.body_name2id("robot0_base")
R_wb = sim.data.body_xmat[bid].reshape(3, 3).copy()
p_wb = sim.data.body_xpos[bid].copy()
world2robot_homo = np.eye(4)
world2robot_homo[:3, :3] = R_wb.T
world2robot_homo[:3, 3] = -R_wb.T @ p_wb

arm_qpos_initial = sim.data.qpos[qa:qa + 7].copy()
reset_joint_pos = np.append(arm_qpos_initial, 0.0)
ee_initial = live_ee_xyzw()
can_initial = sim.data.body_xpos[CAN_BODY_ID].copy()
basket_initial = sim.data.body_xpos[BASKET_BODY_ID].copy()
print(f"[state] ee={np.array2string(ee_initial[:3], precision=3)}  "
      f"can={np.array2string(can_initial, precision=3)}  "
      f"basket={np.array2string(basket_initial, precision=3)}")

SDF_BOUNDS_MIN = np.array([-0.40, -0.50, 0.04])
SDF_BOUNDS_MAX = np.array([ 0.70,  0.50, 0.60])
SDF_VOXEL = 0.02
print()
sdf_cache = build_sdf_for_env(
    env, args.suite, args.task,
    bounds_min=SDF_BOUNDS_MIN,
    bounds_max=SDF_BOUNDS_MAX,
    voxel_size=SDF_VOXEL,
)
sdf_full = sdf_cache["sdf_full"]
sdf_per_body = sdf_cache["sdf_per_body"]


def select_sdf(grasped_indices, about_to_grasp_kp=None):
    excluded = set()
    for kp in list(grasped_indices) + ([about_to_grasp_kp] if about_to_grasp_kp is not None else []):
        body_name = KEYPOINT_TO_BODY.get(kp)
        if body_name in sdf_per_body:
            excluded.add(body_name)
    if not excluded:
        return sdf_full
    return sdf_per_body[next(iter(excluded))]

print("\n[solvers] building IKSolver + SubgoalSolver (warmups run)")
ik_solver = FrankaIKSolver(reset_joint_pos=reset_joint_pos, world2robot_homo=world2robot_homo)

sg_cfg = get_config(config_path="./configs/config.yaml")["subgoal_solver"]
sg_cfg["bounds_min"] = SDF_BOUNDS_MIN.tolist()
sg_cfg["bounds_max"] = SDF_BOUNDS_MAX.tolist()
sg_cfg["sampling_maxfun"] = args.sampling_maxfun_subgoal
subgoal_solver = SubgoalSolver(sg_cfg, ik_solver, reset_joint_pos)

print("[solvers] building CuRoboMotionPlanner (warmup ~5 s)...")
motion_planner = CuRoboMotionPlanner(world2robot_homo=world2robot_homo)

init_boxes = _enumerate_obstacle_boxes(sim)[0]
n_obs = motion_planner.update_world(init_boxes)
print(f"[solvers] pushed {n_obs} obstacle cuboids into cuRobo's world model")

state = {
    "ee_pose_xyzw": ee_initial.copy(),
    "arm_qpos_8": reset_joint_pos.copy(),
    "keypoints_full": np.concatenate([[ee_initial[:3]], init_keypoints], axis=0),
    "grasped_indices": [],
    "movable_mask": make_movable_mask([], n_keypoints),
}


def execute_traj_logged(env, traj, gripper_cmd, video_writer, slip_log,
                         steps_per_wp, settle_steps, render_every):
    sim = env.env.sim
    arm_act, grip_acts = info["arm_act"], info["grip_acts"]

    def _one_step(q_target, gc):
        sim.data.ctrl[arm_act] = np.asarray(q_target)
        if gc > 0:
            sim.data.ctrl[grip_acts[0]] = 0.0
            sim.data.ctrl[grip_acts[1]] = 0.0
        elif gc < 0:
            sim.data.ctrl[grip_acts[0]] = 0.04
            sim.data.ctrl[grip_acts[1]] = -0.04
        sim.step()
        # Slip metric: |can - tcp|
        delta = sim.data.body_xpos[CAN_BODY_ID] - sim.data.site_xpos[GRIP_SITE_ID]
        slip_log.append(float(np.linalg.norm(delta)))

    step_idx = 0
    for q_target in traj:
        for _ in range(steps_per_wp):
            _one_step(q_target, gripper_cmd)
            if video_writer is not None and step_idx % render_every == 0:
                img = sim.render(camera_name="agentview", width=480, height=480)[::-1]
                video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            step_idx += 1
    last_q = np.asarray(traj[-1])
    settle_used = 0
    for _ in range(settle_steps):
        cur_q = sim.data.qpos[qa:qa + 7]
        if np.max(np.abs(last_q - cur_q)) < 0.001:
            break
        _one_step(last_q, gripper_cmd)
        settle_used += 1
        if video_writer is not None and step_idx % render_every == 0:
            img = sim.render(camera_name="agentview", width=480, height=480)[::-1]
            video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        step_idx += 1
    return step_idx, settle_used


def hold_with_logging(env, gripper_cmd, n_steps, video_writer, slip_log, render_every):
    sim = env.env.sim
    arm_act, grip_acts = info["arm_act"], info["grip_acts"]
    hold_q = sim.data.qpos[qa:qa + 7].copy()
    for step in range(n_steps):
        sim.data.ctrl[arm_act] = hold_q
        if gripper_cmd > 0:
            sim.data.ctrl[grip_acts[0]] = 0.0
            sim.data.ctrl[grip_acts[1]] = 0.0
        elif gripper_cmd < 0:
            sim.data.ctrl[grip_acts[0]] = 0.045  # was 0.04
            sim.data.ctrl[grip_acts[1]] = -0.045
        sim.step()
        delta = sim.data.body_xpos[CAN_BODY_ID] - sim.data.site_xpos[GRIP_SITE_ID]
        slip_log.append(float(np.linalg.norm(delta)))
        if video_writer is not None and step % render_every == 0:
            img = sim.render(camera_name="agentview", width=480, height=480)[::-1]
            video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(os.path.join(RUN_OUT, "task.mp4"), fourcc, 30.0, (480, 480))
stage_summaries = []
total_t0 = time.perf_counter()

for stage_idx in range(num_stages):
    stage_num = stage_idx + 1
    print(f"\n{'=' * 70}")
    print(f"Stage {stage_num} of {num_stages}")
    print('=' * 70)

    is_grasp = grasp_keypoints[stage_idx] != -1
    is_release = release_keypoints[stage_idx] != -1
    should_top_down = bool(state["grasped_indices"]) or is_grasp
    holding_now = bool(state["grasped_indices"])
    gripper_during_motion = +1 if holding_now else -1
    print(f"  events: grasp={is_grasp}({grasp_keypoints[stage_idx]})  "
          f"release={is_release}({release_keypoints[stage_idx]})  "
          f"top_down={should_top_down}  gripper_during_motion={gripper_during_motion:+d}")

    for kp_idx, body_name in KEYPOINT_TO_BODY.items():
        body_id = sim.model.body_name2id(body_name)
        state["keypoints_full"][kp_idx + 1] = sim.data.body_xpos[body_id].copy()

    about_to_grasp_kp = grasp_keypoints[stage_idx] if grasp_keypoints[stage_idx] != -1 else None
    sdf_voxels = select_sdf(state["grasped_indices"], about_to_grasp_kp)
    if sdf_voxels is sdf_full:
        sdf_label = "sdf_full"
    else:
        excluded = []
        for kp in list(state["grasped_indices"]) + ([about_to_grasp_kp] if about_to_grasp_kp is not None else []):
            body = KEYPOINT_TO_BODY.get(kp)
            if body in sdf_per_body and body not in excluded:
                excluded.append(body)
        sdf_label = "excluding " + str(excluded)
    print(f"  sdf used         : {sdf_label}")

    # ---------- Subgoal ----------
    sgp = os.path.join(constraint_dir, f"stage{stage_num}_subgoal_constraints.txt")
    pp = os.path.join(constraint_dir, f"stage{stage_num}_path_constraints.txt")
    sg_constraints = load_functions_from_txt(sgp, _grasping_cost_stub) if os.path.exists(sgp) else []
    path_constraints = load_functions_from_txt(pp, _grasping_cost_stub) if os.path.exists(pp) else []

    top_down_xyzw = np.array([1.0, 0.0, 0.0, 0.0])
    init_pose = None
    if about_to_grasp_kp is not None and KEYPOINT_TO_BODY.get(about_to_grasp_kp):
        body_id = sim.model.body_name2id(KEYPOINT_TO_BODY[about_to_grasp_kp])
        target_xyz = sim.data.body_xpos[body_id].copy()
        target_xyz[2] = max(target_xyz[2], SDF_BOUNDS_MIN[2])
        init_pose = np.concatenate([target_xyz, top_down_xyzw])
        print(f"  subgoal seed     : {np.array2string(init_pose, precision=3)}  (about-to-grasp)")
    elif state["grasped_indices"]:
        live_basket = sim.data.body_xpos[BASKET_BODY_ID].copy()
        kp_basket = np.mean(state["keypoints_full"][5:9], axis=0)
        basket_xy = live_basket[:2]
        basket_rim_z = float(kp_basket[2])
        if is_release:
            target_xyz = np.array([basket_xy[0], basket_xy[1], basket_rim_z + 0.05])
            seed_label = "release-at-basket(live-xy)"
        else:
            target_xyz = np.array([basket_xy[0], basket_xy[1], basket_rim_z + 0.10])
            seed_label = "10cm-above-basket(live-xy)"
        init_pose = np.concatenate([target_xyz, top_down_xyzw])
        print(f"  subgoal seed     : {np.array2string(init_pose, precision=3)}  ({seed_label})")

    t0 = time.perf_counter()
    sg_sol, sg_debug = subgoal_solver.solve(
        ee_pose=state["ee_pose_xyzw"],
        keypoints=state["keypoints_full"],
        keypoint_movable_mask=state["movable_mask"],
        goal_constraints=sg_constraints,
        path_constraints=path_constraints,
        sdf_voxels=sdf_voxels,
        collision_points=state["keypoints_full"][:1],
        is_grasp_stage=should_top_down,
        initial_joint_pos=state["arm_qpos_8"],
        from_scratch=True,
        init_pose=init_pose,
    )
    sg_time = time.perf_counter() - t0
    end_pose = sg_sol.copy()
    sg_rot = T.quat2mat(end_pose[3:])
    sg_ee_z = sg_rot[:, 2]
    top_down_align = float(-sg_ee_z[2])
    print(f"  subgoal: end_pos={np.array2string(end_pose[:3], precision=3)}  "
          f"violation={sg_debug.get('subgoal_violation')}  "
          f"feasible={sg_debug['ik_feasible']}  ({sg_time:.0f}s)")
    print(f"  subgoal QUAT     : {np.array2string(end_pose[3:], precision=3)} (xyzw)")
    print(f"  subgoal EE-z     : {np.array2string(sg_ee_z, precision=3)}  "
          f"top_down_align={top_down_align:+.3f}  (want +1.0)")

    cur_boxes = _enumerate_obstacle_boxes(sim)[0]
    cur_excluded = set()
    for kp in list(state["grasped_indices"]) + ([about_to_grasp_kp] if about_to_grasp_kp is not None else []):
        body_name = KEYPOINT_TO_BODY.get(kp)
        if body_name is not None:
            cur_excluded.add(body_name)
    n_obs = motion_planner.update_world(cur_boxes, exclude_bodies=cur_excluded)
    print(f"  cuRobo world     : {n_obs} cuboids "
          f"(excluded {sorted(cur_excluded) if cur_excluded else 'none'})")

    start_pose = state["ee_pose_xyzw"].copy()
    t0 = time.perf_counter()
    if is_grasp:
        joint_traj = motion_planner.plan_grasp(
            start_joint_pos=state["arm_qpos_8"][:7],
            grasp_pose_xyzw=end_pose,
            approach_offset=0.15,
            n_waypoints_approach=80,
            n_waypoints_grasp=30,
            n_settle_steps=30,
        )
    else:
        joint_traj = motion_planner.plan(
            start_joint_pos=state["arm_qpos_8"][:7],
            end_pose_xyzw=end_pose,
            n_waypoints=100,
        )
    ps_time = time.perf_counter() - t0
    if not joint_traj:
        print(f"  cuRobo MotionPlanner FAILED in {ps_time*1000:.0f}ms — skipping stage")
        continue
    print(f"  cuRobo plan: {len(joint_traj)} waypoints in {ps_time*1000:.0f}ms")
    sample_idxs = [0, len(joint_traj) // 4, len(joint_traj) // 2,
                   (3 * len(joint_traj)) // 4, len(joint_traj) - 1]
    print(f"  trajectory shape:")
    for idx in sample_idxs:
        # FK each sample via the IK solver's forward() — base-frame TCP pose
        fk = ik_solver.forward(np.append(joint_traj[idx], 0.0))
        p = fk[:3, 3]
        print(f"    wp {idx:>3}/{len(joint_traj)}: "
              f"xy=({p[0]:+.3f},{p[1]:+.3f})  z={p[2]:.3f}")

    slip_log = []
    print(f"  executing trajectory: gripper_during_motion={gripper_during_motion:+d}")
    t0 = time.perf_counter()
    n_sim_steps, n_settle = execute_traj_logged(
        env, joint_traj, gripper_during_motion, video, slip_log,
        steps_per_wp=args.steps_per_wp,
        settle_steps=args.settle_steps,
        render_every=args.render_every,
    )
    exec_time = time.perf_counter() - t0
    ee_after_traj = live_ee_xyzw()
    traj_err = float(np.linalg.norm(ee_after_traj[:3] - end_pose[:3]))
    print(f"  arrival: ee={np.array2string(ee_after_traj[:3], precision=3)}  "
          f"err={traj_err*100:.2f}cm  ({n_sim_steps} steps, {n_settle} settle, {exec_time:.1f}s)")
    slip_during_motion = np.array(slip_log)
    if holding_now:
        print(f"  slip during motion: mean={slip_during_motion.mean()*100:.2f}cm  "
              f"max={slip_during_motion.max()*100:.2f}cm  "
              f"end={slip_during_motion[-1]*100:.2f}cm")

    if is_grasp:
        print(f"  GRASP: hold close for {args.grasp_steps} steps")
        hold_with_logging(env, +1, args.grasp_steps, video, slip_log, args.render_every)
    if is_release:
        print(f"  RELEASE: hold open for {args.release_steps} steps")
        hold_with_logging(env, -1, args.release_steps, video, slip_log, args.render_every)

    # Final post-event slip
    can_pos_final = sim.data.body_xpos[CAN_BODY_ID].copy()
    tcp_pos_final = sim.data.site_xpos[GRIP_SITE_ID].copy()
    final_slip = float(np.linalg.norm(can_pos_final - tcp_pos_final))
    print(f"  post-stage: can={np.array2string(can_pos_final, precision=3)}  "
          f"tcp={np.array2string(tcp_pos_final, precision=3)}  "
          f"|can-tcp|={final_slip*100:.2f}cm")

    state["ee_pose_xyzw"] = live_ee_xyzw()
    state["arm_qpos_8"] = np.append(sim.data.qpos[qa:qa + 7].copy(), 0.0)
    state["keypoints_full"][0] = state["ee_pose_xyzw"][:3]

    if is_grasp:
        kp = grasp_keypoints[stage_idx]
        state["grasped_indices"].append(kp)
    if is_release:
        kp = release_keypoints[stage_idx]
        if kp in state["grasped_indices"]:
            state["grasped_indices"].remove(kp)

    for kp in state["grasped_indices"]:
        body_name = KEYPOINT_TO_BODY.get(kp)
        if body_name is not None:
            body_id = sim.model.body_name2id(body_name)
            state["keypoints_full"][kp + 1] = sim.data.body_xpos[body_id].copy()
    state["movable_mask"] = make_movable_mask(state["grasped_indices"], n_keypoints)

    stage_summaries.append({
        "stage": stage_num,
        "subgoal_pos": end_pose[:3].copy(),
        "ee_after_traj": ee_after_traj[:3].copy(),
        "traj_err_cm": traj_err * 100,
        "slip_max_cm": slip_during_motion.max() * 100 if len(slip_during_motion) else 0.0,
        "slip_end_cm": slip_during_motion[-1] * 100 if len(slip_during_motion) else 0.0,
        "final_slip_cm": final_slip * 100,
        "can_pos": can_pos_final,
        "subgoal_time_s": sg_time,
        "path_time_s": ps_time,
        "exec_time_s": exec_time,
    })

video.release()
total_time = time.perf_counter() - total_t0


try:
    success = bool(env.env._check_success())
except Exception as e:
    print(f"\n_check_success() raised: {e}")
    success = None

can_final_pos = sim.data.body_xpos[CAN_BODY_ID].copy()
basket_final_pos = sim.data.body_xpos[BASKET_BODY_ID].copy()
can_to_basket = np.linalg.norm(can_final_pos - basket_final_pos)
env.close()

print(f"\n{'=' * 70}")
print("Phase 2 multi-stage execution summary")
print('=' * 70)
print(f"{'stage':>5}  {'sub→':<22}  {'ee actual':<22}  "
      f"{'traj_err':>8}  {'slip max':>8}  {'slip end':>8}")
for s in stage_summaries:
    print(f"  {s['stage']:>3}  "
          f"{np.array2string(s['subgoal_pos'], precision=3):<22}  "
          f"{np.array2string(s['ee_after_traj'], precision=3):<22}  "
          f"{s['traj_err_cm']:>5.1f}cm  "
          f"{s['slip_max_cm']:>5.1f}cm  "
          f"{s['slip_end_cm']:>5.1f}cm")
print()
print(f"  total wall time      : {total_time:.0f}s "
      f"({sum(s['subgoal_time_s']+s['path_time_s']+s['exec_time_s'] for s in stage_summaries):.0f}s solver+exec)")
print(f"  can final position   : {np.array2string(can_final_pos, precision=3)}")
print(f"  basket position      : {np.array2string(basket_final_pos, precision=3)}")
print(f"  |can - basket|       : {can_to_basket*100:.2f} cm")
if success is True:
    print(f"  TASK SUCCESS         : ✓ TRUE")
elif success is False:
    print(f"  TASK SUCCESS         : ✗ FALSE")
else:
    print(f"  TASK SUCCESS         : (unknown)")

print(f"\n  artifacts:")
print(f"    {os.path.join(RUN_OUT, 'task.mp4')}")
sys.exit(0 if success else 1)
