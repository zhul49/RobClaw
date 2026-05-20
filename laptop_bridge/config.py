# laptop_bridge config — single source of truth for ports, serials, controller choice.

# --- Phase A findings (Deoxys API on this laptop) ---
# Deoxys lives at ~/deoxys_control/deoxys, venv at ~/deoxys_control/.venv.
# Run bridges with: ~/deoxys_control/.venv/bin/python3 laptop_bridge/<file>.py
# - FrankaInterface(config_root + "/franka_right.yml") boots the client. Already
#   spawns a background state-subscriber thread.
# - EE pose: robot_interface.last_eef_quat_and_pos -> (quat_xyzw, pos[3,1]).
#   robot_interface.last_q -> 7-vec joints. robot_interface.last_gripper_q -> width.
# - OSC_POSE controller cfg: config/osc-pose-controller.yml.
#     is_delta: true, action_scale.translation=0.05, action_scale.rotation=1.0.
#   Action format: [dx, dy, dz, ax, ay, az, gripper] (7-dim). Position is a delta
#   in meters (scaled internally by 0.05); rotation is axis-angle delta in rad.
# - Gripper is sent via the same action[-1]: <0 open with width=0.08*|a|, >=0 close
#   (Grasp msg, fixed force=30N). Deoxys treats 0 as close, so we suppress
#   gripper sends when our requested cmd is 0.
# - control() is NOT thread-safe (mutates internal state + non-threadsafe ZMQ
#   socket). Call from one thread only. State subscriber runs internally in its
#   own thread on a separate socket; reading last_q / last_eef_pose properties
#   from other threads is fine (just reads latest buffer entry).
# - Finger force is NOT in FrankaGripperStateMessage (only width / max_width /
#   is_grasped / temperature). We publish finger_force = 0.0.
# - PORT CONFLICT: Deoxys binds tcp://*:5555 and tcp://*:5557 on the laptop
#   already. We use 5570/5571/5572 for our bridge instead of the spec's
#   5555/5556/5557. Sheep must be told to use these.

# --- ZMQ ports (laptop-side bind addresses) ---
ROBOT_STATE_PUB_PORT   = 5570  # bridge PUB -> sheep SUB
CAMERA_FRAMES_PUB_PORT = 5571  # bridge PUB -> sheep SUB
ACTION_TARGET_SUB_PORT = 5572  # bridge SUB <- sheep PUB

# Bind on all interfaces so sheep (campus IP) and local smokes both reach us.
BIND_ADDR = "tcp://*"
LOCAL_ADDR = "tcp://127.0.0.1"

# --- Watchdog ---
ACTION_DEFAULT_MAX_AGE_MS = 200       # default if msg omits max_age_ms
ACTION_STALE_FACTOR = 2               # hold pose after now-t > max_age_ms * factor

# --- Control loop ---
CONTROL_HZ = 20.0                     # Deoxys POLICY_RATE
STATE_PUB_HZ = 50.0
CAMERA_HZ = 15

# --- Deoxys config files (relative to deoxys config_root) ---
DEOXYS_INTERFACE_CFG = "franka_right.yml"
DEOXYS_OSC_POSE_CFG = "osc-pose-controller.yml"
DEOXYS_JOINT_POS_CFG = "joint-position-controller.yml"  # used for watchdog hold

# --- Cameras (RealSense D435i) ---
# Filled after enumeration step (see smoke/01). User must map physical mount
# -> serial.  Placeholder None means "not yet assigned".
# Provisional mapping (enumerated 2026-05-19). The bus order can't reveal which
# physical mount each camera is on — verify with smoke/02 and swap if wrong.
# All three are D435I.
CAMERA_SERIALS = {
    "front_left":  "233522075872",
    "front_right": "342522070195",
    "wrist":       "938422076824",
}
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_JPEG_QUALITY = 80

# --- Action clipping (for delta computed from target pose - current pose) ---
# Match deoxys/examples/osc_control.py style: pos delta gain 10 then clip to [-1,1];
# axis-angle gain 1 then clip to [-0.5, 0.5]. These produce action units that the
# OSC controller cfg will then re-scale (0.05 trans, 1.0 rot).
POS_GAIN = 10.0
POS_CLIP = 1.0
ROT_GAIN = 1.0
ROT_CLIP = 0.5
