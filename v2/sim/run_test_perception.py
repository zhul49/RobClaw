import os
os.environ['MUJOCO_GL'] = 'egl'

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
V2_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(V2_ROOT)

LIBERO_PATH = os.environ.get("LIBERO_PATH", os.path.expanduser("~/LIBERO"))

sys.path.insert(0, LIBERO_PATH)
sys.path.insert(0, HERE)                       # sibling imports inside sim/
sys.path.insert(0, PROJECT_ROOT)               # 'v2.X.Y' resolution

import argparse
import time
import numpy as np
import cv2
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from v2.sim.label_cleaner import (clean_label, extract_prompt_vocab,
                                  merge_detections_to_unique_regions,
                                  best_label_for_cluster)


# --- CLI --------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("suite", nargs="?", default="libero_object")
parser.add_argument("task", nargs="?", type=int, default=0)
parser.add_argument("--prompt", default=None,
                    help="GroundingDINO text prompt (period-separated noun phrases). "
                         "Used by GSAM2 to decide what counts as an object.")
parser.add_argument("--instruction", default=None,
                    help="Task instruction sent to GPT-4o. Defaults to LIBERO's "
                         "filename-derived language string. Override to test "
                         "alternate phrasings, e.g. --instruction \"put the soup "
                         "into the basket\".")
parser.add_argument("--out", default=None,
                    help="Override the output dir. Default: outputs/v2_{suite}_{task}/")
args = parser.parse_args()

# Per-suite default text prompt for GSAM2. Each phrase ends with '.', which
DEFAULT_PROMPTS = {
    "libero_object":  "can. bottle. basket. carton.",
    # haven't tried the other situations yet
    "libero_spatial": "bowl. plate. ramekin. cup. dish. drawer. cabinet. wooden box. cookies.",
    "libero_goal":    "bowl. pot. bottle. cabinet. drawer. stove. cup. food container.",
    "libero_10":      "can. box. bottle. basket. bowl. plate. cabinet. pot. stove. container.",
}
PROMPT = args.prompt or DEFAULT_PROMPTS[args.suite]

EXCLUDE_VOCAB = {"robot", "robot arm", "arm", "mount", "robotic arm"}

PROMPT_FULL = PROMPT + " " + " ".join(f"{w}." for w in EXCLUDE_VOCAB)
PROMPT_VOCAB = extract_prompt_vocab(PROMPT_FULL)

PROJECT_OUTPUTS = os.environ.get("LIBERO_KP_OUTPUTS",
                                 os.path.join(PROJECT_ROOT, "outputs"))
OUTPUT_DIR = args.out or os.path.join(PROJECT_OUTPUTS, f"v2_{args.suite}_{args.task}")
CONSTRAINTS_DIR = os.path.join(OUTPUT_DIR, "constraints")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONSTRAINTS_DIR, exist_ok=True)

print("=" * 70)
print(f"v2 perception: suite={args.suite}, task={args.task}")
print(f"Output: {OUTPUT_DIR}")
print(f"Prompt: {PROMPT}")
print("=" * 70)

t_start = time.perf_counter()


# --- render RGB+depth from LIBERO --------------------------------
suite = benchmark.get_benchmark_dict()[args.suite]()
task = suite.get_task(args.task)
INSTRUCTION = args.instruction or task.language
print(f"\n[1/8] Task: {INSTRUCTION}")
if args.instruction:
    print(f"     (overriding LIBERO default: {task.language!r})")
# open the simulator with specified resolution and depth
env = OffScreenRenderEnv(
    bddl_file_name=suite.get_task_bddl_file_path(args.task),
    camera_heights=480, camera_widths=480, camera_depths=True,
)
obs = env.reset()
# muJoCo's renderer returns images with the y-axis pointing up and standard image convention has y pointing down
rgb = obs["agentview_image"][::-1].copy()
depth_norm = obs["agentview_depth"][::-1, :, 0].copy()
H, W = rgb.shape[:2]
# gets the ground truth of where the objects actually are
gt_positions = {k[:-4]: np.array(v) for k, v in obs.items()
                if k.endswith("_pos") and "_to_" not in k and not k.startswith("robot")}
print(f"     GT objects ({len(gt_positions)}):")
for n, p in gt_positions.items():
    print(f"       {n}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
cv2.imwrite(os.path.join(OUTPUT_DIR, "01_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
t_render = time.perf_counter()


# --- GSAM2 (object segmentation HTTP service on port 8765) --------
print(f"\n[2/8] Calling GSAM2...")
tmp = "/tmp/libero_test_rgb.png"
cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
with open(tmp, "rb") as f:
    resp = requests.post("http://127.0.0.1:8765/predict_all_masks",
        files={"image": f},
        data={"prompt": PROMPT_FULL, "box_threshold": 0.15, "text_threshold": 0.15},
        timeout=60)

n_raw = int(resp.headers.get("X-Detections", "0"))
raw_mask = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_GRAYSCALE)

raw_labels = resp.headers.get("X-Labels", "").split("|")
raw_boxes = [list(map(float, b.split(','))) for b in resp.headers.get("X-Boxes", "").split(';') if b]
raw_confs = list(map(float, resp.headers.get("X-Confidences", "").split(",")
                     if resp.headers.get("X-Confidences") else []))
cleaned_labels = [clean_label(r, PROMPT_VOCAB) for r in raw_labels]
print(f"     Raw detections: {n_raw}")
print(f"     Cleaned labels (top 10):")
for i, (lbl, c) in enumerate(zip(cleaned_labels[:10], raw_confs[:10])):
    print(f"       det {i+1}: '{lbl}' (conf={c:.3f})")

# if their (iou) overlap is at least 50% of their combined area, they're the same object
clusters = merge_detections_to_unique_regions(raw_boxes, cleaned_labels, raw_confs,
                                              iou_threshold=0.5)
print(f"     After IoU merge: {len(clusters)} unique regions")

# drop clusters whose label contains any EXCLUDE_VOCAB term as a substring.
def _is_excluded(cluster):
    label = best_label_for_cluster(cluster).lower()
    return any(w in label for w in EXCLUDE_VOCAB)

n_before = len(clusters)
clusters = [c for c in clusters if not _is_excluded(c)]
if len(clusters) < n_before:
    print(f"     Excluded {n_before - len(clusters)} clusters matching {sorted(EXCLUDE_VOCAB)}")
t_gsam = time.perf_counter()


# drop tiny masks <200 px and big top-half background blobs (likely wall projections).
combined_mask = np.zeros_like(raw_mask)
mask_labels = {}
new_idx = 0
for old in np.unique(raw_mask):
    if old == 0:
        continue
    binary = (raw_mask == old)
    n_pixels = binary.sum()
    if n_pixels < 200:
        continue
    ys, xs = np.where(binary)
    cx, cy = xs.mean(), ys.mean()
    best_label = None
    for c in clusters:
        bx1, by1, bx2, by2 = c["box"]
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            best_label = best_label_for_cluster(c)
            break
    if best_label is None:
        continue
    new_idx += 1
    combined_mask[binary] = new_idx
    mask_labels[new_idx] = best_label

print(f"Final masks: {new_idx} objects")
for i, lbl in mask_labels.items():
    print(f"Mask {i}: '{lbl}'")

# colored overlay of the per-object mask
np.random.seed(7)
colors = np.random.randint(50, 255, (256, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]
overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.5,
                          colors[combined_mask][..., ::-1], 0.5, 0)
for v in np.unique(combined_mask):
    if v == 0:
        continue
    ys, xs = np.where(combined_mask == v)
    text = f"{v}:{mask_labels.get(int(v), '?')}"
    cv2.putText(overlay, text, (int(xs.mean())-20, int(ys.mean())),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(overlay, text, (int(xs.mean())-20, int(ys.mean())),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.imwrite(os.path.join(OUTPUT_DIR, "02_masks.png"), overlay)


# --- depth -> world points ---------------------------------------
# muJoCo's depth buffer is normalized [0, 1]; convert to metric depth using the camera's near/far, then unproject to camera frame, then transform into world coordinates with the camera's pose.
print(f"\n[3/8] Depth -> 3D")
sim = env.env.sim
cam_id = sim.model.camera_name2id("agentview")
cam_pos = sim.model.cam_pos[cam_id].copy()
cam_quat = sim.model.cam_quat[cam_id].copy()
fovy = float(sim.model.cam_fovy[cam_id])
near = float(sim.model.vis.map.znear * sim.model.stat.extent)
far = float(sim.model.vis.map.zfar * sim.model.stat.extent)
depth_metric = near * far / (far - (far - near) * depth_norm)
fy = (H/2) / np.tan(np.deg2rad(fovy)/2)
fx = fy
cx_pix, cy_pix = W/2, H/2
u, v = np.meshgrid(np.arange(W), np.arange(H))
points_cam = np.stack([(u - cx_pix) * depth_metric / fx,
                       (v - cy_pix) * depth_metric / fy,
                       depth_metric], axis=-1)

def quat2mat(q):
    """MuJoCo quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])

R = quat2mat(cam_quat)
points_mj = points_cam.copy()
points_mj[..., 1] *= -1
points_mj[..., 2] *= -1  # MuJoCo camera convention: -y up, -z forward
points = (R @ points_mj.reshape(-1, 3).T).T + cam_pos
points = points.reshape(H, W, 3).astype(np.float32)
env.close()
t_3d = time.perf_counter()


# --- KeypointProposer ---------------------------------
print(f"\n[4/8] KeypointProposer...")
os.chdir(HERE)
from v2.sim.keypoint_proposal import KeypointProposer
from v2.common.utils import get_config
cfg = get_config(config_path="./configs/config.yaml")['keypoint_proposer']
cfg['bounds_min'] = [-2.0, -2.0, -0.5]
cfg['bounds_max'] = [ 2.0,  2.0,  2.0]
kp = KeypointProposer(cfg)
keypoints_3d, annotated_rgb = kp.get_keypoints(rgb, points, combined_mask)
print(f"     Detected {len(keypoints_3d)} keypoints")

# per-keypoint shape label by projecting back to pixel and looking up combined_mask
def world_to_pixel(world_pt):
    w_pt = np.array(world_pt) - cam_pos
    cam_pt = R.T @ w_pt
    cam_pt[1] *= -1
    cam_pt[2] *= -1
    if cam_pt[2] <= 0:
        return None
    u_px = cam_pt[0] * fx / cam_pt[2] + cx_pix
    v_px = cam_pt[1] * fy / cam_pt[2] + cy_pix
    return int(u_px), int(v_px)

keypoint_labels = {}
for i, k in enumerate(keypoints_3d):
    px = world_to_pixel(k)
    if px is None or not (0 <= px[0] < W and 0 <= px[1] < H):
        keypoint_labels[i] = "object"
        continue
    mask_val = int(combined_mask[px[1], px[0]])
    keypoint_labels[i] = mask_labels.get(mask_val, "object")

print(f"     Keypoint labels:")
for i, lbl in keypoint_labels.items():
    print(f"       Keypoint {i}: '{lbl}'")

# drop outliers: any keypoint > 60 cm from the median (migh need to change in the future)
kp_arr_check = np.array(keypoints_3d)
if len(kp_arr_check) > 2:
    median_xyz = np.median(kp_arr_check, axis=0)
    distances = np.linalg.norm(kp_arr_check - median_xyz, axis=1)
    keep = distances < 0.6
    n_dropped = (~keep).sum()
    if n_dropped > 0:
        print(f"     Dropped {n_dropped} keypoints as outliers (>60cm from median)")
        keypoints_3d = kp_arr_check[keep]
        keypoint_labels = {new_i: keypoint_labels[old_i]
                           for new_i, old_i in enumerate(np.where(keep)[0])}

cv2.imwrite(os.path.join(OUTPUT_DIR, "03_keypoints_annotated.png"),
            cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
t_kp = time.perf_counter()


# --- nearest-GT match for diagnostics + 3D viz -------------------
matches = []
for i, k in enumerate(keypoints_3d):
    if not gt_positions:
        matches.append((i, k, None, None, None))
        continue
    distances = {n: np.linalg.norm(k - g) for n, g in gt_positions.items()}
    best = min(distances.items(), key=lambda x: x[1])
    matches.append((i, k, best[0], gt_positions[best[0]], best[1]))

print(f"\n[5/8] 3D viz...")
kp_arr = np.array(keypoints_3d)
xc, yc, zc = kp_arr[:, 0].mean(), kp_arr[:, 1].mean(), kp_arr[:, 2].mean()
half = 0.5
mask_box = ((points[..., 0] > xc - half) & (points[..., 0] < xc + half) &
            (points[..., 1] > yc - half) & (points[..., 1] < yc + half) &
            (points[..., 2] > zc - 0.3) & (points[..., 2] < zc + 0.3))
pts_viz = points[mask_box][::5]
clr_viz = rgb[mask_box][::5] / 255.0
fig = plt.figure(figsize=(18, 9))
for idx, (elev, azim, ttl) in enumerate([(20, -65, "Perspective"), (90, -90, "Top-Down")]):
    ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
    ax.scatter(pts_viz[:, 0], pts_viz[:, 1], pts_viz[:, 2], c=clr_viz, s=1, alpha=0.3)
    for n, gt in gt_positions.items():
        ax.scatter(*gt, c='lime', s=200, marker='D',
                   edgecolor='black', linewidth=1.5,
                   label='Ground Truth' if n == list(gt_positions.keys())[0] else "")
        short = n.replace('_1', '').replace('_', ' ')
        ax.text(gt[0]+0.015, gt[1]+0.015, gt[2]+0.05,
                short, fontsize=8, color='darkgreen')
    for i, k, gn, gp, d in matches:
        ax.scatter(*k, c='red', s=200, marker='o',
                   edgecolor='black', linewidth=1.5,
                   label='Detected Keypoint' if i == 0 else "")
        ax.text(k[0]+0.01, k[1]+0.01, k[2]+0.025,
                str(i), fontsize=11, fontweight='bold', color='red')
        if d is not None and d < 0.30:
            ax.plot([k[0], gp[0]], [k[1], gp[1]], [k[2], gp[2]],
                    'b--', alpha=0.5, linewidth=1)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(ttl, fontsize=12)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([xc - half, xc + half])
    ax.set_ylim([yc - half, yc + half])
    ax.set_zlim([zc - 0.3, zc + 0.3])
    if idx == 0:
        ax.legend(loc='upper right')
plt.suptitle(f'{args.suite} Task {args.task}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_keypoints_3d.png"),
            dpi=120, bbox_inches='tight')
plt.close()


# --- GPT-4o constraints ---------------
print(f"\n[6/8] GPT-4o constraints (with shape labels)...")
if not os.environ.get("OPENAI_API_KEY"):
    key_file = os.environ.get("OPENAI_API_KEY_FILE",
                              os.path.expanduser("~/.openai_key"))
    if os.path.exists(key_file):
        with open(key_file) as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
    else:
        raise RuntimeError(
            "No OpenAI API key found. Set the OPENAI_API_KEY env var or "
            f"create a key file at {key_file} (or set OPENAI_API_KEY_FILE)."
        )
from v2.sim.constraint_generation import ConstraintGenerator
global_config = get_config(config_path="./configs/config.yaml")
cg = ConstraintGenerator(global_config['constraint_generator'])
cg.base_dir = CONSTRAINTS_DIR

# group keypoints by shape label so the prompt sees clusters per object
_shape_groups = {}
for _i, _lbl in keypoint_labels.items():
    _shape_groups.setdefault(_lbl, []).append(_i)
keypoint_label_str = "; ".join(f"keypoints {_v} are on the {_k}"
                                for _k, _v in _shape_groups.items())
augmented = f"{INSTRUCTION}\n\n[Detected shapes for each keypoint: {keypoint_label_str}]"
print(f"     Shape hints: {keypoint_label_str}")

metadata = {"init_keypoint_positions": keypoints_3d.tolist(),
            "num_keypoints": len(keypoints_3d),
            "keypoint_shapes": keypoint_labels}
rekep_dir = cg.generate(annotated_rgb, augmented, metadata)
print(f"     Saved: {rekep_dir}")
t_vlm = time.perf_counter()


# --- timing summary ----------------------------------------------
print(f"\n[7/8] Timing")
print("-" * 70)
print(f"  Render:       {(t_render-t_start)*1000:7.0f} ms")
print(f"  GSAM2:        {(t_gsam-t_render)*1000:7.0f} ms")
print(f"  3D project:   {(t_3d-t_gsam)*1000:7.0f} ms")
print(f"  KeypointProp: {(t_kp-t_3d)*1000:7.0f} ms")
print(f"  GPT-4o:       {(t_vlm-t_kp)*1000:7.0f} ms")
print(f"  TOTAL:        {(t_vlm-t_start)*1000:7.0f} ms")


# --- keypoint accuracy + show generated constraints --------------
print(f"\n[8/8] Keypoint accuracy + identity")
print(f"{'KP':>3}  {'Detected':>30}  {'Shape':>15}  {'Nearest GT':>20}  {'Err':>7}")
print("-" * 85)
for i, k, gn, gp, d in matches:
    pos = f"({k[0]:+.3f}, {k[1]:+.3f}, {k[2]:+.3f})"
    shape = keypoint_labels.get(i, "?")
    if gn is None:
        print(f"{i:>3}  {pos:>30}  {shape:>15}  {'(no GT)':>20}  {'--':>7}")
    else:
        print(f"{i:>3}  {pos:>30}  {shape:>15}  {gn.replace('_1', ''):>20}  {d*100:>4.1f}cm")

print(f"\nGenerated constraints:")
import os.path as osp
for f in sorted(os.listdir(rekep_dir)):
    if f.startswith('stage') and f.endswith('.txt'):
        print(f"\n=== {f} ===")
        with open(osp.join(rekep_dir, f)) as fh:
            print(fh.read())
print("=" * 70)
