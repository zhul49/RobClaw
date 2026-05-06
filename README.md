# LIBERO Keypoint Manipulation Pipeline

End-to-end ReKep-style manipulation on the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark. Perceives a scene, asks GPT-4o for constraint code, and uses cuRobo motion planning to execute the task.

## Pipeline

```
Render scene → Segment objects (GSAM2) → Extract 3D keypoints (DINOv3 + k-means)
            → Generate constraints (GPT-4o)
            → Solve subgoal poses (SubgoalSolver)
            → Plan trajectories (cuRobo MotionPlanner)
            → Execute in MuJoCo
```

Two entry points:

- `v2/run_test.py` — perception only. Outputs constraint files to `outputs/v2_<suite>_<task>/constraints/<timestamp>/`.
- `v2/run_full_task_v2.py` — full execution. Loads the latest constraint set and runs the task end-to-end in LIBERO.

## Prerequisites

- Python env (see `v2/requirements.txt` — cuRobo, torch, libero, opencv, etc.)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) cloned locally; set `LIBERO_PATH` to its location (defaults to `~/LIBERO`)
- OpenAI API key in env: `export OPENAI_API_KEY=...` (used by GPT-4o for constraint generation)
- GSAM2 service running on port 8765 (see below)
- (Optional) DINOv3 weights — falls back to DINOv2 (auto-downloaded) if missing

## Setup

### 1. GSAM2 service

The perception step calls a local Grounded-SAM-2 inference server.

```bash
tmux new-session -d -s gsam2 'bash gsam2_service/run_server.sh'
sleep 35   # wait for model load
curl http://127.0.0.1:8765/health   # → {"status":"ok","device":"cuda"}
```

### 2. (Optional) DINOv3

DINOv3 produces sharper keypoint features than DINOv2 but its weights are gated by Meta. To use it:

1. Clone the source: `git clone https://github.com/facebookresearch/dinov3 v2/third_party/dinov3`
2. Request weights at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ and save the `.pth` to `v2/weights/`
3. Or set `DINOV3_REPO` and `DINOV3_WEIGHTS` env vars to point elsewhere

Without this, the `auto` backbone falls back to DINOv2 (downloaded by torch.hub on first use).

## Usage

### Generate constraints for a task (perception)

```bash
cd v2
python run_test.py --suite libero_object --task 0
```

Outputs:
- `outputs/v2_libero_object_0/constraints/<timestamp>/stage{N}_subgoal_constraints.txt`
- `outputs/v2_libero_object_0/constraints/<timestamp>/stage{N}_path_constraints.txt`
- `outputs/v2_libero_object_0/constraints/<timestamp>/metadata.json`

### Run the full task (execution)

```bash
cd v2
python run_full_task_v2.py --suite libero_object --task 0
```

Defaults pick up the latest constraint set automatically. Outputs a video at `outputs/v2_<suite>_<task>/08_phase2_run/<timestamp>/task.mp4` plus per-stage diagnostics.

Common flags:
- `--task N` — task index within the suite
- `--steps-per-wp 8` — sim steps per trajectory waypoint (lower = faster, may not track)
- `--render-every 10` — video frame interval

## Configuration

`v2/configs/config.yaml` holds solver bounds, KeypointProposer settings (DINOv2/DINOv3), and constraint-generator model. Most values are tuned for `libero_object/0` and `libero_object/7`.

## Output layout

```
outputs/v2_<suite>_<task>/
├── sdf_cache.pkl              # cached SDF for the scene
├── constraints/<timestamp>/   # GPT-4o-written constraint code per run
└── 08_phase2_run/<timestamp>/
    └── task.mp4               # execution recording
```

## Acknowledgments

Built on top of:
- [ReKep](https://github.com/huangwl18/ReKep) — keypoint-constraint formulation, prompt template, KeypointProposer / SubgoalSolver / PathSolver
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — benchmark and MuJoCo scenes
- [cuRobo](https://github.com/NVlabs/curobo) — IK and motion planning
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) — text-promptable segmentation
- [DINOv3](https://github.com/facebookresearch/dinov3) / DINOv2 — keypoint feature extraction
