# LIBERO Keypoint Detection Pipeline

End-to-end perception pipeline that:
1. Renders a LIBERO simulation scene
2. Segments objects via Grounded-SAM-2 (text-promptable)
3. Extracts 3D keypoints using DINOv3 + k-means clustering
4. Generates manipulation constraints via GPT-4o

## Prerequisites

- conda env `rekep` with the dependencies in ~/Grounded-SAM-2 and ~/ReKep
- LIBERO installed at ~/LIBERO
- ReKep at ~/ReKep
- OpenAI API key at ~/.openai_key (just the key, no `export` prefix)
- GSAM2 server running (see below)

## Setup: Start GSAM2 server

The pipeline calls a local GSAM2 inference server on port 8765:

```bash
tmux new-session -d -s gsam2 'bash ~/libero_keypoint_project/gsam2_service/run_server.sh'
sleep 35   # wait for model loading
curl http://127.0.0.1:8765/health   # should print {"status":"ok","device":"cuda"}
```

## Usage

### Run on default scene (libero_object task 0 — alphabet soup)
```bash
python ~/libero_keypoint_project/scripts/run_test.py
```

### Run on any LIBERO suite + task
```bash
python ~/libero_keypoint_project/scripts/run_test.py libero_object 7      # milk
python ~/libero_keypoint_project/scripts/run_test.py libero_spatial 0     # bowl on plate
python ~/libero_keypoint_project/scripts/run_test.py libero_goal 0        # open drawer
python ~/libero_keypoint_project/scripts/run_test.py libero_10 2          # stove + moka pot
```

### Custom GSAM2 prompt
```bash
python ~/libero_keypoint_project/scripts/run_test.py libero_spatial 0 \
    --prompt "bowl. plate. ramekin. drawer. cookies."
```

## Output

Each run produces a directory `~/libero_test_<suite>_<task>/` with:

- `01_rgb.png` — raw rendered scene
- `02_masks.png` — GSAM2 segmentation overlay
- `03_keypoints_annotated.png` — numbered keypoints on RGB (input to GPT-4o)
- `04_keypoints_3d.png` — 3D scatter: detected (red) vs ground truth (green)

Plus the script prints:
- Timing breakdown per stage
- Keypoint accuracy table (detected vs ground truth in cm)
- Generated constraint code (from GPT-4o)

## Pipeline Performance

On `libero_object` task 0:
- Mean keypoint error: ~3 cm vs LIBERO ground truth (sub-5cm on every tabletop object)
- End-to-end runtime: ~3-4 sec/frame (GSAM2 ~500ms, DINOv3 ~300ms, GPT-4o ~3s)

## Limitations

- Robot self-detection: GSAM2 sometimes detects the Franka arm as an object;
  pixel-based filter (top of image, large mask) handles most cases
- Dark/metallic objects: GroundingDINO weaker on these, may miss bowls
- VLM identification: requires labels offset from objects (we patched
  `ReKep/keypoint_proposal.py` to draw labels with leader lines)

## Files

- `scripts/demo.py` — original demo (single fixed task)
- `scripts/run_test.py` — parametrized test runner (recommended)
- `scripts/eval_keypoints.py` — keypoint vs ground truth comparison
- `scripts/timing.py` — per-stage timing measurements
- `gsam2_service/` — FastAPI server wrapping Grounded-SAM-2
- `outputs/` — saved results from previous runs
