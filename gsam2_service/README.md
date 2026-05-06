# GSAM2 Service

A small FastAPI wrapper around [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
that exposes open-vocabulary object segmentation over HTTP. Used by
`v2/run_test.py` for object detection during the perception phase.

The service runs locally on `127.0.0.1:8765`. It loads the model into GPU
memory once at startup (~30 s) and then serves per-request inferences in
~500 ms each.

## Prerequisites

- NVIDIA GPU with CUDA installed (the model runs on GPU; CPU is unsupported).
- Python 3.10+.
- `~/Grounded-SAM-2` checkout (cloned separately — see below).

## One-time setup

### 1. Clone the upstream model code

```bash
cd ~
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
# follow the upstream README to download model weights into checkpoints/
```

### 2. Create the dedicated virtualenv

A separate venv keeps GSAM2's heavy CUDA/torch dependencies isolated from
the main project's `rekep_curobo` env.

```bash
python3 -m venv ~/gsam2_server_venv
source ~/gsam2_server_venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn python-multipart
# install the model deps from the upstream repo
pip install -r ~/Grounded-SAM-2/requirements.txt
```

## Running

```bash
cd ~/libero_keypoint_project/gsam2_service
bash run_server.sh
```

To keep it running after you close the terminal, use `tmux`:

```bash
tmux new -s gsam2
bash run_server.sh
# Ctrl+B, D to detach. Re-attach later with: tmux attach -t gsam2
```

## Verifying

```bash
curl http://127.0.0.1:8765/health
# → {"status":"ok","device":"cuda"}
```

If you see `"device":"cpu"` the model didn't find a GPU — check CUDA install.

## API

### `POST /predict_all_masks`

Body (multipart):
- `image` — PNG/JPG file
- `prompt` — period-separated noun phrases (`"can. bottle. basket."`)
- `box_threshold` — float in `[0, 1]`, default 0.15
- `text_threshold` — float in `[0, 1]`, default 0.15

Response:
- Body — grayscale PNG; pixel value = object index (0 = background, 1..N = detected objects)
- Headers:
  - `X-Detections` — total count
  - `X-Labels` — pipe-separated: `"can|basket|bottle"`
  - `X-Boxes` — semicolon-separated `x1,y1,x2,y2`: `"10,20,80,120;..."`
  - `X-Confidences` — comma-separated floats: `"0.85,0.62,..."`

### `POST /predict_mask`

Single-object variant. Same input format, returns one mask.

### `GET /health`

Quick liveness check. Returns `{"status":"ok","device":"cuda"|"cpu"}`.

## Troubleshooting

- **Port 8765 already in use** — another GSAM2 instance is running. Find it
  with `ss -ltnp | grep 8765` and stop it before starting a new one.
- **Slow first request** — expected. The model loads lazily on first call;
  subsequent calls are fast.
- **`CUDA out of memory`** — close other GPU processes (e.g. another
  `run_full_task_v2.py` run still hogging GPU memory).
