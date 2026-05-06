#!/usr/bin/env bash
# Start the GSAM2 service. Resolves paths from the script's own location
# so it works regardless of cwd. The dedicated venv keeps GSAM2's heavy
# CUDA/torch deps out of the main project env.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"

source "$HOME/gsam2_server_venv/bin/activate"

# PyTorch's bundled CUDA libs first, then system libs
export LD_LIBRARY_PATH="$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
):/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Make the upstream Grounded-SAM-2 code importable
export PYTHONPATH="$HOME/Grounded-SAM-2:${PYTHONPATH:-}"

cd "$HERE"
exec uvicorn app:app --host 127.0.0.1 --port 8765
