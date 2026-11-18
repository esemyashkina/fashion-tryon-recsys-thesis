#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

base_python="${PYTHON:-python}"
"$base_python" - <<'PY'
import sys
import torch
import torchvision

if sys.version_info[:2] != (3, 12):
    raise SystemExit("Select a Vast PyTorch template with Python 3.12.")
if not torch.cuda.is_available():
    raise SystemExit("Select a CUDA-enabled Vast PyTorch template.")
print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | torchvision {torchvision.__version__}")
PY

"$base_python" -m venv --system-site-packages .venv
mkdir -p runs third_party
.venv/bin/python - <<'PY' > runs/vast-torch-constraints.txt
import torch
import torchvision
print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
PY

fashn_dir=third_party/fashn-vton-1.5
fashn_revision=7c0f10af3f91ad4048fe9729c470a13ef905d25a
if [[ ! -d "$fashn_dir/.git" ]]; then
    git clone https://github.com/fashn-AI/fashn-vton-1.5.git "$fashn_dir"
fi
git -C "$fashn_dir" checkout --detach "$fashn_revision"
.venv/bin/python -m pip install -c runs/vast-torch-constraints.txt -r requirements.txt "$fashn_dir"
.venv/bin/python "$fashn_dir/scripts/download_weights.py" --weights-dir "$PWD/weights"
.venv/bin/python -c 'import gradio, torch; from fashn_vton import TryOnPipeline; assert torch.cuda.is_available()'
.venv/bin/python -m unittest discover -s tests
