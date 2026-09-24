#!/usr/bin/env bash
# One-time setup on the server (login node — it needs network).
#
#   TORCH_CHANNEL=https://download.pytorch.org/whl/cu126 bash setup.sh
#
# Check the node's driver with `nvidia-smi` first and pick the matching channel.
set -euo pipefail

EEGFM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_CHANNEL="${TORCH_CHANNEL:-https://download.pytorch.org/whl/cu126}"

# Site profile first (sets WORK / CONDA_BASE), else the plain layout.
if [ -n "${SITE:-}" ] && [ -f "$EEGFM_HOME/site/$SITE.sh" ]; then
    # shellcheck disable=SC1090
    source "$EEGFM_HOME/site/$SITE.sh"
else
    # shellcheck disable=SC1091
    source "$EEGFM_HOME/env.sh"
fi

_conda_base="${CONDA_BASE:-$(conda info --base)}"
# shellcheck disable=SC1091
source "$_conda_base/etc/profile.d/conda.sh"

ENV_NAME="${CONDA_ENV:-bci26}"
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -y -n "$ENV_NAME" python=3.12 pip
fi
conda activate "$ENV_NAME"

# The competition benchmark is a separate repo — it is what benchopt runs, and
# it is not vendored here so it can be pulled independently.
if [ ! -d "$BENCH_REPO/.git" ]; then
    git clone https://github.com/neural-interfaces26/2026-competition.git "$BENCH_REPO"
else
    git -C "$BENCH_REPO" pull --ff-only || true
fi

# torch FIRST, from the CUDA channel, so requirements.txt does not pull the
# default wheel. torch/torchvision/torchaudio MUST come from the same index —
# mixing channels fails at import with "operator torchvision::nms does not exist".
#
# --extra-index-url is not optional. --index-url alone *replaces* PyPI, so
# torch's own dependencies have to resolve against the pytorch index too, and
# pip >= 26 rejects its typing_extensions wheel over a name-normalisation
# mismatch ("expected 'typing-extensions', but metadata has
# 'typing_extensions'"), falls back to the sdist, and then cannot find
# flit_core to build it. Keeping PyPI in the search path avoids all of that.
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio \
    --index-url "$TORCH_CHANNEL" \
    --extra-index-url https://pypi.org/simple
pip install -r "$BENCH_REPO/requirements.txt"

mkdir -p "$HOME/.neuralbench"
cat > "$HOME/.neuralbench/config.json" <<EOF
{
  "USER": "$USER",
  "ENTITY_NAME": "$USER",
  "PROJECT_NAME": "neuralbench",
  "DATA_DIR": "$INPUT/data",
  "CACHE_DIR": "$INPUT/cache",
  "SAVE_DIR": "$RESULTS",
  "WANDB_HOST": "",
  "SLURM_PARTITION": "",
  "SLURM_CONSTRAINT": "",
  "N_CPUS": 8,
  "CLUSTER": "slurm"
}
EOF

bash "$EEGFM_HOME/install_solver.sh"

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

echo
echo "Setup done. Next, on a LOGIN node (needs network, ~19 GB):"
echo "    source env.sh && bash prepare_data.sh"
