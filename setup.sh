#!/usr/bin/env bash
# One-time setup on the server (login node — it needs network).
#
#   TORCH_CHANNEL=https://download.pytorch.org/whl/cu126 bash setup.sh
#
# Check the node's driver with `nvidia-smi` first and pick the matching channel.
set -euo pipefail

EEGFM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$EEGFM_HOME/work}"
TORCH_CHANNEL="${TORCH_CHANNEL:-https://download.pytorch.org/whl/cu126}"
BENCH_REPO="$WORK/2026-competition"

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "$WORK"/{data,cache,results,logs}

if ! conda env list | grep -q "^bci26 "; then
    conda create -y -n bci26 python=3.12 pip
fi
conda activate bci26

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
pip install torch torchvision torchaudio --index-url "$TORCH_CHANNEL"
pip install -r "$BENCH_REPO/requirements.txt"

mkdir -p "$HOME/.neuralbench"
cat > "$HOME/.neuralbench/config.json" <<EOF
{
  "USER": "$USER",
  "ENTITY_NAME": "$USER",
  "PROJECT_NAME": "neuralbench",
  "DATA_DIR": "$WORK/data",
  "CACHE_DIR": "$WORK/cache",
  "SAVE_DIR": "$WORK/results",
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
