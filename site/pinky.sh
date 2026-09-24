# UdeSA cluster (pinky). Source this INSTEAD of env.sh:
#   source ~/projects/bci-eegfm/site/pinky.sh
#
# Site policy: only code in $HOME (which is 83% full), everything else on
# /share/data1. Conda already lives on scratch there.
export WORK="${WORK:-/share/data1/mossf/data/bci-eegfm}"
export CONDA_BASE="/share/data1/mossf/miniconda3"
export CONDA_ENV="${CONDA_ENV:-bci26}"

# $HOME here is 83% full (~26 GB free) and already holds 14 GB of caches, so
# push everything that grows onto scratch.
#
# conda needs nothing: ~/.condarc already sets envs_dirs and pkgs_dirs to
# /share/data1/mossf/miniconda3. pip has no such config and defaults to
# $HOME/.cache/pip, and the CUDA torch wheels alone are ~2.5 GB.
export SCRATCH_CACHE="${SCRATCH_CACHE:-/share/data1/mossf/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_CACHE/pip}"
# Catch-all for everything else that honours the XDG spec (matplotlib, fontconfig,
# and pip/HF when their own variables are unset).
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_CACHE}"
# torch.hub downloads (braindecode pulls some checkpoints through it).
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_CACHE/torch}"
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME"

# HF_HOME is set per-project by env.sh ($WORK/input/hf) so REVE's checkpoints
# sit with this project's staged input and can be warmed once, then used with
# HF_HUB_OFFLINE=1. Point it at "$SCRATCH_CACHE/huggingface" instead if you
# would rather share one Hub cache across projects.

# The cluster's own python module, loaded for the underlying library paths
# exactly as the other projects here do. conda supplies the interpreter.
command -v module >/dev/null 2>&1 && { module load python/3.11.3 2>/dev/null; \
                                       module load cuda/12.1 2>/dev/null; }

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
