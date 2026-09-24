# UdeSA cluster (pinky). Source this INSTEAD of env.sh:
#   source ~/projects/bci-eegfm/site/pinky.sh
#
# Site policy: only code in $HOME (which is 83% full), everything else on
# /share/data1. Conda already lives on scratch there.
export WORK="${WORK:-/share/data1/mossf/data/bci-eegfm}"
export CONDA_BASE="/share/data1/mossf/miniconda3"
export CONDA_ENV="${CONDA_ENV:-bci26}"

# $HOME is 83% full (~26 GB free) and the CUDA torch wheels alone are ~2.5 GB,
# so keep pip's cache off it too. conda's own pkgs cache already lives beside
# CONDA_BASE on scratch.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/share/data1/mossf/.cache/pip}"
mkdir -p "$PIP_CACHE_DIR"

# The cluster's own python module, loaded for the underlying library paths
# exactly as the other projects here do. conda supplies the interpreter.
command -v module >/dev/null 2>&1 && { module load python/3.11.3 2>/dev/null; \
                                       module load cuda/12.1 2>/dev/null; }

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
