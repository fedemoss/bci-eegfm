# Source this before anything else, on the laptop or the server:
#   source ~/bci-eegfm/env.sh
EEGFM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EEGFM_HOME
export EEGFM_WEIGHTS="$EEGFM_HOME/weights"

# Where the benchmark and its ~19 GB of data live. Override WORK if the server
# gives you a scratch filesystem (recommended — the data does not belong in $HOME).
export WORK="${WORK:-$EEGFM_HOME/work}"
export BENCH_REPO="$WORK/2026-competition"
export BENCH="$BENCH_REPO/tracks/bci_decoding"

export BENCHOPT_DATA_HOME="$WORK/data"
export MNE_DATA="$WORK/data/mne_data"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

# REVE checkpoints are pulled from the Hub on first use (0.28 GB base,
# 1.56 GB large) and cached here, so they survive between jobs and are shared
# by every run. Compute nodes usually have no network: warm the cache once on
# a login node with
#     python -m eegfm.smoke_test reve_large
# and then jobs can run with HF_HUB_OFFLINE=1.
export HF_HOME="${HF_HOME:-$WORK/hf}"
mkdir -p "$HF_HOME"

mkdir -p "$WORK"/{data,cache,results,logs}

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate bci26 2>/dev/null || echo "[env] conda env 'bci26' not active — run setup.sh first"
