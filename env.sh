# Source this before anything else, on the laptop or the server:
#   source ~/bci-cbramod/env.sh
CBRAMOD_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CBRAMOD_HOME
export CBRAMOD_WEIGHTS="$CBRAMOD_HOME/weights"

# Where the benchmark and its ~19 GB of data live. Override WORK if the server
# gives you a scratch filesystem (recommended — the data does not belong in $HOME).
export WORK="${WORK:-$CBRAMOD_HOME/work}"
export BENCH_REPO="$WORK/2026-competition"
export BENCH="$BENCH_REPO/tracks/bci_decoding"

export BENCHOPT_DATA_HOME="$WORK/data"
export MNE_DATA="$WORK/data/mne_data"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK"/{data,cache,results,logs}

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate bci26 2>/dev/null || echo "[env] conda env 'bci26' not active — run setup.sh first"
