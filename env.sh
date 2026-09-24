# Source this before anything else, on a laptop or a cluster:
#   source ~/projects/bci-eegfm/env.sh
#
# Code lives with this file; everything generated lives under $WORK, split into
# input/ (staged, re-downloadable) and output/ (results). On a cluster set WORK
# to a scratch filesystem before sourcing -- see site/ for ready-made profiles.
EEGFM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EEGFM_HOME
export EEGFM_WEIGHTS="$EEGFM_HOME/weights"

export WORK="${WORK:-$EEGFM_HOME/work}"
export INPUT="$WORK/input"
export OUTPUT="$WORK/output"

# The competition benchmark is a dependency, not our code, and benchopt writes
# into its tree -- so it is staged under input/ rather than kept in $HOME.
export BENCH_REPO="$INPUT/2026-competition"
export BENCH="$BENCH_REPO/tracks/bci_decoding"

export BENCHOPT_DATA_HOME="$INPUT/data"     # Dreyer2023, ~19 GB
export MNE_DATA="$INPUT/data/mne_data"
# REVE checkpoints (0.28 GB base, 1.56 GB large) plus its electrode position
# bank are pulled from the Hub on first use and cached here, so they survive
# between jobs. Compute nodes usually have no network: warm the cache once on a
# login node with `python -m eegfm.smoke_test reve_large`, then jobs can set
# HF_HUB_OFFLINE=1.
export HF_HOME="$INPUT/hf"

export RESULTS="$OUTPUT/results"            # per-arm submission dirs + weights
export LOGS="$OUTPUT/logs"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$INPUT"/{data,hf} "$RESULTS" "$LOGS"

# Conda: CONDA_BASE lets a site profile point at an install outside $HOME.
_conda_base="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
if [ -n "$_conda_base" ] && [ -f "$_conda_base/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$_conda_base/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-bci26}" 2>/dev/null \
        || echo "[env] conda env '${CONDA_ENV:-bci26}' not found — run setup.sh first"
fi
