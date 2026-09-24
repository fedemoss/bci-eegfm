#!/usr/bin/env bash
#SBATCH --job-name=cbramod-matrix
#SBATCH --output=cbramod-%j.out
#SBATCH --error=cbramod-%j.out
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# --- EDIT these two for your site, then uncomment ---
##SBATCH --partition=gpu
##SBATCH --account=CHANGEME
#
#   sbatch run/cbramod.sbatch
#   sbatch --export=ALL,GAIN_SWEEP=1,ARMS= run/cbramod.sbatch     # stage 0 only
#   sbatch --export=ALL,ARMS=neurottt,EPOCHS=40 run/cbramod.sbatch
#
# The data must already be staged (compute nodes have no network) — run
# prepare_data.sh on a login node first.
set -euo pipefail

EEGFM_HOME="${EEGFM_HOME:-$HOME/bci-eegfm}"
source "$EEGFM_HOME/env.sh"

nvidia-smi || true
bash "$EEGFM_HOME/run/matrix.sh"
