#!/bin/bash

#SBATCH -J eegfm_smoke
#SBATCH -N 1
#SBATCH -o /share/data1/mossf/data/bci-eegfm/output/logs/eegfm_smoke_output_%j.out
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:l4:1

# Plumbing check on the small 4-class study: every arm, 2 epochs each.
# The SCORES ARE MEANINGLESS -- this only proves the code paths run on a GPU.
ml python/3.11.3
source /home/mossf/projects/bci-eegfm/site/pinky.sh

nvidia-smi || true
bash "$EEGFM_HOME/run/smoke.sh"
