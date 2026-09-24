#!/bin/bash

#SBATCH -J eegfm_matrix
#SBATCH -N 1
#SBATCH -o /share/data1/mossf/data/bci-eegfm/output/logs/eegfm_matrix_output_%j.out
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task 8
# GPUs are a consumable resource here (GresTypes=gpu, select/cons_tres), so a
# job that does not ask for one does not get one. L4 = 24 GB (c3, c6);
# T4 = 16 GB (c2) is too tight for reve_large fine-tuning.
#SBATCH --gres=gpu:l4:1

# Prepare the cluster's underlying library paths
ml python/3.11.3

# Site profile: scratch conda + /share/data1 for everything generated
source /home/mossf/projects/bci-eegfm/site/pinky.sh

nvidia-smi || true
echo "WORK=$WORK"

bash "$EEGFM_HOME/run/matrix.sh"
