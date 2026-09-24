#!/usr/bin/env bash
# CBraMod / NeuroTTT experiment matrix on Dreyer2023.
#
#   bash run/matrix.sh                                   # full matrix
#   GAIN_SWEEP=1 ARMS= bash run/matrix.sh                # stage 0 only
#   ARMS="probe" INITS="pretrained" bash run/matrix.sh    # one cell
#   EPOCHS=40 GAIN=2.0 bash run/matrix.sh
#
# Each arm trains into its OWN submission dir, then is re-scored inference-only
# with Tent on top. That separation matters: every benchopt parameter config
# shares `outputs/<solver-name>/` by default, so without a per-arm
# COMPET_SUBMISSION_DIR the arms would read each other's weights.
set -euo pipefail

CBRAMOD_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CBRAMOD_HOME/env.sh"

DATASET="${DATASET:-BCI[study=dreyer2023,num_workers=4]}"
GAIN="${GAIN:-1.0}"
EPOCHS="${EPOCHS:-20}"
ARMS="${ARMS-probe finetune neurottt}"
INITS="${INITS-pretrained speech}"
GAIN_SWEEP="${GAIN_SWEEP:-0}"
TENT_DIV="${TENT_DIV:-1.0}"
TTT_CHUNK="${TTT_CHUNK:-1}"    # paper: 1 sample per adaptation step

OUT="$WORK/results/cbramod"
LOGS="$WORK/logs/cbramod"
mkdir -p "$OUT" "$LOGS"

cd "$BENCH_REPO"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# --- stage 0: pick the input gain (and head) on the cheap frozen-backbone arm.
# CBraMod was pretrained on uV/100; Dreyer arrives RobustScaler'd, so whether
# the scales match is an empirical question. Selected on the SUBJECT-GROUPED
# validation split the solver prints, never on the test column.
if [ "$GAIN_SWEEP" = "1" ]; then
    echo "=== stage 0: gain x head sweep (frozen backbone) ==="
    rm -rf "$BENCH/outputs/CBraMod"
    benchopt run "$BENCH" -d "$DATASET" --no-plot \
        -s "CBraMod[arm=probe,gain=0.25,0.5,1.0,2.0,4.0,head=all_patch_reps,avgpool,n_epochs=$EPOCHS]" \
        -o "BCI-decoding[training=True]" 2>&1 | tee "$LOGS/gain_sweep.log"
fi

# --- stages 1 & 2: train each arm, then re-score it with Tent.
for arm in $ARMS; do
  for init in $INITS; do
    tag="${arm}_${init}"
    sub="$OUT/$tag"
    mkdir -p "$sub"
    export COMPET_SUBMISSION_DIR="$sub"

    echo "=== ${tag}: train (reported without Tent) ==="
    benchopt run "$BENCH" -d "$DATASET" --no-plot \
        -s "CBraMod[arm=$arm,init=$init,gain=$GAIN,n_epochs=$EPOCHS]" \
        -o "BCI-decoding[training=True]" 2>&1 | tee "$LOGS/${tag}_train.log"

    # Inference-only: reloads the weights just written, no retraining.
    echo "=== ${tag}: + Tent (stage II-b) ==="
    benchopt run "$BENCH" -d "$DATASET" --no-plot \
        -s "CBraMod[arm=$arm,init=$init,gain=$GAIN,adapt=tent]" \
        2>&1 | tee "$LOGS/${tag}_tent.log"

    echo "=== ${tag}: + Tent + diversity ($TENT_DIV) ==="
    benchopt run "$BENCH" -d "$DATASET" --no-plot \
        -s "CBraMod[arm=$arm,init=$init,gain=$GAIN,adapt=tent,tent_diversity=$TENT_DIV]" \
        2>&1 | tee "$LOGS/${tag}_tentdiv.log"

    # Stage II-a needs the SSL heads, so it only exists on the neurottt arm.
    # ttt_chunk=1 is the paper's setting (one sample, then reset); raise it if
    # the full pass is too slow -- it is the dominant cost of this arm.
    if [ "$arm" = "neurottt" ]; then
        echo "=== ${tag}: + TTT with SSL (stage II-a, chunk=$TTT_CHUNK) ==="
        benchopt run "$BENCH" -d "$DATASET" --no-plot \
            -s "CBraMod[arm=$arm,init=$init,gain=$GAIN,adapt=ssl,ttt_chunk=$TTT_CHUNK]" \
            2>&1 | tee "$LOGS/${tag}_tttssl.log"
    fi

    unset COMPET_SUBMISSION_DIR
  done
done

echo
echo "=== summary ==="
python "$CBRAMOD_HOME/run/summarize.py"
