#!/usr/bin/env bash
# 5-minute plumbing check before burning GPU hours. Trains 2 epochs of every
# arm on the small 4-class study (BNCI2014_001), which is ~1 min/model.
# The SCORES ARE MEANINGLESS at 2 epochs — this only proves the code paths run.
set -euo pipefail
CBRAMOD_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CBRAMOD_HOME/env.sh"

cd "$BENCH_REPO"
benchopt prepare "$BENCH" -d "BCI[study=tangermann2012]"
rm -rf "$BENCH/outputs/CBraMod"
benchopt run "$BENCH" -d "BCI[study=tangermann2012]" --no-plot \
    -s "CBraMod[arm=probe,n_epochs=2]" \
    -s "CBraMod[arm=finetune,n_epochs=2]" \
    -s "CBraMod[arm=neurottt,n_epochs=2]" \
    -s "CBraMod[arm=probe,n_epochs=2,tent=True]" \
    -o "BCI-decoding[training=True]"
echo
echo "If all four solvers reported 'done', the plumbing is good."
