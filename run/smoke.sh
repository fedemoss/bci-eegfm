#!/usr/bin/env bash
# 5-minute plumbing check before burning GPU hours. Trains 2 epochs of every
# arm on the small 4-class study (BNCI2014_001), which is ~1 min/model.
# The SCORES ARE MEANINGLESS at 2 epochs — this only proves the code paths run.
set -euo pipefail
EEGFM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${SITE:-}" ] && [ -f "$EEGFM_HOME/site/$SITE.sh" ]; then
    source "$EEGFM_HOME/site/$SITE.sh"
else
    source "$EEGFM_HOME/env.sh"
fi

cd "$BENCH_REPO"
benchopt prepare "$BENCH" -d "BCI[study=tangermann2012]"
rm -rf "$BENCH/outputs/EEGFM"
benchopt run "$BENCH" -d "BCI[study=tangermann2012]" --no-plot \
    -s "EEGFM[arm=probe,n_epochs=2]" \
    -s "EEGFM[arm=finetune,n_epochs=2]" \
    -s "EEGFM[arm=neurottt,n_epochs=2]" \
    -s "EEGFM[arm=probe,n_epochs=2,adapt=tent]" \
    -s "EEGFM[arm=neurottt,n_epochs=2,adapt=ssl,ttt_chunk=32]" \
    -o "BCI-decoding[training=True]"
echo
echo "If all five solvers reported 'done', the plumbing is good."
