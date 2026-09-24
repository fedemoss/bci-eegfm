#!/usr/bin/env bash
# Copy the solver into the benchmark's solvers/ directory. Re-run after every
# `git pull` of this repo — benchopt discovers solvers by scanning that folder,
# so the file has to physically live there.
set -euo pipefail
EEGFM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$EEGFM_HOME/work}"
DEST="$WORK/2026-competition/tracks/bci_decoding/solvers"

[ -d "$DEST" ] || { echo "benchmark not found at $DEST — run setup.sh first"; exit 1; }
cp "$EEGFM_HOME/solvers/eegfm.py" "$DEST/eegfm.py"
echo "installed $DEST/eegfm.py"
