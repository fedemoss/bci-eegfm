#!/usr/bin/env bash
# Copy the solver into the benchmark's solvers/ directory. Re-run after every
# `git pull` of this repo — benchopt discovers solvers by scanning that folder,
# so the file has to physically live there.
set -euo pipefail
CBRAMOD_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$CBRAMOD_HOME/work}"
DEST="$WORK/2026-competition/tracks/bci_decoding/solvers"

[ -d "$DEST" ] || { echo "benchmark not found at $DEST — run setup.sh first"; exit 1; }
cp "$CBRAMOD_HOME/solvers/cbramod.py" "$DEST/cbramod.py"
echo "installed $DEST/cbramod.py"
