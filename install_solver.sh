#!/usr/bin/env bash
# Copy the solver into the benchmark's solvers/ directory. Re-run after every
# `git pull` of this repo — benchopt discovers solvers by scanning that folder,
# so the file has to physically live there.
set -euo pipefail
EEGFM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SITE:-}" ] && [ -f "$EEGFM_HOME/site/$SITE.sh" ]; then
    source "$EEGFM_HOME/site/$SITE.sh"
else
    source "$EEGFM_HOME/env.sh"
fi
DEST="$BENCH/solvers"

[ -d "$DEST" ] || { echo "benchmark not found at $DEST — run setup.sh first"; exit 1; }
cp "$EEGFM_HOME/solvers/eegfm.py" "$DEST/eegfm.py"
echo "installed $DEST/eegfm.py"

# benchopt discovers solvers by scanning this directory, so a file left behind
# by an earlier version of this repo still registers a solver -- one that now
# imports a package that no longer exists. Remove it.
for stale in cbramod.py; do
    if [ -f "$DEST/$stale" ]; then
        rm -f "$DEST/$stale"
        echo "removed stale $DEST/$stale (superseded by eegfm.py)"
    fi
done
