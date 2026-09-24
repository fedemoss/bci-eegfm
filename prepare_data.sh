#!/usr/bin/env bash
# Stage Dreyer2023Large (~19 GB). Needs network -> LOGIN NODE, not a compute node.
# Idempotent: downloads, then runs the extraction once so later runs hit warm caches.
set -euo pipefail
EEGFM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EEGFM_HOME/env.sh"

echo "Staging Dreyer2023 into $BENCHOPT_DATA_HOME (this takes a while)..."
benchopt prepare "$BENCH" -d "BCI[study=dreyer2023]"
echo "done — expect train 12,392 / val 3,360 / test 5,040 windows, 27 ch @ 120 Hz"
