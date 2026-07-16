#!/usr/bin/env bash
# Evaluate a trained model on clean (reliable) communication.
#
# Usage: ./scripts/test_clean.sh MODEL DIFFICULTY [RUN]
#   RUN defaults per difficulty: easy=1, medium=2, hard=3.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?usage: test_clean.sh MODEL DIFFICULTY [RUN]}"
DIFF="${2:?usage: test_clean.sh MODEL DIFFICULTY [RUN]}"
RUN="${3:-}"
source scripts/_common.sh
prepare_dirs
require_ckpt
run_case "" "clean"
