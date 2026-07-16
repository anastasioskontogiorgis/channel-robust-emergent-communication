#!/usr/bin/env bash
# Evaluate a trained model under ONE communication disturbance.
#
# Usage: ./scripts/test_noise.sh MODEL DIFFICULTY TYPE [PARAMS...]
#   TYPE and PARAMS:
#     gaussian LEVEL              e.g. gaussian 0.5
#     uniform  LEVEL              e.g. uniform 0.8
#     drops    WHOLE PART         e.g. drops 0.3 0.4   (either may be 0)
#     jumble   PROB               e.g. jumble 0.4
#     delay    PROB [STEP]        e.g. delay 0.7 2
#   Checkpoint run number: override with RUN=n (defaults: easy=1, medium=2, hard=3).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?usage: test_noise.sh MODEL DIFFICULTY TYPE [PARAMS...]}"
DIFF="${2:?usage: test_noise.sh MODEL DIFFICULTY TYPE [PARAMS...]}"
TYPE="${3:?usage: test_noise.sh MODEL DIFFICULTY TYPE [PARAMS...]}"
source scripts/_common.sh
prepare_dirs
require_ckpt

case "$TYPE" in
  gaussian|uniform)
    LEVEL="${4:?usage: ... $TYPE LEVEL}"
    EXTRA="--comm_constraints simple --noise_type $TYPE --noise_level $LEVEL"
    TAG="${TYPE}${LEVEL}" ;;
  drops)
    WHOLE="${4:?usage: ... drops WHOLE PART}"
    PART="${5:?usage: ... drops WHOLE PART}"
    EXTRA="--comm_constraints drops --drop_prob_whole $WHOLE --drop_prob_part $PART"
    TAG="drops_whole${WHOLE}_part${PART}" ;;
  jumble)
    PROB="${4:?usage: ... jumble PROB}"
    EXTRA="--comm_constraints jumble --jumble_prob $PROB"
    TAG="jumble${PROB}" ;;
  delay)
    PROB="${4:?usage: ... delay PROB [STEP]}"
    STEP="${5:-2}"
    EXTRA="--comm_constraints delay --delay_prob $PROB --delay_step $STEP"
    TAG="delay${PROB}_${STEP}" ;;
  *) echo "unknown TYPE: $TYPE (gaussian|uniform|drops|jumble|delay)" >&2; exit 1 ;;
esac

run_case "$EXTRA" "$TAG"
