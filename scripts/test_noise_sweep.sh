#!/usr/bin/env bash
# Run the FULL published disturbance schedule (EJAI paper) against one
# trained model: additive Gaussian/uniform noise, partial / whole / combined
# message loss, message jumbling, and message delays.
#
# Usage: ./scripts/test_noise_sweep.sh MODEL DIFFICULTY [RUN]
#   RUN defaults per difficulty: easy=1, medium=2, hard=3.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?usage: test_noise_sweep.sh MODEL DIFFICULTY [RUN]}"
DIFF="${2:?usage: test_noise_sweep.sh MODEL DIFFICULTY [RUN]}"
RUN="${3:-}"
source scripts/_common.sh
prepare_dirs
require_ckpt

run_case "" "clean"

# Additive noise
for s in 0.2 0.5 0.8; do
  run_case "--comm_constraints simple --noise_type gaussian --noise_level $s" "gaussian${s}"
done
for s in 0.5 0.8; do
  run_case "--comm_constraints simple --noise_type uniform --noise_level $s" "uniform${s}"
done

# Message loss (partial / whole / combined at matched settings)
for p in 0.4 0.7; do
  run_case "--comm_constraints drops --drop_prob_part $p" "drops_part${p}"
done
for p in 0.3 0.6; do
  run_case "--comm_constraints drops --drop_prob_whole $p" "drops_whole${p}"
done
run_case "--comm_constraints drops --drop_prob_whole 0.3 --drop_prob_part 0.4" "drops_combined0.58"
run_case "--comm_constraints drops --drop_prob_whole 0.6 --drop_prob_part 0.7" "drops_combined0.88"

# Message jumbling
for p in 0.4 0.7; do
  run_case "--comm_constraints jumble --jumble_prob $p" "jumble${p}"
done

# Message delays (max delay 2 steps)
for p in 0.4 0.7; do
  run_case "--comm_constraints delay --delay_prob $p --delay_step 2" "delay${p}_2"
done

echo "Sweep complete — framework stats in data/ (auto-named), console logs in logs/."
