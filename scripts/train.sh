#!/usr/bin/env bash
# Train one model on one difficulty of the Traffic Junction benchmark.
#
# Usage: ./scripts/train.sh MODEL DIFFICULTY [SEED]
#   MODEL:      commnet | ic3net | tarcomm | gacomm | magic
#   DIFFICULTY: easy | medium | hard
#
# Notes:
#  * Training is always clean: run_baselines.py force-disables noise in
#    train mode (matching the papers' methodology).
#  * --save writes the checkpoint; --export (default on) writes training
#    stats to data/<model>_train_<difficulty>.txt.
#  * Train each model easy -> medium -> hard so checkpoints land in the
#    paper's run1/run2/run3 convention that the test scripts assume.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: train.sh MODEL DIFFICULTY [SEED]}"
DIFF="${2:?usage: train.sh MODEL DIFFICULTY [SEED]}"
SEED="${3:-0}"
export OMP_NUM_THREADS=1
source scripts/_common.sh
prepare_dirs

python -u run_baselines.py \
  --env_name traffic_junction --mode train \
  $GEOM_FLAGS $ADD_FLAGS $TRAIN_SCHED $MODEL_FLAGS $ARCH_FLAGS \
  --nprocesses 16 --epoch_size 10 --lrate 0.001 --value_coeff 0.01 \
  --save --seed "$SEED" \
  | tee "logs/train_${MODEL}_${DIFF}_seed${SEED}.log"

echo "checkpoint saved under saved/traffic_junction/${MODEL_KEY}/ (latest runN)"
