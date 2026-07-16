#!/usr/bin/env bash
# Shared setup for the runbooks. Sourced (with MODEL and DIFF set), not executed.
#
# Conventions encoded here:
#  * Model flag bundles follow run_baselines.py's selection logic exactly.
#  * Difficulty settings follow the paper's "Training and testing settings"
#    table (epochs, add rates, curricula). Grid geometry (dim / max_steps /
#    vision) is not in that table: easy is from the author's original script;
#    medium/hard use the benchmark's standard values — VERIFY vs Methodology.
#  * Checkpoints: training appends saved/traffic_junction/<key>/runN/model.pt.
#    Training each model in easy -> medium -> hard order yields the paper's
#    convention run1=easy, run2=medium, run3=hard, which the test scripts
#    assume by default (override with the RUN argument / variable).

case "$MODEL" in
  commnet) MODEL_FLAGS="--commnet";          MODEL_KEY="commnet"    ;;
  ic3net)  MODEL_FLAGS="--ic3net";           MODEL_KEY="ic3net"     ;;
  tarcomm) MODEL_FLAGS="--tarcomm --ic3net"; MODEL_KEY="tar_ic3net" ;;
  gacomm)  MODEL_FLAGS="--gacomm";           MODEL_KEY="gacomm"     ;;
  magic)   MODEL_FLAGS="--magic";            MODEL_KEY="magic"      ;;
  *) echo "unknown model: $MODEL (commnet|ic3net|tarcomm|gacomm|magic)" >&2; exit 1 ;;
esac

case "$DIFF" in
  easy)
    GEOM_FLAGS="--difficulty easy --nagents 5 --dim 6 --max_steps 20 --vision 1"
    ADD_FLAGS="--add_rate_min 0.1 --add_rate_max 0.3"
    TRAIN_SCHED="--num_epochs 2000 --curr_start 250 --curr_end 1250"
    TEST_SCHED="--num_epochs 1000 --curr_start 125 --curr_end 625"
    DEFAULT_RUN=1 ;;
  medium)
    GEOM_FLAGS="--difficulty medium --nagents 10 --dim 14 --max_steps 40 --vision 1"  
    ADD_FLAGS="--add_rate_min 0.02 --add_rate_max 0.05"
    TRAIN_SCHED="--num_epochs 3000 --curr_start 375 --curr_end 1875"
    TEST_SCHED="--num_epochs 1000 --curr_start 125 --curr_end 625"
    DEFAULT_RUN=2 ;;
  hard)
    GEOM_FLAGS="--difficulty hard --nagents 20 --dim 18 --max_steps 80 --vision 1"    
    ADD_FLAGS="--add_rate_min 0.05 --add_rate_max 0.05"
    TRAIN_SCHED="--num_epochs 4000 --curr_start 0 --curr_end 0"   # no curriculum (paper: '-')
    TEST_SCHED="--num_epochs 1000 --curr_start 0 --curr_end 0"    # no curriculum (paper: '-')
    DEFAULT_RUN=3 ;;
  *) echo "unknown difficulty: $DIFF (easy|medium|hard)" >&2; exit 1 ;;
esac

RUN="${RUN:-$DEFAULT_RUN}"
CKPT="saved/traffic_junction/${MODEL_KEY}/run${RUN}/model.pt"

# Architecture flags shared by train and test (the loaded checkpoint must be
# rebuilt with identical dimensions).
ARCH_FLAGS="--recurrent --hid_size 128 --detach_gap 10"

require_ckpt () {
  if [ ! -f "$CKPT" ]; then
    echo "checkpoint not found: $CKPT" >&2
    echo "(train first — run1=easy, run2=medium, run3=hard — or override RUN)" >&2
    exit 1
  fi
}

# Stats export is ON by default in run_baselines.py and writes to ./data/,
# which the framework does NOT create — so the runbooks do.
prepare_dirs () { mkdir -p data logs saved; }

run_case () {
  local extra="$1" tag="$2"
  echo "=== ${MODEL} / ${DIFF} :: ${tag} ==="
  python -u run_baselines.py \
    --env_name traffic_junction --mode test \
    $GEOM_FLAGS $ADD_FLAGS $TEST_SCHED $MODEL_FLAGS $ARCH_FLAGS \
    --load "$CKPT" $extra \
    | tee "logs/${MODEL}_${DIFF}_${tag}.log"
}
