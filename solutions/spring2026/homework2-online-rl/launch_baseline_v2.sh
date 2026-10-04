#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/cs229-r-hw2
CODE="$ROOT/workspace/code"
ENV=/root/miniconda3
RUNS="$ROOT/runs"
LOGS="$ROOT/logs"
mkdir -p "$RUNS" "$LOGS" /root/demos
cp -f "$CODE"/demos/*.npz /root/demos/

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1

pids=()
for seed in 11 22 33 44 55; do
  name="off_base_v2_seed${seed}"
  (
    cd "$CODE"
    "$ENV/bin/python" -u train_off_policy.py \
      device=cuda use_wandb=false save_video=false seed="$seed" \
      num_train_frames=100000 num_eval_episodes=50 \
      hydra.run.dir="$RUNS/$name"
  ) > "$LOGS/$name.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "off_base_v2_seed$((index + 1)),success" \
      >> "$RUNS/baseline_v2_status.csv"
  else
    echo "off_base_v2_seed$((index + 1)),failed" \
      >> "$RUNS/baseline_v2_status.csv"
    status=1
  fi
done
exit "$status"
