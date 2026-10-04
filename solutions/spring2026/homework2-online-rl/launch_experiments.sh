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
names=()

launch() {
  local name="$1"
  shift
  (
    cd "$CODE"
    "$ENV/bin/python" -u "$@"
  ) > "$LOGS/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  echo "$!" > "$LOGS/$name.pid"
  echo "launched $name pid=$!"
}

"$ENV/bin/python" "$CODE/run_grid_experiments.py" \
  --output "$RUNS/gridworld/metrics.json" --seeds 30 \
  > "$LOGS/gridworld.log" 2>&1 &
pids+=("$!")
names+=("gridworld")

for seed in 11 22 33; do
  name="ppo_seed${seed}"
  launch "$name" train_on_policy.py \
    device=cuda use_wandb=false save_video=false seed="$seed" \
    num_train_frames=1000000 num_eval_episodes=50 \
    hydra.run.dir="$RUNS/$name"
done

for seed in 11 22 33 44 55; do
  name="off_base_seed${seed}"
  launch "$name" train_off_policy.py \
    device=cuda use_wandb=false save_video=false seed="$seed" \
    num_train_frames=100000 num_eval_episodes=50 \
    hydra.run.dir="$RUNS/$name"

  name="off_utd5_seed${seed}"
  launch "$name" train_off_policy.py \
    device=cuda use_wandb=false save_video=false seed="$seed" \
    num_train_frames=40000 num_eval_episodes=50 \
    agent.num_critics=10 utd=5 \
    hydra.run.dir="$RUNS/$name"
done

for gae_lambda in 0.0 0.95 0.99 1.0; do
  safe_lambda="${gae_lambda/./p}"
  for seed in 101 102; do
    name="ppo_gae${safe_lambda}_seed${seed}"
    launch "$name" train_on_policy.py \
      device=cuda use_wandb=false save_video=false seed="$seed" \
      num_train_frames=200000 num_eval_episodes=20 \
      agent.gae_lambda="$gae_lambda" \
      hydra.run.dir="$RUNS/$name"
  done
done

monitor_pid=""
(
  echo "timestamp,gpu_util_pct,memory_used_mib,power_w,cpu_load_1m"
  while true; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
        break
      fi
    done
    [[ "$alive" -eq 0 ]] && break
    gpu=$(
      nvidia-smi \
        --query-gpu=utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits
    )
    load=$(cut -d' ' -f1 /proc/loadavg)
    echo "$(date --iso-8601=seconds),$gpu,$load"
    sleep 30
  done
) > "$RUNS/resource_usage.csv" &
monitor_pid="$!"

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "${names[$index]},success" >> "$RUNS/status.csv"
  else
    echo "${names[$index]},failed" >> "$RUNS/status.csv"
    status=1
  fi
done
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
exit "$status"
