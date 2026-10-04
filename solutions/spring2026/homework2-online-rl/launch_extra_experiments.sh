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

for seed in 44 55; do
  name="ppo_fast_seed${seed}"
  launch "$name" train_on_policy.py \
    device=cuda use_wandb=false save_video=false seed="$seed" \
    num_train_frames=1000000 num_eval_episodes=50 eval_every_frames=10000 \
    hydra.run.dir="$RUNS/$name"
done

for seed in 66 77 88; do
  name="off_utd5_c2_seed${seed}"
  launch "$name" train_off_policy.py \
    device=cuda use_wandb=false save_video=false seed="$seed" \
    num_train_frames=40000 num_eval_episodes=50 \
    agent.num_critics=2 utd=5 \
    hydra.run.dir="$RUNS/$name"

  name="off_utd1_c10_seed${seed}"
  launch "$name" train_off_policy.py \
    device=cuda use_wandb=false save_video=false seed="$seed" \
    num_train_frames=40000 num_eval_episodes=50 \
    agent.num_critics=10 utd=1 \
    hydra.run.dir="$RUNS/$name"

  name="off_nobc_seed${seed}"
  launch "$name" train_off_policy.py \
    device=cuda use_wandb=false save_video=false seed="$seed" \
    num_train_frames=40000 num_eval_episodes=50 \
    pretrain_steps=0 bc_freq=0 \
    hydra.run.dir="$RUNS/$name"
done

for seed in 201 202; do
  name="ppo_nobc_seed${seed}"
  launch "$name" train_on_policy.py \
    device=cuda use_wandb=false save_video=false seed="$seed" \
    num_train_frames=200000 num_eval_episodes=20 eval_every_frames=5000 \
    pretrain_steps=0 agent.reverse_kl_coef=0 \
    hydra.run.dir="$RUNS/$name"
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "${names[$index]},success" >> "$RUNS/extra_status.csv"
  else
    echo "${names[$index]},failed" >> "$RUNS/extra_status.csv"
    status=1
  fi
done
exit "$status"
