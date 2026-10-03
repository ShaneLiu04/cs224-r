# Homework 2

This repository contains the starter code for CS224R Homework 2 on Grid World and onthe Meta-World `hammer-v2` task. The codebase includes three RL baselines:

- `gridworld_q_learning.py`: q-learning algorithm on the grid world
- `off_policy.py`: off-policy actor-critic with behavior cloning pretraining on the hammer task
- `on_policy.py`: on-policy PPO with behavior cloning pretraining on the hammer task

The training scripts, configs, replay buffer, logging code, and environment wrappers are already provided. As a student, your main job is to implement the missing learning updates in the agent files.

## What Students Should Read

Start with these files:

- `gridworld_q_learning.py`: the q-learning agent implementation. This file contains missing student code for sampling the actions and training the q learning algorithm. 
- `off_policy.py`: the off-policy agent implementation. This file contains the missing student code for critic updates, actor updates, and behavior cloning.
- `on_policy.py`: the PPO agent implementation. This file contains the missing student code for GAE, PPO losses, and rollout updates.
- `train_off_policy.py`: local training loop for the off-policy agent.
- `train_on_policy.py`: local training loop for the PPO agent.
- `cfgs/off_policy_config.yaml`: default hyperparameters for the off-policy baseline.
- `cfgs/on_policy_config.yaml`: default hyperparameters for the PPO baseline.

These support files are also worth understanding:

- `mw.py`: wraps the Meta-World hammer environment in a `dm_env`-style interface used by the training code.
- `replay_buffer.py`: replay buffer storage and dataloader logic for demonstrations and off-policy training.
- `utils.py`: helper utilities such as seeding, tensor conversion, target-network updates, and action distributions.
- `logger.py`: CSV and Weights & Biases logging.
- `video.py`: rollout video recording during evaluation.

You usually do not need to modify the Modal wrappers unless you specifically want to run on Modal:

- `modal_gridworld_q_learning.py`
- `modal_on_policy.py`
- `modal_off_policy.py`

## Files Students Are Expected To Edit

The intended student-editable files are:

- `gridworld_q_learning.py`
- `off_policy.py`
- `on_policy.py`

These are the only files in this checkout that contain `YOUR CODE HERE` blocks.

## Repository Structure

Top-level layout:

- `cfgs/`: Hydra config files for each training setup.
- `demos/`: demonstration trajectories used for behavior cloning pretraining.
- `train_off_policy.py`: local entrypoint for off-policy training.
- `train_on_policy.py`: local entrypoint for PPO training.
- `modal_train_off_policy.py`: Modal entrypoint for off-policy training.
- `modal_train_on_policy.py`: Modal entrypoint for PPO training.
- `gridworld_q_learning.py`: q-learning agent definition.
- `off_policy.py`: off-policy agent definition.
- `on_policy.py`: PPO agent definition.
- `mw.py`: Meta-World environment construction.
- `replay_buffer.py`: replay buffer implementation.
- `logger.py`, `utils.py`, `video.py`: shared infrastructure.
- `setup.sh`: one-time Ubuntu/cloud bootstrap script for MuJoCo and dependencies.
- `conda_env_modal.yml`: Conda environment definition for launching on modal.
- `conda_env_local.yml`: Conda environment definition if you want to run locally.


## Running The Code

The local training entrypoints are:

```bash
python gridworld_q_learning.py
python train_off_policy.py
python train_on_policy.py
```

Both use Hydra configs from `cfgs/`, so you can override config values from the command line:

```bash
python train_off_policy.py batch_size=128 utd=2 agent.num_critics=10
python train_on_policy.py rollout_length=2048 pretrain_steps=1000 agent.clip_eps=0.2
```

Hydra writes outputs under `Logdir/`. A run directory typically contains:

- `wandb/` metadata when W&B logging is enabled

## Gridworld Q-Learning

Main files:

- `gridworld_q_learning.py`

Everything, environment + algorithm is implemented in this file.

## Off-Policy Training

Main files:

- `train_off_policy.py`
- `off_policy.py`
- `cfgs/off_policy_config.yaml`

Default behavior:

1. Copy demonstration trajectories into local `demos/` and `buffer/` directories inside the Hydra run folder.
2. Run behavior cloning pretraining.
3. Train an off-policy actor-critic agent with replay-buffer updates.

Useful config values in `cfgs/off_policy_config.yaml`:

- `num_train_frames`
- `batch_size`
- `utd`
- `bc_freq`
- `warmup`
- `agent.num_critics`
- `agent.hidden_dim`

## On-Policy PPO Training

Main files:

- `train_on_policy.py`
- `on_policy.py`
- `cfgs/on_policy_config.yaml`

Default behavior:

1. Copy demonstration trajectories into a local `demos/` directory inside the Hydra run folder.
2. Run behavior cloning pretraining.
3. Collect on-policy rollouts and optimize the PPO objective.

Useful config values in `cfgs/on_policy_config.yaml`:

- `num_train_frames`
- `rollout_length`
- `batch_size`
- `pretrain_steps`
- `bc_freq`
- `agent.clip_eps`
- `agent.ppo_epochs`
- `agent.value_coef`
- `agent.entropy_coef`
- `agent.gae_lambda`

## Modal Entry Points

If you are running on Modal instead of locally, use:

```bash
modal run modal_gridworld_q_learning.py
modal run modal_on_policy.py
modal run modal_off_policy.py
```

Those files are wrappers around the same core training logic and configs. The main student implementation work still lives in `gridworld_q_learning`, `off_policy.py` and `on_policy.py`.
