### Local environment

Create a new conda environment `cs224r-hw3` and activate it:

```bash
conda create -n cs224r-hw3 python=3.10.19
conda activate cs224r-hw3
```


### Modal (for training)

All training jobs run on [Modal](https://modal.com).

### 1. Install Modal in the conda env

```bash
pip install modal
```

### 2. Authenticate

```bash
modal setup
```

### 3. Redeem course compute credits

Redeem the credits from the course email at https://modal.com/credits.

### Running training jobs

Use `modal_train.py` as the entry point for all algorithms. It handles image build, dependency installation, and volume persistence automatically.

**Basic usage:**

```bash
modal run --detach modal_train.py --algo <bc|awac|iql> --env-name <env> --exp-name <name> [extra flags]
```

**Example**:

```bash
modal run --detach modal_train.py --algo awac --env-name antmaze-umaze-v0 --exp-name awac_antmaze_umaze --use-wandb --seed 1
```

The job continues on Modal's servers after you close your terminal. Monitor progress from the [Modal dashboard](https://modal.com).


### Parallel runs (3 seeds)

Use `modal_train_para.py` to launch 3 seeds simultaneously, each on its own GPU container. For example, 

```bash
# AWAC
modal run --detach modal_train_para.py --algo awac --env-name antmaze-umaze-v0 --exp-name awac_antmaze_umaze --use-wandb
```

The script accepts the same flags as `modal_train.py`. Results are saved to separate directories per seed (`<exp-name>_seed1`, `<exp-name>_seed2`, `<exp-name>_seed3`).

### Stopping a running job

**From the terminal:**

```bash
# List running apps
modal app list

# Stop by app ID
modal app stop <app-id>

# Stop by name
modal app stop --name cs224r-hw3
```

**From the dashboard:** go to [modal.com](https://modal.com) → Apps → find your job → click Stop.

**If attached** (no `--detach`): press `Ctrl+C` in the terminal.

### Retrieving results

Training outputs (logs, checkpoints, videos) are saved to the `cs224r-hw3-results` Modal Volume, mounted at `/root/hw/data` inside the container.

**Download all logs and videos to your local machine:**

```bash
mkdir -p data
modal volume get cs224r-hw3-results / ./data/
```

This copies everything from the volume into a local `data/` folder. You can then open the video files (`.mp4`) directly to visualize evaluation rollouts.

**Browse the volume without downloading:**

```bash
modal volume ls cs224r-hw3-results
modal volume ls cs224r-hw3-results /<exp-name>/
```

**Download only a specific experiment:**

```bash
modal volume get cs224r-hw3-results /<exp-name>/ ./data/
```

### W&B integration

Create a Modal secret with your API key (run once):

```bash
modal secret create wandb WANDB_API_KEY=<your_key>
```

The secret is already wired into `modal_train.py`. Pass `--use-wandb` to any training command to enable logging. If you have not created the secret yet, comment out the `_secrets` line in `modal_train.py` before running without W&B.

## Codebase Structure

The code is organized under `cs224r/` with the following key components:

```
cs224r/
├── agents/              # Agent classes
│   ├── awac_agent.py    # AWAC agent
│   ├── iql_agent.py     # IQL agent
│   ├── bc_agent.py      # Behavior cloning agent (reference)
│   └── base_agent.py    # Abstract base class for agents
│
├── critics/             # Critic / value network implementations
│   ├── awac_critic.py   # Q-network and update logic for AWAC
│   ├── iql_critic.py    # Q/V-network and update logic for IQL
│   └── base_critic.py   # Abstract base class for critics
│
├── policies/            # Policy network implementations
│   ├── MLP_policy.py    # MLP policy used by AWAC and IQL
│   ├── argmax_policy.py # Greedy argmax policy wrapper
│   └── base_policy.py   # Abstract base class for policies
│
├── infrastructure/      # Training loop and utilities
│   ├── rl_trainer_awac.py  # Main training loop (RL_Trainer)
│   ├── pytorch_util.py     # PyTorch helper functions
│   ├── utils.py            # Rollout sampling, env registration, and env utilities
│   └── wrappers.py         # Gym environment wrappers
│
├── scripts/             # Entry points
│   ├── run_algo.py         # Unified entry point for all algorithms
│
└── envs/                # Custom environments
    └── pointmass/       # PointMass environment
```

### How training works

1. **`scripts/run_algo.py`** — instantiates a `Trainer`, which selects the right agent class (`BCAgent`, `AWACAgent`, or `IQLAgent`) based on the `--algo` flag.

2. **`infrastructure/rl_trainer_awac.py`** (`RL_Trainer`) — runs the main training loop. It handles offline dataset loading, evaluation, and logging to WandB. 

3. **`agents/awac_agent.py` / `agents/iql_agent.py`** — each agent holds references to its critic and actor, and implements `train()`, which samples a batch from the replay buffer and calls the critic and actor update methods.

4. **`critics/awac_critic.py` / `critics/iql_critic.py`** — contain the Q (and V for IQL) networks and implement the core `update()` loss computation.

5. **`policies/MLP_policy.py`** — defines `MLPPolicyAWAC`, a stochastic MLP policy. Its `update()` method computes the AWAC policy loss using advantage-weighted regression.

## Files to Implement

You are expected to fill in the `TODO` blocks in the following files:

- `cs224r/critics/awac_critic.py`
- `cs224r/critics/iql_critic.py`
- `cs224r/agents/awac_agent.py`
- `cs224r/agents/iql_agent.py`
- `cs224r/policies/MLP_policy.py`

The descriptions for TODOs have been provided in the homework PDF. Each TODO block in the code also includes inline hints to guide your implementation.
