# CS224R Homework 1: Imitation Learning on Flappy Bird

This folder contains the starter code for Homework 1 of CS224R (Deep Reinforcement Learning), focused on imitation learning with action chunking in a custom Flappy Bird environment.

## 1. What You Are Building

You are implementing a standard imitation learning pipeline widely used across robotics, autonomous driving, and language modeling. In the LLM world this same idea is called **Supervised Fine-Tuning (SFT)**, cloning expert demonstrations to train a policy. Here you will apply it to a control task with continuous actions:

1. `BC Reg` (Behavior Cloning with MSE): learn a policy from expert demonstrations via MSE regression.
2. `BC Flow` (Behavior Cloning with Flow Matching): learn a policy from the expert demos via a generative model.
3. `DAgger` (Dataset Aggregation): iteratively relabel policy rollouts with a deterministic expert.
4. `Evaluation`: evaluates the control policy by letting it run in a couple environments and hundred episodes.

## 2. Current Project Status (What Is and Is Not Implemented)

Implemented (provided):
- Flappy Bird Gymnasium environment with easy/hard modes and PD-controlled physics.
- Expert policy (`Expert.act`) and data collection with action-chunk windowing.
- Full training pipeline orchestration, evaluation, and video recording.
- U-Net architecture.
- DAgger outer loop (`run_dagger`) and `DeterministicExpert` scaffolding.
- All visualization and policy wrapper utilities.

Not implemented (you need to write these):
- `networks.py`: `BCPolicy.__init__`, `BCPolicy.forward`, `FlowMatchingSchedule.interpolate`, and `FlowMatchingSchedule.sample` raise `NotImplementedError`.
- `losses.py`: `mse_loss` and `flow_matching_loss` raise `NotImplementedError`.
- `dagger.py`: `DeterministicExpert.act`, `rollout_episode` and `rollout_and_relabel` raise `NotImplementedError`.

## 3. Flappy Bird Task Overview

The environment is a physics-based Flappy Bird where the agent controls a target y-position (normalised to [0, 1]). A PD controller converts the target into thrust internally, creating momentum-based dynamics that require anticipation.

### Observation (4-D, all normalised)

| Index | Name | Description |
|-------|------|-------------|
| 0 | `dist_to_pipe` | Horizontal distance to next pipe |
| 1 | `gap1_y` | Vertical position of gap 1 (upper gap in hard mode) |
| 2 | `gap2_y` | Vertical position of gap 2 (same as gap1 in easy mode) |
| 3 | `bird_y` | Current bird vertical position |

### Action

A single float in [0, 1] representing the target y-position. The environment's PD controller converts this to thrust.

### Difficulty Modes

- **Easy mode**: each pipe has one gap opening.
- **Hard mode**: pipes alternate between two and one openings.

### Action Chunking

Policies predict `ACTION_CHUNK=20` future target positions at once. During rollout, only the first `EXECUTE_STEPS=10` are executed before re-querying the policy (receding horizon).

### Episode Termination

- **Collision**: bird hits a pipe wall or the screen boundary (reward = -1).
- **Timeout**: episode reaches 1000 steps (the goal).

## 4. Folder Map

```text
├── README.md                    # This file
├── installation.md              # Environment setup instructions
├── requirements.txt             # Python dependencies
├── assets/                      # Sprites for Flappy Bird rendering
│   ├── background2.png
│   ├── pipe.png
│   ├── robobird_up.png
│   └── robobird_down.png
│
├── main.py                      # Training pipeline & CLI entrypoint (read-only)
├── networks.py                  # Network architectures               [TODO: BCPolicy, FlowMatchingSchedule]
├── losses.py                    # Loss functions                      [TODO: mse_loss, flow_matching_loss]
├── expert.py                    # Expert policy + data collection (read-only)
├── dagger.py                    # DAgger relabeling + training loop   [TODO: DeterministicExpert, rollout_episode, rollout_and_relabel]
├── flappy_bird_env.py           # Gymnasium environment (read-only)
├── visualization.py             # Evaluation, video, policy wrappers (read-only)
│
├── models/                      # Saved model checkpoints (generated at runtime)
├── videos/                      # Generated .mp4 episode videos (generated at runtime)
└── results/                     # Text summaries with trajectory logs (generated at runtime)
```

## 5. Setup

See [installation.md](installation.md) for detailed instructions, or [colab_instructions.md](colab_instructions.md) if running on Colab.

### Quickstart

```bash
conda create -n hw1 python=3.10 -y
conda activate hw1
pip install torch gymnasium pygame "imageio[ffmpeg]"
```

For GPU support, use the pip command from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) instead of `pip install torch`.

Notes:
- A CUDA-capable GPU is not strictly required but speeds up training significantly.
- The code supports automatic device selection: CUDA > MPS (Apple Silicon) > CPU.
- If you encounter display-related errors when rendering videos, set `export SDL_VIDEODRIVER=dummy`.

## 6. Running the Pipeline

Tip: use specific `--method` and `--env` flags to run one part at a time instead of the full pipeline.

### Individual Methods

```bash
python main.py --method bc_reg --env easy
python main.py --method bc_reg --env hard
python main.py --method bc_flow --env hard
python main.py --method dagger --env hard
```

### Full Pipeline (all methods, all environments)

```bash
python main.py
```

### Output

Each run creates timestamped directories:
- `videos/<timestamp>/` -- `.mp4` episode videos for each method.
- `models/<timestamp>/` -- saved PyTorch model checkpoints (`.pt` files).
- `results/<timestamp>/` -- text summaries with evaluation stats and trajectory logs.

## 7. Environment Details

### Physics

| Parameter | Value | Description |
|-----------|-------|-------------|
| Gravity | 0.5 px/step^2 | Constant downward acceleration |
| Thrust scale | 2.5 px/step^2 | Per-unit upward acceleration |
| Max velocity | 10.0 px/step | Velocity clamp |
| PD Kp | 3.5 | Proportional gain (position error) |
| PD Kd | 1.2 | Derivative gain (velocity damping) |
| Pipe speed | 3.0 px/step | Horizontal pipe scroll rate |
| Pipe gap size | 75 px | Vertical size of each opening |
| Hard gap separation | 150 px | Distance between gap centers in hard mode |
| Screen size | 800 x 448 px | Width x Height |
| Max steps | 1000 | Episode timeout |

### Expert Behavior

The expert uses a commitment-based strategy with EMA smoothing (factor 0.15):
- **Easy mode**: always targets `gap1_y`.
- **Hard mode**: hovers at the midpoint of both gaps while far away. When within `COMMIT_DIST=0.18` normalised distance, randomly commits to gap1 or gap2 for the remainder of that pipe. Resets commitment when a new pipe is detected (via a rounded gap-position signature).

## 8. Troubleshooting

**Pygame display errors / `SDL` issues:**
- Set `export SDL_VIDEODRIVER=dummy` before running.
- Ensure `pygame` is installed: `pip install pygame`.

**Video encoding errors:**
- Ensure `imageio[ffmpeg]` is installed. If ffmpeg is missing: `pip install imageio[ffmpeg]`.

**Slow training:**
- Training runs on CPU by default if no GPU is detected. Each method takes a few minutes on CPU.

## 9. Quick Command Cheat Sheet

```bash
# Part 1: BC Reg on easy (after implementing BCPolicy, mse_loss)
python main.py --method bc_reg --env easy

# Part 2: BC Reg on hard (observe failure)
python main.py --method bc_reg --env hard

# Part 3: BC Flow on hard (after implementing FlowMatchingSchedule + flow_matching_loss)
python main.py --method bc_flow --env hard

# Part 4: DAgger on hard (after filling in DeterministicExpert gap choice + rollout_and_relabel)
python main.py --method dagger --env hard

# Full pipeline (all methods, all envs)
python main.py
```
