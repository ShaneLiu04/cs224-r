"""Main pipeline: imitation learning on Flappy Bird with action chunks.

Part 1 -- Easy mode: BC regression works on unimodal expert data.

Part 2 -- Hard mode: BC regression averages bimodal expert actions and crashes.

Part 3 -- Hard mode: Flow Matching models the full distribution and succeeds.

Part 4 -- Hard mode (DAgger): starting from BC, DAgger relabels with a
deterministic expert (always upper gap), resolving the ambiguity so BC succeeds.

Action chunking: policies predict ACTION_CHUNK future target positions at once.
During rollout, only the first EXECUTE_STEPS are executed before re-predicting
(receding horizon).

Usage:
    python main.py                              # run full episodes
    python main.py --method bc_reg --env easy       # just BC on easy mode
    python main.py --method bc_reg --env hard       # BC on hard mode (observe failure)
    python main.py --method bc_flow --env hard  # flow matching on hard
    python main.py --method dagger --env hard   # DAgger on hard mode
    python main.py --plot                          # uses latest run in results/
    python main.py --plot results/20260330_120000  # specific run directory
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from networks import (
    BCPolicy, FlowMatchingPolicy,
)
from losses import mse_loss, flow_matching_loss
from expert import collect_expert_data
from dagger import run_dagger
from flappy_bird_env import FlappyBirdEnv
from visualization import (
    ExpertWrapper, FlowMatchingWrapper,
    ChunkExecutor, evaluate_policy,
    plot_summary, plot_dagger_iterations,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BC_BATCH_SIZE = 2048
BC_EPOCHS = 100
BC_LR = 1e-5
NUM_DIFFUSION_ITERS = 20
PIPE_SPEED = 3.0
ACTION_CHUNK = 20
EXECUTE_STEPS = ACTION_CHUNK // 2

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Using device: {DEVICE}")
print(f"Action chunk: {ACTION_CHUNK}, execute first {EXECUTE_STEPS}")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_policy(model, loss_fn, states, actions, epochs=50, batch_size=256,
                 lr=1e-3, log_every=10, verbose=False, device=DEVICE):
    """Generic training loop for any policy.

    Args:
        model: the network to train (parameters will be optimized).
        loss_fn: callable(model, s_batch, a_batch) -> scalar loss.
        states/actions: numpy arrays of training data.
        epochs, batch_size, lr: training hyperparameters.
        log_every: print loss every N epochs when verbose=True.
        verbose: whether to print training progress.
        device: torch device.

    Returns:
        The trained model.
    """
    s_tensor = torch.tensor(states, dtype=torch.float32)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    loader = DataLoader(TensorDataset(s_tensor, a_tensor),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            loss = loss_fn(model, s_batch, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s_batch.size(0)
            n += s_batch.size(0)

        if verbose and (epoch + 1) % log_every == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n:.6f}")

    return model


def train_bc_policy(states, actions, epochs=50, batch_size=256, lr=1e-3,
                    verbose=False, device=DEVICE):
    """Train BC policy that outputs ACTION_CHUNK actions."""
    action_dim = np.array(actions).shape[1]
    policy = BCPolicy(action_dim=action_dim).to(device)
    return train_policy(policy, mse_loss, states, actions,
                        epochs=epochs, batch_size=batch_size, lr=lr,
                        log_every=10, verbose=verbose, device=device)


def train_flow_matching_policy(states, actions, epochs=100, batch_size=256,
                               lr=1e-4, num_steps=20, verbose=False,
                               device=DEVICE):
    """Train a Flow Matching policy (same U-Net architecture as diffusion)."""
    action_dim = np.array(actions).shape[1]
    state_dim = np.array(states).shape[1]

    policy = FlowMatchingPolicy(
        state_dim=state_dim, pred_horizon=action_dim, action_dim=1,
        num_steps=num_steps, device=device,
    ).to(device)
    train_policy(policy, flow_matching_loss, states, actions,
                 epochs=epochs, batch_size=batch_size, lr=lr,
                 log_every=20, verbose=verbose, device=device)
    return FlowMatchingWrapper(policy.model, policy.schedule)


# ---------------------------------------------------------------------------
# Results / trajectory capture
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_result_file(policy, difficulty, file_label, mean, std,
                     num_eval_episodes, results_dir, use_chunks=True,
                     dagger_rounds=None):
    """Run 5 episodes, record pipe-by-pipe trajectory text, and save to file."""
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=PIPE_SPEED)
    if hasattr(policy, "set_env"):
        policy.set_env(env)
    executor = ChunkExecutor(execute_steps=EXECUTE_STEPS) if use_chunks else None

    lines = [
        f"Method: {file_label}",
        f"Env:    {difficulty}",
        f"Eval:   {mean:.1f} +/- {std:.1f}  ({num_eval_episodes} episodes)",
        f"Chunks: predict {ACTION_CHUNK}, execute {EXECUTE_STEPS}",
    ]

    if dagger_rounds is not None:
        round_means, round_stds = dagger_rounds
        lines.append("")
        lines.append("DAgger per-round performance:")
        for rnd, (m, s) in enumerate(zip(round_means, round_stds), 1):
            lines.append(f"  Round {rnd}: {m:.1f} +/- {s:.1f}")

    lines.append("")

    seed_base = 7000
    traj_lengths = []
    for ep in range(5):
        obs, _ = env.reset(seed=seed_base + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        if executor:
            executor.reset()
        done = False
        traj = []

        while not done:
            if executor:
                if executor.needs_query():
                    state_t = torch.tensor(obs, dtype=torch.float32,
                                           device=DEVICE).unsqueeze(0)
                    pred = policy(state_t).cpu().numpy().flatten()
                    executor.set_chunk(pred)
                action = executor.get_action()
            else:
                state_t = torch.tensor(obs, dtype=torch.float32,
                                       device=DEVICE).unsqueeze(0)
                action = float(policy(state_t).cpu().numpy().flat[0])

            traj.append((obs.copy(), action))
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated

        outcome = "TIMEOUT" if truncated else "CRASHED"
        traj_lengths.append(len(traj))
        lines.append(f"=== Episode {ep+1}: {len(traj)} steps, {outcome} ===")

        pipe_num = 0
        min_dist = float("inf")
        crossing_snap = None
        prev_dist = None

        for step, (o, a) in enumerate(traj):
            dist, g1, g2, bird_y = o[0], o[1], o[2], o[3]

            if dist < min_dist:
                min_dist = dist
                crossing_snap = (step, bird_y, g1, g2, a)

            if prev_dist is not None and dist > prev_dist + 0.05:
                pipe_num += 1
                s, by, g1s, g2s, act = crossing_snap
                lines.append(
                    f"  pipe {pipe_num:2d} (t={s:4d}): "
                    f"bird={by:.3f}  gaps=({g1s:.3f}, {g2s:.3f})  act={act:.3f}"
                )
                min_dist = float("inf")
                crossing_snap = None

            prev_dist = dist

        if outcome == "CRASHED" and traj:
            o_fin, _ = traj[-1]
            lines.append(
                f"  CRASH  (t={len(traj):4d}): "
                f"bird={o_fin[3]:.3f}  gaps=({o_fin[1]:.3f}, {o_fin[2]:.3f})"
            )

        lines.append("")

    env.close()

    filepath = os.path.join(results_dir, f"{file_label}_{difficulty}.txt")
    os.makedirs(results_dir, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved results: {filepath}")


# ---------------------------------------------------------------------------
# Individual run functions
# ---------------------------------------------------------------------------

def _collect_data(difficulty, videos_dir):
    """Collect expert data and evaluate the expert baseline."""
    print(f"\nCollecting {difficulty}-mode expert data...")
    states, actions = collect_expert_data(
        difficulty, num_episodes=500, action_chunk=ACTION_CHUNK,
        seed=1 if difficulty == "hard" else 2000)
    print(f"    Collected {len(states)} chunk transitions")

    print(f"    Evaluating expert on {difficulty} (with video)...")
    expert = ExpertWrapper(difficulty)
    expert_mean, expert_std = evaluate_policy(
        expert, difficulty, num_episodes=10, seed=500,
        use_chunks=False,
        video_path=f"{videos_dir}/expert_{difficulty}.mp4", video_episodes=3)
    print(f"    Expert {difficulty}: {expert_mean:.1f} +/- {expert_std:.1f}")
    return states, actions


def run_bc_reg(difficulty, states, actions, videos_dir, models_dir, results_dir):
    print(f"\nTraining BC (MSE regression) on {difficulty} data...")
    policy = train_bc_policy(states, actions, epochs=BC_EPOCHS,
                             lr=BC_LR, verbose=True, batch_size=BC_BATCH_SIZE)
    mean, std = evaluate_policy(
        policy, difficulty, num_episodes=50, seed=500,
        video_path=f"{videos_dir}/bc_reg_{difficulty}.mp4", video_episodes=3)
    print(f"    BC on {difficulty}: {mean:.1f} +/- {std:.1f}")
    torch.save(policy.state_dict(), f"{models_dir}/bc_reg_{difficulty}_chunk.pt")
    save_result_file(policy, difficulty, "bc_reg", mean, std, 50, results_dir)
    return policy, mean, std


def run_bc_flow(difficulty, states, actions, videos_dir, models_dir,
                results_dir):
    print(f"\nTraining Flow Matching policy on {difficulty} data...")
    policy = train_flow_matching_policy(
        states, actions, epochs=50, num_steps=NUM_DIFFUSION_ITERS,
        batch_size=BC_BATCH_SIZE, verbose=True)
    mean, std = evaluate_policy(
        policy, difficulty, num_episodes=50, seed=500,
        video_path=f"{videos_dir}/bc_flow_{difficulty}.mp4",
        video_episodes=3)
    print(f"    Flow Matching on {difficulty}: {mean:.1f} +/- {std:.1f}")
    torch.save(policy.state_dict(),
               f"{models_dir}/bc_flow_{difficulty}_chunk.pt")
    save_result_file(policy, difficulty, "bc_flow", mean, std, 50, results_dir)
    return policy, mean, std


def run_dagger_method(difficulty, states, actions, bc_policy, videos_dir,
                      models_dir, results_dir):
    """Run DAgger. Trains BC first if no bc_policy provided."""
    if bc_policy is None:
        print("\n    Training BC policy as DAgger prerequisite...")
        bc_policy = train_bc_policy(states, actions, epochs=BC_EPOCHS,
                                    lr=BC_LR, batch_size=BC_BATCH_SIZE)

    print(f"\nRunning DAgger on {difficulty} mode "
          "(deterministic expert -> upper gap)...")
    policy, means, stds = run_dagger(
        difficulty=difficulty,
        initial_states=states,
        initial_actions=actions,
        rounds=5,
        episodes_per_round=30,
        epochs=BC_EPOCHS,
        pipe_speed=PIPE_SPEED,
        seed=5000,
        action_chunk=ACTION_CHUNK,
        device=DEVICE,
        train_bc_fn=train_bc_policy,
        eval_episodes=50,
        verbose=True,
        initial_policy=bc_policy,
    )
    final_mean, final_std = means[-1], stds[-1]
    print(f"    DAgger final on {difficulty}: {final_mean:.1f} +/- {final_std:.1f}")

    torch.save(policy.state_dict(), f"{models_dir}/dagger_{difficulty}_chunk.pt")

    eval_mean, eval_std = evaluate_policy(
        policy, difficulty, num_episodes=100, seed=500,
        video_path=f"{videos_dir}/dagger_{difficulty}.mp4", video_episodes=3)
    print(f"    DAgger eval: {eval_mean:.1f} +/- {eval_std:.1f}")

    save_result_file(policy, difficulty, "dagger", eval_mean, eval_std,
                     100, results_dir, dagger_rounds=(means, stds))

    return policy, means, stds, final_mean, final_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Imitation learning on Flappy Bird with action chunks.")
    parser.add_argument("--method", type=str, default="all",
                        choices=["bc_reg", "bc_flow", "dagger", "all"],
                        help="Which method to run (default: all)")
    parser.add_argument("--env", type=str, default="all",
                        choices=["easy", "hard", "all"],
                        help="Which environment difficulty (default: all)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    videos_dir = f"videos/{timestamp}"
    models_dir = f"models/{timestamp}"
    results_dir = f"results/{timestamp}"
    os.makedirs("data", exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Run timestamp: {timestamp}")
    print(f"  Videos  -> {videos_dir}/")
    print(f"  Models  -> {models_dir}/")
    print(f"  Results -> {results_dir}/")
    print(f"  Method  -> {args.method}")
    print(f"  Env     -> {args.env}")

    run_all = args.method == "all"
    envs = ["easy", "hard"] if args.env == "all" else [args.env]
    methods = (["bc_reg", "bc_flow", "dagger"]
               if run_all else [args.method])

    results = {}
    dagger_data = {}  # difficulty -> (dag_means, dag_stds)

    for difficulty in envs:
        states, actions = _collect_data(difficulty, videos_dir)

        bc_policy = None

        if "bc_reg" in methods:
            policy, mean, std = run_bc_reg(difficulty, states, actions,
                                           videos_dir, models_dir, results_dir)
            results[("bc_reg", difficulty)] = (mean, std)
            bc_policy = policy

        if "bc_flow" in methods:
            _, mean, std = run_bc_flow(difficulty, states, actions,
                                       videos_dir, models_dir,
                                       results_dir)
            results[("bc_flow", difficulty)] = (mean, std)

        if "dagger" in methods:
            _, dag_means, dag_stds, final_mean, final_std = run_dagger_method(
                difficulty, states, actions, bc_policy, videos_dir, models_dir,
                results_dir)
            results[("dagger", difficulty)] = (final_mean, final_std)
            dagger_data[difficulty] = (dag_means, dag_stds)

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nAction chunks: predict {ACTION_CHUNK}, execute first "
          f"{EXECUTE_STEPS}")
    for (method, difficulty), (mean, std) in sorted(results.items()):
        print(f"  {method:20s} [{difficulty:4s}]: {mean:.1f} +/- {std:.1f}")
    # ----- Plots -----
    if results:
        plot_summary(results, results_dir)
    for difficulty, (dag_means, dag_stds) in dagger_data.items():
        bc_mean = results.get(("bc_reg", difficulty), (None,))[0]
        if bc_mean is not None:
            plot_dagger_iterations(dag_means, dag_stds, bc_mean,
                                   difficulty, results_dir)

    print(f"\nRun timestamp: {timestamp}")
    print(f"Videos saved to {videos_dir}/")
    print(f"Models saved to {models_dir}/")
    print(f"Results saved to {results_dir}/")
    print("=" * 60)


def _find_latest_results_dir():
    """Return the most recent results/{timestamp}/ directory."""
    base = "results"
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No '{base}/' directory found.")
    runs = sorted(
        [d for d in os.listdir(base)
         if os.path.isdir(os.path.join(base, d))],
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError("No runs found in results/.")
    return os.path.join(base, runs[0])


def _parse_result_file(path):
    """Parse a result .txt file, return (method, difficulty, mean, std, dagger_rounds)."""
    import re
    method = difficulty = None
    mean = std = None
    dag_means, dag_stds = [], []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Method:"):
                method = line.split(":", 1)[1].strip()
            elif line.startswith("Env:"):
                difficulty = line.split(":", 1)[1].strip()
            elif line.startswith("Eval:"):
                m = re.search(r"([\d.]+)\s*\+/-\s*([\d.]+)", line)
                if m:
                    mean, std = float(m.group(1)), float(m.group(2))
            elif line.startswith("Round"):
                m = re.search(r"Round\s+\d+:\s*([\d.]+)\s*\+/-\s*([\d.]+)", line)
                if m:
                    dag_means.append(float(m.group(1)))
                    dag_stds.append(float(m.group(2)))

    dagger_rounds = (dag_means, dag_stds) if dag_means else None
    return method, difficulty, mean, std, dagger_rounds


def plot_from_results(results_dir=None):
    """Re-generate plots from saved result .txt files.

    Usage:
        python main.py --plot              # use latest run
        python main.py --plot results/20260330_120000  # specific run
    """
    if results_dir is None:
        results_dir = _find_latest_results_dir()
    print(f"Generating plots from: {results_dir}")

    txt_files = [f for f in os.listdir(results_dir) if f.endswith(".txt")]
    if not txt_files:
        raise FileNotFoundError(
            f"No result .txt files found in {results_dir}. "
            "Run training first with: python main.py"
        )

    results = {}
    dagger_data = {}
    missing = []

    for fname in txt_files:
        path = os.path.join(results_dir, fname)
        method, difficulty, mean, std, dagger_rounds = _parse_result_file(path)
        if method is None or mean is None:
            missing.append(f"  {fname}: could not parse method/eval stats")
            continue
        results[(method, difficulty)] = (mean, std)
        if dagger_rounds is not None:
            dagger_data[difficulty] = dagger_rounds

    if missing:
        print("WARNING — could not parse these files:")
        for m in missing:
            print(m)

    if not results:
        raise FileNotFoundError(
            f"No valid result files found in {results_dir}."
        )

    plot_summary(results, results_dir)

    for difficulty, (dag_means, dag_stds) in dagger_data.items():
        bc_mean = results.get(("bc_reg", difficulty), (None,))[0]
        if bc_mean is None:
            print(f"  WARNING: no bc_reg result for {difficulty}, "
                  f"skipping DAgger iteration plot (need bc_reg_{difficulty}.txt)")
            continue
        plot_dagger_iterations(dag_means, dag_stds, bc_mean,
                               difficulty, results_dir)

    print("Done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "--plot":
        plot_from_results(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        main()
