"""Run reproducible, report-oriented Assignment 1 experiments."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from dagger import run_dagger
from expert import collect_expert_data
from flappy_bird_env import FlappyBirdEnv
from losses import flow_matching_loss, mse_loss
from main import ACTION_CHUNK, PIPE_SPEED
from networks import BCPolicy, FlowMatchingPolicy
from visualization import ChunkExecutor, FlowMatchingWrapper


def set_seed(seed):
    """Set all random seeds used by this experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_with_history(model, loss_fn, states, actions, epochs, batch_size,
                       learning_rate, device):
    """Train one policy and return epoch losses and elapsed seconds."""
    states_tensor = torch.as_tensor(
        states, dtype=torch.float32, device=device,
    )
    actions_tensor = torch.as_tensor(
        actions, dtype=torch.float32, device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        total, count = 0.0, 0
        permutation = torch.randperm(len(states_tensor), device=device)
        for offset in range(0, len(states_tensor), batch_size):
            indices = permutation[offset:offset + batch_size]
            states_batch = states_tensor[indices]
            actions_batch = actions_tensor[indices]
            loss = loss_fn(model, states_batch, actions_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(states_batch)
            count += len(states_batch)
        history.append(total / count)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return history, time.perf_counter() - start


def train_bc_fast(states, actions, epochs=50, batch_size=256, lr=1e-3,
                  verbose=False, device=None):
    """Train BC with GPU-resident tensors while preserving optimizer settings."""
    del verbose
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    policy = BCPolicy(action_dim=np.asarray(actions).shape[1]).to(device)
    train_with_history(
        policy, mse_loss, states, actions, epochs, batch_size, lr, device,
    )
    return policy


@torch.no_grad()
def evaluate_lengths(policy, difficulty, seeds, device, execute_steps=10):
    """Evaluate a policy and return every episode length plus latency."""
    policy.eval()
    environment = FlappyBirdEnv(
        difficulty=difficulty, pipe_speed=PIPE_SPEED,
    )
    lengths, calls = [], 0
    policy_seconds = 0.0
    for seed in seeds:
        observation, _ = environment.reset(seed=int(seed))
        if hasattr(policy, "reset"):
            policy.reset()
        executor = ChunkExecutor(
            chunk_size=ACTION_CHUNK, execute_steps=execute_steps,
        )
        done = False
        while not done:
            if executor.needs_query():
                state = torch.as_tensor(
                    observation, dtype=torch.float32, device=device,
                ).unsqueeze(0)
                start = time.perf_counter()
                chunk = policy(state).detach().cpu().numpy().reshape(-1)
                policy_seconds += time.perf_counter() - start
                calls += 1
                executor.set_chunk(chunk)
            action = executor.get_action()
            observation, _, terminated, truncated, _ = environment.step(
                np.array([action], dtype=np.float32),
            )
            done = terminated or truncated
        lengths.append(environment.step_count)
    environment.close()
    latency_ms = 1000.0 * policy_seconds / max(calls, 1)
    return lengths, latency_ms


def summarize(lengths):
    """Return descriptive statistics for episode lengths."""
    values = np.asarray(lengths, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "success_rate": float(np.mean(values >= 1000)),
        "n": int(len(values)),
        "lengths": [int(value) for value in values],
    }


def evaluate_configuration(policy, difficulty, eval_seeds, device,
                           execute_steps=10):
    """Evaluate and attach policy-call latency to summary statistics."""
    lengths, latency = evaluate_lengths(
        policy, difficulty, eval_seeds, device, execute_steps,
    )
    result = summarize(lengths)
    result["latency_ms"] = latency
    result["execute_steps"] = execute_steps
    return result


def system_metadata(device):
    """Capture software and hardware versions used by the experiment."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--flow-epochs", type=int, default=50)
    parser.add_argument("--dagger-rounds", type=int, default=5)
    parser.add_argument("--dagger-episodes", type=int, default=30)
    args = parser.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "seed": args.seed,
        "metadata": system_metadata(device),
        "config": vars(args) | {"output": str(args.output)},
        "methods": {},
        "ablations": {},
    }

    datasets = {}
    for difficulty, offset in (("easy", 0), ("hard", 10000)):
        start = time.perf_counter()
        states, actions = collect_expert_data(
            difficulty=difficulty,
            num_episodes=args.expert_episodes,
            action_chunk=ACTION_CHUNK,
            pipe_speed=PIPE_SPEED,
            seed=args.seed + offset,
        )
        datasets[difficulty] = (states, actions)
        metrics[f"{difficulty}_dataset"] = {
            "transitions": int(len(states)),
            "collection_seconds": time.perf_counter() - start,
        }
        np.savez_compressed(
            args.output / f"dataset_summary_{difficulty}_{args.seed}.npz",
            states=states[::10],
            actions=actions[::10],
        )

    trained = {}
    for difficulty in ("easy", "hard"):
        states, actions = datasets[difficulty]
        policy = BCPolicy(action_dim=ACTION_CHUNK).to(device)
        history, seconds = train_with_history(
            policy, mse_loss, states, actions,
            epochs=args.bc_epochs, batch_size=2048,
            learning_rate=1e-5, device=device,
        )
        result = evaluate_configuration(
            policy, difficulty,
            range(500 + args.seed, 500 + args.seed + args.eval_episodes),
            device,
        )
        result |= {"train_seconds": seconds, "loss": history}
        metrics["methods"][f"bc_reg_{difficulty}"] = result
        trained[f"bc_reg_{difficulty}"] = policy
        torch.save(
            policy.state_dict(),
            args.output / f"bc_reg_{difficulty}_{args.seed}.pt",
        )

    hard_states, hard_actions = datasets["hard"]
    flow = FlowMatchingPolicy(
        pred_horizon=ACTION_CHUNK, num_steps=20, device=device,
    ).to(device)
    history, seconds = train_with_history(
        flow, flow_matching_loss, hard_states, hard_actions,
        epochs=args.flow_epochs, batch_size=2048,
        learning_rate=1e-4, device=device,
    )
    flow_wrapper = FlowMatchingWrapper(flow.model, flow.schedule)
    result = evaluate_configuration(
        flow_wrapper, "hard",
        range(1500 + args.seed, 1500 + args.seed + args.eval_episodes),
        device,
    )
    result |= {"train_seconds": seconds, "loss": history}
    metrics["methods"]["bc_flow_hard"] = result
    torch.save(
        flow.model.state_dict(), args.output / f"bc_flow_hard_{args.seed}.pt",
    )

    flow_steps = {}
    for steps in (5, 10, 20, 40):
        flow.schedule.num_steps = steps
        flow_steps[str(steps)] = evaluate_configuration(
            flow_wrapper, "hard",
            range(2500 + args.seed, 2500 + args.seed + args.eval_episodes),
            device,
        )
    flow.schedule.num_steps = 20
    metrics["ablations"]["flow_integration_steps"] = flow_steps

    dagger, round_means, round_stds = run_dagger(
        difficulty="hard",
        initial_states=hard_states,
        initial_actions=hard_actions,
        rounds=args.dagger_rounds,
        episodes_per_round=args.dagger_episodes,
        epochs=args.bc_epochs,
        pipe_speed=PIPE_SPEED,
        seed=3500 + args.seed,
        action_chunk=ACTION_CHUNK,
        device=device,
        train_bc_fn=train_bc_fast,
        eval_episodes=args.eval_episodes,
        lr=1e-5,
        batch_size=2048,
        verbose=False,
        initial_policy=trained["bc_reg_hard"],
    )
    dagger_result = evaluate_configuration(
        dagger, "hard",
        range(4500 + args.seed, 4500 + args.seed + args.eval_episodes),
        device,
    )
    dagger_result["round_means"] = [float(value) for value in round_means]
    dagger_result["round_stds"] = [float(value) for value in round_stds]
    metrics["methods"]["dagger_hard"] = dagger_result
    torch.save(
        dagger.state_dict(), args.output / f"dagger_hard_{args.seed}.pt",
    )

    execute_steps = {}
    for steps in (1, 5, 10, 20):
        execute_steps[str(steps)] = {
            "bc_easy": evaluate_configuration(
                trained["bc_reg_easy"], "easy",
                range(5500 + args.seed, 5500 + args.seed + args.eval_episodes),
                device, execute_steps=steps,
            ),
            "dagger_hard": evaluate_configuration(
                dagger, "hard",
                range(6500 + args.seed, 6500 + args.seed + args.eval_episodes),
                device, execute_steps=steps,
            ),
        }
    metrics["ablations"]["execute_steps"] = execute_steps

    output_path = args.output / f"metrics_seed_{args.seed}.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
