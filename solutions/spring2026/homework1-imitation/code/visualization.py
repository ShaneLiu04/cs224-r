"""Visualization and evaluation utilities.

All code in this file is fully provided -- no TODOs.

Contents:
    - ExpertWrapper: wraps Expert for the evaluate_policy interface.
    - DiffusionWrapper: wraps (model, schedule) for the same interface.
    - FlowMatchingWrapper: wraps (model, schedule) for flow matching.
    - GaussianWrapper: wraps GaussianBCPolicy (det or stochastic).
    - ChunkExecutor: receding-horizon action-chunk execution.
    - _draw_chunk_overlay: overlay predicted targets on video frames.
    - evaluate_policy: run episodes, optionally record videos, return stats.
"""

import os
import time

import numpy as np
import torch
import imageio.v3 as iio

from expert import Expert
from flappy_bird_env import FlappyBirdEnv, SCREEN_W, SCREEN_H, BIRD_X


# ---------------------------------------------------------------------------
# Policy wrappers
# ---------------------------------------------------------------------------

class ExpertWrapper:
    """Wrap Expert so it has the same call interface as a learned policy.

    Returns a single action (not a chunk) -- evaluated step-by-step.
    """

    def __init__(self, difficulty, env=None):
        self.expert = Expert()
        self.difficulty = difficulty
        self.env = env

    def eval(self):
        return self

    def reset(self):
        self.expert.reset()

    def set_env(self, env):
        self.env = env

    def __call__(self, state_t):
        obs = state_t.cpu().numpy()[0]
        a = self.expert.act(obs, self.difficulty)
        return torch.tensor([[a]], dtype=torch.float32, device=state_t.device)


class DiffusionWrapper:
    """Wrap (model, schedule) so it has the same call interface as BCPolicy."""

    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, state):
        return self.schedule.sample(self.model, state)

    def state_dict(self):
        return self.model.state_dict()


class FlowMatchingWrapper:
    """Wrap (model, schedule) for flow matching -- same interface as DiffusionWrapper."""

    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, state):
        return self.schedule.sample(self.model, state)

    def state_dict(self):
        return self.model.state_dict()


class GaussianWrapper:
    """Wrap GaussianBCPolicy for evaluate_policy (det or stochastic)."""

    def __init__(self, model, stochastic=False):
        self.model = model
        self.stochastic = stochastic

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, state):
        if self.stochastic:
            return self.model.sample(state)
        return self.model.deterministic(state)

    def state_dict(self):
        return self.model.state_dict()


# ---------------------------------------------------------------------------
# Chunk executor (receding horizon)
# ---------------------------------------------------------------------------

class ChunkExecutor:
    """Execute action chunks with a receding horizon.

    Every ``execute_steps`` steps the policy is queried for a new chunk of
    ``chunk_size`` actions.  Only the first ``execute_steps`` are executed
    before re-querying.
    """

    def __init__(self, chunk_size=20, execute_steps=10):
        self.chunk_size = chunk_size
        self.execute_steps = execute_steps
        self.action_chunk = None
        self.step_in_chunk = 0

    def reset(self):
        self.action_chunk = None
        self.step_in_chunk = 0

    def needs_query(self):
        return self.action_chunk is None or self.step_in_chunk >= self.execute_steps

    def set_chunk(self, chunk):
        """chunk: numpy array or list of length chunk_size."""
        self.action_chunk = np.array(chunk, dtype=np.float32).flatten()
        self.step_in_chunk = 0

    def get_action(self):
        action = self.action_chunk[self.step_in_chunk]
        self.step_in_chunk += 1
        return float(np.clip(action, 0.0, 1.0))

    def get_all_targets(self):
        """Return the full predicted chunk (for visualisation)."""
        if self.action_chunk is None:
            return np.array([])
        return self.action_chunk.copy()

    def current_index(self):
        return self.step_in_chunk


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def _draw_chunk_overlay(frame, chunk_targets, current_idx, execute_steps=10):
    """Draw predicted chunk positions on a rendered frame.

    - Current action: bright green circle.
    - Future actions: blue circles that fade with distance.
    - Already-executed actions: dim gray.
    """
    if len(chunk_targets) == 0:
        return frame
    frame = frame.copy()
    h, w = frame.shape[:2]
    n = len(chunk_targets)
    spacing = min(8, (w - BIRD_X - 20) // max(n, 1))

    for i, target_y in enumerate(chunk_targets):
        py = int(np.clip(target_y * SCREEN_H, 0, SCREEN_H - 1))
        px = BIRD_X + 20 + i * spacing
        if px >= w:
            break
        r = 4 if i == current_idx else 3

        if i < current_idx:
            color = np.array([100, 100, 100], dtype=np.uint8)
        elif i == current_idx:
            color = np.array([60, 255, 60], dtype=np.uint8)
        else:
            fade = max(0.2, 1.0 - (i - current_idx) / n)
            color = np.array([int(60 * fade), int(120 * fade), int(255 * fade)],
                             dtype=np.uint8)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    yy, xx = py + dy, px + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        frame[yy, xx] = color

    boundary_x = BIRD_X + 20 + execute_steps * spacing
    if boundary_x < w:
        for y in range(h):
            frame[y, boundary_x] = [255, 255, 0]

    return frame


# ---------------------------------------------------------------------------
# Evaluation (with optional video from the same rollouts)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")


@torch.no_grad()
def evaluate_policy(policy, difficulty, num_episodes, pipe_speed=3.0,
                    seed=100, use_chunks=True, video_path=None,
                    video_episodes=3, execute_steps=10):
    """Evaluate policy and return (mean, std) of episode lengths.

    If *use_chunks* is True, uses ChunkExecutor (for learned policies).
    If False, runs step-by-step (for the expert baseline).
    If *video_path* is given, the first *video_episodes* episodes are recorded.
    """
    import pygame

    recording = video_path is not None
    render_mode = "rgb_array" if recording else None

    policy.eval()
    if recording:
        pygame.init()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed,
                        render_mode=render_mode)
    if hasattr(policy, "set_env"):
        policy.set_env(env)
    executor = ChunkExecutor(execute_steps=execute_steps) if use_chunks else None
    episode_lengths = []
    all_frames = []
    episode_outcomes = []
    t_eval_start = time.time()
    t_policy_total = 0.0
    t_render_total = 0.0
    n_policy_calls = 0

    for ep in range(num_episodes):
        if recording and ep == video_episodes:
            env.close()
            pygame.quit()
            if all_frames:
                os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
                iio.imwrite(video_path, np.stack(all_frames), fps=30)
                outcomes_str = ", ".join(
                    [f"Ep{i+1}: {steps}steps ({outcome})"
                     for i, (outcome, steps) in enumerate(episode_outcomes)])
                print(f"  Saved video: {video_path}")
                print(f"    {outcomes_str}")
            recording = False
            env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed,
                                render_mode=None)
            if hasattr(policy, "set_env"):
                policy.set_env(env)

        obs, _ = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        if executor:
            executor.reset()
        done = False
        terminated = False
        truncated = False
        frames = []
        chunk_targets = np.array([])
        chunk_idx = 0

        while not done:
            if executor:
                if executor.needs_query():
                    t0 = time.time()
                    state_t = torch.tensor(obs, dtype=torch.float32,
                                           device=DEVICE).unsqueeze(0)
                    pred = policy(state_t).cpu().numpy().flatten()
                    t_policy_total += time.time() - t0
                    n_policy_calls += 1
                    executor.set_chunk(pred)
                    chunk_targets = executor.get_all_targets()
                chunk_idx = executor.current_index()
                action = executor.get_action()
            else:
                t0 = time.time()
                state_t = torch.tensor(obs, dtype=torch.float32,
                                       device=DEVICE).unsqueeze(0)
                action = float(policy(state_t).cpu().numpy().flat[0])
                t_policy_total += time.time() - t0
                n_policy_calls += 1
                chunk_targets = np.array([action])
                chunk_idx = 0

            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            if recording and ep < video_episodes:
                t0 = time.time()
                frame = env.render()
                t_render_total += time.time() - t0
                if frame is not None:
                    frame = _draw_chunk_overlay(frame, chunk_targets, chunk_idx,
                                               execute_steps=execute_steps)
                    frames.append(frame)
            done = terminated or truncated

        episode_lengths.append(env.step_count)

        if (ep + 1) % max(1, num_episodes // 5) == 0 or ep == 0:
            elapsed = time.time() - t_eval_start
            avg_so_far = np.mean(episode_lengths)
            per_call = (t_policy_total / n_policy_calls * 1000) if n_policy_calls else 0
            print(f"    [eval] ep {ep+1}/{num_episodes} "
                  f"({elapsed:.1f}s elapsed, avg_len={avg_so_far:.0f}, "
                  f"policy: {t_policy_total:.1f}s/{n_policy_calls} calls "
                  f"= {per_call:.1f}ms/call, render: {t_render_total:.1f}s)")

        if recording and ep < video_episodes and frames:
            last_frame = frames[-1].copy()
            surface = pygame.surfarray.make_surface(
                np.transpose(last_frame, (1, 0, 2)))
            font = pygame.font.SysFont(None, 48)
            font_small = pygame.font.SysFont(None, 36)

            if truncated:
                text = font.render("TIMEOUT", True, (0, 255, 0))
                bg_color = (0, 100, 0)
            else:
                text = font.render("CRASHED", True, (255, 0, 0))
                bg_color = (100, 0, 0)

            pygame.draw.rect(surface, bg_color,
                             (10, 10, text.get_width() + 20,
                              text.get_height() + 20))
            surface.blit(text, (20, 20))

            ep_text = font_small.render(
                f"Episode {ep+1}/{video_episodes} - Steps: {len(frames)}",
                True, (255, 255, 255))
            pygame.draw.rect(surface, (0, 0, 0),
                             (10, 60, ep_text.get_width() + 20,
                              ep_text.get_height() + 20))
            surface.blit(ep_text, (20, 70))

            last_frame_annotated = np.transpose(
                pygame.surfarray.array3d(surface), (1, 0, 2))
            frames[-1] = last_frame_annotated
            for _ in range(60):
                frames.append(last_frame_annotated)
            all_frames.extend(frames)
            episode_outcomes.append(
                ("TIMEOUT" if truncated else "CRASHED", len(frames) - 60))

    env.close()
    elapsed = time.time() - t_eval_start
    per_call = (t_policy_total / n_policy_calls * 1000) if n_policy_calls else 0
    print(f"    [eval done] {num_episodes} eps in {elapsed:.1f}s | "
          f"policy: {t_policy_total:.1f}s ({n_policy_calls} calls, "
          f"{per_call:.1f}ms/call) | render: {t_render_total:.1f}s")

    if recording and all_frames:
        pygame.quit()
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        iio.imwrite(video_path, np.stack(all_frames), fps=30)
        outcomes_str = ", ".join(
            [f"Ep{i+1}: {steps}steps ({outcome})"
             for i, (outcome, steps) in enumerate(episode_outcomes)])
        print(f"  Saved video: {video_path}")
        print(f"    {outcomes_str}")

    return np.mean(episode_lengths), np.std(episode_lengths)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary(results, results_dir):
    """Bar chart of all methods with error bars, grouped by environment.

    Args:
        results: dict mapping (method, difficulty) -> (mean, std).
        results_dir: directory to save the plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    difficulties = sorted({d for _, d in results})
    methods = sorted({m for m, _ in results})

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.8 / len(difficulties)

    for i, diff in enumerate(difficulties):
        means = [results.get((m, diff), (0, 0))[0] for m in methods]
        stds = [results.get((m, diff), (0, 0))[1] for m in methods]
        offset = (i - (len(difficulties) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               label=diff, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Episode length")
    ax.set_title("Method comparison")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(results_dir, "summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {path}")


def plot_dagger_iterations(dag_means, dag_stds, bc_mean, difficulty,
                           results_dir):
    """Per-iteration DAgger plot with BC baseline as horizontal line.

    Args:
        dag_means: list of mean episode lengths per DAgger round.
        dag_stds: list of std episode lengths per DAgger round.
        bc_mean: mean episode length of BC regression (horizontal line).
        difficulty: 'easy' or 'hard'.
        results_dir: directory to save the plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = np.arange(1, len(dag_means) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(rounds, dag_means, yerr=dag_stds, fmt="o-", capsize=4,
                label="DAgger")
    ax.axhline(bc_mean, color="red", linestyle="--", label="BC (regression)")
    ax.set_xlabel("DAgger round")
    ax.set_ylabel("Episode length")
    ax.set_title(f"DAgger progress — {difficulty}")
    ax.set_xticks(rounds)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(results_dir, f"dagger_iterations_{difficulty}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {path}")
