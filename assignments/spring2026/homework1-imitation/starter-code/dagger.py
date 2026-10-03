"""DAgger (Dataset Aggregation) for resolving bimodal expert ambiguity.

The key idea: roll out the current policy, but *relabel* each visited state
with a deterministic expert that always picks the upper gap (gap1). Over
rounds the relabeled data resolves the bimodal ambiguity so that a simple
BC policy can succeed on hard mode.

Structure:
    Provided (read-only):
        - DeterministicExpert class skeleton, reset, and act scaffolding.
        - run_dagger(): the full DAgger training loop.

    TODO (students implement):
        - DeterministicExpert.act gap choice (Problem 4): two small TODOs
          inside the provided scaffolding -- set raw_target to gap1_y.
        - rollout_episode (Problem 4): roll out policy for one episode,
          collect states and expert actions.
        - rollout_and_relabel (Problem 4): loop over episodes, call
          rollout_episode, relabel/window into action chunks.
"""

import numpy as np
import torch

from expert import COMMIT_DIST
from flappy_bird_env import FlappyBirdEnv
EXECUTE_STEPS = 10


class DeterministicExpert:
    """Expert that always picks gap1 (upper gap) in hard mode.

    Mimics the real Expert's behavior (hover at midpoint, commit when close,
    EMA smoothing) but replaces the random gap choice with a deterministic
    one.  This produces smooth, consistent relabeling.

    Behaviour:
        - While far from the pipe: target = midpoint of (gap1_y, gap2_y).
        - When dist < commit_dist: commit to gap1 (upper gap).
        - Apply EMA smoothing to the raw target.
        - Detect new pipes via a rounded (gap1, gap2) signature and reset
          commitment.

    Args:
        commit_dist: distance threshold to commit (default COMMIT_DIST).
        smoothing: EMA smoothing factor (default 0.15).
    """

    def __init__(self, commit_dist: float = COMMIT_DIST, smoothing: float = 0.15):
        self.commit_dist = commit_dist
        self.smoothing = smoothing
        self._last_gap_sig = None
        self._committed = False
        self._smooth_target = None

    def reset(self):
        self._last_gap_sig = None
        self._committed = False
        self._smooth_target = None

    def act(self, obs: np.ndarray) -> float:
        """Return a deterministic target y position in [0, 1].

        Args:
            obs: 4-D observation [dist_to_pipe, gap1_y, gap2_y, bird_y].

        Returns:
            Target y clipped to [0, 1].
        """
        dist = obs[0]
        gap1_y = obs[1]
        gap2_y = obs[2]

        gap_sig = (round(gap1_y, 3), round(gap2_y, 3))
        if self._last_gap_sig != gap_sig:
            self._committed = False
            self._last_gap_sig = gap_sig

        midpoint = (gap1_y + gap2_y) / 2.0

        if dist < self.commit_dist: # very close to the pipe
            self._committed = True
            # ======================================================
            # TODO (Problem 4): Set the raw target.
            # ======================================================
            raise NotImplementedError
        else:
            raw_target = float(midpoint)

        if self._smooth_target is None:
            self._smooth_target = raw_target
        else:
            self._smooth_target += self.smoothing * (
                raw_target - self._smooth_target)

        return float(np.clip(self._smooth_target, 0.0, 1.0))


@torch.no_grad()
def rollout_episode(env, policy, seed, action_chunk, device):
    """Roll out the policy for one episode, collecting states and expert actions.

    Uses a receding-horizon scheme: the policy predicts ``action_chunk``
    actions at once, but only the first ``EXECUTE_STEPS`` are used before
    re-querying.

    Args:
        env: FlappyBirdEnv instance.
        policy: trained policy (callable, takes (1, state_dim) tensor).
        seed: random seed for env.reset().
        action_chunk: prediction horizon length K.
        device: torch device for policy inference.

    Returns:
        ep_states: list of float32 observation arrays, each shape (4,).
        ep_expert_actions: list of float expert actions.

    Hints:
        - env.reset(seed=seed) returns (obs, info).
        - Maintain a ``chunk_buf`` (None initially) and ``step_in_chunk``
          counter.  When chunk_buf is None or step_in_chunk >= EXECUTE_STEPS,
          query the policy for a new chunk.
        - env.step(np.array([action])) returns (obs, reward, terminated,
          truncated, info).
    """
    # ============================================================
    # TODO (Problem 4): Implement single-episode policy rollout.
    # ============================================================
    raise NotImplementedError("TODO: Implement rollout_episode")


@torch.no_grad()
def rollout_and_relabel(policy, difficulty, num_episodes, pipe_speed, seed,
                        action_chunk, device):
    """Roll out policy for multiple episodes and relabel with expert.

    For each episode:
        1. Call rollout_episode() to collect per-step (state, expert_action).
        2. For each episode relabel the policy rollouts with the expert actions.

    Args:
        policy: trained policy (callable, takes (1, state_dim) tensor).
        difficulty: "easy" or "hard".
        num_episodes: episodes to collect.
        pipe_speed: environment pipe speed.
        seed: base random seed.
        action_chunk: prediction horizon length K.
        device: torch device for policy inference.

    Returns:
        new_states: float32 array (N, 4).
        new_actions: float32 array (N, action_chunk).

    Hints:
        - Call policy.eval().
        - Use seed + ep as the per-episode seed.
        - det_expert.act(obs) returns the expert's target y.
        - After collecting each episodes, relabel with the expert.

    """
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    det_expert = DeterministicExpert()
    new_states, new_actions = [], []


    # ============================================================
    # TODO (Problem 4): Loop over episodes, call rollout_episode,
    #   window into (state, action_chunk) pairs, return arrays.
    # ============================================================
    raise NotImplementedError("TODO: Implement rollout_and_relabel")


def run_dagger(difficulty, initial_states, initial_actions, rounds,
               episodes_per_round, epochs, pipe_speed, seed,
               action_chunk, device, train_bc_fn,
               eval_episodes=100, lr=1e-3, batch_size=2048, verbose=False,
               initial_policy=None):
    """Run DAgger training loop with automatic relabeling.

    If *initial_policy* is provided it is used for evaluation and rollout
    in the first round (skipping a redundant BC training step).  In every
    subsequent round the policy is retrained on the aggregated data.

    Args:
        difficulty: "easy" or "hard".
        initial_states: initial expert states array.
        initial_actions: initial expert action-chunk array.
        rounds: number of DAgger rounds.
        episodes_per_round: rollout episodes per round.
        epochs: BC training epochs per round.
        pipe_speed: environment pipe speed.
        seed: base random seed.
        action_chunk: prediction horizon length K.
        device: torch device.
        train_bc_fn: callable(states, actions, epochs, batch_size, lr,
                     verbose, device) -> policy.
        eval_episodes: episodes for evaluation.
        lr: learning rate for BC.
        batch_size: batch size for BC.
        verbose: print training progress.
        initial_policy: optional pre-trained BC policy for round 1.

    Returns:
        policy: final trained policy.
        performance_means: list of mean episode lengths per round.
        performance_stds: list of std episode lengths per round.
    """
    from visualization import evaluate_policy

    all_states = initial_states.copy()
    all_actions = initial_actions.copy()
    performance_means = []
    performance_stds = []
    policy = initial_policy
    best_policy_state = None
    best_mean = -float('inf')

    for rnd in range(1, rounds + 1):
        if policy is None:
            print(f"  Round {rnd}/{rounds}: Training on {len(all_states)} "
                  f"transitions...")
            policy = train_bc_fn(all_states, all_actions, epochs=epochs,
                                 verbose=verbose, batch_size=batch_size, lr=lr,
                                 device=device)
        else:
            print(f"  Round {rnd}/{rounds}: Using "
                  f"{'pretrained' if rnd == 1 else 'retrained'} "
                  f"policy ({len(all_states)} transitions in dataset)")

        avg_len, std_len = evaluate_policy(
            policy, difficulty, num_episodes=eval_episodes,
            pipe_speed=pipe_speed, seed=seed + rnd * 1000)
        performance_means.append(avg_len)
        performance_stds.append(std_len)
        print(f"    Evaluation: {avg_len:.1f} +/- {std_len:.1f} avg length "
              f"({eval_episodes} episodes)")

        if avg_len > best_mean:
            best_mean = avg_len
            best_policy_state = {k: v.clone()
                                 for k, v in policy.state_dict().items()}

        print(f"    Collecting {episodes_per_round} episodes with "
              f"auto-relabeling...", end=" ", flush=True)
        new_states, new_actions = rollout_and_relabel(
            policy, difficulty, episodes_per_round,
            pipe_speed, seed + rnd * 10000,
            action_chunk=action_chunk, device=device)
        print(f"Got {len(new_states)} new transitions")

        if len(new_states) > 0:
            all_states = np.concatenate([all_states, new_states], axis=0)
            all_actions = np.concatenate([all_actions, new_actions], axis=0)

        print(f"  Round {rnd}/{rounds}: Retraining on {len(all_states)} "
              f"transitions...")
        policy = train_bc_fn(all_states, all_actions, epochs=epochs,
                             verbose=verbose, batch_size=batch_size, lr=lr,
                             device=device)

    if best_policy_state is not None:
        policy.load_state_dict(best_policy_state)
        print(f"  Restored best policy (eval {best_mean:.1f})")

    return policy, performance_means, performance_stds
