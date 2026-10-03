"""Expert policy and data collection for Flappy Bird.

The expert outputs a target y position (normalised 0-1) for each timestep.
The environment's internal PD controller converts the target into thrust.

This file is fully provided -- do not modify.
"""

import numpy as np
from flappy_bird_env import FlappyBirdEnv

COMMIT_DIST = 0.18  # normalised pipe distance at which the expert picks a gap


class Expert:
    """Expert that outputs target y positions (normalised 0-1).

    Easy mode: target the single gap centre (gap1_y).

    Hard mode: hover at the midpoint of both gaps while far away.  When the
    bird gets within ``commit_dist`` of the pipe, randomly pick one of the
    two gaps and target it for the remainder of that pipe.  Reset commitment
    when a new pipe appears (detected by a change in gap positions).

    Actions are temporally smoothed with an EMA:
        smooth_target += smoothing * (raw_target - smooth_target)

    Observation layout:
        obs[0] = dist_to_pipe  (normalised)
        obs[1] = gap1_y        (normalised)
        obs[2] = gap2_y        (normalised)
        obs[3] = bird_y        (normalised)

    Args:
        commit_dist: distance threshold to commit to a gap (default 0.18).
        smoothing: EMA smoothing factor (default 0.15).
    """

    def __init__(self, commit_dist: float = COMMIT_DIST, smoothing: float = 0.15):
        self.commit_dist = commit_dist
        self.smoothing = smoothing
        self.target_gap_idx = None
        self._last_gap_sig = None
        self._committed = False
        self._smooth_target = None

    def reset(self):
        self.target_gap_idx = None
        self._last_gap_sig = None
        self._committed = False
        self._smooth_target = None

    def act(self, obs: np.ndarray, difficulty: str) -> float:
        """Return target y position in [0, 1].

        Args:
            obs: 4-D observation [dist_to_pipe, gap1_y, gap2_y, bird_y].
            difficulty: "easy" or "hard".

        Returns:
            Target y position clipped to [0, 1].
        """
        dist = obs[0]
        gap1_y = obs[1]
        gap2_y = obs[2]

        if difficulty == "easy":
            raw_target = float(gap1_y)
        else:
            gap_sig = (round(gap1_y, 3), round(gap2_y, 3))
            if self._last_gap_sig != gap_sig:
                self._committed = False
                self.target_gap_idx = None
                self._last_gap_sig = gap_sig

            midpoint = (gap1_y + gap2_y) / 2.0

            if not self._committed:
                if dist < self.commit_dist:
                    self.target_gap_idx = np.random.choice([0, 1])
                    self._committed = True
                else:
                    raw_target = float(midpoint)

            if self._committed:
                raw_target = float(
                    gap1_y if self.target_gap_idx == 0 else gap2_y)

        if self._smooth_target is None:
            self._smooth_target = raw_target
        else:
            self._smooth_target += self.smoothing * (
                raw_target - self._smooth_target)

        return float(np.clip(self._smooth_target, 0.0, 1.0))


def collect_expert_data(difficulty, num_episodes, action_chunk,
                        pipe_speed=3.0, seed=0):
    """Collect expert demos step-by-step, then window into action chunks.

    Training pairs are (s_t, [a_t, a_{t+1}, ..., a_{t+K-1}]) where
    K = action_chunk.

    Args:
        difficulty: "easy" or "hard".
        num_episodes: number of episodes to collect.
        action_chunk: prediction horizon length K.
        pipe_speed: environment pipe speed.
        seed: base random seed.

    Returns:
        states: float32 array of shape (N, 4).
        actions: float32 array of shape (N, action_chunk).
    """
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    expert = Expert()
    all_states, all_actions = [], []
    all_steps = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        done = False
        ep_states, ep_actions = [], []
        while not done:
            action = expert.act(obs, difficulty)
            ep_states.append(obs.copy())
            ep_actions.append(action)
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated
        all_steps.append(len(ep_states))
        for i in range(len(ep_states) - action_chunk + 1):
            all_states.append(ep_states[i])
            all_actions.append(ep_actions[i:i + action_chunk])
    print(f"Average steps: {np.mean(all_steps):.1f}")
    print(f"Chunk={action_chunk}: {len(all_states)} chunk pairs "
          f"from {num_episodes} episodes")
    env.close()
    return (np.array(all_states, dtype=np.float32),
            np.array(all_actions, dtype=np.float32))
