from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEFT = 0
RIGHT = 1
DOWN = 2
UP = 3
ACTION_NAMES = {LEFT: "left", RIGHT: "right", DOWN: "down", UP: "up"}


@dataclass(frozen=True)
class Scenario:
    name: str
    goal_1_reward: float
    goal_2_reward: float
    step_reward: float
    expected_outcome: str
    episodes: int = 5000
    horizon: int = 20
    alpha: float = 0.2
    gamma: float = 0.98
    epsilon_start: float = 0.4
    epsilon_end: float = 0.02
    seed: int = 0


@dataclass(frozen=True)
class Rollout:
    states: tuple[tuple[int, int], ...]
    actions: tuple[str, ...]
    total_reward: float
    outcome: str


SCENARIOS = (
    Scenario(
        name="scenario_1",
        goal_1_reward=10.0,
        goal_2_reward=5.0,
        step_reward=-1.0,
        expected_outcome="goal_1",
    ),
    Scenario(
        name="scenario_2",
        goal_1_reward=10.0,
        goal_2_reward=5.0,
        step_reward=-2.0,
        expected_outcome="goal_2",
    ),
    Scenario(
        name="scenario_3",
        goal_1_reward=1.0,
        goal_2_reward=1.0,
        step_reward=1.0,
        expected_outcome="timeout",
    ),
)


class GridWorld:
    def __init__(self, goal_1_reward: float, goal_2_reward: float, step_reward: float):
        """Initialize the grid layout, terminal states, and reward settings."""
        self.start = (0, 0)
        self.goal_2 = (4, 0)
        self.goal_1 = (4, 3)
        self.width = 5
        self.height = 4
        self.goal_1_reward = goal_1_reward
        self.goal_2_reward = goal_2_reward
        self.step_reward = step_reward

    def step(self, state: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool]:
        """Apply an action and return the next state, reward, and terminal flag."""
        assert action in (LEFT, RIGHT, DOWN, UP), f"invalid action: {action}"
        x, y = state
        if action == LEFT:
            x = max(0, x - 1)
        if action == RIGHT:
            x = min(self.width - 1, x + 1)
        if action == DOWN:
            y = max(0, y - 1)
        if action == UP:
            y = min(self.height - 1, y + 1)
        next_state = (x, y)

        reward = self.step_reward
        done = False
        if next_state == self.goal_2:
            reward += self.goal_2_reward
            done = True
        if next_state == self.goal_1:
            reward += self.goal_1_reward
            done = True
        return next_state, reward, done


def epsilon_for_episode(scenario: Scenario, episode_idx: int) -> float:
    """Compute the epsilon value for a given episode using linear decay."""
    fraction = episode_idx / max(1, scenario.episodes - 1)
    return scenario.epsilon_start + fraction * (scenario.epsilon_end - scenario.epsilon_start)


def action_values(q_table: np.ndarray, state: tuple[int, int]) -> np.ndarray:
    """Return the Q-values for all actions at a given state."""
    x, y = state
    return q_table[y, x]


def choose_action(
    q_table: np.ndarray, state: tuple[int, int], epsilon: float, rng: np.random.Generator
) -> int:
    """Select an action with epsilon-greedy exploration."""


    ### YOUR CODE HERE ###
    pass
    ### YOUR CODE HERE ###

def greedy_action(q_table: np.ndarray, state: tuple[int, int]) -> int:
    ### YOUR CODE HERE ###
    pass
    ### YOUR CODE HERE ###



def train_q_learning(scenario: Scenario) -> tuple[np.ndarray, GridWorld]:
    """Train a Q-learning agent for one scenario and return the learned table."""
    env = GridWorld(
        goal_1_reward=scenario.goal_1_reward,
        goal_2_reward=scenario.goal_2_reward,
        step_reward=scenario.step_reward,
    )
    q_table = np.zeros((env.height, env.width, 4), dtype=np.float64)
    rng = np.random.default_rng(scenario.seed)

    for episode_idx in range(scenario.episodes):
        state = env.start
        epsilon = epsilon_for_episode(scenario, episode_idx)
        for _ in range(scenario.horizon):


            ### YOUR CODE HERE ###
            pass
            ### YOUR CODE HERE ###


    return q_table, env


def rollout_policy(q_table: np.ndarray, env: GridWorld, horizon: int) -> Rollout:
    """Follow the greedy policy from the start state to collect one rollout."""
    state = env.start
    states = [state]
    actions = []
    total_reward = 0.0
    outcome = "timeout"

    for _ in range(horizon):
        action = greedy_action(q_table, state)
        next_state, reward, done = env.step(state, action)
        actions.append(ACTION_NAMES[action])
        total_reward += reward
        states.append(next_state)
        state = next_state
        if done:
            if state == env.goal_1:
                outcome = "goal_1"
            if state == env.goal_2:
                outcome = "goal_2"
            break

    return Rollout(
        states=tuple(states),
        actions=tuple(actions),
        total_reward=total_reward,
        outcome=outcome,
    )


def summarize_scenario(scenario: Scenario) -> dict[str, object]:
    """Run training and rollout, then package the scenario results."""
    q_table, env = train_q_learning(scenario)
    rollout = rollout_policy(q_table, env, scenario.horizon)
    return {
        "scenario": scenario.name,
        "expected_outcome": scenario.expected_outcome,
        "observed_outcome": rollout.outcome,
        "trajectory": rollout.states,
        "actions": rollout.actions,
        "total_reward": rollout.total_reward,
        "start_q_values": tuple(np.round(action_values(q_table, env.start), 3)),
    }


def run_all_scenarios() -> list[dict[str, object]]:
    """Evaluate all predefined scenarios and return their summaries."""
    return [summarize_scenario(scenario) for scenario in SCENARIOS]


def main() -> None:
    """Print the summary for each scenario."""
    for result in run_all_scenarios():
        print(result["scenario"])
        print(f"  expected: {result['expected_outcome']}")
        print(f"  observed: {result['observed_outcome']}")
        print(f"  trajectory: {result['trajectory']}")
        print(f"  actions: {result['actions']}")
        print(f"  total_reward: {result['total_reward']:.2f}")
        print(f"  start_q_values: {result['start_q_values']}")


if __name__ == "__main__":
    main()
