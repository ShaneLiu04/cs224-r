"""Return of the starter's CQL data collector (greedy on shortest-path distance) per maze.

Run from homework3-offline-rl/code with PYTHONPATH=. so that cs224r.envs registers the mazes.
"""
import json

import gym
import numpy as np

from cs224r.infrastructure.dqn_utils import register_custom_envs

EP_LEN = {"PointmassEasy-v0": 50, "PointmassMedium-v0": 150, "PointmassHard-v0": 100}


def rollout(env, ep_len, eps, rng):
    env.reset()
    ret, success, cells = 0.0, False, set()
    for _ in range(ep_len):
        action = env.get_optimal_action(env.state)
        if rng.random() < eps:
            action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        cells.add(tuple(np.round(np.asarray(env.state), 0).astype(int).tolist()))
        ret += reward
        if done:
            success = reward >= 0
            break
    return ret, success, len(cells)


def main():
    register_custom_envs()
    rng = np.random.default_rng(0)
    out = {}
    for name, ep_len in EP_LEN.items():
        env = gym.make(name)
        env.unwrapped.set_logdir(f"/tmp/oracle_{name}_")
        for eps in (0.0, 0.2):
            stats = [rollout(env, ep_len, eps, rng) for _ in range(20)]
            out[f"{name}/eps{eps}"] = {
                "return": float(np.mean([s[0] for s in stats])),
                "success": float(np.mean([s[1] for s in stats])),
                "distinct_cells": float(np.mean([s[2] for s in stats])),
            }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
