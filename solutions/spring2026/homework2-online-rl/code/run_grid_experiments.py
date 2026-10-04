"""Run reproducible Gridworld Q-learning sweeps for the HW2 report."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from gridworld_q_learning import SCENARIOS, summarize_scenario


def run_scenario(scenario):
    """Run one scenario and convert NumPy values to JSON-compatible values."""
    result = summarize_scenario(scenario)
    return {
        "scenario": result["scenario"],
        "seed": scenario.seed,
        "episodes": scenario.episodes,
        "alpha": scenario.alpha,
        "gamma": scenario.gamma,
        "expected_outcome": result["expected_outcome"],
        "observed_outcome": result["observed_outcome"],
        "trajectory": [list(state) for state in result["trajectory"]],
        "actions": list(result["actions"]),
        "total_reward": float(result["total_reward"]),
        "start_q_values": [
            float(value) for value in result["start_q_values"]
        ],
    }


def aggregate(records):
    """Aggregate expected-outcome rate and return statistics."""
    rewards = np.asarray(
        [record["total_reward"] for record in records], dtype=np.float64,
    )
    correct = np.asarray([
        record["observed_outcome"] == record["expected_outcome"]
        for record in records
    ])
    return {
        "expected_outcome_rate": float(correct.mean()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "n": len(records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()

    results = {
        "scenario_definitions": [asdict(scenario) for scenario in SCENARIOS],
        "official_runs": [
            run_scenario(scenario) for scenario in SCENARIOS
        ],
        "sample_efficiency": {},
        "alpha_ablation": {},
        "gamma_ablation": {},
    }

    episode_budgets = (50, 100, 250, 500, 1000, 2000, 5000)
    for scenario in SCENARIOS:
        scenario_results = {}
        for episodes in episode_budgets:
            records = [
                run_scenario(
                    replace(scenario, seed=seed, episodes=episodes)
                )
                for seed in range(args.seeds)
            ]
            scenario_results[str(episodes)] = aggregate(records)
        results["sample_efficiency"][scenario.name] = scenario_results

    for alpha in (0.05, 0.1, 0.2, 0.5):
        records = {}
        for scenario in SCENARIOS:
            runs = [
                run_scenario(
                    replace(
                        scenario, seed=seed, episodes=1000, alpha=alpha,
                    )
                )
                for seed in range(args.seeds)
            ]
            records[scenario.name] = aggregate(runs)
        results["alpha_ablation"][str(alpha)] = records

    for gamma in (0.9, 0.95, 0.98, 0.999):
        records = {}
        for scenario in SCENARIOS:
            runs = [
                run_scenario(
                    replace(
                        scenario, seed=seed, episodes=1000, gamma=gamma,
                    )
                )
                for seed in range(args.seeds)
            ]
            records[scenario.name] = aggregate(runs)
        results["gamma_ablation"][str(gamma)] = records

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved Gridworld experiments to {args.output}")


if __name__ == "__main__":
    main()
