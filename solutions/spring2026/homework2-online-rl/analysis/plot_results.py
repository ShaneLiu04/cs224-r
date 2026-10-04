"""Aggregate Homework 2 metrics and produce publication-quality figures."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


COLORS = {
    "ppo": "#4472C4",
    "off_base": "#ED7D31",
    "off_utd5": "#70AD47",
}
LABELS = {
    "ppo": "PPO + GAE",
    "off_base": "Actor-Critic (2 critics, UTD=1)",
    "off_utd5": "Actor-Critic (10 critics, UTD=5)",
}


def configure_style():
    """Configure a consistent print-friendly style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_figure(figure, output_dir, name):
    """Save one figure as both PNG and PDF."""
    figure.savefig(output_dir / f"{name}.png")
    figure.savefig(output_dir / f"{name}.pdf")
    plt.close(figure)


def load_eval_runs(runs_dir):
    """Load every valid evaluation CSV and infer its experiment group."""
    groups = {"ppo": [], "off_base": [], "off_utd5": []}
    extra_groups = {
        "off_utd5_c2": [],
        "off_utd1_c10": [],
        "off_nobc": [],
        "ppo_nobc": [],
    }
    fast_ppo = []
    base_v2 = []
    gae_groups = {}
    for path in sorted(runs_dir.glob("*/eval.csv")):
        name = path.parent.name
        frame = pd.read_csv(path)
        frame = frame.rename(columns={
            "episode_success": "eval/episode_success",
            "eval_total_time": "eval/eval_total_time",
        })
        if "eval/episode_success" not in frame.columns:
            continue
        frame = frame.sort_values("frame").drop_duplicates("frame")
        record = {"name": name, "path": path, "data": frame}
        if name.startswith("ppo_fast_seed"):
            fast_ppo.append(record)
        elif name.startswith("ppo_seed"):
            groups["ppo"].append(record)
        elif name.startswith("off_base_v2"):
            base_v2.append(record)
        elif name.startswith("off_base"):
            groups["off_base"].append(record)
        elif name.startswith("off_utd5_c2"):
            extra_groups["off_utd5_c2"].append(record)
        elif name.startswith("off_utd1_c10"):
            extra_groups["off_utd1_c10"].append(record)
        elif name.startswith("off_nobc"):
            extra_groups["off_nobc"].append(record)
        elif name.startswith("ppo_nobc"):
            extra_groups["ppo_nobc"].append(record)
        elif name.startswith("off_utd5"):
            groups["off_utd5"].append(record)
        else:
            match = re.match(r"ppo_gae([0-9p]+)_seed", name)
            if match:
                value = float(match.group(1).replace("p", "."))
                gae_groups.setdefault(value, []).append(record)
    if fast_ppo:
        groups["ppo"] = fast_ppo
    if base_v2:
        groups["off_base"] = base_v2
    return groups, gae_groups, extra_groups


def interpolate_group(records, max_frame=None, points=300):
    """Interpolate runs onto a shared frame grid."""
    if not records:
        return np.array([]), np.empty((0, 0))
    if max_frame is None:
        max_frame = min(record["data"]["frame"].max() for record in records)
    grid = np.linspace(0, max_frame, points)
    curves = []
    for record in records:
        data = record["data"]
        curves.append(np.interp(
            grid,
            data["frame"].to_numpy(),
            data["eval/episode_success"].to_numpy(),
        ))
    return grid, np.asarray(curves)


def run_statistics(records, threshold):
    """Compute best, final, AUC and threshold-crossing metrics."""
    rows = []
    for record in records:
        data = record["data"]
        frames = data["frame"].to_numpy(dtype=float)
        success = data["eval/episode_success"].to_numpy(dtype=float)
        reached = frames[success >= threshold]
        rows.append({
            "run": record["name"],
            "best_success": float(success.max()),
            "final_success": float(success[-1]),
            "auc": float(
                np.trapezoid(success, frames) / max(frames[-1], 1.0)
            ),
            "threshold_frame": (
                float(reached[0]) if len(reached) else None
            ),
            "wall_seconds": float(
                data["eval/eval_total_time"].iloc[-1]
                if "eval/eval_total_time" in data.columns else np.nan
            ),
        })
    return rows


def plot_main_curves(groups, output_dir):
    """Plot sample-efficiency curves for the three required methods."""
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    for key in ("ppo", "off_base", "off_utd5"):
        grid, curves = interpolate_group(groups[key])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1) if len(curves) > 1 else 0.0
        axis.plot(grid, mean, color=COLORS[key], label=LABELS[key])
        axis.fill_between(
            grid, np.maximum(mean - std, 0), np.minimum(mean + std, 1),
            color=COLORS[key], alpha=0.16,
        )
    axis.axhline(0.25, color="gray", linestyle=":", linewidth=1)
    axis.axhline(0.90, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Environment frames",
        ylabel="Evaluation success rate",
        title="Online RL Sample Efficiency on Meta-World Hammer",
        ylim=(-0.02, 1.03),
    )
    axis.legend(frameon=False, loc="lower right")
    save_figure(figure, output_dir, "main_learning_curves")


def plot_off_policy_zoom(groups, output_dir):
    """Compare standard and high-UTD actor-critic within 100k frames."""
    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    for key in ("off_base", "off_utd5"):
        grid, curves = interpolate_group(groups[key])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1) if len(curves) > 1 else 0.0
        axis.plot(grid, mean, color=COLORS[key], label=LABELS[key])
        axis.fill_between(
            grid, np.maximum(mean - std, 0), np.minimum(mean + std, 1),
            color=COLORS[key], alpha=0.18,
        )
    axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Environment frames",
        ylabel="Evaluation success rate",
        title="Critic Ensemble and Update-to-Data Ablation",
        ylim=(-0.02, 1.03),
    )
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "off_policy_utd_ablation")


def plot_final_distribution(groups, output_dir):
    """Plot best and final success distributions across seeds."""
    keys = ("ppo", "off_base", "off_utd5")
    best, final = [], []
    for key in keys:
        stats = run_statistics(
            groups[key], 0.25 if key == "ppo" else 0.9,
        )
        best.append([row["best_success"] for row in stats])
        final.append([row["final_success"] for row in stats])
    positions = np.arange(len(keys))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.4, 4.3))
    axis.bar(
        positions - width / 2,
        [np.mean(values) for values in best],
        width,
        yerr=[np.std(values, ddof=1) for values in best],
        capsize=4,
        label="Best",
        color="#5B9BD5",
    )
    axis.bar(
        positions + width / 2,
        [np.mean(values) for values in final],
        width,
        yerr=[np.std(values, ddof=1) for values in final],
        capsize=4,
        label="Final",
        color="#A5A5A5",
    )
    axis.set_xticks(positions, [LABELS[key] for key in keys], rotation=8)
    axis.set(
        ylabel="Success rate",
        title="Across-Seed Reliability",
        ylim=(0, 1.08),
    )
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "success_distribution")


def plot_gae_ablation(gae_groups, output_dir):
    """Plot short-budget PPO sensitivity to GAE lambda."""
    values = sorted(gae_groups)
    auc_means, auc_stds, best_means, best_stds = [], [], [], []
    for value in values:
        stats = run_statistics(gae_groups[value], threshold=0.25)
        auc = [row["auc"] for row in stats]
        best = [row["best_success"] for row in stats]
        auc_means.append(np.mean(auc))
        auc_stds.append(np.std(auc, ddof=1) if len(auc) > 1 else 0)
        best_means.append(np.mean(best))
        best_stds.append(np.std(best, ddof=1) if len(best) > 1 else 0)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].errorbar(
        values, auc_means, yerr=auc_stds, marker="o", capsize=4,
        color=COLORS["ppo"],
    )
    axes[0].set(
        xlabel="GAE lambda", ylabel="Normalized success AUC",
        title="Learning Efficiency",
    )
    axes[1].errorbar(
        values, best_means, yerr=best_stds, marker="o", capsize=4,
        color="#8064A2",
    )
    axes[1].set(
        xlabel="GAE lambda", ylabel="Best success rate",
        title="Best Performance",
        ylim=(-0.02, 1.03),
    )
    figure.suptitle("PPO Bias–Variance Sensitivity", y=1.03)
    figure.tight_layout()
    save_figure(figure, output_dir, "gae_lambda_ablation")


def plot_grid_trajectories(grid_metrics, output_dir):
    """Visualize the learned greedy trajectory in each reward scenario."""
    runs = grid_metrics["official_runs"]
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    for axis, run in zip(axes, runs):
        trajectory = np.asarray(run["trajectory"])
        axis.set_xticks(range(5))
        axis.set_yticks(range(4))
        axis.grid(True, color="#D9D9D9")
        axis.plot(
            trajectory[:, 0], trajectory[:, 1],
            marker="o", color="#4472C4",
        )
        axis.scatter([0], [0], color="black", s=55, label="start", zorder=3)
        axis.scatter([4], [0], color="#ED7D31", s=65, label="goal 2")
        axis.scatter([4], [3], color="#70AD47", s=65, label="goal 1")
        axis.set(
            xlim=(-0.2, 4.2), ylim=(-0.2, 3.2),
            title=(
                f"{run['scenario']}: {run['observed_outcome']}\n"
                f"return={run['total_reward']:.1f}"
            ),
            aspect="equal",
        )
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle("Greedy Policies under Three Reward Designs", y=1.04)
    figure.tight_layout()
    save_figure(figure, output_dir, "gridworld_trajectories")


def plot_grid_sample_efficiency(grid_metrics, output_dir):
    """Plot probability of learning the expected outcome versus episodes."""
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    for index, (scenario, records) in enumerate(
        grid_metrics["sample_efficiency"].items()
    ):
        episodes = np.asarray([int(value) for value in records])
        order = np.argsort(episodes)
        rates = np.asarray([
            records[str(value)]["expected_outcome_rate"]
            for value in episodes
        ])
        axis.plot(
            episodes[order], rates[order], marker="o",
            label=scenario, color=list(COLORS.values())[index],
        )
    axis.set_xscale("log")
    axis.set(
        xlabel="Training episodes (log scale)",
        ylabel="Expected-outcome rate over seeds",
        title="Tabular Q-Learning Sample Efficiency",
        ylim=(-0.02, 1.03),
    )
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "gridworld_sample_efficiency")


def best_until(records, max_frame):
    """Return each run's best success up to a fixed frame budget."""
    values = []
    for record in records:
        data = record["data"]
        subset = data[data["frame"] <= max_frame]
        values.append(float(subset["eval/episode_success"].max()))
    return values


def plot_factorial_ablation(groups, extra_groups, output_dir):
    """Disentangle critic-count and UTD effects at a common frame budget."""
    configurations = [
        ("2 critics\nUTD=1", groups["off_base"]),
        ("10 critics\nUTD=1", extra_groups["off_utd1_c10"]),
        ("2 critics\nUTD=5", extra_groups["off_utd5_c2"]),
        ("10 critics\nUTD=5", groups["off_utd5"]),
    ]
    values = [best_until(records, 40000) for _, records in configurations]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(len(configurations))
    axis.bar(
        positions,
        [np.mean(item) for item in values],
        yerr=[
            np.std(item, ddof=1) if len(item) > 1 else 0
            for item in values
        ],
        capsize=4,
        color=["#A5A5A5", "#5B9BD5", "#FFC000", "#70AD47"],
    )
    axis.set_xticks(
        positions, [label for label, _ in configurations],
    )
    axis.set(
        ylabel="Best success by 40k frames",
        title="Factorial Critic-Ensemble × UTD Ablation",
        ylim=(0, 1.08),
    )
    save_figure(figure, output_dir, "critic_utd_factorial")


def plot_bc_ablation(groups, extra_groups, output_dir):
    """Compare online RL with and without demonstration initialization."""
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    comparisons = [
        (
            axes[0], groups["ppo"], extra_groups["ppo_nobc"],
            200000, "PPO",
        ),
        (
            axes[1], groups["off_base"], extra_groups["off_nobc"],
            40000, "Off-policy Actor-Critic",
        ),
    ]
    for axis, with_bc, without_bc, budget, title in comparisons:
        for records, label, color in (
            (with_bc, "With demonstrations", "#4472C4"),
            (without_bc, "Without demonstrations", "#C00000"),
        ):
            grid, curves = interpolate_group(
                records, max_frame=budget, points=200,
            )
            mean = curves.mean(axis=0)
            std = curves.std(axis=0, ddof=1) if len(curves) > 1 else 0.0
            axis.plot(grid, mean, label=label, color=color)
            axis.fill_between(
                grid, np.maximum(mean - std, 0), np.minimum(mean + std, 1),
                color=color, alpha=0.15,
            )
        axis.set(
            xlabel="Environment frames",
            ylabel="Success rate",
            title=title,
            ylim=(-0.02, 1.03),
        )
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.ticklabel_format(
            style="sci", axis="x", scilimits=(0, 0),
        )
    axes[1].legend(frameon=False)
    figure.suptitle("Demonstration Initialization Ablation", y=1.03)
    figure.tight_layout()
    save_figure(figure, output_dir, "demonstration_ablation")


def plot_resource_usage(runs_dir, output_dir):
    """Plot measured GPU utilization and memory over wall-clock samples."""
    path = runs_dir / "resource_usage.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    x = np.arange(len(frame)) * 0.5
    axes[0].plot(x, frame["gpu_util_pct"], color="#4472C4")
    axes[0].set(
        xlabel="Wall time (minutes)",
        ylabel="GPU utilization (%)",
        title="GPU Utilization",
        ylim=(0, 105),
    )
    axes[1].plot(x, frame["memory_used_mib"] / 1024, color="#70AD47")
    axes[1].set(
        xlabel="Wall time (minutes)",
        ylabel="GPU memory (GiB)",
        title="GPU Memory",
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "resource_utilization")


def write_summaries(groups, gae_groups, output_dir, macros_path):
    """Write tidy CSV/JSON summaries and LaTeX macros."""
    all_rows = []
    summary = {}
    for key, records in groups.items():
        threshold = 0.25 if key == "ppo" else 0.9
        rows = run_statistics(records, threshold)
        for row in rows:
            row["method"] = key
        all_rows.extend(rows)
        crossings = [
            row["threshold_frame"] for row in rows
            if row["threshold_frame"] is not None
        ]
        summary[key] = {
            "seeds": len(rows),
            "best_mean": float(np.mean([
                row["best_success"] for row in rows
            ])),
            "best_std": float(np.std([
                row["best_success"] for row in rows
            ], ddof=1)) if len(rows) > 1 else 0.0,
            "final_mean": float(np.mean([
                row["final_success"] for row in rows
            ])),
            "threshold_reached": len(crossings),
            "threshold_frame_median": (
                float(np.median(crossings)) if crossings else None
            ),
        }
    pd.DataFrame(all_rows).to_csv(
        output_dir / "summary_per_seed.csv", index=False,
    )
    summary["gae_ablation"] = {
        str(value): run_statistics(records, 0.25)
        for value, records in gae_groups.items()
    }
    (output_dir / "summary_aggregate.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    command_names = {
        "ppo": "PPO",
        "off_base": "OffBase",
        "off_utd5": "OffUTD",
    }
    lines = []
    for key, prefix in command_names.items():
        value = summary[key]
        lines.extend([
            f"\\renewcommand{{\\{prefix}Best}}"
            f"{{{100 * value['best_mean']:.1f}\\%}}",
            f"\\renewcommand{{\\{prefix}Final}}"
            f"{{{100 * value['final_mean']:.1f}\\%}}",
            f"\\renewcommand{{\\{prefix}Seeds}}{{{value['seeds']}}}",
        ])
    macros_path.parent.mkdir(parents=True, exist_ok=True)
    macros_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_submission_csv(groups, output_dir):
    """Export the three learning curves requested by the handout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "ppo": "on_policy.csv",
        "off_base": "off_policy_2critics_utd1.csv",
        "off_utd5": "off_policy_10critics_utd5.csv",
    }
    for key, records in groups.items():
        rows = []
        for record in records:
            match = re.search(r"seed(\d+)", record["name"])
            seed = int(match.group(1)) if match else -1
            for _, row in record["data"].iterrows():
                rows.append({
                    "seed": seed,
                    "frame": int(row["frame"]),
                    "episode_success": float(
                        row["eval/episode_success"]
                    ),
                })
        with (output_dir / filenames[key]).open(
            "w", newline="", encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["seed", "frame", "episode_success"],
            )
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--macros", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_style()

    groups, gae_groups, extra_groups = load_eval_runs(args.runs)
    for key, records in groups.items():
        if not records:
            raise RuntimeError(f"No completed runs found for {key}")
    grid_metrics = json.loads(args.grid.read_text(encoding="utf-8"))

    plot_main_curves(groups, args.output)
    plot_off_policy_zoom(groups, args.output)
    plot_final_distribution(groups, args.output)
    plot_gae_ablation(gae_groups, args.output)
    plot_grid_trajectories(grid_metrics, args.output)
    plot_grid_sample_efficiency(grid_metrics, args.output)
    plot_factorial_ablation(groups, extra_groups, args.output)
    plot_bc_ablation(groups, extra_groups, args.output)
    plot_resource_usage(args.runs, args.output)
    write_summaries(groups, gae_groups, args.output, args.macros)
    write_submission_csv(groups, args.submission)
    print("Homework 2 report assets generated")


if __name__ == "__main__":
    main()
