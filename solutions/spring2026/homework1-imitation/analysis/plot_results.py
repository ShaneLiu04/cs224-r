"""Aggregate raw Assignment 1 metrics and generate publication figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "bc_reg_easy": "#4472C4",
    "bc_reg_hard": "#ED7D31",
    "bc_flow_hard": "#70AD47",
    "dagger_hard": "#A64D79",
}
LABELS = {
    "bc_reg_easy": "BC (easy)",
    "bc_reg_hard": "BC (hard)",
    "bc_flow_hard": "Flow Matching (hard)",
    "dagger_hard": "DAgger (hard)",
}


def configure_style():
    """Set a consistent, print-friendly plotting style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_runs(input_dir):
    """Load all completed seed metric files."""
    paths = sorted(input_dir.glob("seed_*/metrics_seed_*.json"))
    if not paths:
        paths = sorted(input_dir.glob("metrics_seed_*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def save_figure(figure, output_dir, name):
    """Save one figure as PNG and PDF."""
    figure.savefig(output_dir / f"{name}.png")
    figure.savefig(output_dir / f"{name}.pdf")
    plt.close(figure)


def write_summary(runs, output_dir):
    """Write tidy per-seed and aggregate metrics."""
    rows = []
    for run in runs:
        for method, result in run["methods"].items():
            rows.append({
                "seed": run["seed"],
                "method": method,
                "mean": result["mean"],
                "std": result["std"],
                "median": result["median"],
                "success_rate": result["success_rate"],
                "latency_ms": result["latency_ms"],
                "train_seconds": result.get("train_seconds", ""),
            })
    with (output_dir / "summary_per_seed.csv").open(
        "w", newline="", encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    for method in LABELS:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            "mean": float(np.mean([row["mean"] for row in method_rows])),
            "between_seed_std": float(
                np.std([row["mean"] for row in method_rows], ddof=1)
                if len(method_rows) > 1 else 0.0
            ),
            "success_rate": float(
                np.mean([row["success_rate"] for row in method_rows])
            ),
            "seeds": len(method_rows),
        }
    (output_dir / "summary_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8",
    )
    return rows, aggregate


def write_latex_macros(runs, aggregate, path):
    """Write experiment values consumed by both LaTeX reports."""
    names = {
        "bc_reg_easy": "BCEasy",
        "bc_reg_hard": "BCHard",
        "bc_flow_hard": "FlowHard",
        "dagger_hard": "DaggerHard",
    }
    lines = []
    for method, prefix in names.items():
        lines.append(
            f"\\renewcommand{{\\{prefix}Mean}}"
            f"{{{aggregate[method]['mean']:.1f}}}"
        )
        lines.append(
            f"\\renewcommand{{\\{prefix}Std}}"
            f"{{{aggregate[method]['between_seed_std']:.1f}}}"
        )
    gpu = runs[0]["metadata"].get("gpu") or "CPU"
    lines.extend([
        f"\\renewcommand{{\\NumSeeds}}{{{len(runs)}}}",
        f"\\renewcommand{{\\GPUName}}{{{gpu}}}",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_submission_summaries(runs, aggregate, output_dir):
    """Write the four text result files requested by the handout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("bc_reg_easy", "bc_reg_easy.txt", "Behavior Cloning (MSE)", "easy"),
        ("bc_reg_hard", "bc_reg_hard.txt", "Behavior Cloning (MSE)", "hard"),
        (
            "bc_flow_hard", "bc_flow_hard.txt",
            "Conditional Flow Matching", "hard",
        ),
        ("dagger_hard", "dagger_hard.txt", "DAgger", "hard"),
    )
    for method, filename, label, environment in specifications:
        result = aggregate[method]
        lines = [
            f"Method: {label}",
            f"Environment: {environment}",
            "Metric: episode length over 50 evaluation episodes per seed",
            f"Seeds: {', '.join(str(run['seed']) for run in runs)}",
            (
                "Aggregate: "
                f"{result['mean']:.1f} +/- "
                f"{result['between_seed_std']:.1f} "
                "(mean +/- between-seed SD)"
            ),
            f"Success rate: {100 * result['success_rate']:.1f}%",
            "",
            "Per-seed results:",
        ]
        for run in runs:
            value = run["methods"][method]
            lines.append(
                f"  seed {run['seed']:>2}: "
                f"{value['mean']:.1f} +/- {value['std']:.1f}, "
                f"success={100 * value['success_rate']:.1f}%"
            )
        if method == "dagger_hard":
            lines.extend(["", "DAgger per-round means:"])
            for run in runs:
                values = ", ".join(
                    f"{value:.1f}"
                    for value in run["methods"][method]["round_means"]
                )
                lines.append(f"  seed {run['seed']:>2}: {values}")
        (output_dir / filename).write_text(
            "\n".join(lines) + "\n", encoding="utf-8",
        )


def plot_method_comparison(aggregate, output_dir):
    """Plot the primary comparison with between-seed error bars."""
    methods = list(LABELS)
    means = [aggregate[method]["mean"] for method in methods]
    errors = [aggregate[method]["between_seed_std"] for method in methods]
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    bars = axis.bar(
        range(len(methods)), means, yerr=errors, capsize=5,
        color=[COLORS[method] for method in methods],
    )
    axis.set_xticks(range(len(methods)), [LABELS[method] for method in methods])
    axis.set_ylabel("Episode length (steps)")
    axis.set_title("Imitation Learning Performance")
    axis.set_ylim(0, 1080)
    axis.axhline(1000, color="black", linestyle="--", linewidth=1)
    axis.text(3.45, 1015, "success horizon", ha="right", fontsize=8)
    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2, value + 25,
            f"{value:.1f}", ha="center", fontsize=9,
        )
    save_figure(figure, output_dir, "method_comparison")


def plot_dagger_curve(runs, output_dir):
    """Plot DAgger learning dynamics across random seeds."""
    values = np.array([
        run["methods"]["dagger_hard"]["round_means"] for run in runs
    ])
    rounds = np.arange(1, values.shape[1] + 1)
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    baseline = np.mean([
        run["methods"]["bc_reg_hard"]["mean"] for run in runs
    ])
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(rounds, mean, marker="o", color=COLORS["dagger_hard"])
    axis.fill_between(
        rounds, mean - std, mean + std,
        color=COLORS["dagger_hard"], alpha=0.18, label="across-seed SD",
    )
    axis.axhline(
        baseline, color=COLORS["bc_reg_hard"], linestyle="--",
        label="BC hard baseline",
    )
    axis.set(
        xlabel="DAgger round", ylabel="Episode length (steps)",
        title="Dataset Aggregation Learning Curve",
        xticks=rounds,
    )
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "dagger_learning_curve")


def plot_training_curves(runs, output_dir):
    """Plot loss convergence for all trainable baselines."""
    methods = ("bc_reg_easy", "bc_reg_hard", "bc_flow_hard")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, method in zip(axes, methods):
        histories = np.array([
            run["methods"][method]["loss"] for run in runs
        ])
        epochs = np.arange(1, histories.shape[1] + 1)
        mean = histories.mean(axis=0)
        std = histories.std(axis=0, ddof=1)
        axis.plot(epochs, mean, color=COLORS[method])
        axis.fill_between(
            epochs, np.maximum(mean - std, 1e-8), mean + std,
            color=COLORS[method], alpha=0.18,
        )
        axis.set_yscale("log")
        axis.set_title(LABELS[method])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Training loss")
    figure.suptitle("Optimization Convergence", y=1.03)
    figure.tight_layout()
    save_figure(figure, output_dir, "training_curves")


def plot_flow_ablation(runs, output_dir):
    """Plot quality-latency trade-off for Euler integration steps."""
    steps = [5, 10, 20, 40]
    means = np.array([
        [
            run["ablations"]["flow_integration_steps"][str(step)]["mean"]
            for step in steps
        ]
        for run in runs
    ])
    latency = np.array([
        [
            run["ablations"]["flow_integration_steps"][str(step)][
                "latency_ms"
            ]
            for step in steps
        ]
        for run in runs
    ])
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].errorbar(
        steps, means.mean(axis=0), yerr=means.std(axis=0, ddof=1),
        marker="o", capsize=4, color=COLORS["bc_flow_hard"],
    )
    axes[0].set(
        xlabel="Euler integration steps",
        ylabel="Episode length",
        title="Control Quality",
        xticks=steps,
    )
    axes[1].errorbar(
        steps, latency.mean(axis=0), yerr=latency.std(axis=0, ddof=1),
        marker="o", capsize=4, color="#5B9BD5",
    )
    axes[1].set(
        xlabel="Euler integration steps",
        ylabel="Latency per policy call (ms)",
        title="Inference Cost",
        xticks=steps,
    )
    figure.suptitle("Flow Matching Quality–Latency Trade-off", y=1.03)
    figure.tight_layout()
    save_figure(figure, output_dir, "flow_steps_ablation")


def plot_execute_ablation(runs, output_dir):
    """Plot receding-horizon execution-step sensitivity."""
    steps = [1, 5, 10, 20]
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    for key, label, color in (
        ("bc_easy", "BC easy", COLORS["bc_reg_easy"]),
        ("dagger_hard", "DAgger hard", COLORS["dagger_hard"]),
    ):
        values = np.array([
            [
                run["ablations"]["execute_steps"][str(step)][key]["mean"]
                for step in steps
            ]
            for run in runs
        ])
        axis.errorbar(
            steps, values.mean(axis=0),
            yerr=values.std(axis=0, ddof=1),
            marker="o", capsize=4, label=label, color=color,
        )
    axis.set(
        xlabel="Executed actions before replanning",
        ylabel="Episode length",
        title="Receding-Horizon Sensitivity",
        xticks=steps,
    )
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "execute_steps_ablation")


def plot_action_distribution(input_dir, output_dir):
    """Plot hard-expert choices in coordinates normalized by the two gaps."""
    files = sorted(input_dir.glob(
        "seed_*/dataset_summary_hard_*.npz",
    ))
    states, actions = [], []
    for path in files:
        data = np.load(path)
        states.append(data["states"])
        actions.append(data["actions"])
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    separation = states[:, 2] - states[:, 1]
    midpoint = (states[:, 1] + states[:, 2]) / 2.0
    near = (states[:, 0] < 0.18) & (np.abs(separation) > 0.1)
    relative_target = (
        (actions[near, 0] - midpoint[near]) / np.abs(separation[near])
    )
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.hist(
        relative_target, bins=np.linspace(-0.8, 0.8, 49),
        color=COLORS["bc_reg_hard"], alpha=0.8,
    )
    axis.axvline(-0.5, color="black", linestyle="--", linewidth=1)
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Target relative to midpoint (gap separation units)",
        ylabel="Count",
        title="Hard-Mode Expert Gap Choice Near Commitment",
        xlim=(-0.8, 0.8),
    )
    save_figure(figure, output_dir, "hard_action_distribution")


def plot_efficiency(rows, output_dir):
    """Plot measured training time for directly trained policies."""
    methods = ("bc_reg_easy", "bc_reg_hard", "bc_flow_hard")
    values = [
        [
            float(row["train_seconds"]) for row in rows
            if row["method"] == method
        ]
        for method in methods
    ]
    means = [np.mean(value) for value in values]
    errors = [np.std(value, ddof=1) for value in values]
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.bar(
        range(len(methods)), means, yerr=errors, capsize=4,
        color=[COLORS[method] for method in methods],
    )
    axis.set_xticks(range(len(methods)), [LABELS[method] for method in methods])
    axis.set(
        ylabel="Training time (s)",
        title="Measured Training Cost",
    )
    save_figure(figure, output_dir, "training_efficiency")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--macros", type=Path)
    parser.add_argument("--submission-output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_style()
    runs = load_runs(args.input)
    if len(runs) < 2:
        raise RuntimeError("At least two completed seeds are required")
    rows, aggregate = write_summary(runs, args.output)
    if args.macros is not None:
        write_latex_macros(runs, aggregate, args.macros)
    if args.submission_output is not None:
        write_submission_summaries(
            runs, aggregate, args.submission_output,
        )
    plot_method_comparison(aggregate, args.output)
    plot_dagger_curve(runs, args.output)
    plot_training_curves(runs, args.output)
    plot_flow_ablation(runs, args.output)
    plot_execute_ablation(runs, args.output)
    plot_action_distribution(args.input, args.output)
    plot_efficiency(rows, args.output)
    print(f"Generated figures from {len(runs)} seeds in {args.output}")


if __name__ == "__main__":
    main()
