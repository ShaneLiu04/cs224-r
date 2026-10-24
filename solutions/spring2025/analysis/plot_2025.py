"""Figures, summary JSON, and LaTeX macros for 2025 HW3 (IQL/CQL) and HW4 (HER, meta-RL)."""
import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def split_seed(run):
    m = re.match(r"(.+)_s(\d+)$", run)
    return (m.group(1), int(m.group(2))) if m else (run, 0)


def curves(df, tag):
    """config -> DataFrame indexed by step, one column per seed."""
    sub = df[df.tag == tag].copy()
    if sub.empty:
        return {}
    sub[["config", "seed"]] = sub.run.apply(lambda r: pd.Series(split_seed(r)))
    out = {}
    for config, g in sub.groupby("config"):
        out[config] = g.pivot_table(index="step", columns="seed", values="value").sort_index()
    return out


def band(ax, table, label, color, smooth=1):
    mean = table.mean(axis=1).rolling(smooth, min_periods=1).mean()
    std = table.std(axis=1).fillna(0).rolling(smooth, min_periods=1).mean()
    ax.plot(table.index, mean, color=color, label=f"{label} (n={table.shape[1]})", lw=1.4)
    ax.fill_between(table.index, mean - std, mean + std, color=color, alpha=0.18, lw=0)


def final_stats(table, last=3):
    """Per-seed mean of the last `last` evaluations -> (mean, std, per-seed list)."""
    per_seed = table.apply(lambda col: col.dropna().tail(last).mean())
    return float(per_seed.mean()), float(per_seed.std(ddof=0)), [float(x) for x in per_seed]


def hw3_figures(df, out, summary):
    ret = curves(df[df.group == "hw3"], "Eval_AverageReturn")
    if not ret:
        return
    summary["hw3"] = {k: dict(zip(("mean", "std", "seeds"), final_stats(v))) for k, v in ret.items()}

    def panel(ax, keys, labels, title):
        for i, (k, lab) in enumerate(zip(keys, labels)):
            if k in ret:
                band(ax, ret[k], lab, PALETTE[i % len(PALETTE)], smooth=3)
        ax.axvline(20000, color="gray", ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("eval average return")
        ax.legend(fontsize=7)

    zetas = ["0.2", "0.5", "0.7", "0.9"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    panel(axes[0], [f"iql_easy_z{z}_rnd" for z in zetas], [f"ζ={z}" for z in zetas],
          "IQL, PointmassEasy, RND data")
    means = [summary["hw3"].get(f"iql_easy_z{z}_rnd", {}).get("mean", np.nan) for z in zetas]
    stds = [summary["hw3"].get(f"iql_easy_z{z}_rnd", {}).get("std", 0) for z in zetas]
    xs = np.arange(len(zetas))
    axes[1].bar(xs, np.nan_to_num(means), yerr=stds, color=PALETTE[:4], capsize=3)
    axes[1].set_xticks(xs, zetas)
    for x, z in zip(xs, zetas):
        for s in summary["hw3"].get(f"iql_easy_z{z}_rnd", {}).get("seeds", []):
            axes[1].scatter([x], [s], color="k", s=8, zorder=3)
    axes[1].set_xlabel("expectile ζ")
    axes[1].set_ylabel("final return (last 3 evals)")
    axes[1].set_title("expectile sweep")
    save(fig, out, "hw3_fig1_iql_expectile")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, z in zip(axes, ["0.7", "0.9"]):
        panel(ax, [f"iql_medium_z{z}_rnd", f"iql_medium_z{z}_random"], ["RND", "random"],
              f"IQL ζ={z}, PointmassMedium")
    save(fig, out, "hw3_fig2_iql_exploration")

    alphas = ["0.0", "0.1", "1.0"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    panel(axes[0], [f"cql_medium_a{a}_rnd" for a in alphas], [f"α={a}" for a in alphas],
          "CQL, PointmassMedium, RND data")
    panel(axes[1], ["cql_medium_a0.0_rnd", "cql_medium_a0.0_random",
                    "cql_medium_a0.1_rnd", "cql_medium_a0.1_random"],
          ["α=0 RND", "α=0 random", "α=0.1 RND", "α=0.1 random"], "CQL: exploration data")
    save(fig, out, "hw3_fig3_cql_alpha")

    data_q = curves(df[df.group == "hw3"], "Exploitation_Data_q-values")
    ood_q = curves(df[df.group == "hw3"], "Exploitation_OOD_q-values")
    if data_q:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharey=True)
        for ax, a in zip(axes, alphas):
            k = f"cql_medium_a{a}_rnd"
            if k in data_q:
                band(ax, data_q[k], "Q(s, a_data)", PALETTE[0], smooth=5)
                band(ax, ood_q[k], "logsumexp_a Q(s,a)", PALETTE[1], smooth=5)
            ax.set_title(f"α={a}")
            ax.set_xlabel("environment steps")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Q value")
        save(fig, out, "hw3_fig4_cql_qvalues")
        sweep = ["0.0", "0.01", "0.03", "0.1", "0.3", "1.0", "3.0"]
        summary["hw3_q"] = {
            a: {"data_q": float(data_q[f"cql_medium_a{a}_rnd"].tail(5).mean().mean()),
                "lse_q": float(ood_q[f"cql_medium_a{a}_rnd"].tail(5).mean().mean())}
            for a in sweep if f"cql_medium_a{a}_rnd" in data_q}

    sweep = [a for a in ["0.0", "0.01", "0.03", "0.1", "0.3", "1.0", "3.0"]
             if f"cql_medium_a{a}_rnd" in summary["hw3"]]
    zsweep = [z for z in ["0.5", "0.7", "0.9", "0.95"] if f"iql_medium_z{z}_rnd" in summary["hw3"]]
    if len(sweep) > 3 or len(zsweep) > 2:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
        xs = np.arange(len(sweep))
        stats = [summary["hw3"][f"cql_medium_a{a}_rnd"] for a in sweep]
        axes[0].errorbar(xs, [s["mean"] for s in stats], yerr=[s["std"] for s in stats],
                         color=PALETTE[0], marker="o", capsize=3, label="final return")
        for x, s in zip(xs, stats):
            axes[0].scatter([x] * len(s["seeds"]), s["seeds"], color=PALETTE[0], s=8, alpha=0.5)
        axes[0].set_xticks(xs, sweep)
        axes[0].set_xlabel("CQL α")
        axes[0].set_ylabel("final return (last 3 evals)", color=PALETTE[0])
        gap = [summary.get("hw3_q", {}).get(a) for a in sweep]
        if all(g is not None for g in gap):
            twin = axes[0].twinx()
            twin.plot(xs, [g["lse_q"] - g["data_q"] for g in gap], color=PALETTE[1], marker="s",
                      ls="--", label="logsumexp Q − Q(s,a_data)")
            twin.set_ylabel("OOD gap", color=PALETTE[1])
            twin.grid(False)
        axes[0].set_title("CQL α sweep, PointmassMedium, RND")
        xs = np.arange(len(zsweep))
        stats = [summary["hw3"][f"iql_medium_z{z}_rnd"] for z in zsweep]
        axes[1].bar(xs, [s["mean"] for s in stats], yerr=[s["std"] for s in stats],
                    color=PALETTE[2], capsize=3, alpha=0.8)
        for x, s in zip(xs, stats):
            axes[1].scatter([x] * len(s["seeds"]), s["seeds"], color="k", s=8, zorder=3)
        axes[1].set_xticks(xs, zsweep)
        axes[1].set_xlabel("expectile ζ")
        axes[1].set_ylabel("final return (last 3 evals)")
        axes[1].set_title("IQL ζ sweep, PointmassMedium, RND")
        save(fig, out, "hw3_fig6_medium_sweeps")

    oracle = summary.get("oracle", {})
    hard = [("iql_hard_z0.9_rnd", "IQL ζ=0.9"), ("cql_hard_a0.0_rnd", "CQL α=0"),
            ("cql_hard_a0.1_rnd", "CQL α=0.1")]
    if any(k in ret for k, _ in hard):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        panel(ax, [k for k, _ in hard], [l for _, l in hard], "PointmassHard, RND flag")
        if "PointmassHard-v0/eps0.0" in oracle:
            ax.axhline(oracle["PointmassHard-v0/eps0.0"]["return"], color="k", ls="--", lw=1,
                       label="CQL data collector")
            ax.legend(fontsize=7)
        save(fig, out, "hw3_fig7_hard")

    sources = [("", "oracle collector (starter)"), ("_rnd_explore", "RND exploration"),
               ("_random_explore", "uniform random")]
    alphas = ["0.0", "0.1", "1.0"]
    if any(f"cql_medium_a0.1{s}" in summary["hw3"] for s, _ in sources[1:]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
        width = 0.26
        for i, (suffix, label) in enumerate(sources):
            keys = [f"cql_medium_a{a}{suffix or '_rnd'}" for a in alphas]
            stats = [summary["hw3"].get(k, {"mean": np.nan, "std": 0, "seeds": []}) for k in keys]
            xs = np.arange(len(alphas)) + (i - 1) * width
            axes[0].bar(xs, [s["mean"] for s in stats], width, yerr=[s["std"] for s in stats],
                        color=PALETTE[i], capsize=2, label=label, alpha=0.85)
            for x, s in zip(xs, stats):
                axes[0].scatter([x] * len(s["seeds"]), s["seeds"], color="k", s=6, zorder=3)
        if "PointmassMedium-v0/eps0.0" in oracle:
            axes[0].axhline(oracle["PointmassMedium-v0/eps0.0"]["return"], color="k", ls="--", lw=1,
                            label="oracle collector return")
        axes[0].set_xticks(np.arange(len(alphas)), [f"α={a}" for a in alphas])
        axes[0].set_ylabel("final return (last 3 evals)")
        axes[0].set_title("CQL on PointmassMedium: which data?")
        axes[0].legend(fontsize=6.5, loc="lower right")
        panel(axes[1], ["cql_medium_a0.1_rnd", "cql_medium_a0.1_rnd_explore",
                        "cql_medium_a0.1_random_explore"],
              ["oracle data", "RND data", "random data"], "CQL α=0.1 learning curves")
        axes[1].axvline(10000, color="gray", ls=":", lw=1)
        save(fig, out, "hw3_fig8_cql_data_source")


def traj_montage(traj_dir, out):
    rows = [("iql_medium_z0.9_rnd_s0", "IQL ζ=0.9, RND"), ("iql_medium_z0.9_random_s0", "IQL ζ=0.9, random"),
            ("cql_medium_a0.0_rnd_s0", "CQL α=0, RND"), ("cql_medium_a0.1_rnd_s0", "CQL α=0.1, RND"),
            ("cql_medium_a0.1_random_s0", "CQL α=0.1, random")]
    rows = [r for r in rows if os.path.exists(os.path.join(traj_dir, f"{r[0]}__eval_last_traj.png"))]
    if not rows:
        return
    cols = [("curr_state_density", "exploration state density"), ("eval_last_traj", "final eval trajectory")]
    fig, axes = plt.subplots(len(rows), 2, figsize=(5.2, 2.5 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (run, title) in enumerate(rows):
        for j, (img, label) in enumerate(cols):
            path = os.path.join(traj_dir, f"{run}__{img}.png")
            ax = axes[i, j]
            ax.axis("off")
            if os.path.exists(path):
                ax.imshow(mpimg.imread(path))
            ax.set_title(f"{title}\n{label}", fontsize=7)
    save(fig, out, "hw3_fig5_trajectories")


def her_figures(df, out, summary):
    sr = curves(df[df.group == "her"], "eval_metrics/success_rate")
    if sr:
        types = ["no_hindsight", "final", "random", "future"]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharey=True)
        summary["her"] = {}
        for ax, bits in zip(axes, [6, 15, 25]):
            for i, t in enumerate(types):
                k = f"bits{bits}_{t}"
                if k in sr:
                    band(ax, sr[k], t, PALETTE[i], smooth=5)
                    summary["her"][k] = dict(zip(("mean", "std", "seeds"), final_stats(sr[k], last=10)))
                    curve = sr[k].mean(axis=1).rolling(5, min_periods=1).mean()
                    hit = curve[curve >= 0.9]
                    summary["her"][k]["epoch_to_0.9"] = float(hit.index[0]) if len(hit) else None
            ax.set_title(f"bit flip, n={bits}")
            ax.set_xlabel("epoch")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("success rate")
        save(fig, out, "hw4_fig1_bitflip_success")

        grid = np.full((3, 4), np.nan)
        for i, bits in enumerate([6, 15, 25]):
            for j, t in enumerate(types):
                grid[i, j] = summary["her"].get(f"bits{bits}_{t}", {}).get("mean", np.nan)
        fig, ax = plt.subplots(figsize=(4.8, 2.6))
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(4), types, rotation=20)
        ax.set_yticks(range(3), ["6 bits", "15 bits", "25 bits"])
        for i in range(3):
            for j in range(4):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            color="w" if grid[i, j] < 0.6 else "k", fontsize=8)
        fig.colorbar(im, ax=ax, label="final success (last 10 logs)")
        ax.grid(False)
        save(fig, out, "hw4_fig2_bitflip_heatmap")

    kr = curves(df[df.group == "herk"], "eval_metrics/success_rate")
    if kr and "bits15_future" in sr:
        kr["k4"] = sr["bits15_future"]
        ks = [k for k in ["k1", "k2", "k4", "k8"] if k in kr]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
        summary["herk"] = {}
        for i, k in enumerate(ks):
            band(axes[0], kr[k], f"k={k[1:]}", PALETTE[i], smooth=5)
            summary["herk"][k] = dict(zip(("mean", "std", "seeds"), final_stats(kr[k], last=10)))
            first = kr[k].mean(axis=1).rolling(5, min_periods=1).mean()
            hit = first[first >= 0.9]
            summary["herk"][k]["epoch_to_0.9"] = float(hit.index[0]) if len(hit) else None
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("success rate")
        axes[0].set_title("15-bit flip, future HER, k relabels per transition")
        axes[0].legend(fontsize=7)
        e90 = [summary["herk"][k]["epoch_to_0.9"] or np.nan for k in ks]
        axes[1].bar(range(len(ks)), e90, color=PALETTE[:len(ks)])
        axes[1].set_xticks(range(len(ks)), [k[1:] for k in ks])
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("first epoch with success ≥ 0.9")
        axes[1].set_title("sample efficiency vs relabel ratio")
        save(fig, out, "hw4_fig5_her_k")

    sw = curves(df[df.group == "sawyer"], "eval_metrics/success_rate")
    rw = curves(df[df.group == "sawyer"], "eval_metrics/total_reward")
    if sw:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
        summary["sawyer"] = {}
        for i, t in enumerate(["no_hindsight", "final", "future", "random"]):
            if t in sw:
                band(axes[0], sw[t], t, PALETTE[i], smooth=10)
                band(axes[1], rw[t], t, PALETTE[i], smooth=10)
                summary["sawyer"][t] = dict(zip(("mean", "std", "seeds"), final_stats(sw[t], last=20)))
                curve = sw[t].mean(axis=1).rolling(10, min_periods=1).mean()
                hit = curve[curve >= 0.8]
                summary["sawyer"][t]["epoch_to_0.8"] = float(hit.index[0]) if len(hit) else None
        axes[0].set_ylabel("success rate")
        axes[1].set_ylabel("epoch total reward (16 episodes)")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.legend(fontsize=7)
        fig.suptitle("SawyerReachXYEnv-v1 (MuJoCo 2.1, multiworld)", fontsize=9)
        save(fig, out, "hw4_fig3_sawyer")


def meta_figures(df, out, summary):
    meta = df[df.group == "meta"].copy()
    if meta.empty:
        return
    meta = meta[meta.run.str.endswith("/episode")]
    meta["algo"] = meta.run.str.extract(r"^(dreamnopen|dream|rl2)_s\d+")[0]
    meta["seed"] = meta.run.str.extract(r"_s(\d+)/")[0].astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0))
    summary["meta"] = {}
    names = {"dream": "DREAM (c=0.1)", "rl2": "RL²", "dreamnopen": "DREAM, step penalty c=0"}
    for i, algo in enumerate(["dream", "rl2", "dreamnopen"]):
        for ax, tag in zip(axes[:2], ["reward/test", "steps/test_exploration_per_episode"]):
            g = meta[(meta.algo == algo) & (meta.tag == tag)]
            if g.empty and algo == "rl2" and tag.startswith("steps/test_exploration"):
                g = meta[(meta.algo == algo) & (meta.tag == "steps/test_steps_per_episode")]
            if g.empty:
                continue
            table = g.pivot_table(index="step", columns="seed", values="value").sort_index()
            band(ax, table, names[algo], PALETTE[i])
            if tag == "reward/test":
                summary["meta"][algo] = dict(zip(("mean", "std", "seeds"), final_stats(table, last=3)))
            else:
                summary.setdefault("meta_len", {})[algo] = dict(
                    zip(("mean", "std", "seeds"), final_stats(table, last=3)))
    axes[0].set_title("meta-test return (rewards/test)")
    axes[1].set_title("test episode length")
    for i, algo in enumerate(["dream", "dreamnopen"]):
        g = meta[(meta.algo == algo) & (meta.tag == "exploitation_DQN/decoder_loss")]
        if not g.empty:
            band(axes[2], g.pivot_table(index="step", columns="seed", values="value").sort_index(),
                 names[algo], PALETTE[2 * i], smooth=10)
        axes[2].set_title("DREAM decoder ||g(τ) − z||²")
    for ax in axes:
        ax.set_xlabel("meta-training trials")
        ax.legend(fontsize=7)
    save(fig, out, "hw4_fig4_meta_rl")


def macros(summary, path):
    lines = ["% Auto-generated by analysis/plot_2025.py"]

    def name(s):
        return re.sub(r"[^A-Za-z]", "", s.title())

    for group in ("hw3", "her", "herk", "sawyer", "meta", "meta_len"):
        for config, stats in summary.get(group, {}).items():
            key = re.sub(r"(\d)", lambda m: "ABCDEFGHIJ"[int(m.group(1))], config)
            lines.append(f"\\newcommand{{\\r{name(group)}{name(key)}}}"
                         f"{{{stats['mean']:.2f} \\pm {stats['std']:.2f}}}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", default="/root/autodl-tmp/hw5/export")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or os.path.join(args.export, "figures")
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(os.path.join(args.export, "scalars_2025.csv"))
    summary = {}
    oracle_path = os.path.join(args.export, "oracle_baseline.json")
    if os.path.exists(oracle_path):
        with open(oracle_path, encoding="utf-8") as f:
            summary["oracle"] = json.load(f)
    hw3_figures(df, out, summary)
    traj_montage(os.path.join(args.export, "traj"), out)
    her_figures(df, out, summary)
    meta_figures(df, out, summary)
    with open(os.path.join(args.export, "summary_2025.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    macros(summary, os.path.join(args.export, "results_2025.tex"))
    print(json.dumps({g: {k: round(v["mean"], 3) for k, v in d.items()}
                      for g, d in summary.items() if g not in ("hw3_q", "oracle")}, indent=1))


if __name__ == "__main__":
    main()
