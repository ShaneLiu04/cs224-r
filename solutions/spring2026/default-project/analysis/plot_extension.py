"""Figures and macros for the Default Project extension (long-think SFT, RL variants, selection)."""
import argparse
import glob
import gzip
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"base": "#7f7f7f", "sft_short": "#ff7f0e", "sft_long": "#1f77b4", "official_sft": "#17becf",
     "rl_rloo_long": "#d62728", "rl_grpo_long": "#2ca02c", "rl_drgrpo_long": "#9467bd",
     "rl_curr_long": "#8c564b", "rl_rloo_short": "#e377c2", "rl_rloo_official": "#bcbd22"}
LABEL = {"base": "Base", "sft_short": "SFT (short)", "sft_long": "SFT (long think)",
         "official_sft": "official SFT ckpt", "rl_rloo_long": "RLOO", "rl_grpo_long": "GRPO",
         "rl_drgrpo_long": "Dr. GRPO", "rl_curr_long": "RLOO + curriculum",
         "rl_rloo_short": "RLOO from short SFT", "rl_rloo_official": "RLOO from official"}
SELECT = [("pass@1", "random sample"), ("max_logprob", "max mean log-prob"),
          ("majority", "majority vote"), ("number_check", "number-use check"),
          ("verifier", "full verifier")]


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def load_evals(eval_dir):
    evals = {}
    for path in glob.glob(os.path.join(eval_dir, "*.json")):
        name = os.path.basename(path)[:-5]
        m = re.match(r"(.+)_(test|heldout)$", name)
        if m:
            evals.setdefault(m.group(1), {})[m.group(2)] = json.load(open(path))
    return evals


def family(name):
    """rl_rloo_long_s1 -> rl_rloo_long; sft_long_ep3 stays separate."""
    return re.sub(r"_s\d+$", "", name)


def seed_mean(evals, fam, split, key):
    vals = [v[split]["summary"][key] for n, v in evals.items()
            if family(n) == fam and split in v and key in v[split]["summary"]]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (np.nan, 0.0, 0)


def main_models(evals):
    order = ["base", "sft_short", "sft_long", "official_sft", "rl_rloo_short", "rl_rloo_official",
             "rl_rloo_long", "rl_grpo_long", "rl_drgrpo_long", "rl_curr_long"]
    fams = {family(n) for n in evals}
    return [m for m in order if m in fams]


def fig_passk(evals, out, split):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ks = [1, 2, 4, 8, 16]
    for m in main_models(evals):
        means = [seed_mean(evals, m, split, f"pass@{k}")[0] for k in ks]
        ax.plot(ks, means, marker="o", ms=3, color=C.get(m), label=LABEL.get(m, m))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("k (samples)")
    ax.set_ylabel(f"pass@k ({split})")
    ax.legend(fontsize=6.5, ncol=2)
    save(fig, out, f"ext_fig1_passk_{split}")


def fig_selection(evals, out, split):
    models = [m for m in main_models(evals) if m != "base"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    width = 0.8 / len(SELECT)
    for j, (key, label) in enumerate(SELECT):
        k = key if key == "pass@1" else f"{key}@16"
        vals = [seed_mean(evals, m, split, k) for m in models]
        ax.bar(np.arange(len(models)) + j * width, [v[0] for v in vals], width,
               yerr=[v[1] for v in vals], label=label, capsize=2)
    ax.set_xticks(np.arange(len(models)) + 0.4 - width / 2,
                  [LABEL.get(m, m) for m in models], rotation=18, fontsize=7)
    ax.set_ylabel(f"accuracy of selected answer ({split})")
    ax.set_title("turning 16 samples into one answer")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save(fig, out, f"ext_fig2_selection_{split}")


def fig_selection_budget(evals, out, split, models=("sft_long", "rl_rloo_long")):
    models = [m for m in models if any(family(n) == m for n in evals)]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    ks = [1, 2, 4, 8, 16]
    for ax, m in zip(axes, models):
        for key, label in SELECT[1:]:
            ax.plot(ks, [seed_mean(evals, m, split, f"{key}@{k}")[0] for k in ks], marker="o", ms=3,
                    label=label)
        ax.plot(ks, [seed_mean(evals, m, split, f"pass@{k}")[0] for k in ks], "k--", lw=1,
                label="pass@k (oracle)")
        ax.axhline(seed_mean(evals, m, split, "pass@1")[0], color="gray", ls=":", lw=1,
                   label="pass@1")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("sample budget k")
        ax.set_title(LABEL.get(m, m))
    axes[0].set_ylabel(f"selected-answer accuracy ({split})")
    axes[-1].legend(fontsize=7)
    save(fig, out, f"ext_fig3_selection_budget_{split}")


def fig_sft_epochs(evals, sft_metrics, out):
    eps, p1, p16 = [], [], []
    for e in range(1, 7):
        name = "sft_long" if e == 6 else f"sft_long_ep{e}"
        if name in evals and "heldout" in evals[name]:
            s = evals[name]["heldout"]["summary"]
            eps.append(e)
            p1.append(s["pass@1"])
            p16.append(s["pass@16"])
    nll = [(r["epoch"] + 1, r["eval_loss"]) for r in sft_metrics if "eval_loss" in r]
    if not eps:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(eps, p1, "o-", label="pass@1")
    ax.plot(eps, p16, "s-", label="pass@16")
    ax.set_xlabel("epoch over 1,000 cog_behav demonstrations")
    ax.set_ylabel("held-out pass@k")
    if nll:
        ax2 = ax.twinx()
        ax2.plot(*zip(*nll), "^:", color="gray", label="held-out NLL")
        ax2.set_ylabel("response NLL", color="gray")
        ax2.grid(False)
    ax.legend(fontsize=7, loc="center right")
    save(fig, out, "ext_fig4_sft_epochs")


def samples(eval_dir, name, split):
    path = os.path.join(eval_dir, f"{name}_{split}.samples.jsonl.gz")
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fig_lengths(eval_dir, evals, out, split="heldout"):
    models = [m for m in ["sft_long", "official_sft", "rl_rloo_long_s0", "rl_grpo_long_s0"]
              if m in evals]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 1024, 33)
    drawn = 0
    for ax, m in zip(axes, models):
        rows = samples(eval_dir, m, split)
        if not rows:
            continue
        drawn += 1
        ok = [l for r in rows for l, c in zip(r["lengths"], r["correct"]) if c]
        bad = [l for r in rows for l, c in zip(r["lengths"], r["correct"]) if not c]
        ax.hist(bad, bins=bins, alpha=0.6, density=True, label=f"wrong (n={len(bad)})", color="#d62728")
        ax.hist(ok, bins=bins, alpha=0.6, density=True, label=f"correct (n={len(ok)})", color="#2ca02c")
        ax.set_title(LABEL.get(family(m), m), fontsize=8)
        ax.set_xlabel("response tokens")
        ax.legend(fontsize=6)
    if drawn == 0:
        plt.close(fig)
        return
    axes[0].set_ylabel("density")
    save(fig, out, "ext_fig5_lengths")


def fig_difficulty(evals, out, split="heldout"):
    models = [m for m in main_models(evals) if m != "base"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    width = 0.38
    for j, n in enumerate([3, 4]):
        vals = []
        for m in models:
            per = []
            for name, v in evals.items():
                if family(name) == m and split in v:
                    recs = [r for r in v[split]["records"] if len(r["nums"]) == n]
                    if recs:
                        per.append(np.mean([r["pass@1"] for r in recs]))
            vals.append(np.mean(per) if per else np.nan)
        ax.bar(np.arange(len(models)) + j * width, vals, width, label=f"{n} numbers")
    ax.set_xticks(np.arange(len(models)) + width / 2, [LABEL.get(m, m) for m in models],
                  rotation=18, fontsize=7)
    ax.set_ylabel(f"pass@1 ({split})")
    ax.legend(fontsize=7)
    save(fig, out, "ext_fig6_difficulty")


def load_rl(rl_dir):
    runs = {}
    for path in glob.glob(os.path.join(rl_dir, "*", "metrics.jsonl")):
        name = path.split(os.sep)[-2]
        if name == "smoke":
            continue
        runs[name] = [json.loads(line) for line in open(path)]
    return runs


def fig_rl(runs, out):
    if not runs:
        return
    fams = {}
    for name, rows in runs.items():
        fams.setdefault("rl_" + re.sub(r"_s\d+$", "", name), []).append(rows)
    panels = [("reward_exact", "train exact-match rate", 5), ("eval_pass@1", "held-out pass@1 (n=4)", 1),
              ("mean_tokens", "response tokens", 5), ("zero_variance_groups", "groups with zero advantage", 5),
              ("kl", "KL(π‖π_SFT)", 5), ("entropy", "token entropy", 5)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4))
    for ax, (key, title, smooth) in zip(axes.flat, panels):
        for fam, seeds in sorted(fams.items()):
            curves = []
            for rows in seeds:
                pts = [(r["step"], r[key]) for r in rows if key in r]
                if pts:
                    curves.append(pts)
            if not curves:
                continue
            steps = sorted({s for c in curves for s, _ in c})
            mat = np.full((len(curves), len(steps)), np.nan)
            for i, c in enumerate(curves):
                for s, v in c:
                    mat[i, steps.index(s)] = v
            mean = np.nanmean(mat, axis=0)
            if smooth > 1:
                mean = np.convolve(np.pad(mean, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"),
                                   np.ones(smooth) / smooth, mode="valid")
            ax.plot(steps, mean, color=C.get(fam), label=f"{LABEL.get(fam, fam)} ({len(curves)})", lw=1.3)
            if len(curves) > 1:
                std = np.nanstd(mat, axis=0)
                ax.fill_between(steps, mean - std, mean + std, color=C.get(fam), alpha=0.15, lw=0)
        ax.set_title(title)
        ax.set_xlabel("RL step")
    axes[0, 0].legend(fontsize=6)
    save(fig, out, "ext_fig7_rl_curves")


def fig_sharpening(eval_dir, out, before="sft_long", after="rl_rloo_long_s0", split="heldout"):
    a, b = samples(eval_dir, before, split), samples(eval_dir, after, split)
    if not a or not b:
        return
    ca = np.array([sum(r["correct"]) for r in a])
    cb = np.array([sum(r["correct"]) for r in b])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    grid = np.zeros((17, 17))
    for x, y in zip(ca, cb):
        grid[y, x] += 1
    ax.imshow(np.log1p(grid), origin="lower", cmap="magma_r")
    ax.plot([0, 16], [0, 16], "c--", lw=1)
    ax.set_xlabel(f"correct / 16, {LABEL.get(before, before)}")
    ax.set_ylabel(f"correct / 16, {LABEL.get(family(after), after)}")
    ax.set_title(f"lost {np.sum((ca > 0) & (cb == 0))}, gained {np.sum((ca == 0) & (cb > 0))} prompts",
                 fontsize=8)
    ax.grid(False)
    save(fig, out, "ext_fig8_sharpening")
    return {"lost": int(np.sum((ca > 0) & (cb == 0))), "gained": int(np.sum((ca == 0) & (cb > 0))),
            "mean_before": float(ca.mean() / 16), "mean_after": float(cb.mean() / 16)}


def macro_name(text):
    text = re.sub(r"@(\d+)", lambda m: "At" + "".join("ZOTHFVSENX"[int(d)] for d in m.group(1)), text)
    text = re.sub(r"\d", lambda m: "ZOTHFVSENX"[int(m.group(0))], text)
    return "".join(part.capitalize() for part in re.split(r"[^A-Za-z]+", text) if part)


def write_macros(evals, runs, extra, path):
    lines = ["% Auto-generated by analysis/plot_extension.py"]
    keys = ["pass@1", "pass@4", "pass@8", "pass@16", "majority@16", "max_logprob@16",
            "number_check@16", "verifier@16", "mean_tokens", "format_rate", "truncation_rate"]
    for fam in sorted({family(n) for n in evals}):
        for split in ("test", "heldout"):
            for k in keys:
                mean, std, n = seed_mean(evals, fam, split, k)
                if n == 0:
                    continue
                fmt = f"{mean:.0f}" if k == "mean_tokens" else f"{mean:.3f}"
                if n > 1 and k != "mean_tokens":
                    fmt += f"\\pm{std:.3f}"
                lines.append(f"\\newcommand{{\\x{macro_name(fam + ' ' + split + ' ' + k)}}}{{{fmt}}}")
            lines.append(f"\\newcommand{{\\x{macro_name(fam + ' ' + split + ' seeds')}}}{{{max(n, 1)}}}")
    for name, rows in runs.items():
        last = [r for r in rows if "eval_pass@1" in r]
        if last:
            lines.append(f"\\newcommand{{\\x{macro_name('rltrain ' + name + ' evalpass')}}}"
                         f"{{{last[-1]['eval_pass@1']:.3f}}}")
            sec = np.mean([r["t_gen"] + r["t_train"] for r in rows if "t_gen" in r])
            lines.append(f"\\newcommand{{\\x{macro_name('rltrain ' + name + ' secperstep')}}}{{{sec:.1f}}}")
    for k, v in (extra or {}).items():
        lines.append(f"\\newcommand{{\\x{macro_name('sharp ' + k)}}}{{{v:.3f}}}" if isinstance(v, float)
                     else f"\\newcommand{{\\x{macro_name('sharp ' + k)}}}{{{v}}}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="/root/autodl-tmp/hw5/dp/eval")
    parser.add_argument("--rl_dir", default="/root/autodl-tmp/hw5/dp/runs/rl")
    parser.add_argument("--sft_metrics", default="/root/autodl-tmp/hw5/dp/runs/sft/dp/sft_long/metrics.jsonl")
    parser.add_argument("--out", default="/root/autodl-tmp/hw5/export_dp")
    args = parser.parse_args()
    figs = os.path.join(args.out, "figures")
    os.makedirs(figs, exist_ok=True)
    evals = load_evals(args.eval_dir)
    runs = load_rl(args.rl_dir)
    sft_metrics = [json.loads(l) for l in open(args.sft_metrics)] if os.path.exists(args.sft_metrics) else []
    for split in ("test", "heldout"):
        fig_passk(evals, figs, split)
        fig_selection(evals, figs, split)
    fig_selection_budget(evals, figs, "heldout")
    fig_sft_epochs(evals, sft_metrics, figs)
    fig_lengths(args.eval_dir, evals, figs)
    fig_difficulty(evals, figs)
    fig_rl(runs, figs)
    extra = fig_sharpening(args.eval_dir, figs)
    write_macros(evals, runs, extra, os.path.join(args.out, "extension_macros.tex"))
    table = {n: {s: {k: v for k, v in d["summary"].items() if isinstance(v, (int, float))}
                 for s, d in e.items()} for n, e in evals.items()}
    json.dump({"evals": table, "sharpening": extra}, open(os.path.join(args.out, "extension_summary.json"), "w"),
              indent=1)
    print("models:", sorted(evals), "rl runs:", sorted(runs))


if __name__ == "__main__":
    main()
