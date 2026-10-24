"""Montage of meta-test rollouts (last frame of each episode) for DREAM and RL^2."""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence


def frames(path):
    with Image.open(path) as gif:
        return [f.convert("RGB") for f in ImageSequence.Iterator(gif)]


def latest(run_dir):
    vis = os.path.join(run_dir, "visualize")
    step = max(os.listdir(vis), key=int)
    return step, os.path.join(vis, step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="/root/autodl-tmp/hw5/runs/meta")
    parser.add_argument("--out", default="/root/autodl-tmp/hw5/export/figures")
    args = parser.parse_args()
    rows = []
    for run, label in [("dream_s4", "DREAM"), ("dreamnopen_s0", "DREAM, c=0")]:
        step, d = latest(os.path.join(args.meta, run))
        panels = []
        for ep in (0, 1):
            panels.append((frames(os.path.join(d, f"{ep}-exploration.gif"))[-1], f"ep{ep} explore"))
            panels.append((frames(os.path.join(d, f"{ep}-exploitation.gif"))[-1], f"ep{ep} exploit"))
        rows.append((f"{label} @ {step}", panels))
    step, d = latest(os.path.join(args.meta, "rl2_s3"))
    seq = frames(os.path.join(d, "0.gif"))
    picks = sorted({0, len(seq) // 3, 2 * len(seq) // 3, len(seq) - 1})
    rows.append((f"RL² @ {step}", [(seq[i], f"t={i}") for i in picks]))

    ncol = max(len(p) for _, p in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.3 * ncol, 2.6 * len(rows)))
    for r, (title, panels) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(panels):
                ax.imshow(panels[c][0])
                ax.set_title(f"{title}\n{panels[c][1]}" if c == 0 else panels[c][1], fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"hw4_fig6_meta_frames.{ext}"), bbox_inches="tight", dpi=150)
    print("ok", [t for t, _ in rows])


if __name__ == "__main__":
    main()
