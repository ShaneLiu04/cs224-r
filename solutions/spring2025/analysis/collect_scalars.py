"""Dump every TensorBoard scalar of the 2025 HW3/HW4 runs into one CSV.

Columns: group, run, tag, step, value. `group` is hw3, her, sawyer, or meta;
`run` encodes the configuration (e.g. cql_medium_a0.1_rnd_s0).
"""
import argparse
import csv
import glob
import os
import re
import shutil

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalars(path):
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()
    for tag in acc.Tags().get("scalars", []):
        for event in acc.Scalars(tag):
            yield tag, event.step, event.value


def hw3_runs(root):
    for d in sorted(glob.glob(os.path.join(root, "hw3_*"))):
        m = re.match(r"hw3_(.+)_Pointmass\w+-v0_", os.path.basename(d))
        if m and glob.glob(os.path.join(d, "events*")):
            yield "hw3", m.group(1), d


def gcrl_runs(root):
    for d in sorted(glob.glob(os.path.join(root, "bit_flip", "bits*", "*", "s*"))):
        bits, her, seed = d.split(os.sep)[-3:]
        yield "her", f"{bits}_{her}_{seed}", d
    for d in sorted(glob.glob(os.path.join(root, "bit_flip_k", "k*", "s*"))):
        k, seed = d.split(os.sep)[-2:]
        yield "herk", f"{k}_{seed}", d
    for d in sorted(glob.glob(os.path.join(root, "sawyer_reach", "*", "s*"))):
        her, seed = d.split(os.sep)[-2:]
        yield "sawyer", f"{her}_{seed}", d


def meta_runs(root):
    for d in sorted(glob.glob(os.path.join(root, "*", "tensorboard"))):
        name = d.split(os.sep)[-2]
        for sub in sorted(os.listdir(d)) or [""]:
            path = os.path.join(d, sub)
            if os.path.isdir(path) and glob.glob(os.path.join(path, "events*")):
                yield "meta", f"{name}/{sub}", path
        if glob.glob(os.path.join(d, "events*")):
            yield "meta", name, d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw3", default="/root/autodl-tmp/hw5/spring2025/homework3-offline-rl/code/data")
    parser.add_argument("--gcrl", default="/root/autodl-tmp/hw5/runs/gcrl")
    parser.add_argument("--meta", default="/root/autodl-tmp/hw5/runs/meta")
    parser.add_argument("--out", default="/root/autodl-tmp/hw5/export")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "traj"), exist_ok=True)
    rows = 0
    with open(os.path.join(args.out, "scalars_2025.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "run", "tag", "step", "value"])
        sources = list(hw3_runs(args.hw3)) + list(gcrl_runs(args.gcrl)) + list(meta_runs(args.meta))
        for group, run, path in sources:
            for tag, step, value in scalars(path):
                writer.writerow([group, run, tag, step, value])
                rows += 1
            if group == "hw3":
                for name in ("eval_last_traj", "curr_state_density", "curr_exploitation_value"):
                    img = os.path.join(path, f"{name}.png")
                    if os.path.exists(img):
                        shutil.copy(img, os.path.join(args.out, "traj", f"{run}__{name}.png"))
    print("wrote", rows, "rows from", len(sources), "runs")


if __name__ == "__main__":
    main()
