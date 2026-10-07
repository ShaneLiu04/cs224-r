"""Run the HW3 experiment grid on one machine with bounded concurrency.

Usage (on AutoDL, after `source env.sh`):
    python run_experiments.py --groups core --max-procs 24
    python run_experiments.py --groups ablation --max-procs 24 --dry-run

Each job writes <runs>/<name>/<hw3_...>/metrics.csv; a job is skipped if its
`done` marker exists, so the script can be re-launched safely.
"""
import argparse
import glob
import itertools
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
RUNS = os.path.join(HERE, "artifacts", "raw", "runs")
STITCH = os.path.join(CODE, "offline_datasets", "pointmass_stitching_dataset.npz")
SEEDS = (1, 2, 3)
# The public v0 medium-diverse HDF5 is truncated (header eof 231095454, hosted
# copy 100917248). medium-diverse-v2 is the fixed release of the same maze.
UMAZE, MEDIUM, PM = "antmaze-umaze-v0", "antmaze-medium-diverse-v2", "PointmassMedium-v0"


def job(name, algo, env, seed, **flags):
    args = ["--algo", algo, "--env_name", env, "--exp_name", name, "--seed", str(seed)]
    for k, v in flags.items():
        args += [f"--{k}"] if v is True else [f"--{k}", str(v)]
    return name, args


def core_jobs():
    jobs = []
    for s in SEEDS:
        jobs.append(job(f"awac_umaze_seed{s}", "awac", UMAZE, s))
        jobs.append(job(f"awac_medium_seed{s}", "awac", MEDIUM, s))
        for z in (0.2, 0.9):
            jobs.append(job(f"iql_z{z}_umaze_seed{s}", "iql", UMAZE, s, iql_expectile=z))
        jobs.append(job(f"iql_z0.9_medium_seed{s}", "iql", MEDIUM, s, iql_expectile=0.9))
        for z in (0.2, 0.9):
            jobs.append(job(f"iql_z{z}_stitch_seed{s}", "iql", PM, s, iql_expectile=z,
                            offline_dataset=STITCH))
        jobs.append(job(f"fbc10_stitch_seed{s}", "bc", PM, s, offline_dataset=STITCH,
                        filter_top_percent=10))
    return jobs


def ablation_jobs():
    jobs = []
    for s in SEEDS:
        for z in (0.5, 0.7, 0.95):
            jobs.append(job(f"iql_z{z}_umaze_seed{s}", "iql", UMAZE, s, iql_expectile=z))
        for lam in (0.03, 0.3, 1.0):
            jobs.append(job(f"awac_lam{lam}_umaze_seed{s}", "awac", UMAZE, s, awac_lambda=lam))
        jobs.append(job(f"iql_z0.2_medium_seed{s}", "iql", MEDIUM, s, iql_expectile=0.2))
        # stitching: expectile and filtering sweeps on the cheap PointMass task
        for z in (0.5, 0.7, 0.8, 0.95):
            jobs.append(job(f"iql_z{z}_stitch_seed{s}", "iql", PM, s, iql_expectile=z,
                            offline_dataset=STITCH))
        jobs.append(job(f"awac_stitch_seed{s}", "awac", PM, s, offline_dataset=STITCH))
        for pct in (1, 25, 50, 100):
            jobs.append(job(f"fbc{pct}_stitch_seed{s}", "bc", PM, s, offline_dataset=STITCH,
                            filter_top_percent=pct))
    return jobs


GROUPS = {"core": core_jobs, "ablation": ablation_jobs}


def is_antmaze(args):
    return "antmaze" in args[args.index("--env_name") + 1]


def _cmdlines():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            raw = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")
        except OSError:
            continue
        if b"cs224r/scripts/run_algo.py" in raw:
            yield raw.decode(errors="ignore")


def algo_count():
    return sum(1 for _ in _cmdlines())


def live_exp_names():
    names = set()
    for text in _cmdlines():
        parts = text.split()
        if "--exp_name" in parts:
            names.add(parts[parts.index("--exp_name") + 1])
    return names


def occupied(name):
    """Skip a job that already finished or is still training."""
    run = os.path.join(RUNS, name)
    if os.path.exists(os.path.join(run, "done")):
        return True
    if glob.glob(os.path.join(run, "*", "metrics.csv")):
        return True
    return name in live_exp_names()


def priority(name):
    """Required PointMass runs first, then other PointMass, then leftover AntMaze."""
    if name.startswith(("iql_z0.2_stitch", "iql_z0.9_stitch", "fbc10_stitch")):
        return 0
    if "stitch" in name:
        return 1
    return 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs="+", default=["core"], choices=list(GROUPS))
    p.add_argument("--max-procs", type=int, default=os.cpu_count())
    p.add_argument("--gpus", type=int, default=None)
    p.add_argument("--only", default=None, help="substring filter on job names")
    p.add_argument("--exclude", nargs="*", default=[], help="exact job names to skip")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    import torch
    n_gpu = a.gpus if a.gpus is not None else torch.cuda.device_count()
    jobs = list(itertools.chain.from_iterable(GROUPS[g]() for g in a.groups))
    if a.only:
        jobs = [j for j in jobs if a.only in j[0]]
    if a.exclude:
        ban = set(a.exclude)
        jobs = [j for j in jobs if j[0] not in ban]
    # PointMass stitching is required and short. Do not let leftover AntMaze
    # ablations occupy the slots that open when the first wave finishes.
    jobs.sort(key=lambda j: (priority(j[0]), is_antmaze(j[1])))
    todo = [j for j in jobs if not occupied(j[0])]
    print(f"{len(todo)}/{len(jobs)} jobs to run, max_procs={a.max_procs}, "
          f"gpus={n_gpu}, alive={algo_count()}", flush=True)
    if a.dry_run:
        for name, args in todo:
            print(name, " ".join(args))
        return

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    running, gpu_cycle = {}, itertools.cycle(range(max(n_gpu, 1)))
    while todo or running:
        while todo and occupied(todo[0][0]):
            name, _ = todo.pop(0)
            print(f"[skip] {name}", flush=True)
        while todo and len(running) < a.max_procs and algo_count() < a.max_procs:
            name, args = todo.pop(0)
            if occupied(name):
                print(f"[skip] {name}", flush=True)
                continue
            out = os.path.join(RUNS, name)
            os.makedirs(out, exist_ok=True)
            extra = ["--no_video", "--logdir_root", out]
            extra += ["--which_gpu", str(next(gpu_cycle))] if n_gpu else ["--no_gpu"]
            log = open(os.path.join(out, "stdout.log"), "w")
            cmd = [sys.executable, "cs224r/scripts/run_algo.py", *args, *extra]
            running[name] = (subprocess.Popen(cmd, cwd=CODE, env=env, stdout=log,
                                              stderr=subprocess.STDOUT), log, time.time())
            print(f"[start] {name}", flush=True)
        time.sleep(10)
        for name in list(running):
            proc, log, t0 = running[name]
            if proc.poll() is None:
                continue
            log.close()
            ok = proc.returncode == 0
            if ok:
                open(os.path.join(RUNS, name, "done"), "w").write(f"{time.time() - t0:.0f}\n")
            print(f"[{'done' if ok else 'FAIL'}] {name} rc={proc.returncode} "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            del running[name]


if __name__ == "__main__":
    main()
