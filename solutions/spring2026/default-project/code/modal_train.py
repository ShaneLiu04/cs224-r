"""Modal launcher for the existing SFT, IPO, and RLOO training entrypoints.

Run from the `default_proj` directory, for example:

    modal run modal_train.py sft --model_name Qwen/Qwen2.5-0.5B
    modal run modal_train.py ipo --model_name your/model --dataset_name your/dataset
    modal run modal_train.py rloo --model_name your/model --dataset_name your/dataset
    modal run modal_train.py eval --model_path your/model --output_name your_eval
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

import modal


LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_ROOT = Path("/root/default_proj")
REMOTE_VOLUME_ROOT = Path("/vol")
REMOTE_REQUIREMENTS_PATH = REMOTE_PROJECT_ROOT / "modal_requirements.txt"

APP_NAME = os.environ.get("MODAL_APP_NAME", "default-proj-training")
GPU_CONFIG = os.environ.get("MODAL_GPU", "H100!")
TIMEOUT_SECONDS = int(os.environ.get("MODAL_TIMEOUT_SECONDS", "86400"))
STARTUP_TIMEOUT_SECONDS = int(os.environ.get("MODAL_STARTUP_TIMEOUT_SECONDS", "1800"))
CPU_COUNT = int(os.environ.get("MODAL_CPU_COUNT", "8"))
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "default-proj-training")
PIP_EXTRA_INDEX_URL = os.environ.get(
    "MODAL_PIP_EXTRA_INDEX_URL",
    "https://download.pytorch.org/whl/cu128",
)

TRAINING_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _build_secret_list() -> list[modal.Secret]:
    secret_values = {}
    for key in (
        "HF_TOKEN",
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_USERNAME",
        "WANDB_USER_EMAIL",
    ):
        value = os.environ.get(key)
        if value:
            secret_values[key] = value

    if not secret_values:
        return []

    return [modal.Secret.from_dict(secret_values)]


base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(str(LOCAL_PROJECT_ROOT), remote_path=str(REMOTE_PROJECT_ROOT), copy=True)
    .run_commands(
        (
            f"cd {shlex.quote(str(REMOTE_PROJECT_ROOT))} && "
            "python -m pip install --upgrade "
            "pip==25.3 setuptools==80.10.2 wheel==0.46.3"
        ),
        (
            f"cd {shlex.quote(str(REMOTE_PROJECT_ROOT))} && "
            "python -m pip install "
            f"--extra-index-url {shlex.quote(PIP_EXTRA_INDEX_URL)} "
            f"-r {shlex.quote(str(REMOTE_REQUIREMENTS_PATH))}"
        ),
        f"cd {shlex.quote(str(REMOTE_PROJECT_ROOT))} && python -m pip install --no-deps -e .",
    )
)

app = modal.App(APP_NAME)


def _run_training(script_path: str, trainer_args: list[str]) -> str:
    REMOTE_VOLUME_ROOT.mkdir(parents=True, exist_ok=True)
    (REMOTE_VOLUME_ROOT / "cache" / "huggingface" / "datasets").mkdir(parents=True, exist_ok=True)
    (REMOTE_VOLUME_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    hf_home = REMOTE_VOLUME_ROOT / "cache" / "huggingface"
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    env.setdefault("WANDB__SERVICE_WAIT", "300")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    command = ["python", script_path, *trainer_args]
    print(f"Executing command in Modal: {shlex.join(command)}")
    print(f"Checkpoints and cache are persisted in Modal volume '{VOLUME_NAME}' under {REMOTE_VOLUME_ROOT}.")

    try:
        subprocess.run(
            command,
            cwd=str(REMOTE_PROJECT_ROOT),
            env=env,
            check=True,
        )
    finally:
        TRAINING_VOLUME.commit()

    return f"Finished {script_path}. Persisted artifacts to Modal volume '{VOLUME_NAME}'."


def _run_eval(eval_args: list[str]) -> str:
    REMOTE_VOLUME_ROOT.mkdir(parents=True, exist_ok=True)
    (REMOTE_VOLUME_ROOT / "cache" / "huggingface" / "datasets").mkdir(parents=True, exist_ok=True)
    (REMOTE_VOLUME_ROOT / "evaluation" / "eval_results").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    hf_home = REMOTE_VOLUME_ROOT / "cache" / "huggingface"
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    env.setdefault("WANDB__SERVICE_WAIT", "300")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    command = ["python", "evaluation/countdown_eval.py", *eval_args]
    print(f"Executing command in Modal: {shlex.join(command)}")
    print(
        f"Evaluation outputs are persisted in Modal volume '{VOLUME_NAME}' under "
        f"{REMOTE_VOLUME_ROOT / 'evaluation' / 'eval_results'}."
    )

    try:
        subprocess.run(
            command,
            cwd=str(REMOTE_PROJECT_ROOT),
            env=env,
            check=True,
        )
    finally:
        TRAINING_VOLUME.commit()

    return "Finished evaluation run and persisted results to the Modal volume."


@app.function(
    image=base_image,
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    timeout=TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={str(REMOTE_VOLUME_ROOT): TRAINING_VOLUME},
    secrets=_build_secret_list(),
)
def run_sft(trainer_args: list[str]) -> str:
    return _run_training("sft_trainer/sft.py", trainer_args)


@app.function(
    image=base_image,
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    timeout=TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={str(REMOTE_VOLUME_ROOT): TRAINING_VOLUME},
    secrets=_build_secret_list(),
)
def run_ipo(trainer_args: list[str]) -> str:
    return _run_training("ipo_trainer/ipo.py", trainer_args)


@app.function(
    image=base_image,
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    timeout=TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={str(REMOTE_VOLUME_ROOT): TRAINING_VOLUME},
    secrets=_build_secret_list(),
)
def run_rloo(trainer_args: list[str]) -> str:
    return _run_training("rloo_trainer/rloo.py", trainer_args)


@app.function(
    image=base_image,
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    timeout=TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={str(REMOTE_VOLUME_ROOT): TRAINING_VOLUME},
    secrets=_build_secret_list(),
)
def run_eval(eval_args: list[str]) -> str:
    return _run_eval(eval_args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch one of the existing training entrypoints on Modal.",
    )
    parser.add_argument("trainer", choices=("sft", "ipo", "rloo", "eval"))
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the underlying trainer script.",
    )
    return parser


@app.local_entrypoint()
def main(*raw_args: str) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(raw_args))

    trainer_args = list(args.trainer_args)
    if trainer_args[:1] == ["--"]:
        trainer_args = trainer_args[1:]

    if args.trainer == "sft":
        result = run_sft.remote(trainer_args)
    elif args.trainer == "ipo":
        result = run_ipo.remote(trainer_args)
    elif args.trainer == "eval":
        result = run_eval.remote(trainer_args)
    else:
        result = run_rloo.remote(trainer_args)

    print(result)
