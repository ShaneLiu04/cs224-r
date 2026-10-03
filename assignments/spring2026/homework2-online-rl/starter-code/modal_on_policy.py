import modal
import subprocess

app = modal.App("cs224r-hw2-on-policy")
volume = modal.Volume.from_name("cs224r-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1-mesa-glx",
        "libosmesa6-dev",
        "libglew-dev",
        "libglfw3",
        "libgl1-mesa-dev",
        "patchelf",
        "git",
        "build-essential",
        "libhdf5-dev",
        "wget",
        "libglib2.0-0",
    )
    .run_commands(
        "mkdir -p /root/.mujoco",
        "wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz",
        "tar -xzf /tmp/mujoco.tar.gz -C /root/.mujoco",
        "rm /tmp/mujoco.tar.gz",
    )
    .env({
        "MUJOCO_PY_MUJOCO_PATH": "/root/.mujoco/mujoco210",
        "LD_LIBRARY_PATH": "/root/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
    })
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "jax-jumpy",
        "numpy==1.24.3", 
        "gym==0.26.2",
        "scikit-image>0.18.1",
        "pandas>=1.4.0,<2.0",
        "matplotlib>=3.5.0",
        "opencv-python",
        "termcolor==1.1.0",
        "dm_control",
        "tb-nightly",
        "wandb",
        "Cython==0.29.33",
        "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb",
        # extra_options="--no-deps",
        "imageio>=2.33.0",
        "imageio-ffmpeg>=0.4.4",
        "hydra-core==1.1.0",
        "hydra-submitit-launcher==1.1.5",
        "yapf==0.31.0",
        "mujoco_py==2.1.2.14",
        "scikit-learn"    
    )
    .pip_install(
        "numpy==1.24.3"
    )
    .add_local_python_source("on_policy", "train_on_policy", "mw", "utils", "logger", "replay_buffer", "video")
    .add_local_dir("cfgs", remote_path="/root/cfgs")
    .add_local_dir("demos", remote_path="/root/demos")
)


@app.function(
    image=image,
    volumes={"/output": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    gpu="A10",
    timeout=12000,
)
def train():
    import os
    import yaml

    subprocess.run(["find", "/root", "-name", "on_policy*"], check=True)
    subprocess.run(["python", "-c", "import on_policy; print(on_policy.__file__)"], cwd="/root")
    
    # Patch the config in-place from its already-mounted location
    config_path = "/root/cfgs/on_policy_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["use_tb"] = False
    cfg["defaults"] = [
        d for d in cfg.get("defaults", [])
        if d != "override hydra/launcher: submitit_local"
    ]
    if "hydra" in cfg:
        cfg["hydra"].pop("launcher", None)
        cfg["hydra"].pop("sweep", None)
        cfg["hydra"]["run"] = {"dir": "/output/on_policy_run"}

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    result = subprocess.run(
        [
            "python", "-u", "train_on_policy.py",
            "--config-path", "/root/cfgs",
            "--config-name", "on_policy_config",
        ],
        cwd="/root",
        capture_output=False,
        env={
            **os.environ,
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "MUJOCO_PY_MUJOCO_PATH": "/root/.mujoco/mujoco210",
            "LD_LIBRARY_PATH": "/root/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu",
        },
    )

    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")


@app.local_entrypoint()
def main():
    train.remote()