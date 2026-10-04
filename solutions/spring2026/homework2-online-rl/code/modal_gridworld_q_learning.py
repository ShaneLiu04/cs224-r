import modal
import subprocess

app = modal.App("cs224r-hw2-gridworld")
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
        "torch",
        "torchvision",
        "torchaudio",
    )
    .pip_install(
        "jax-jumpy",
        "numpy==1.24", 
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
        "imageio>=2.33.0",
        "imageio-ffmpeg>=0.4.4",
        "hydra-core==1.1.0",
        "hydra-submitit-launcher==1.1.5",
        "yapf==0.31.0",
        "mujoco_py==2.1.2.14",
        "scikit-learn"    
    )
    .add_local_python_source("gridworld_q_learning")
)


@app.function(
    image=image,
    volumes={"/output": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    gpu="A10",
    timeout=3600,
)
def train():
    import os
    import yaml

    subprocess.run(["find", "/root", "-name", "gridworld_q_learning*"], check=True)
    subprocess.run(["python", "-c", "import gridworld_q_learning; print(gridworld_q_learning.__file__)"], cwd="/root")
    
    result = subprocess.run(
        [
            "python", "-u", "gridworld_q_learning.py",
        ],
        cwd="/root",
        capture_output=False,
    )

    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")


@app.local_entrypoint()
def main():
    train.remote()