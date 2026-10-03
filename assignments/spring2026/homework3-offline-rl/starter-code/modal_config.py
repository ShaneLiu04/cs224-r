"""
Shared Modal infrastructure (app, image, volume, train function).
Imported by modal_train.py and modal_train_para.py.
No local entrypoints are defined here to avoid duplicate registration.
"""
import modal

app = modal.App("cs224r-hw3")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "curl",
        "git",
        "libegl1",
        "libgl1-mesa-glx",
        "libosmesa6-dev",
        "patchelf",
        "g++",
        "make",
        "swig",
        "libglfw3",
        "libglew-dev",
    )
    .run_commands(
        "mkdir -p /root/.mujoco",
        "curl -L -O https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz",
        "tar -xvf mujoco210-linux-x86_64.tar.gz -C /root/.mujoco",
        "rm mujoco210-linux-x86_64.tar.gz",
    )
    .env({
        "MUJOCO_GL": "egl",
        "D4RL_SUPPRESS_IMPORT_ERROR": "1",
        "MUJOCO_PY_MUJOCO_PATH": "/root/.mujoco/mujoco210",
        "LD_LIBRARY_PATH": "/root/.mujoco/mujoco210/bin:/usr/local/lib",
    })
    .pip_install(
        "torch==1.13.1",
        "numpy==1.26",
        "gym==0.23.1",
        "protobuf==3.20.1",
        "matplotlib==3.5.3",
        "moviepy==1.0.0",
        "pyvirtualdisplay==1.3.2",
        "opencv-python-headless",
        "networkx==2.5",
        "ipdb==0.13.3",
        "scipy",
        "imageio-ffmpeg==0.6.0",
        "Cython==0.29.37",
        "tqdm==4.67.3",
        "wandb==0.25.0",
        "dm_control==1.0.37",
        "mujoco-py==2.1.2.14",
        "D4RL @ git+https://github.com/Farama-Foundation/d4rl@89141a689b0353b0dac3da5cba60da4b1b16254d",
    )
    # add_local_dir must be last per Modal's requirement to avoid full rebuilds on every change.
    # Exclude data/ so the volume can mount cleanly at /root/hw/data.
    .add_local_dir(".", remote_path="/root/hw", ignore=["data/"])
)

# Persist training outputs (logs, checkpoints, videos) across container runs.
# Mounted at /root/hw/data to match the hardcoded data_path in run_algo.py.
volume = modal.Volume.from_name("cs224r-hw3-results", create_if_missing=True)

# Attach W&B secret when it exists.
# Create it once with: modal secret create wandb WANDB_API_KEY=<your_key>
# If the secret does not exist, remove this line and omit --use-wandb.
_secrets = [modal.Secret.from_name("wandb")]


@app.function(
    image=image,
    volumes={"/root/hw/data": volume},
    secrets=_secrets,
    gpu="T4",
    timeout=14400,
)
def train(algo, env_name, exp_name, extra_args):
    import os
    import subprocess

    os.chdir("/root/hw")

    # Install the local cs224r package so imports resolve.
    subprocess.run(["pip", "install", "-e", "."], check=True)

    cmd = [
        "python", "cs224r/scripts/run_algo.py",
        "--algo", algo,
        "--env_name", env_name,
        "--exp_name", exp_name,
    ] + extra_args

    subprocess.run(cmd, check=True)

    # Flush volume writes so results survive after the container shuts down.
    volume.commit()
