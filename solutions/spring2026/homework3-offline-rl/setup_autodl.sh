#!/usr/bin/env bash
# HW3 environment on AutoDL: gym 0.23.1 + mujoco-py 2.1 + D4RL (pinned commit).
# The runs in artifacts/ were produced with /root/autodl-tmp/envs/hw3py
# (Python 3.12 venv --system-site-packages, torch 2.8.0+cu128). A separate
# Python 3.10 env could not finish the cu128 torch wheel download.
# Before training, export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# and MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210.
set -euo pipefail

ROOT=/root/autodl-tmp/cs229-r-hw3
ENV=/root/autodl-tmp/envs/hw3
PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
PIP="--index-url $PIP_INDEX --extra-index-url https://pypi.org/simple"
# AutoDL's GitHub accelerator; only enabled for GitHub downloads (it stalls the pip mirror).
turbo() { ( [ -f /etc/network_turbo ] && source /etc/network_turbo; "$@" ); }

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential git curl wget patchelf swig libegl1 libgl1-mesa-dev \
  libgl1-mesa-glx libglew-dev libglfw3 libosmesa6-dev libhdf5-dev > /dev/null

mkdir -p /root/.mujoco
if [ ! -d /root/.mujoco/mujoco210 ]; then
  turbo curl -fsSL -o /tmp/mujoco210.tar.gz \
    https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
  tar -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco
fi

if [ ! -x "$ENV/bin/python" ]; then
  /root/miniconda3/bin/conda create -y -q -p "$ENV" python=3.10
fi
PY="$ENV/bin/python"
"$PY" -m pip install -q $PIP --upgrade "pip<24.1" "setuptools<70" wheel
# PyPI torch 2.8 linux wheels are cu128 builds (needed for Blackwell sm_120).
"$PY" -m pip install -q $PIP torch==2.8.0
"$PY" -m pip install -q $PIP \
  numpy==1.26.4 gym==0.23.1 "Cython<3" mujoco-py==2.1.2.14 h5py termcolor click \
  matplotlib scipy networkx opencv-python-headless imageio imageio-ffmpeg tqdm wandb \
  pandas seaborn pytest protobuf==3.20.3
turbo "$PY" -m pip install -q --no-deps \
  "git+https://github.com/Farama-Foundation/d4rl@89141a689b0353b0dac3da5cba60da4b1b16254d"
"$PY" -m pip install -q --no-deps -e "$ROOT/code"

cat > "$ROOT/env.sh" <<EOF
export PATH=$ENV/bin:\$PATH
export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:\${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export D4RL_DATASET_DIR=/root/autodl-tmp/d4rl_datasets
export WANDB_MODE=disabled
EOF
source "$ROOT/env.sh"
"$PY" -c "import mujoco_py, d4rl, gym, torch; print('mujoco_py ok', torch.__version__)"
echo "HW3 environment ready: $ENV"
