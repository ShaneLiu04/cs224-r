#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/cs229-r-hw2
ENV=/root/miniconda3

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  build-essential git libegl1-mesa libgl1-mesa-dev libgl1-mesa-glx \
  libglew-dev libglfw3 libglib2.0-0 libhdf5-dev libosmesa6-dev \
  libopengl0 patchelf unzip wget

"$ENV/bin/pip" install \
  --index-url https://pypi.org/simple \
  gymnasium scikit-image pandas matplotlib opencv-python \
  termcolor dm_control tensorboard wandb Cython \
  metaworld mujoco \
  imageio imageio-ffmpeg hydra-core==1.3.2 \
  hydra-submitit-launcher ipdb yapf \
  scikit-learn pytest seaborn

echo "AutoDL environment ready: $ENV"
