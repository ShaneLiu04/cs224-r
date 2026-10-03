#!/bin/bash
cd /home/ubuntu
sudo apt-get install unzip
cd /home/ubuntu/
mkdir .mujoco
cd .mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvf mujoco210-linux-x86_64.tar.gz
curl micro.mamba.pm/install.sh | bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
source /opt/conda/etc/profile.d/conda.sh
sudo apt-get update
sudo apt-get install libglew-dev
sudo apt install patchelf
sudo apt-get install libegl1-mesa
sudo apt-get install libgl1-mesa-glx
sudo apt install libopengl0
conda config --set channel_priority flexible 
cd /home/ubuntu/hw2/ac
mamba env create --file=conda_env.yml
conda init
source /home/ubuntu/.bashrc
conda activate PixelAC
git clone https://github.com/rlworkgroup/metaworld.git
cd metaworld
pip install -e .
cd /home/ubuntu/hw2
