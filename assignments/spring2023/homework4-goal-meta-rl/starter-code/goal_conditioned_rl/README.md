# Setup (AWS)
Consider using an AWS EC2 instance if local installation fails.

Create a c4.4xlarge instance. Follow the CS 224R AWS Guide which has instructions for setting up and accessing an AWS spot instance.

SSH into the virtual machine and copy the starter code. The machine may be in a conda environment by default. If this is the case, deactivate by running `conda deactivate`. To check the current conda environment, `conda env list` should have an asterisk in front of the `base` environment. Now, we can start the setup:

## (1) Installing MuJoCo:
For all the commands, change `mujoco200_linux` to `mujoco200_macos` if installing MuJoCo on a Mac device.
```bash
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
rm mujoco200_linux.zip
mkdir ~/.mujoco
mv mujoco200_linux ~/.mujoco/mujoco200
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

```

## (2) Copy MuJoCo Key
Place the mjkey.txt (from https://www.roboti.us/license.html) into `~/.mujoco/` folder.

## (3) Install dependencies:

Create the conda virtual environment, and install the dependencies in the requirements.txt.
```bash
conda create -n gcrl python=3.7
conda init bash
conda activate gcrl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco200/bin
pip install -r requirements.txt
```

Run `python test_installation.py` to ensure that `mujoco-py` was installed correctly.

# Troubleshooting
It is likely you will run into issues while installing on `mujoco-py`. Some of these commands are likely to help with installation issues on linux:
```bash
sudo apt update
sudo apt-get install libosmesa6-dev
sudo apt-get install patchelf
```
Then rerun
```bash
pip install -r requirements.txt
```

Run `python test_installation.py` to ensure that `mujoco-py` was installed correctly.