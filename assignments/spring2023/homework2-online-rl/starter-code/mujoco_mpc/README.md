## Overview
This codebase makes small modifications to the original MuJoCo MPC software application (https://github.com/deepmind/mujoco_mpc).

## Installation
We assume that you are running these steps in an AWS EC2 c4.4xlarge instance.

### Get clang-14
```
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 14
```

### Get cmake
```
sudo apt install cmake
```

### Install MJPC packages
```
sudo apt-get install libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev ninja-build
```

### Clone and build OpenCV
```
git clone https://github.com/opencv/opencv.git ~/opencv
cd ~/opencv
mkdir -p build && cd build
cmake ~/opencv -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang-14 -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++-14
make -j12
sudo make install
```

### Build MJPC
From the `hw2/mujoco_mpc` directory, run:
```
cmake -Bbuild/ -Smjpc/ -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang-14 -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++-14
cmake --build build/ --config Release --target mjpc -j 12 --
```

### Set up display
```
sudo apt install xvfb
xvfb-run -a -s "-screen 0 1400x900x24" bash
```

NOTE: You will need to rerun the second line each time you start a new session. If you see the following error
```
ERROR: could not initialize GLFW

Press Enter to exit ...
```
rerun `xvfb-run -a -s "-screen 0 1400x900x24" bash` to resolve it.

### Run an example
```
./build/bin/mjpc --task="Quadruped Flat" --steps=100 \
--horizon=0.35 --w0=0.0 --w1=0.0 --w2=0.0 --w3=0.0
```
