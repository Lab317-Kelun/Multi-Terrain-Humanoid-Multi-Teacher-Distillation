#!/bin/bash

conda create -n terrain python=3.8
conda activate terrain
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/shiki-ta/Humanoid-Terrain-Bench.git
cd Humanoid-Terrain-Bench
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
cd isaacgym/python && pip install -e .
cd rsl_rl && pip install -e .
cd legged_gym && pip install -e .
cd challenging_terrain && pip install -e .
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask