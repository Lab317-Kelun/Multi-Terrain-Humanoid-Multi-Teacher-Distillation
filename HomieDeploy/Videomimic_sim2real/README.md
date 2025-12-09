# VideoMimic Real

We provide our final deployment code for your convenience. Do note that this code is not thouroughly tested and cleaned up, but provide it as reference.

This deployment can be done on either the Jetson or a secondary computer connected to the G1. Most of the code currently assumes G1 but only minor edits should be required to get things running offboard.

Setup the Jetson on the Unitree with ROS 1 Noetic.

### Install Elevation Mapping Code

This is available [here](https://github.com/ArthurAllshire/elevation_mapping_humanoid), modified slightly from the original package [here](https://github.com/smoggy-P/elevation_mapping_humanoid).

### Get checkpoints

You can export your own checkpoints from the simulation 

Download the JIT checkpoints from [here](https://drive.google.com/drive/u/0/folders/1VzfDTnC4KbeNbzWW0l3LiZHu11ajDFhr). You will need to change the paths for `default_model_path` and `heightmap_model_path` in the [videomimic_inference_real](videomimic_real/videomimic_inference_real.cpp) script to wherever you extract them on your system (by default it assumes they are at `/home/unitree/Desktop/20250414_170842_g1_deepmimic_dict.pt` and `/home/unitree/Desktop/20250502_124756_g1_deepmimic_dict.pt`).

### Build examples

To build the examples inside this repository:

First make a build directory:
```bash
mkdir build
cd build
```

Then run cmake. I wanted to avoid using unitree's ROS stuff explicitly for my node for flexbility. For this reason, you you need to build it against ros manually. So you need to add ROS installation directory and devel folder to CMAKE_PREFIX_PATH. For torchscript we also need to build against torch and CUDA.

The following commands should work on the Jetson which ships by default on the G1 (you will need to change the ros workspace path to whatever you have):

```bash
export CMAKE_PREFIX_PATH=/home/unitree/.local/lib/python3.8/site-packages/torch:/home/unitree/noetic_ws/devel:/opt/ros/noetic
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CUDA_ARCHITECTURES="87" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

```bash
make
```

Now, you need to start the elevation mapping etc. Example of how to do this is in `scripts/start_robot.sh`. This script will also run the inference (again you will need to change the paths to wherever your ROS workspace is located). Now executable should be in `build/bin/videomimic_inference_real`. Note you need to pass in the interface (usually `eth0` if you are running locally on the robot.).
