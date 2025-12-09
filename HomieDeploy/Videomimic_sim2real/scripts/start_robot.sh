# !/bin/bash

# ---- Launch the ROS node ----
# Source your ros workspaces, 
cd /home/unitree/catkin_ws || exit
source /opt/ros/noetic/setup.sh
source /home/unitree/noetic_ws/devel/setup.bash
source devel/setup.bash
export livox_ros_driver_DIR=/home/unitree/noetic_ws/devel/share/livox_ros_driver2/cmake


# Set these if you are running the scripts on a different machine to the robot
# export ROS_MASTER_URI=http://192.168.123.164:11311  
# export ROS_IP=192.168.123.164
# Start the ROS launch command in the background so the script can continue
# roslaunch elevation_mapping_demos realsense_demo.launch 

roslaunch elevation_mapping_demos realsense_demo.launch &

# wait some time before running real inference
sleep 20

# replace with path to your executable.
exec /home/unitree/code/vmg_sdk/build/bin/videomimic_inference_real eth0
