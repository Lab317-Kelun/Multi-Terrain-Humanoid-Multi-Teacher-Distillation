cmake_minimum_required(VERSION 3.0.2)

# Find ROS packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs pcl_conversions tf2_ros tf2_eigen)

# Find LibTorch
find_package(Torch REQUIRED)

# Find Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# Find PCL
# Note: pcl_conversions often brings in core PCL, but let's be explicit
# You might need to adjust components based on your PCL installation
find_package(PCL REQUIRED COMPONENTS common kdtree search filters)

# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

# Note: Torch include directories are usually handled via target_link_libraries

# Add the new combined example
add_executable(videomimic_inference_real videomimic_inference_real.cpp)
target_link_libraries(videomimic_inference_real 
  ${catkin_LIBRARIES} 
  unitree_sdk2 
  ${TORCH_LIBRARIES} 
  ${PCL_LIBRARIES}
)
