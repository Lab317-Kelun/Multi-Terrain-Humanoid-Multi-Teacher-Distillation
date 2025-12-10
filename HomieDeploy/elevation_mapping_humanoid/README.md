# 高度图映射（Elevation Mapping）人形机器人版

## 概述

![](./figs/demo.gif)

本仓库提供了使用单个 MID-360 激光雷达为人形机器人生成高度图的实现。它主要基于 [Robot-Centric Elevation Mapping](https://github.com/ANYbotics/elevation_mapping) 和 [Fast Lio Mid360](https://github.com/SylarAnh/fast_lio_mid360)。该包在里程计坐标系中生成稳定、完整、平滑的高度图，可以进一步使用来自激光雷达、IMU 和机器人位姿的数据将其转换到 `torso_link` 坐标系。

## 功能说明

### 核心功能

1. **高度图生成**：从激光雷达点云数据生成 2.5D 高度图（elevation map）
2. **实时处理**：实时处理点云数据并更新高度图
3. **多传感器融合**：融合 LiDAR、IMU 和机器人位姿数据
4. **坐标系转换**：支持从里程计坐标系转换到机器人本体坐标系（torso_link）

### 主要组件

1. **fast_lio_mid360**：LiDAR-IMU 里程计，提供点云和位姿估计
2. **elevation_mapping**：核心高度图生成模块
3. **pose_publisher**：发布机器人位姿信息
4. **辅助工具**：点云过滤、坐标系发布等

### 工作流程

```
MID-360 LiDAR → Fast LIO → 点云 + 位姿 → Elevation Mapping → 高度图
                                                              ↓
                                                      ROS Topic 发布
                                                      /elevation_mapping/elevation_map
```

**注意**：本仓库的贡献主要在工程实现方面，大部分功劳应归功于下面列出的原始研究论文。如果您觉得本仓库有用，请考虑引用它们：

```bibtex
@article{fankhauser_probabilistic_2018,
	title = {Probabilistic Terrain Mapping for Mobile Robots With Uncertain Localization},
	volume = {3},
	url = {https://ieeexplore.ieee.org/document/8392399/},
	doi = {10.1109/LRA.2018.2849506},
	pages = {3019--3026},
	journaltitle = {{IEEE} Robotics and Automation Letters},
	author = {Fankhauser, Peter and Bloesch, Michael and Hutter, Marco},
	date = {2018-10}
}

@misc{xu_fast-lio_2021,
	title = {{FAST}-{LIO}: A Fast, Robust {LiDAR}-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter},
	url = {http://arxiv.org/abs/2010.08196},
	doi = {10.48550/arXiv.2010.08196},
	publisher = {{arXiv}},
	author = {Xu, Wei and Zhang, Fu},
	date = {2021-04-14}
}
```

本仓库已应用于以下论文：
```bibtex
@misc{long_learning_2024,
	title = {Learning Humanoid Locomotion with Perceptive Internal Model},
	url = {http://arxiv.org/abs/2411.14386},
	doi = {10.48550/arXiv.2411.14386},
	publisher = {{arXiv}},
	author = {Long, Junfeng and Ren, Junli and Shi, Moji and Wang, Zirui and Huang, Tao and Luo, Ping and Pang, Jiangmiao},
	date = {2024-11-21},
}

@misc{wang_beamdojo_2025,
	title = {{BeamDojo}: Learning Agile Humanoid Locomotion on Sparse Footholds},
	url = {http://arxiv.org/abs/2502.10363},
	doi = {10.48550/arXiv.2502.10363},
	publisher = {{arXiv}},
	author = {Wang, Huayi and Wang, Zirui and Ren, Junli and Ben, Qingwei and Huang, Tao and Zhang, Weinan and Pang, Jiangmiao},
	date = {2025-02-14},
}

@misc{ren_vb-com_2025,
	title = {{VB}-Com: Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception},
	url = {http://arxiv.org/abs/2502.14814},
	doi = {10.48550/arXiv.2502.14814},
	publisher = {{arXiv}},
	author = {Ren, Junli and Huang, Tao and Wang, Huayi and Wang, Zirui and Ben, Qingwei and Pang, Jiangmiao and Luo, Ping},
	date = {2025-02-20},
}
```
## 安装

### 依赖项

本包基于机器人操作系统（[ROS](http://www.ros.org)），需要先[安装 ROS](http://wiki.ros.org)。

此外，还依赖以下包：

- [Grid Map](https://github.com/anybotics/grid_map) (grid map library for mobile robots)
- [kindr](http://github.com/anybotics/kindr) (kinematics and dynamics library for robotics)
- [Point Cloud Library (PCL)](http://pointclouds.org/) (point cloud processing)
- [Eigen](http://eigen.tuxfamily.org) (linear algebra library)
- [Livox Ros Driver](https://github.com/Livox-SDK/livox_ros_driver) or [Livox Ros Driver2](https://github.com/Livox-SDK/livox_ros_driver2) (driver packages for connecting Livox LiDARs)

### 编译

1. **编译 Livox ROS 驱动：**
   
   按照 [Livox ROS Driver 文档](https://github.com/Livox-SDK/livox_ros_driver?tab=readme-ov-file#livox-ros-driver%E8%A7%88%E6%B2%83ros%E9%A9%B1%E5%8A%A8%E7%A8%8B%E5%BA%8F%E4%B8%AD%E6%96%87%E8%AF%B4%E6%98%8E) 或 [Livox ROS Driver2 文档](https://github.com/Livox-SDK/livox_ros_driver2?tab=readme-ov-file#livox-ros-driver-2) 中的说明进行编译。编译完成后，source Livox ROS 驱动：

    ```
        source $Livox_ros_driver_dir$/devel/setup.bash
    ```


2. **克隆并编译本仓库：**
   
   将本仓库克隆到您的 catkin 工作空间并编译：

    ```
        cd catkin_workspace/src
        git clone https://github.com/smoggy-P/elevation_mapping_humanoid.git
        cd ../
        catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
        catkin build
        source devel/setup.bash
    ```

    **注意**：如果您使用 Livox ROS Driver2，请在编译前将以下文件中的所有 `livox_ros_driver` 替换为 `livox_ros_driver2`：

   - `fast_lio_mid360/src/preprocess.h`
   - `fast_lio_mid360/src/preprocess.cpp`
   - `fast_lio_mid360/src/laserMapping.cpp`

## 运行

启动高度图映射：

```
roslaunch elevation_mapping_demos realsense_demo.launch
```

一旦进程成功启动，您可以在 `rviz` 中可视化高度图。此外，通过检查 `/elevation_mapping/elevation_map` 话题来确保高度图数据发布正确。该话题应包含有效数据（即不是所有 `NaN` 值）：

```
rostopic echo /elevation_mapping/elevation_map
```

## 适配到您的机器人

要将本包适配到您的机器人，请修改 [publish_tf.py](./fast_lio_mid360/script/publish_tf.py) 脚本中的静态变换，以匹配您的机器人配置。

## ROS 话题

### 订阅的话题

- `/cloud_registered`：注册后的点云数据（来自 Fast LIO）
- `/odometry`：机器人里程计信息
- `/pose`：机器人位姿信息

### 发布的话题

- `/elevation_mapping/elevation_map`：生成的高度图（grid_map_msgs/GridMap 类型）
- `/elevation_mapping/raw_elevation_map`：原始高度图数据

## 集成到策略网络

高度图数据可以通过以下方式集成到您的策略网络中：

1. **订阅 ROS 话题**：在 Python 代码中订阅 `/elevation_mapping/elevation_map`
2. **提取高度值**：从 GridMap 消息中提取高度值
3. **转换为观测向量**：将高度图转换为策略网络所需的观测格式（例如 225 维高度扫描点）

详细说明请参考 [代码功能说明.md](./代码功能说明.md) 和 [README_CN.md](./README_CN.md)。
