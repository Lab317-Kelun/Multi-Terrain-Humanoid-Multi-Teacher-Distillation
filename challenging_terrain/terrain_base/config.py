class terrain_config:
    """PLY Mesh 加载配置"""
    
    # === 网格类型 ===
    mesh_type = "ply"  # 网格类型："ply"=从PLY文件加载, "None"=平地
    
    # === PLY文件加载配置 ===
    ply_path = "/home/cft/kelun/Humanoid-Terrain-Bench/mesh/small_scane.ply"  # PLY文件路径
    ply_scale = 1.0  # PLY mesh缩放因子（1.0=不缩放）
    ply_offset = None  # PLY mesh偏移量 [x, y, z]，None=不偏移
    
    # === 地形网格参数（PLY模式使用默认值） ===
    border_size = 0.0  # 边界大小（PLY模式不需要）
    horizontal_scale = 0.1  # 水平缩放（用于高度采样网格索引转换）
    vertical_scale = 0.005  # 垂直缩放（用于高度采样）
    
    # === 物理属性 ===
    static_friction = [0.4, 1.0]  # 静摩擦系数范围
    dynamic_friction = [0.4, 1.0]  # 动摩擦系数范围
    restitution = [0.0, 0.3]  # 弹性系数范围
    
    # === 高度测量配置 ===
    measure_heights = True  # 是否生成高度采样点
    measured_points_x = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    measured_points_y = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    measured_points_x_origin = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    measured_points_y_origin = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    # 高度测量的域随机化参数（模拟真实传感器误差）
    measure_horizontal_noise = 0.0  # 水平噪声幅度，单位：米
    measure_horizontal_offset = 0.0  # 水平偏移量，单位：米
    measure_vertical_offset = 0.03  # 垂直偏移量范围，单位：米
    measure_vertical_noise = 0.03  # 垂直噪声范围，单位：米
    measure_map_roll_pitch_noise = 0.03  # 地图倾斜噪声，单位：米
    measure_map_yaw_noise = 0.0  # 地图偏航噪声，单位：弧度
    map_repeat_prob = 0.2  # 地图更新延迟概率
    
    # === 数据集采样点配置 ===
    dataset_points_x = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dataset_points_y = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    # === 目标点配置 ===
    num_goals = 10  # 目标点数量
