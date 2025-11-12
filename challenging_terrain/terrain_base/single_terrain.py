import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from .config import terrain_config
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation
import math

class single_terrain:
    def __init__(self, cfg: terrain_config) -> None:
        self.cfg = cfg
    
    def parkour(terrain, 
            length_x=18.,
            length_y=4.,
            num_goals=6, 
            start_x=0,
            start_y=0,
            platform_size=2.5, 
            difficulty=0.5,
            x_range=[0.5, 1.0],
            y_range=[0.3, 0.4],
            stone_len_range=[0.8, 1.0],
            stone_width_range=[0.6, 0.8],
            incline_height=0.1,
            pit_depth=[0.5, 1.]):
    
        goals = np.zeros((num_goals, 2))
        pit_depth_val = np.random.uniform(pit_depth[0], pit_depth[1])
        pit_depth_grid = -round(pit_depth_val / terrain.vertical_scale)
        
        h_scale = terrain.horizontal_scale
        v_scale = terrain.vertical_scale
    
        length_y_grid = round(length_y / h_scale)
        mid_y = length_y_grid // 2

        length_x_grid = round(length_x / h_scale)
        
        stone_len = round(((stone_len_range[0] - stone_len_range[1]) * difficulty + stone_len_range[1]) / h_scale)
        stone_width = round(((stone_width_range[0] - stone_width_range[1]) * difficulty + stone_width_range[1]) / h_scale)
        gap_x = round(((x_range[1] - x_range[0]) * difficulty + x_range[0]) / h_scale)
        gap_y = round(((y_range[1] - y_range[0]) * difficulty + y_range[0]) / h_scale)
        
        platform_size_grid = int(round(platform_size / h_scale))
        incline_height_grid = int(round(incline_height / v_scale))
        
        terrain.height_field_raw[start_x+platform_size_grid:start_x + length_x_grid, start_y:start_y+length_y_grid*2] = pit_depth_grid
        
        dis_x = start_x +platform_size_grid - gap_x + stone_len // 2
        goals[0] = [start_x + platform_size_grid - stone_len // 2, start_y + mid_y]
        left_right_flag = np.random.randint(0, 2)
        
        for i in range(num_goals - 2):
            dis_x += gap_x
            pos_neg = 2 * (left_right_flag - 0.5)  # 1 或 -1
            dis_y = mid_y + pos_neg * gap_y
            
            x_start = int(dis_x - stone_len // 2)
            x_end = x_start + stone_len
            y_start = int(dis_y - stone_width // 2)
            y_end = y_start + stone_width
            
            heights = np.tile(np.linspace(-incline_height_grid, incline_height_grid, stone_width),(stone_len, 1)) * pos_neg
            heights = heights.astype(int)
            
            if x_end > terrain.height_field_raw.shape[0]:
                x_end = terrain.height_field_raw.shape[0]
            if y_end > terrain.height_field_raw.shape[1]:
                y_end = terrain.height_field_raw.shape[1]
    
            actual_height = heights[:x_end - x_start, :y_end - y_start]
            terrain.height_field_raw[x_start:x_end, y_start:y_end] = actual_height
            goals[i + 1] = [dis_x, dis_y]
            left_right_flag = 1 - left_right_flag
        
        final_dis_x = dis_x + gap_x
        goals[-1] = [final_dis_x, mid_y]

        # terrain.height_field_raw[final_dis_x:round(length_x/terrain.horizontal_scale), start_y:start_y+mid_y*2] = 0
        return terrain, goals, final_dis_x
    
    def hurdle(
            terrain,
            length_x=18.,
            length_y=4.,
            num_goals=8,
            start_x=0,
            start_y=0,
            platform_size=1., 
            difficulty = 0.5,
            hurdle_range=[0.1, 0.2],
            hurdle_height_range=[0.05, 0.15],
            flat_size = 0.6
            ):
        
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale)// 2  
        per_x = (round(length_x/ terrain.horizontal_scale)- platform_size) // num_goals


        hurdle_size = round(((hurdle_range[1]-hurdle_range[0])*difficulty +hurdle_range[0])/terrain.horizontal_scale)
        hurdle_height = round(((hurdle_height_range[1]-hurdle_height_range[0])*difficulty + hurdle_height_range[0])/terrain.vertical_scale)

        platform_size = round(platform_size / terrain.horizontal_scale)
        # terrain.height_field_raw[start_x:start_x+platform_size, start_y:start_y+2*mid_y] = 0

        terrain.height_field_raw[start_x:start_x +round(length_x/ terrain.horizontal_scale), start_y:start_y+mid_y*2] = 0

        flat_size = round(flat_size / terrain.horizontal_scale)
        dis_x = start_x + platform_size

        for i in range(num_goals):
            goals[i]=[dis_x+per_x*i,start_y+mid_y]

        for i in range(num_goals):

            terrain.height_field_raw[dis_x-hurdle_size//2:dis_x+hurdle_size//2, start_y:start_y+mid_y*2] = hurdle_height
            dis_x += flat_size + hurdle_size

        return terrain,goals,dis_x
        
    def bridge(terrain,
               length_x=18.0,
                length_y=4.0,
                num_goals=8,
                start_x = 0,
                start_y = 0,
                platform_size=1.0, 
                difficulty = 0.5,
                bridge_width_range=[0.3,0.4],  
                bridge_height=0.7,
                ):
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y / terrain.horizontal_scale) // 2  
        bridge_width = round(((bridge_width_range[1]-bridge_width_range[0])*difficulty +bridge_width_range[0])/terrain.horizontal_scale)
        bridge_height = round(bridge_height / terrain.vertical_scale)
        platform_size = round(platform_size / terrain.horizontal_scale)
        terrain.height_field_raw[start_x:start_x+platform_size, start_y:start_y+2*mid_y] = 0
        bridge_start_x = platform_size + start_x
        bridge_length = round(length_x / terrain.horizontal_scale)
        bridge_end_x = start_x + bridge_length

        for i in range(num_goals):
            goals[i] = [bridge_start_x + bridge_length/num_goals*i, mid_y]  
       
        left_y1 = 0
        left_y2 = int(mid_y - bridge_width // 2) 
        right_y1 = int(mid_y + bridge_width // 2)
        right_y2 = mid_y*2
        terrain.height_field_raw[bridge_start_x:bridge_end_x, left_y1:left_y2] = -bridge_height
        terrain.height_field_raw[bridge_start_x:bridge_end_x, right_y1:right_y2] = -bridge_height

        # terrain.height_field_raw[bridge_start_x:bridge_end_x, left_y2:right_y1] = 0

        return terrain,goals,bridge_end_x

    def flat(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            ):
        goals = np.zeros((num_goals, 2))
        length_x = round(length_x / terrain.horizontal_scale)
        length_y = round(length_y / terrain.horizontal_scale)
        platform_size = round(platform_size / terrain.horizontal_scale)

        for i in range(num_goals):
            # y_pos = round(random.uniform(0,length_y))
            y_pos = length_y//2
            goals[i]=[start_x+platform_size+length_x/num_goals*i,start_y+y_pos]

        return terrain,goals,length_x

    def uneven(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            num_range=[150,200],
            size_range=[0.4,0.7],
            height_range=[0.1,0.2],
            ):   

        goals = np.zeros((num_goals, 2))
        platform_size = round(platform_size/ terrain.horizontal_scale)
        per_x = (round(length_x/ terrain.horizontal_scale) - platform_size)// num_goals
        mid_y = round(length_y/ terrain.horizontal_scale) // 2

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+per_x*i,start_y+mid_y]

        height = round(((height_range[1]-height_range[0])*difficulty + height_range[0])/terrain.vertical_scale)


        min_size = round(size_range[0]/ terrain.horizontal_scale)
        max_size = round(size_range[1]/ terrain.horizontal_scale)

        discrete_start_x = start_x+platform_size
        discrete_start_y = start_y

        discrete_end_x = discrete_start_x +round(length_x/ terrain.horizontal_scale) - platform_size
        discrete_end_y = discrete_start_y +round(length_y/ terrain.horizontal_scale)

        num_rects = round((num_range[1]-num_range[0])*difficulty + num_range[0])

        for _ in range(num_rects):
            width = round(random.uniform(min_size, max_size))
            length = round(random.uniform(min_size, max_size))
            start_i = round(random.uniform(discrete_start_x, discrete_end_x-width))
            start_j = round(random.uniform(discrete_start_y, discrete_end_y-length))

            terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = random.uniform(-height//2, height)

        terrain.height_field_raw[start_x:start_x+platform_size , start_y:start_y+mid_y*2] = 0
        terrain.height_field_raw[discrete_end_x:discrete_end_x+platform_size , start_y:start_y+mid_y*2] = 0

        return terrain,goals,discrete_end_x+platform_size

    def stair(terrain,
                length_x=18.0,
                length_y=4.0,
                num_goals=8,
                start_x = 0,
                start_y = 0,
                platform_size=1.0, 
                difficulty = 0.5,
                height_range=[0.08,0.2],
                size_range=[0.4,0.5],
                upstair = True,
                start_z = 3.0
                ):

        goals = np.zeros((num_goals, 2))
        platform_size = round(platform_size/ terrain.horizontal_scale)
        per_x = (round(length_x/ terrain.horizontal_scale)- platform_size) // num_goals
        per_y = round(length_y/ terrain.horizontal_scale) // 2
        step_height = round(((height_range[1]-height_range[0])*difficulty + height_range[0])/terrain.vertical_scale)
        step_x = round(((size_range[0]-size_range[1])*difficulty +size_range[1])/terrain.horizontal_scale)

        if(upstair):
            total_step_height = 0
        else:
            total_step_height = round(start_z/terrain.vertical_scale)

        dis_x = start_x + platform_size

        for i in range(num_goals):
            goals[i]=[dis_x+per_x*i,start_y+per_y]

        for i in range(num_goals):
            if(upstair):
                total_step_height += step_height
            else :
                total_step_height -= step_height

            terrain.height_field_raw[dis_x : dis_x + step_x, start_y : start_y + per_y*2] = total_step_height
            dis_x += step_x

        # terrain.height_field_raw[start_x:start_x+platform_size,start_y:start_y + per_y*2] = 0
        terrain.height_field_raw[dis_x:start_x+round(length_x/ terrain.horizontal_scale),start_y:start_y + per_y*2] = total_step_height

        return terrain,goals,start_x+round(length_x/ terrain.horizontal_scale)

    def wave(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            amplitude_range=[0.05,0.1]
            ):   
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale) //2
        platform_size = round(1.5/ terrain.horizontal_scale)
        mid_x =  (round(length_x/ terrain.horizontal_scale) - platform_size)// num_goals

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+mid_x*i,start_y+mid_y]
        
        x_indices = np.arange(start_x, start_x + mid_x*num_goals + platform_size)
        amplitude = round(((amplitude_range[1]-amplitude_range[0])*difficulty + amplitude_range[0])/terrain.vertical_scale)
        wave_pattern = amplitude * np.sin(2 * np.pi * x_indices / length_x)

        for i, wave_height in enumerate(wave_pattern):
            terrain.height_field_raw[x_indices[i], start_y:start_y +mid_y*2] = wave_height

        terrain.height_field_raw[start_x :start_x + platform_size, start_y:start_y+ mid_y*2] = 0

        return terrain,goals,start_x+mid_x*num_goals

    def slope(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            angle_range = [4.1,10.0],
            uphill=True
            ):    

        goals = np.zeros((num_goals, 2))
        length_x_grid = round((length_x - platform_size) / terrain.horizontal_scale)
        length_y_grid = round(length_y / terrain.horizontal_scale)
        platform_size = round(platform_size/ terrain.horizontal_scale)

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+length_x_grid/num_goals*i,start_y+length_y_grid//2]

        slope_angle = (angle_range[1]-angle_range[0])*difficulty + angle_range[0]
        angle_rad = math.radians(slope_angle)
        total_height = length_x * math.tan(angle_rad)

        total_height_units = total_height / terrain.vertical_scale

        start_x += platform_size

        for x in range(start_x, start_x + length_x_grid):
            progress = (x - start_x) / length_x_grid
            if uphill:
                height = progress * total_height_units
            else:
                height = (1 - progress) * total_height_units
            terrain.height_field_raw[x, start_y:start_y + length_y_grid] = round(height)
        
        return terrain,goals,start_x + length_x_grid

    def gap(terrain,
            length_x=18.0,        # 地形总长度(米)
            length_y=4.0,         # 地形总宽度(米)
            num_goals=8,          # 目标点数量
            start_x=0,            # 起始X坐标(网格单位)
            start_y=0,            # 起始Y坐标(网格单位)
            platform_size=1.0,    # 起始平台大小(米)
            difficulty=0.5,       # 难度系数(0-1) → 映射到等级0-8
            gap_depth=1.0,        # 间隙深度(米)
            ):
        """
        BeamDojo标准GAP地形 - 按课程等级生成间隙地形
        
        难度等级(0-8)映射:
        - Level 0: 平台0.7m, 间隙0.1m  (最简单)
        - Level 4: 平台0.35m, 间隙0.3m (中等)
        - Level 8: 平台0.2m, 间隙0.5m  (最难)
        
        参数:
            difficulty: 0.0~1.0 映射到等级0~8
            platform_size: 起始平台大小
            gap_depth: 间隙深度(掉落惩罚)
        """
        
        # 初始化目标点数组
        goals = np.zeros((num_goals, 2))
        
        # 转换基本参数为网格单位
        length_x_grid = round(length_x / terrain.horizontal_scale)
        length_y_grid = round(length_y / terrain.horizontal_scale)
        init_platform_grid = round(platform_size / terrain.horizontal_scale)
        gap_depth_grid = round(gap_depth / terrain.vertical_scale)
        
        # 计算Y方向中心线
        mid_y = start_y + length_y_grid // 2
        
        # === BeamDojo课程等级参数 ===
        # 平台尺寸数组（按难度等级0-8）- 来自BeamDojo论文
        platform_sizes = [0.7, 0.65, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2]
        
        # 根据difficulty计算当前等级
        difficulty_level = min(8, int(difficulty * 8))  # 映射到0-8
        
        # 当前等级对应的平台尺寸和间距
        current_platform_size = platform_sizes[difficulty_level]
        max_gap_distance = 0.1 + 0.05 * difficulty_level  # 间隙宽度: 0.1m → 0.5m
        
        # 转换为网格单位
        platform_size_grid = round(current_platform_size / terrain.horizontal_scale)
        gap_distance_grid = round(max_gap_distance / terrain.horizontal_scale)
        
        # === 第1步: 先将整个区域填充为深坑(背景) ===
        terrain.height_field_raw[start_x:start_x + length_x_grid, 
                                start_y:start_y + length_y_grid] = -gap_depth_grid
        
        # === 第2步: 创建起始平台(平地) ===
        terrain.height_field_raw[start_x:start_x + init_platform_grid, 
                                start_y:start_y + length_y_grid] = 0
        
        # === 第3步: 按照BeamDojo标准生成平台序列 ===
        current_x = start_x + init_platform_grid
        goal_idx = 0
        
        # 循环生成: [平台] → [间隙] → [平台] → [间隙] → ...
        # 注意: 持续生成平台直到填满整个区域,不受目标点数量限制
        while current_x < start_x + length_x_grid:
            # 创建平台(高度=0)
            platform_end = min(current_x + platform_size_grid, start_x + length_x_grid)
            terrain.height_field_raw[current_x:platform_end, 
                                    start_y:start_y + length_y_grid] = 0
            
            # 在平台中心放置目标点(只在还需要目标点时)
            if goal_idx < num_goals:
                platform_center_x = current_x + platform_size_grid // 2
                if platform_center_x < start_x + length_x_grid:
                    goals[goal_idx] = [platform_center_x, mid_y]
                    goal_idx += 1
            
            # 移动到下一个位置
            current_x += platform_size_grid
            
            # 创建间隙(保持深坑,高度=-gap_depth_grid)
            if current_x < start_x + length_x_grid:
                gap_end = min(current_x + gap_distance_grid, start_x + length_x_grid)
                # 间隙区域已经是深坑了,不需要额外操作
                current_x = gap_end
        
        # === 第4步: 如果目标点不足(不应该发生),补齐到num_goals ===
        if goal_idx < num_goals:
            print(f"Warning: Only generated {goal_idx}/{num_goals} goals in gap terrain")
            # 用最后一个有效目标点复制填充
            if goal_idx > 0:
                last_goal = goals[goal_idx - 1]
                for i in range(goal_idx, num_goals):
                    goals[i] = last_goal
        
        return terrain, goals, start_x + length_x_grid
    
    def plot(terrain,
            length_x=18.,
            length_y=4.,
            num_goals=8,
            start_x=0,
            start_y=0,
            platform_size=1., 
            difficulty = 0.5,
            hurdle_range=[0.1, 0.15],
            hurdle_height = 1.2,
            flat_size = 1.0
            ):
        
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale)// 2  
        per_x = (round(length_x/ terrain.horizontal_scale)- platform_size) // num_goals


        hurdle_size = round(((hurdle_range[1]-hurdle_range[0])*difficulty +hurdle_range[0])/terrain.horizontal_scale)// 2
        hurdle_height = round(hurdle_height/terrain.vertical_scale)

        platform_size = round(platform_size / terrain.horizontal_scale)
        # terrain.height_field_raw[start_x:start_x+platform_size, start_y:start_y+2*mid_y] = 0

        terrain.height_field_raw[start_x:start_x +round(length_x/ terrain.horizontal_scale), start_y:start_y+mid_y*2] = 0

        flat_size = round(flat_size / terrain.horizontal_scale)
        dis_x = start_x + platform_size

        for i in range(num_goals):
            goals[i]=[dis_x+per_x*i,start_y+mid_y]

        for i in range(num_goals):

            terrain.height_field_raw[dis_x-hurdle_size:dis_x+hurdle_size, start_y+mid_y - hurdle_size:start_y+mid_y + hurdle_size] = hurdle_height
            dis_x += flat_size + hurdle_size * 2

        return terrain,goals,dis_x
    
    def flat_gap(terrain,
                 length_x=18.0,       # 地形总长度(米) - X方向
                 length_y=8.0,        # 地形总宽度(米) - Y方向
                 num_goals=8,         # 目标点数量
                 start_x=0,           # 起始X坐标(网格单位)
                 start_y=0,           # 起始Y坐标(网格单位)
                 platform_size=1.0,   # 起始平台大小(米)
                 difficulty=0.5,      # 难度系数(0-1)
                 gap_depth=1.0,       # 间隙深度(米)
                 middle_gap_width=1.0,  # 中间间隔宽度(米) - 左右区域之间的缓冲区
                 ):
        
        # 初始化目标点数组
        goals = np.zeros((num_goals, 2))
        
        # 计算基本参数
        length_x_grid = round(length_x / terrain.horizontal_scale)
        length_y_grid = round(length_y / terrain.horizontal_scale)
        # 起始平台长度（平地），左右两侧使用相同的平台
        init_platform_grid = round(platform_size / terrain.horizontal_scale)
        middle_gap_grid = round(middle_gap_width / terrain.horizontal_scale)  # 中间间隔
        
        # Y方向分成三个区域：左边 + 中间间隔 + 右边
        # 可用宽度 = 总宽度 - 中间间隔
        usable_width = length_y_grid - middle_gap_grid
        left_width = usable_width // 2  # 左边区域宽度
        right_width = usable_width - left_width  # 右边区域宽度（可能略有不同）
        
        # === 左边区域：平地（机器人行走区域） ===
        left_y_start = start_y
        left_y_end = start_y + left_width
        
        # 整个左边区域设置为平地（高度为0）
        terrain.height_field_raw[start_x:start_x + length_x_grid, 
                                left_y_start:left_y_end] = 0
        
        # 设置起始平台（左边区域的起点，保持为平地以强调"起步区"）
        terrain.height_field_raw[start_x:start_x + init_platform_grid, 
                                left_y_start:left_y_end] = 0
        
        # 计算目标点位置（都在左边区域的中心线上）
        left_mid_y = start_y + left_width // 2  # 左边区域的中心Y坐标
        per_x = (length_x_grid - init_platform_grid) // num_goals
        for i in range(num_goals):
            goals[i] = [start_x + init_platform_grid + per_x * i, left_mid_y]
            
        # 保存左边区域的中心Y坐标（用于环境原点定位）
        terrain.left_region_center_y = left_mid_y * terrain.horizontal_scale  # 转换为米

        # === 中间区域：平地间隔（缓冲区） ===
        middle_y_start = start_y + left_width
        middle_y_end = middle_y_start + middle_gap_grid
        
        # 中间区域也设置为平地
        terrain.height_field_raw[start_x:start_x + length_x_grid,
                                middle_y_start:middle_y_end] = 0
        
        # === 右边区域：BeamDojo标准GAP地形 ===
        gap_y_start = middle_y_end
        gap_y_end = start_y + length_y_grid
        
        # 先在右侧放置一个起始平地平台，然后再开始BeamDojo标准GAP地形
        terrain.height_field_raw[start_x:start_x + init_platform_grid,
                                gap_y_start:gap_y_end] = 0
        
        # 平台尺寸数组（按难度等级0-8对应）
        platform_sizes = [0.7, 0.65, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2]
        # 转换为网格单位
        gap_depth_grid = round(gap_depth / terrain.vertical_scale)
        
        # 根据难度等级计算当前平台尺寸和间距
        difficulty_level = min(8, int(difficulty * 8))  # 0-8级
        current_platform_size = platform_sizes[difficulty_level]  # 当前难度对应的平台尺寸
        max_gap_distance = 0.1 + 0.05 * difficulty_level  # 最大平台间距
        
        beam_platform_size_grid = round(current_platform_size / terrain.horizontal_scale)
        gap_distance_grid = round(max_gap_distance / terrain.horizontal_scale)
        
        # 在右边区域创建BeamDojo标准GAP地形
        # 从右侧起始平地平台之后开始生成
        current_x = start_x + init_platform_grid
        
        while current_x + beam_platform_size_grid < start_x + length_x_grid:
            # 创建平台
            terrain.height_field_raw[current_x:current_x + beam_platform_size_grid, 
                                    gap_y_start:gap_y_end] = 0
            # 移动到下一个平台位置
            current_x += beam_platform_size_grid
            # 如果还有空间，创建间隙
            if current_x < start_x + length_x_grid:
                gap_size = min(gap_distance_grid, start_x + length_x_grid - current_x)
                terrain.height_field_raw[current_x:current_x + gap_size, 
                                        gap_y_start:gap_y_end] = -gap_depth_grid
                current_x += gap_size
                
        return terrain, goals, start_x + length_x_grid

    def flat_Stones_Everywhere(terrain,
                    length_x=18.0,       # 地形总长度(米) - X方向
                    length_y=8.0,        # 地形总宽度(米) - Y方向
                    num_goals=8,         # 目标点数量
                    start_x=0,           # 起始X坐标(网格单位)
                    start_y=0,           # 起始Y坐标(网格单位)
                    platform_size=1.0,   # 起始平台大小(米)
                    difficulty=0.5,      # 难度系数(0-1)
                    gap_depth=1.0,       # 间隙深度(米)
                    middle_gap_width=1.0,  # 中间间隔宽度(米) - 左右区域之间的缓冲区
                    ):
        """
        组合地形：左边平地起点 + 右边Stones Everywhere
        
        布局类似flat_gap：
        - 左边：平地（机器人行走），从起点开始
        - 右边：Stones Everywhere地形（机器人观测）
        
        训练目标：机器人在左边平地行走，观测右边石头地形
        """
        
        # 初始化目标点数组
        goals = np.zeros((num_goals, 2))
        
        # 计算基本参数
        length_x_grid = round(length_x / terrain.horizontal_scale)
        length_y_grid = round(length_y / terrain.horizontal_scale)
        init_platform_grid = round(platform_size / terrain.horizontal_scale)  # 起始平台
        middle_gap_grid = round(middle_gap_width / terrain.horizontal_scale)
        
        # Y方向分成三个区域：左边平地 + 中间间隔 + 右边Stones
        usable_width = length_y_grid - middle_gap_grid
        left_width = usable_width // 2
        
        # === 左边区域：平地（机器人实际行走区域） ===
        left_y_start = start_y
        left_y_end = start_y + left_width
        
        # 整个左边区域设为平地（高度=0）
        terrain.height_field_raw[start_x:start_x + length_x_grid, 
                                left_y_start:left_y_end] = 0
        
        # 设置起始平台（左边区域的起点）
        terrain.height_field_raw[start_x:start_x + init_platform_grid, 
                                left_y_start:left_y_end] = 0
        
        # 计算目标点位置（都在左边区域的中心线上，从起始平台后开始）
        left_mid_y = start_y + left_width // 2  # 左边区域的中心Y坐标
        per_x = (length_x_grid - init_platform_grid) // num_goals
        for i in range(num_goals):
            goals[i] = [start_x + init_platform_grid + per_x * i, left_mid_y]
            
        # 保存左边区域的中心Y坐标（用于环境原点定位）
        terrain.left_region_center_y = left_mid_y * terrain.horizontal_scale

        # === 中间区域：平地间隔（缓冲区） ===
        middle_y_start = start_y + left_width
        middle_y_end = middle_y_start + middle_gap_grid
        terrain.height_field_raw[start_x:start_x + length_x_grid,
                                middle_y_start:middle_y_end] = 0
        
        # === 右边区域：Stones Everywhere 地形（机器人观测区） ===
        stones_y_start = middle_y_end
        stones_y_end = start_y + length_y_grid
        
        # 先将整个右边区域设为深坑（背景）
        gap_depth_grid = round(gap_depth / terrain.vertical_scale)
        terrain.height_field_raw[start_x:start_x + length_x_grid,
                                stones_y_start:stones_y_end] = -gap_depth_grid
        
        # === Stones Everywhere 参数（来自论文BeamDojo） ===
        difficulty_level = min(8, int(difficulty * 8))  # 难度等级 l ∈ [0, 8]
        
        # 石块尺寸：max{0.25, 1.5(1 - 0.1l)}
        stone_size = max(0.25, 1.5 * (1.0 - 0.1 * difficulty_level))
        stone_size_grid = round(stone_size / terrain.horizontal_scale)
        
        # 石块间距：0.05 × ⌈l/2⌉
        stone_distance = 0.05 * np.ceil(difficulty_level / 2.0)
        stone_distance_grid = round(stone_distance / terrain.horizontal_scale)
        
        # 子网格尺寸 = 石块尺寸 + 间距
        subgrid_size = stone_size_grid + stone_distance_grid
        
        # 在右边区域放置石块"岛屿"（尽可能密集填充整个右边区域）
        current_x = start_x
        # 放宽条件：只要石块起点在范围内就放置（允许部分越界）
        while current_x < start_x + length_x_grid:
            current_y = stones_y_start
            while current_y < stones_y_end:
                # 计算实际可以放置的石块尺寸（避免越界）
                actual_stone_x = min(stone_size_grid, start_x + length_x_grid - current_x)
                actual_stone_y = min(stone_size_grid, stones_y_end - current_y)
                
                # 只有当石块足够大时才创建（至少要有原尺寸的30%）
                if actual_stone_x >= stone_size_grid * 0.3 and actual_stone_y >= stone_size_grid * 0.3:
                    # 创建石块平台（高度=0）
                    terrain.height_field_raw[current_x:current_x + actual_stone_x,
                                            current_y:current_y + actual_stone_y] = 0
                current_y += subgrid_size
            current_x += subgrid_size
        
        return terrain, goals, start_x + length_x_grid

    def flat_Stones_Middle_Start(terrain,
                    length_x=10.0,       # 地形总长度(米) - X方向：左边8 + 中间1 + 右边8
                    length_y=21.0,        # 地形总宽度(米) - Y方向：8米
                    num_goals=6,         # 目标点数量
                    start_x=0,           # 起始X坐标(网格单位)
                    start_y=0,           # 起始Y坐标(网格单位)
                    platform_size=1.0,   # 起始平台大小(米)
                    difficulty=0.5,      # 难度系数(0-1)
                    gap_depth=1.0,       # 间隙深度(米)
                    middle_gap_width=1.0,  # 中间间隔宽度(米) - 左右区域之间的缓冲区
                    ):
        """
        组合地形：机器人出生在左边平地中心，右边石头地形有小平地
        
        布局：
        - 左边：平地（机器人出生在中心区域）
        - 中间：平地缓冲区
        - 右边：Stones Everywhere地形 + 中心小平地（观测区）
        
        训练目标：机器人在左边平地行走，观测右边的石头地形
        """
        
        # 初始化目标点数组
        goals = np.zeros((num_goals, 2))
        
        # 计算基本参数
        length_x_grid = round(length_x / terrain.horizontal_scale)
        length_y_grid = round(length_y / terrain.horizontal_scale)
        init_platform_grid = round(platform_size / terrain.horizontal_scale)  # 起始平台
        middle_gap_grid = round(middle_gap_width / terrain.horizontal_scale)
        
        # Y方向分成三个区域：左边8米 + 中间1米 + 右边8米
        left_region_width = 8.0  # 左边区域宽度(米)
        middle_region_width = 1.0  # 中间区域宽度(米)
        right_region_width = 8.0  # 右边区域宽度(米)
        
        # 转换为网格单位
        left_region_width_grid = round(left_region_width / terrain.horizontal_scale)
        middle_region_width_grid = round(middle_region_width / terrain.horizontal_scale)
        right_region_width_grid = round(right_region_width / terrain.horizontal_scale)
        
        # === 左边区域：平地（机器人出生和行走区域） ===
        left_y_start = start_y
        left_y_end = start_y + left_region_width_grid
        
        # 整个左边区域设为平地（高度=0）
        terrain.height_field_raw[start_x:start_x + length_x_grid, 
                                left_y_start:left_y_end] = 0
        
        # 计算左边区域的中心坐标
        left_mid_y = left_y_start + left_region_width_grid // 2  # 左边区域的中心Y坐标
        left_mid_x = start_x + length_x_grid // 2  # 左边区域的中心X坐标
        
        # 设置起始平台（在左边区域的中心，机器人出生点附近）
        platform_x_start = left_mid_x - init_platform_grid // 2
        platform_x_end = left_mid_x + init_platform_grid // 2
        terrain.height_field_raw[platform_x_start:platform_x_end, 
                                left_y_start:left_y_end] = 0
        
        # 计算目标点位置（都在左边区域的中心线上，只在左边区域内）
        # 目标点在X方向分布，从起始平台后开始，到左边区域结束
        left_region_x_end = start_x + length_x_grid  # 左边区域的X方向结束位置
        available_x_length = left_region_x_end - platform_x_end  # 可用的X方向长度
        per_x = available_x_length // num_goals
        for i in range(num_goals):
            goals[i] = [platform_x_end + per_x * i, left_mid_y]
            
        # 保存左边区域的中心Y坐标（用于环境原点定位）
        terrain.left_region_center_y = left_mid_y * terrain.horizontal_scale
        
        # 保存左边区域的中心X坐标（用于环境原点定位）
        left_mid_x = start_x + length_x_grid // 2  # 左边区域的中心X坐标
        terrain.left_region_center_x = left_mid_x * terrain.horizontal_scale
        
        # === 中间区域：平地缓冲区 ===
        middle_y_start = left_y_end
        middle_y_end = middle_y_start + middle_region_width_grid
        
        # 整个中间区域设为平地（高度=0）
        terrain.height_field_raw[start_x:start_x + length_x_grid, 
                                middle_y_start:middle_y_end] = 0
        
        # === 右边区域：Stones Everywhere 地形 + 中心小平地（观测区） ===
        right_y_start = middle_y_end
        right_y_end = right_y_start + right_region_width_grid
        
        # 先将整个右边区域设为深坑（背景）
        gap_depth_grid = round(gap_depth / terrain.vertical_scale)
        terrain.height_field_raw[start_x:start_x + length_x_grid,
                                right_y_start:right_y_end] = -gap_depth_grid
        
        # === Stones Everywhere 参数（来自论文BeamDojo） ===
        difficulty_level = min(8, int(difficulty * 8))  # 难度等级 l ∈ [0, 8]
        
        # 石块尺寸：max{0.25, 1.5(1 - 0.1l)}
        stone_size = max(0.25, 1.5 * (1.0 - 0.1 * difficulty_level))
        stone_size_grid = round(stone_size / terrain.horizontal_scale)
        
        # 石块间距：0.05 × ⌈l/2⌉
        stone_distance = 0.05 * np.ceil(difficulty_level / 2.0)
        stone_distance_grid = round(stone_distance / terrain.horizontal_scale)
        
        # 子网格尺寸 = 石块尺寸 + 间距
        subgrid_size = stone_size_grid + stone_distance_grid
        
        # 在右边区域放置石块"岛屿"
        current_x = start_x
        while current_x < start_x + length_x_grid:
            current_y = right_y_start
            while current_y < right_y_end:
                # 计算实际可以放置的石块尺寸（避免越界）
                actual_stone_x = min(stone_size_grid, start_x + length_x_grid - current_x)
                actual_stone_y = min(stone_size_grid, right_y_end - current_y)
                
                # 只有当石块足够大时才创建（至少要有原尺寸的30%）
                if actual_stone_x >= stone_size_grid * 0.3 and actual_stone_y >= stone_size_grid * 0.3:
                    # 创建石块平台（高度=0）
                    terrain.height_field_raw[current_x:current_x + actual_stone_x,
                                            current_y:current_y + actual_stone_y] = 0
                current_y += subgrid_size
            current_x += subgrid_size
        
        # === 右边区域中心小平地 ===
        # 在右边区域的中心创建一个小平地
        right_center_y = right_y_start + (right_y_end - right_y_start) // 2
        small_platform_size = round(platform_size / terrain.horizontal_scale)  # 0.5米的小平地
        
        # 小平地位置：右边区域的中心
        small_platform_y_start = right_center_y - small_platform_size // 2
        small_platform_y_end = right_center_y + small_platform_size // 2
        small_platform_x_start = start_x + length_x_grid // 2 - small_platform_size // 2
        small_platform_x_end = start_x + length_x_grid // 2 + small_platform_size // 2
        
        # 确保小平地在右边区域范围内
        small_platform_y_start = max(small_platform_y_start, right_y_start)
        small_platform_y_end = min(small_platform_y_end, right_y_end)
        small_platform_x_start = max(small_platform_x_start, start_x)
        small_platform_x_end = min(small_platform_x_end, start_x + length_x_grid)
        
        # 创建小平地（高度=0）
        terrain.height_field_raw[small_platform_x_start:small_platform_x_end,
                                small_platform_y_start:small_platform_y_end] = 0
        
        return terrain, goals, start_x + length_x_grid