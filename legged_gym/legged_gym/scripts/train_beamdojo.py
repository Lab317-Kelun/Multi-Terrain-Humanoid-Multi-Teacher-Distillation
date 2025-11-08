#!/usr/bin/env python3

import numpy as np
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 按照legged_gym的方式导入
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import wandb

def train_beamdojo(args):
    """BEAMDOJO训练主函数"""
    args.headless = True
    
    # 设置默认参数
    if not hasattr(args, 'proj_name') or args.proj_name is None:
        args.proj_name = 'beamdojo'
    if not hasattr(args, 'exptid') or args.exptid is None:
        args.exptid = f"{args.task}_{datetime.now().strftime('%m%d_%H%M')}"
    
    # 创建日志路径
    from legged_gym import LEGGED_GYM_ROOT_DIR
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + datetime.now().strftime('%b%d_%H-%M-%S--') + args.exptid
    
    try:
        os.makedirs(log_pth)
    except:
        pass
    
    # Wandb设置
    debug_mode = getattr(args, 'debug', False)
    no_wandb = getattr(args, 'no_wandb', False)
    
    if debug_mode:
        mode = "disabled"
        args.rows = 10
        args.cols = 8
        args.num_envs = 64
    else:
        mode = "online"
    
    if no_wandb:
        mode = "disabled"
    
    try:
        wandb.init(project=args.proj_name, name=args.exptid, group=args.exptid[:3] if args.exptid else "beamdojo", mode=mode, dir="../../logs")
    except Exception as e:
        print(f"⚠️ Wandb初始化失败: {e}")
        print("🔄 继续训练，但不使用wandb...")
        mode = "disabled"
    
    # 保存配置文件
    from legged_gym import LEGGED_GYM_ENVS_DIR
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid_robot.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/humanoid/humanoid_beamdojo_config.py", policy="now")

    print(f"🚀 开始BEAMDOJO训练: {args.task}")
    
    # 创建环境和训练器
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args)
    
    # 开始训练
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    
    print(f"✅ BEAMDOJO训练完成! 模型保存在: {log_pth}")

if __name__ == '__main__':
    args = get_args()
    train_beamdojo(args)