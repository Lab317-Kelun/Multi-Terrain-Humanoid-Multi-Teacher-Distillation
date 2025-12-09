"""
JIT 模型转 ONNX 工具
功能：将已保存的 JIT 模型（.pt）转换为 ONNX 格式（.onnx），用于 sim2real 部署

使用流程：
1. 使用 save_jit.py 将 .pt 模型转换为 JIT 模型（.pt）
2. 使用本脚本将 JIT 模型转换为 ONNX 模型（.onnx）
3. 在部署代码中使用 ONNX Runtime 加载模型

观测维度说明（与 save_jit.py 一致）：
- n_proprio = 48（本体感受观测）
- num_scan = 225（高度扫描点）
- n_priv_explicit = 3（显式特权信息：base_lin_vel）
- n_priv_latent = 29（隐式特权信息：质量、摩擦、电机强度）
- history_len = 10（历史长度）
- 总观测维度 = 48 + 225 + 3 + 29 + 10*48 = 785
"""
import torch
import os
import argparse


def export_jit_to_onnx(jit_model, path, dummy_input):
    """
    将 JIT 模型导出为 ONNX 格式
    
    @param jit_model 已加载的 JIT 模型
    @param path ONNX 文件保存路径
    @param dummy_input 示例输入（用于确定输入形状）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 导出 JIT 模型到 ONNX
    torch.onnx.export(
        jit_model,                  # JIT 模型
        dummy_input,                # 示例输入（用于追踪模型结构）
        path,                       # 输出文件路径
        export_params=True,         # 导出模型参数（权重）
        opset_version=11,           # ONNX 操作集版本
        do_constant_folding=True,   # 执行常量折叠优化
        input_names=['input'],      # 输入名称
        output_names=['output'],    # 输出名称
        dynamic_axes={
            'input': {0: 'batch_size'},   # 输入的第 0 维（batch）是动态的
            'output': {0: 'batch_size'}   # 输出的第 0 维（batch）是动态的
        }
    )
    print(f"已导出 ONNX 模型: {path}")


def main(args):
    """
    主函数：加载 JIT 模型并导出为 ONNX
    
    @param args 命令行参数
    """
    # 观测维度配置（与 save_jit.py 中的配置一致）
    # humanoid_beamdojo 配置（来自 humanoid_beamdojo_config.py）
    # 本体感受观测维度（只包含下肢12个关节）：
    # commands(3) + ang_vel(3) + delta_yaw(1) + delta_pose_x(1) + delta_pose_y(1) 
    # + gravity(3) + dof_pos(12) + dof_vel(12) + action_history(12) = 48
    n_proprio = 48
    n_priv_explicit = 3   # base_lin_vel (x, y, z)
    n_priv_latent = 29    # 质量参数(4) + 摩擦系数(1) + 电机强度(12+12) = 29
    num_scan = 225        # 15×15 高度扫描点
    history_len = 10      # 历史长度
    
    # 总观测维度
    # 顺序：proprio + scan + priv_explicit + priv_latent + history*proprio
    num_obs = n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len * n_proprio
    # = 48 + 225 + 3 + 29 + 10*48 = 785
    
    # 加载 JIT 模型
    if not os.path.exists(args.jit_path):
        raise FileNotFoundError(f"JIT 模型文件不存在: {args.jit_path}")
    
    print(f"正在加载 JIT 模型: {args.jit_path}")
    jit_model = torch.jit.load(args.jit_path, map_location='cpu')
    jit_model.eval()  # 设置为评估模式
    
    # 创建示例输入（用于 ONNX 导出）
    # 形状：[batch_size, obs_dim] = [1, 785]
    dummy_input = torch.randn(1, num_obs, device='cpu')
    
    # 确定输出路径
    if args.output_path:
        export_path = args.output_path
    else:
        # 如果没有指定输出路径，使用 JIT 文件所在目录，文件名改为 .onnx
        jit_dir = os.path.dirname(args.jit_path)
        jit_basename = os.path.basename(args.jit_path)
        onnx_basename = jit_basename.replace('_jit.pt', '.onnx').replace('.pt', '.onnx')
        export_path = os.path.join(jit_dir, onnx_basename)
    
    print(f"观测维度: {num_obs}")
    print(f"示例输入形状: {dummy_input.shape}")
    print(f"输出路径: {export_path}")
    
    # 导出为 ONNX
    with torch.no_grad():
        export_jit_to_onnx(jit_model, export_path, dummy_input)
    
    print(f"✓ 转换完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将 JIT 模型转换为 ONNX 格式')
    parser.add_argument('--jit_path', type=str, required=True,
                        help='JIT 模型文件路径（.pt 文件，由 save_jit.py 生成）')
    parser.add_argument('--output_path', type=str, default=None,
                        help='ONNX 输出文件路径（可选，默认与 JIT 文件同目录）')
    
    args = parser.parse_args()
    main(args)