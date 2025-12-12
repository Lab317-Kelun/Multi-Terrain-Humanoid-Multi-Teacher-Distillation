"""
This file is used to transfer a JIT .pt file to a .onnx file
观测维度配置与 save_jit.py 保持一致
"""
import torch
import os
import argparse

def export_jit_to_onnx(jit_model, path, dummy_input):
    """将 JIT 模型导出为 ONNX 格式"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # export jit to onnx
    torch.onnx.export(
        jit_model,                  
        dummy_input,            
        path,                       
        export_params=True,         
        opset_version=11,           
        do_constant_folding=True,   
        input_names=['input'],      
        output_names=['output'],    
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"已导出 ONNX 模型: {path}")

def main(args):
    # humanoid_beamdojo 配置（与 save_jit.py 保持一致）
    # 本体感受观测维度（只包含下肢12个关节）：commands(3) + ang_vel(3) + delta_yaw(1) + delta_pose_x(1) + delta_pose_y(1) + gravity(3) + dof_pos(12) + dof_vel(12) + action_history(12) = 48
    n_proprio = 45
    n_priv_explicit = 3  # base_lin_vel (x, y, z)
    n_priv_latent = 4 + 1 + 12 + 12  # 质量参数(4) + 摩擦系数(1) + 电机强度(12+12) = 29
    num_scan = 225  # 15×15 高度扫描点
    history_len = 10  # 历史长度
    
    # 计算总观测维度
    # num_obs = n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len*n_proprio
    num_obs = n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len * n_proprio
    # = 48 + 225 + 3 + 29 + 10*48 = 48 + 225 + 3 + 29 + 480 = 785
    
    print(f"观测维度配置:")
    print(f"  n_proprio: {n_proprio}")
    print(f"  num_scan: {num_scan}")
    print(f"  n_priv_explicit: {n_priv_explicit}")
    print(f"  n_priv_latent: {n_priv_latent}")
    print(f"  history_len: {history_len}")
    print(f"  总观测维度: {num_obs}")
    
    # 直接使用提供的 JIT 模型路径
    jit_model_path = args.jit_path
    
    if not os.path.exists(jit_model_path):
        raise FileNotFoundError(f"JIT 模型文件不存在: {jit_model_path}")
    
    print(f"\n加载 JIT 模型: {jit_model_path}")
    jit_model = torch.jit.load(jit_model_path)
    jit_model.eval()
    
    # 创建虚拟输入（与 save_jit.py 中的观测维度一致）
    device = torch.device('cpu')
    dummy_input = torch.ones(1, num_obs, device=device)
    
    # 导出 ONNX 模型
    if args.output_path:
        onnx_path = args.output_path
    else:
        # 默认输出路径：与 JIT 模型同目录，文件名改为 .onnx
        jit_dir = os.path.dirname(jit_model_path)
        jit_name = os.path.basename(jit_model_path)
        onnx_name = jit_name.replace('.pt', '.onnx').replace('_jit', '')
        onnx_path = os.path.join(jit_dir, onnx_name)
    
    print(f"\n开始导出 ONNX 模型...")
    export_jit_to_onnx(jit_model, onnx_path, dummy_input)
    print(f"\n完成！ONNX 模型已保存到: {onnx_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将 JIT 模型转换为 ONNX 格式')
    parser.add_argument('--jit_path', type=str, required=True, help='JIT 模型文件路径（.pt 文件）')
    parser.add_argument('--output_path', type=str, default=None, help='ONNX 模型输出路径（默认：与 JIT 模型同目录，文件名自动替换）')
    main(parser.parse_args())
