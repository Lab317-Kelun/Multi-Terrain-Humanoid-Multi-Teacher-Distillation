import torch

# 替换为你实际的 checkpoint 路径
ckpt_path = "/home/cft/yanzhe/Multi-Terrain-Humanoid-Multi-Teacher-Distillation/legged_gym/logs/teachers/14model_43000.pt"

print(f"Loading checkpoint from: {ckpt_path}")
try:
    loaded_dict = torch.load(ckpt_path, map_location="cpu")
    
    print("\n=== Top Level Keys ===")
    print(list(loaded_dict.keys()))

    # 确定参数字典在哪里
    if 'model_state_dict' in loaded_dict:
        state_dict = loaded_dict['model_state_dict']
        print("\n=== Found 'model_state_dict'. Printing keys inside... ===")
    else:
        state_dict = loaded_dict
        print("\n=== No 'model_state_dict' found. Assuming top level is state_dict. ===")

    # 打印所有键（排序后）
    keys = sorted(list(state_dict.keys()))
    print(f"\nTotal keys found: {len(keys)}")
    
    print("\n--- All Keys ---")
    for k in keys:
        print(k)

    # 特别检查归一化层
    print("\n=== Normalization Check ===")
    norm_keys = [k for k in keys if "normalizer" in k or "running_mean" in k]
    if norm_keys:
        print(f"Found {len(norm_keys)} normalization keys:")
        for k in norm_keys:
            print(f"  {k}")
    else:
        print("WARNING: No normalization keys found!")

except Exception as e:
    print(f"Error loading checkpoint: {e}")