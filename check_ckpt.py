import torch
import os

# Checkpoint paths to compare
ckpt_paths = [
    # "/home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/logs/beamdojo/Dec14_20-09-35--homie_stage2_stone/model_32500.pt",
    # "/home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/logs/beamdojo/Dec14_20-08-39--homie_stage2_gap/model_20000.pt",
    # "/home/cft/yanzhe/Multi-Terrain-Humanoid-Multi-Teacher-Distillation/model_7000.pt"
    "/home/cft/zikang/State-Estimation-AMP-Lab/unitree_rl_lab/logs/rsl_rl_MLE/2025-12-29_21-16-34_MLE_ensemble_3/model_18300.pt"
]

output_file = "checkpoint_keys_comparison.txt"

print(f"Comparing {len(ckpt_paths)} checkpoints...")
print(f"Output will be written to: {output_file}")

all_keys_sets = []
ckpt_names = []

with open(output_file, "w") as f:
    for ckpt_path in ckpt_paths:
        f.write(f"\n{'='*50}\n")
        f.write(f"Processing checkpoint: {ckpt_path}\n")
        print(f"Processing: {os.path.basename(ckpt_path)}")
        
        if not os.path.exists(ckpt_path):
            msg = f"File not found: {ckpt_path}\n"
            print(f"  Error: {msg.strip()}")
            f.write(msg)
            continue

        try:
            loaded_dict = torch.load(ckpt_path, map_location="cpu")
            
            # Determine where the state dict is
            if 'model_state_dict' in loaded_dict:
                state_dict = loaded_dict['model_state_dict']
                f.write("Found 'model_state_dict'. Using it.\n")
            else:
                state_dict = loaded_dict
                f.write("No 'model_state_dict' found. Using top level dict.\n")

            # Get keys
            keys = sorted(list(state_dict.keys()))
            f.write(f"Total keys found: {len(keys)}\n")
            
            f.write("\n--- Keys ---\n")
            for k in keys:
                f.write(f"{k}\n")
            
            all_keys_sets.append(set(keys))
            ckpt_names.append(os.path.basename(ckpt_path))

        except Exception as e:
            msg = f"Error loading checkpoint: {e}\n"
            print(f"  Error: {msg.strip()}")
            f.write(msg)

    f.write(f"\n{'='*50}\n")
    f.write("COMPARISON RESULT\n")
    f.write(f"{'='*50}\n")

    if not all_keys_sets:
        f.write("No checkpoints were successfully loaded.\n")
        print("No checkpoints were successfully loaded.")
    else:
        # Compare all against the first one
        base_name = ckpt_names[0]
        base_keys = all_keys_sets[0]
        all_match = True
        
        for i in range(1, len(all_keys_sets)):
            current_name = ckpt_names[i]
            current_keys = all_keys_sets[i]
            
            diff_missing = base_keys - current_keys # Keys in base but not in current
            diff_extra = current_keys - base_keys   # Keys in current but not in base
            
            if diff_missing or diff_extra:
                all_match = False
                f.write(f"\nMismatch between {base_name} and {current_name}:\n")
                if diff_missing:
                    f.write(f"  Keys in {base_name} but MISSING in {current_name} ({len(diff_missing)}):\n")
                    for k in sorted(list(diff_missing)):
                        f.write(f"    {k}\n")
                if diff_extra:
                    f.write(f"  Keys in {current_name} but MISSING in {base_name} ({len(diff_extra)}):\n")
                    for k in sorted(list(diff_extra)):
                        f.write(f"    {k}\n")
            else:
                f.write(f"\n{base_name} and {current_name} have IDENTICAL keys.\n")

        if all_match:
            msg = "\nSUCCESS: All loaded checkpoints have the same set of keys."
            print(msg)
            f.write(msg + "\n")
        else:
            msg = "\nWARNING: Checkpoints have different keys. See details in output file."
            print(msg)
            f.write(msg + "\n")

print(f"Done.")
