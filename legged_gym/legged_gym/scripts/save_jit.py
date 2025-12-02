import os, sys
sys.path.append("../../../rsl_rl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import Actor, get_activation
from rsl_rl.modules.estimator import Estimator
import argparse

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root): 
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint

class HardwareVisionNN(nn.Module):
    def __init__(self,  num_prop,
                        num_scan,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        tanh,
                        actor_hidden_dims=[512, 256, 128],
                        scan_encoder_dims=[128, 64, 32],
                        depth_encoder_hidden_dim=512,
                        activation='elu',
                        priv_encoder_dims=[64, 20]
                        ):
        super(HardwareVisionNN, self).__init__()
        
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        num_obs = num_prop + num_scan + num_priv_explicit + num_priv_latent + num_hist*num_prop
        self.num_obs = num_obs
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=tanh)

        # Estimator 使用历史观测的拼接作为输入: history_len * num_prop
        self.estimator = Estimator(input_dim=num_hist * num_prop, output_dim=num_priv_explicit, hidden_dims=[256, 128, 64])
        
    def forward(self, obs, depth_latent):
        # Estimator 使用历史观测作为输入（观测的最后 history_len*num_prop 维）
        hist_obs = obs[:, -self.num_hist*self.num_prop:]
        obs[:, self.num_prop+self.num_scan : self.num_prop+self.num_scan+self.num_priv_explicit] = self.estimator(hist_obs)
        return self.actor(obs, hist_encoding=True, eval=False, scandots_latent=depth_latent)

class MuJoCoDeployNN(nn.Module):
    def __init__(self, hardware_vision_nn):
        super(MuJoCoDeployNN, self).__init__()
        self.hardware_vision_nn = hardware_vision_nn
        
    def forward(self, obs):
        obs_clone = obs.clone()
        n = self.hardware_vision_nn
        hist_obs = obs_clone[:, -n.num_hist*n.num_prop:]
        obs_clone[:, n.num_prop+n.num_scan : n.num_prop+n.num_scan+n.num_priv_explicit] = \
            n.estimator(hist_obs)
        return n.actor(obs_clone, hist_encoding=True, eval=False, scandots_latent=None)

def main(args):    
    # humanoid_beamdojo 配置（来自 humanoid_beamdojo_config.py）
    # 本体感受观测维度（只包含下肢12个关节）：commands(3) + ang_vel(3) + delta_yaw(1) + delta_pose_x(1) + delta_pose_y(1) + gravity(3) + dof_pos(12) + dof_vel(12) + action_history(12) = 48
    n_proprio = 48
    n_priv_explicit = 3  # base_lin_vel (x, y, z)
    n_priv_latent = 4 + 1 + 12 + 12  # 质量参数(4) + 摩擦系数(1) + 电机强度(12+12) = 29
    num_scan = 225  # 15×15 高度扫描点
    num_actions = 12  # 只控制下肢12个关节（上肢已固定）
    history_len = 10  # 历史长度
    actor_hidden_dims = [1024, 512, 256, 128]
    scan_encoder_dims = [128, 64, 32]
    priv_encoder_dims = [64, 20]
    activation = 'elu'
    tanh_encoder_output = False

    load_run = "../../logs/beamdojo/" + args.exptid
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=args.checkpoint)
    load_run = os.path.dirname(load_path)
    
    device = torch.device('cpu')
    use_tanh = args.tanh if args.tanh else tanh_encoder_output
    policy = HardwareVisionNN(
        n_proprio, num_scan, n_priv_latent, n_priv_explicit, history_len, num_actions, 
        use_tanh,
        actor_hidden_dims=actor_hidden_dims,
        scan_encoder_dims=scan_encoder_dims,
        priv_encoder_dims=priv_encoder_dims,
        activation=activation
    ).to(device)
    ac_state_dict = torch.load(load_path, map_location=device)
    policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    policy.estimator.load_state_dict(ac_state_dict['estimator_state_dict'])
    
    os.makedirs(os.path.join(load_run, "traced"), exist_ok=True)
    deploy_policy = MuJoCoDeployNN(policy).eval()
    
    with torch.no_grad(): 
        obs_input = torch.ones(1, n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len*n_proprio, device=device)
        traced_policy = torch.jit.trace(deploy_policy, obs_input)
        save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-mujoco_jit.pt")
        traced_policy.save(save_path)
        print(f"已保存: {save_path}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exptid', type=str, required=True)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--tanh', action='store_true')
    main(parser.parse_args())
    