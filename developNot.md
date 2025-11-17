# onpolicyRunner:
- 构建estimator
- 构建actor_critic
- init_storage
- 进行learn：rollout，调用act，收集到storage，
- save/load model

观测：本体 + 高程图 + 显式特权 + 隐式特权 + 历史
原始：onpolicyrunner初始化estimater

训练时： 在rollout时，在ppo.py的act()中，用estimater来取代特权信息，然后再调用ac的act，最后在forward中被hist或pri编码后，得到latent，最后输入本体，latent，和显式特权得到动作
        在update时，更新estimator，priv，hist和主体

推理时： 
