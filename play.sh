# python legged_gym/legged_gym/scripts/play.py \
#     --task humanoid_beamdojo \
#     --checkpoint_path /home/cft/yanzhe/Multi-Terrain-Humanoid-Multi-Teacher-Distillation/legged_gym/logs/beamdojo/Dec16_21-44-30--distill_1216_2144/model_8200.pt

    #--checkpoint_path /home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/logs/beamdojo/Dec14_20-08-39--homie_stage2_gap/model_20000.pt
    #--checkpoint_path /home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/logs/beamdojo/Dec14_20-09-35--homie_stage2_stone/model_32500.pt


python legged_gym/legged_gym/scripts/play_distill_beamdojo.py \
     --device cpu \
     --task humanoid_beamdojo \
     --checkpoint_path /home/cft/yanzhe/Multi-Terrain-Humanoid-Multi-Teacher-Distillation/legged_gym/logs/beamdojo/Dec16_21-44-30--distill_1216_2144/model_83800.pt