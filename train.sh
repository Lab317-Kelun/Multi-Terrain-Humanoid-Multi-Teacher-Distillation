export WANDB_API_KEY="d18742a0e125a18ff2990ecbf3542d8270830341"
cd legged_gym/legged_gym/scripts
python train_distill_beamdojo.py --teacher_checkpoint /home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/logs/beamdojo/Nov16_23-06-59--stage0_curvel0.6 --device cuda:0