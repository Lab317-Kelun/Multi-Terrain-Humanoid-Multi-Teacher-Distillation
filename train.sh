export WANDB_API_KEY="d18742a0e125a18ff2990ecbf3542d8270830341"
cd legged_gym/legged_gym/scripts
python train_distill_beamdojo.py --teacher_checkpoint /home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/logs/beamdojo/Nov21_00-23-42--stage2_newest/model_8000.pt \
                                 --device cuda:0 \
                                 --headless \
                                 