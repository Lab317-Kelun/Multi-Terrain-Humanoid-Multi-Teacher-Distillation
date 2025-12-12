export WANDB_API_KEY="d18742a0e125a18ff2990ecbf3542d8270830341"


cd legged_gym/legged_gym/scripts
python train_distill_beamdojo.py \
                                 --device cuda:0 \
                                 --headless \
                                 --run_name 1teacher
                                 