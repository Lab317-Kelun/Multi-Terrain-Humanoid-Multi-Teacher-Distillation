"""Entry point for teacher-student distillation training runs."""

from __future__ import annotations

import os
from datetime import datetime

import wandb

from legged_gym import LEGGED_GYM_ENVS_DIR, LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - task registry side effects
from legged_gym.utils import get_args, task_registry


def _build_log_dir(args) -> str:
	log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name)
	run_stamp = datetime.now().strftime("%b%d_%H-%M-%S--") + (args.exptid or "distill")
	log_dir = os.path.join(log_root, run_stamp)
	os.makedirs(log_dir, exist_ok=True)
	return log_dir


def _init_wandb(args) -> None:
	if args.debug or args.no_wandb:
		mode = "disabled"
	else:
		mode = "online"
	wandb.init(project=args.proj_name, name=args.exptid, group=(args.exptid or "distill")[:3], mode=mode, dir="../../logs")
	wandb.save(os.path.join(LEGGED_GYM_ENVS_DIR, "base/legged_robot_config.py"), policy="now")
	wandb.save(os.path.join(LEGGED_GYM_ENVS_DIR, "base/legged_robot.py"), policy="now")


def train(args) -> None:
	args.headless = True
	log_dir = _build_log_dir(args)
	_init_wandb(args)

	env, _ = task_registry.make_env(name=args.task, args=args)
	runner, train_cfg = task_registry.make_alg_runner(log_root=log_dir, env=env, name=args.task, args=args)
	runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
	train(get_args())
