#!/usr/bin/env python3
"""
Isaac Lab rl_games training wrapper for the Eureka reward-generation pipeline.

Runs INSIDE the Isaac Lab container. Eureka invokes it via:
  apptainer exec --nv <container> /isaac-sim/python.sh <project_root>/isaaclab_train.py \
    --task CartpoleGPT --headless --max_iterations 500 \
    --run_name env_iter0_response0_island0 --runs_dir /path/to/runs

Prints to stdout (required by eureka.py's log parser):
  Network Directory: <path/nn>
  Tensorboard Directory: <path/summaries>

rl_games naturally emits "fps step: ..." which block_until_training() polls for.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isaac Lab rl_games training wrapper for Eureka")
    p.add_argument("--task", required=True, help="Task name, e.g. CartpoleGPT")
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--max_iterations", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default="", help="Unique run sub-directory name")
    p.add_argument("--runs_dir", default="", help="Root directory for all training runs")
    p.add_argument("--states_dir", default="", help="If set, save terminal states to this directory for LOVE Island σ computation")
    p.add_argument("--trajectories_dir", default="", help="If set, save full episode trajectories here for GIF rendering")
    p.add_argument("--load_checkpoint", action="store_true", default=False, help="Resume from a saved checkpoint")
    p.add_argument("--load_path", default="", help="Path to .pth checkpoint file for resuming")
    return p.parse_args()


def load_task_module(project_root: str, task_name: str):
    """Load the generated env module from eureka/envs/isaaclab/{task_name.lower()}.py."""
    filename = task_name.lower() + ".py"
    filepath = os.path.join(project_root, "eureka", "envs", "isaaclab", filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Task file not found: {filepath}\n"
            f"  task_name={task_name!r}, looked for {filename}"
        )
    spec = importlib.util.spec_from_file_location(f"eureka_task_{task_name}", filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_rl_games_cfg(task_name: str, num_envs: int, seed: int,
                        runs_dir: str, run_name: str, rl_device: str,
                        max_epochs: int = 500,
                        load_checkpoint: bool = False,
                        load_path: str = "") -> dict:
    return {
        "params": {
            "seed": seed,
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0.0},
                        "fixed_sigma": True,
                    }
                },
                "mlp": {
                    "units": [32, 32],
                    "activation": "elu",
                    "d2rl": False,
                    "initializer": {"name": "default"},
                    "regularizer": {"name": "None"},
                },
            },
            "load_checkpoint": load_checkpoint,
            "load_path": load_path,
            "config": {
                "name": run_name,
                "env_name": "rlgpu",
                "device": rl_device,
                "device_name": rl_device,
                "ppo": True,
                "mixed_precision": False,
                "normalize_input": True,
                "normalize_value": True,
                "num_actors": num_envs,
                "reward_shaper": {"scale_value": 1.0},
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 3e-4,
                "lr_schedule": "adaptive",
                "kl_threshold": 0.008,
                "score_to_win": 20000,
                "max_epochs": max_epochs,
                "save_best_after": 50,
                "save_frequency": 50,
                "grad_norm": 1.0,
                "entropy_coef": 0.0,
                "truncate_grads": True,
                "e_clip": 0.2,
                "horizon_length": 16,
                "minibatch_size": 8192,
                "mini_epochs": 8,
                "critic_coef": 4,
                "clip_value": True,
                "seq_length": 4,
                "bounds_loss_coef": 0.0001,
                "train_dir": runs_dir,
            },
        }
    }


def main():
    args = parse_args()

    # SimulationApp MUST be created before any other Isaac Sim / Isaac Lab imports.
    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": True})

    import torch

    # Resolve project root (the directory containing this script).
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Load the generated env module.
    task_module = load_task_module(project_root, args.task)

    # Cartpole always uses CartpoleEnv / CartpoleEnvCfg.
    # For additional envs, extend this mapping.
    # Dynamically resolve the env class from the loaded module.
    # Convention: task file exports exactly one DirectRLEnv subclass (e.g. AntEnv)
    # and a matching cfg class (e.g. AntEnvCfg).
    import inspect
    from isaaclab.envs import DirectRLEnv

    env_classes = [
        obj for _, obj in inspect.getmembers(task_module, inspect.isclass)
        if issubclass(obj, DirectRLEnv) and obj is not DirectRLEnv
    ]
    if len(env_classes) != 1:
        raise ValueError(
            f"Expected exactly one DirectRLEnv subclass in {args.task}, "
            f"found: {[c.__name__ for c in env_classes]}"
        )
    env_cls = env_classes[0]
    env_cfg_cls = getattr(task_module, env_cls.__name__ + "Cfg")

    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args.num_envs
    if args.states_dir:
        env_cfg.states_dir = args.states_dir
    if args.trajectories_dir:
        env_cfg.trajectories_dir = args.trajectories_dir

    # Output directories.
    import datetime
    run_name = args.run_name or f"{args.task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runs_dir = args.runs_dir or os.path.join(project_root, "runs")
    nn_dir = os.path.join(runs_dir, run_name, "nn")
    tb_dir = os.path.join(runs_dir, run_name, "summaries")
    os.makedirs(nn_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # These two lines are parsed by eureka.py's log reader.
    print(f"Network Directory: {nn_dir}", flush=True)
    print(f"Tensorboard Directory: {tb_dir}", flush=True)

    # Create the environment.
    env = env_cls(env_cfg, render_mode=None)
    rl_device = str(env.device)

    # Wire up a direct TensorBoard writer on the env.  rl_games logs extras with an
    # "episode/" prefix which Eureka's parser skips; the env writes the keys we need
    # (gt_reward, gpt_reward, consecutive_successes) directly to tb_dir.
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    env._tb_writer = _SummaryWriter(tb_dir)

    # Wrap for rl_games using Isaac Lab 2.x class names.
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    # Isaac Lab 2.x: env is passed directly to RlGamesVecEnvWrapper (positional).
    env_wrapped = RlGamesVecEnvWrapper(env, rl_device, 10.0, 1.0)

    # The env name in rl-games configuration MUST be "rlgpu" — rl-games looks it up
    # by that key when the Runner calls env_creator.
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"env_creator": lambda **kwargs: env_wrapped, "vecenv_type": "IsaacRlgWrapper"},
    )

    cfg = build_rl_games_cfg(
        task_name=args.task,
        num_envs=env.num_envs,       # use actual env count after wrapper
        seed=args.seed,
        runs_dir=runs_dir,
        run_name=run_name,
        rl_device=rl_device,
        max_epochs=args.max_iterations,
        load_checkpoint=args.load_checkpoint,
        load_path=args.load_path,
    )

    runner = Runner()
    runner.load(cfg)
    runner.reset()
    runner.run({"train": True})

    # Flush and close the direct TensorBoard writer before closing the env.
    if getattr(env, "_tb_writer", None) is not None:
        env._tb_writer.close()

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
