"""Sigma (ensemble disagreement) computation for LOVE Island.

Reward functions are plain @torch.jit.script callables — they can be loaded
from generated .py files and evaluated on saved terminal states with no
simulator running.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from glob import glob
from typing import Callable, Dict, List, Optional

import numpy as np
import torch


def load_reward_fn(py_path: str) -> Optional[Callable]:
    """Import compute_reward from a generated environment file.

    Uses the same importlib pattern as isaaclab_train.py:load_task_module().
    Returns None if the file has no compute_reward function.
    """
    if not os.path.exists(py_path):
        logging.warning(f"[sigma] reward file not found: {py_path}")
        return None

    module_name = f"_love_island_reward_{os.path.basename(py_path).replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logging.warning(f"[sigma] failed to load {py_path}: {e}")
        return None

    fn = getattr(module, "compute_reward", None)
    if fn is None:
        logging.warning(f"[sigma] no compute_reward in {py_path}")
    return fn


def load_terminal_states(states_dir: str) -> Optional[Dict[str, torch.Tensor]]:
    """Load and concatenate all terminal_{step}.pt files from a run's states/ dir.

    Returns dict with keys joint_pos, joint_vel, gt_reward (all CPU tensors),
    or None if no state files exist.
    """
    pattern = os.path.join(states_dir, "terminal_*.pt")
    files = sorted(glob(pattern))
    if not files:
        logging.info(f"[sigma] no terminal state files in {states_dir}")
        return None

    joint_pos_parts: List[torch.Tensor] = []
    joint_vel_parts: List[torch.Tensor] = []
    gt_reward_parts: List[torch.Tensor] = []

    for f in files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=True)
            joint_pos_parts.append(data["joint_pos"])
            joint_vel_parts.append(data["joint_vel"])
            gt_reward_parts.append(data["gt_reward"])
        except Exception as e:
            logging.warning(f"[sigma] skipping {f}: {e}")

    if not joint_pos_parts:
        return None

    return {
        "joint_pos": torch.cat(joint_pos_parts, dim=0),
        "joint_vel": torch.cat(joint_vel_parts, dim=0),
        "gt_reward": torch.cat(gt_reward_parts, dim=0),
    }


def compute_sigma_batch(
    reward_fns: List[Callable],
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> np.ndarray:
    """Compute per-state σ = std of R_i(s) across all reward functions.

    Args:
        reward_fns: list of compute_reward callables (None entries are skipped)
        joint_pos:  Tensor[N, J]
        joint_vel:  Tensor[N, J]

    Returns:
        sigma: ndarray[N] — std of ensemble rewards per state
        mean_rewards: ndarray[N] — mean of ensemble rewards per state
    """
    active_fns = [fn for fn in reward_fns if fn is not None]
    if not active_fns:
        n = joint_pos.shape[0]
        return np.zeros(n), np.zeros(n)

    rewards = []
    with torch.no_grad():
        for fn in active_fns:
            try:
                r, _ = fn(joint_pos, joint_vel)
                rewards.append(r.cpu().numpy())
            except Exception as e:
                logging.warning(f"[sigma] reward fn call failed: {e}")

    if not rewards:
        n = joint_pos.shape[0]
        return np.zeros(n), np.zeros(n)

    stacked = np.stack(rewards, axis=0)  # [num_fns, N]
    sigma = stacked.std(axis=0)          # [N]
    mean_r = stacked.mean(axis=0)        # [N]
    return sigma, mean_r


def tb_sigma_prefilter(
    tb_dirs: List[str],
    low_threshold: float = 0.05,
    key: str = "gpt_reward",
) -> float:
    """Cheap TensorBoard-based guard: std of final-epoch mean gpt_reward across islands.

    Returns the scalar σ_tb. If this is below low_threshold the caller can
    skip loading .pt state files entirely. Never use as primary σ.
    """
    from utils.file_utils import load_tensorboard_logs

    final_means = []
    for tb_dir in tb_dirs:
        if not os.path.isdir(tb_dir):
            continue
        try:
            logs = load_tensorboard_logs(tb_dir)
            vals = logs.get(key, [])
            if vals:
                final_means.append(float(vals[-1]))
        except Exception as e:
            logging.debug(f"[sigma] tb prefilter failed for {tb_dir}: {e}")

    if len(final_means) < 2:
        return float("inf")  # can't compute std, don't skip

    return float(np.std(final_means))
