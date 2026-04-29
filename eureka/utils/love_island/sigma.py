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
    pattern = os.path.join(states_dir, "terminal_*.pt")
    files = sorted(glob(pattern))
    if not files:
        logging.info(f"[sigma] no terminal state files in {states_dir}")
        return None

    parts: Dict[str, List[torch.Tensor]] = {}

    for f in files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=True)
            for key, tensor in data.items():
                parts.setdefault(key, []).append(tensor)
        except Exception as e:
            logging.warning(f"[sigma] skipping {f}: {e}")

    if not parts:
        return None

    # Validate consistent shapes before concatenating — mismatched dims means
    # files from different envs (e.g. stale cartpole mixed with ant). Drop bad files.
    result = {}
    for key, tensors in parts.items():
        if len(tensors) == 0:
            continue
        ref_shape = tensors[0].shape[1:]  # everything after batch dim must match
        valid = [t for t in tensors if t.shape[1:] == ref_shape]
        if len(valid) < len(tensors):
            logging.warning(
                f"[sigma] key '{key}': dropped {len(tensors)-len(valid)} files "
                f"with mismatched shape (expected {ref_shape})"
            )
        if valid:
            result[key] = torch.cat(valid, dim=0)

    return result if result else None


def compute_sigma_batch(
    reward_fns: List[Callable],
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> tuple:
    """Compute per-state σ across reward functions.

    Uses schema-based dispatch: reads each fn's argument names from its
    torch.jit schema and looks them up in state_dict. Falls back to
    (joint_pos, joint_vel) for cartpole-style fns that only need those two.
    """
    active_fns = [fn for fn in reward_fns if fn is not None]
    if not active_fns:
        n = joint_pos.shape[0]
        return np.zeros(n), np.zeros(n)

    # Build a lookup table from all available tensors
    available = {"joint_pos": joint_pos, "joint_vel": joint_vel}
    if state_dict:
        available.update({k: v for k, v in state_dict.items()
                          if isinstance(v, torch.Tensor)})

    rewards = []
    with torch.no_grad():
        for fn in active_fns:
            try:
                # Use schema to get param names, same approach as antgpt._get_rewards
                param_names = [arg.name for arg in fn.schema.arguments]
                args = []
                missing = []
                for p in param_names:
                    if p in available:
                        args.append(available[p])
                    else:
                        missing.append(p)
                if missing:
                    logging.warning(f"[sigma] skipping fn — missing tensors: {missing}")
                    continue
                r, _ = fn(*args)
                rewards.append(r.cpu().numpy())
            except Exception as e:
                logging.warning(f"[sigma] reward fn call failed: {e}")

    if not rewards:
        n = joint_pos.shape[0]
        return np.zeros(n), np.zeros(n)

    stacked = np.stack(rewards, axis=0)  # [num_fns, N]
    return stacked.std(axis=0), stacked.mean(axis=0)


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
