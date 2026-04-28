"""Episode trajectory rendering for LOVE Island visual feedback.

Renders saved cartpole trajectories as animated GIFs using matplotlib.
No Isaac Sim required — works entirely from saved .pt files.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for SLURM/headless use
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

CART_WIDTH = 0.4
CART_HEIGHT = 0.2
POLE_LENGTH = 0.6
TRACK_HALF_LEN = 3.2  # slightly wider than cfg.max_cart_pos=3.0 for margin


def load_trajectories(trajectories_dir: str, traj_keys: List[str]) -> Dict[str, dict]:
    """Load trajectory .pt files by traj_key via JSONL index.

    traj_key format: "{step}_{env_idx}"
    """
    if not os.path.isdir(trajectories_dir):
        return {}
    idx_path = os.path.join(trajectories_dir, "index.json")
    if not os.path.exists(idx_path):
        return {}

    key_to_filename: Dict[str, str] = {}
    with open(idx_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                k = f"{entry['step']}_{entry['env_idx']}"
                key_to_filename[k] = entry["filename"]
            except (json.JSONDecodeError, KeyError):
                continue

    result = {}
    for tk in traj_keys:
        fname = key_to_filename.get(tk)
        if fname is None:
            continue
        fpath = os.path.join(trajectories_dir, fname)
        if os.path.exists(fpath):
            result[tk] = torch.load(fpath, map_location="cpu")
    return result


def load_recent_trajectories(trajectories_dir: str, n: int = 2) -> Dict[str, dict]:
    """Load the n most recently saved trajectories from a directory."""
    if not os.path.isdir(trajectories_dir):
        return {}
    idx_path = os.path.join(trajectories_dir, "index.json")
    if not os.path.exists(idx_path):
        return {}

    entries = []
    with open(idx_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    result = {}
    for entry in entries[-n:]:
        fpath = os.path.join(trajectories_dir, entry["filename"])
        if os.path.exists(fpath):
            tk = f"{entry['step']}_{entry['env_idx']}"
            result[tk] = torch.load(fpath, map_location="cpu")
    return result


def render_cartpole_gif(
    traj_data: dict,
    output_path: str,
    title: str = "",
    fps: int = 20,
) -> str:
    """Render a single cartpole trajectory as an animated GIF.

    traj_data keys:
      joint_pos_seq: Tensor[T, J]  — joint positions at each timestep
      joint_vel_seq: Tensor[T, J]  — joint velocities (unused in rendering)
      cart_dof_idx: int            — column index for cart position
      pole_dof_idx: int            — column index for pole angle
    """
    jp = traj_data["joint_pos_seq"].numpy()   # [T, J]
    T = jp.shape[0]
    cart_idx = int(traj_data.get("cart_dof_idx", 0))
    pole_idx = int(traj_data.get("pole_dof_idx", 1))

    cart_pos = jp[:, cart_idx]    # [T]
    pole_ang = jp[:, pole_idx]    # [T], radians from vertical

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(-TRACK_HALF_LEN, TRACK_HALF_LEN)
    ax.set_ylim(-CART_HEIGHT - 0.1, POLE_LENGTH + 0.3)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=7, pad=4)
    ax.set_xlabel("cart position (m)", fontsize=8)
    ax.tick_params(labelsize=7)

    cart_rect = patches.FancyBboxPatch(
        (-CART_WIDTH / 2, -CART_HEIGHT / 2), CART_WIDTH, CART_HEIGHT,
        boxstyle="round,pad=0.02",
        linewidth=1, edgecolor="black", facecolor="steelblue",
    )
    ax.add_patch(cart_rect)
    (pole_line,) = ax.plot([], [], "o-", color="firebrick", linewidth=2.5, markersize=5)
    step_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=7, va="top")

    def _update(frame: int):
        cx = float(cart_pos[frame])
        pa = float(pole_ang[frame])
        cart_rect.set_x(cx - CART_WIDTH / 2)
        pole_tip_x = cx + POLE_LENGTH * np.sin(pa)
        pole_tip_y = POLE_LENGTH * np.cos(pa)
        pole_line.set_data([cx, pole_tip_x], [0.0, pole_tip_y])
        step_text.set_text(f"t={frame}/{T - 1}")
        return cart_rect, pole_line, step_text

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=int(1000 / fps), blit=True
    )
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    return output_path


def render_episode_gifs(
    selected_records: list,
    trajectories_dirs: List[str],
    output_dir: str,
    feedback_type: str,
    sigma_breakdown: dict,
    fps: int = 20,
    max_gifs: int = 2,
) -> List[str]:
    """Render GIFs for selected episodes, returning a list of written GIF paths.

    Tries exact matching by EpisodeRecord.traj_key first, then falls back to
    loading the most recently saved trajectories from each directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    matched: Dict[str, tuple] = {}  # traj_key -> (record | None, traj_data)

    # Attempt precise match by traj_key
    for record in selected_records[:max_gifs]:
        tk = getattr(record, "traj_key", None)
        if tk is not None:
            for tdir in trajectories_dirs:
                if not tdir:
                    continue
                found = load_trajectories(tdir, [tk])
                if tk in found:
                    matched[tk] = (record, found[tk])
                    break

    # Fallback to most recent trajectories when not enough matched
    if len(matched) < max_gifs:
        for tdir in trajectories_dirs:
            if len(matched) >= max_gifs:
                break
            n_needed = max_gifs - len(matched)
            recent = load_recent_trajectories(tdir, n=n_needed)
            for tk, traj_data in recent.items():
                if tk not in matched:
                    matched[tk] = (None, traj_data)
                    if len(matched) >= max_gifs:
                        break

    mean_sigma = sigma_breakdown.get("mean_sigma_overall", 0.0)
    gif_paths = []
    for i, (traj_key, (record, traj_data)) in enumerate(list(matched.items())[:max_gifs]):
        sigma_val = record.sigma if record is not None else mean_sigma
        title = f"σ={sigma_val:.3f}  |  {feedback_type}  |  key: {traj_key}"
        out_path = os.path.join(output_dir, f"episode_{i}.gif")
        try:
            render_cartpole_gif(traj_data, out_path, title=title, fps=fps)
            gif_paths.append(out_path)
            logging.info(f"[renderer] Rendered {out_path}")
        except Exception as exc:
            logging.warning(f"[renderer] Failed to render traj_key={traj_key}: {exc}")

    return gif_paths
