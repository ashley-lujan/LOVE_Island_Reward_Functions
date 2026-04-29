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

# Add this new function alongside render_cartpole_gif:
# Ant geometry constants (from nv_ant.xml, approximate top-down layout)
# Each tuple: (base_angle_rad, upper_len, lower_len)
_ANT_LEGS = [
    (-0.785, 0.28, 0.28),   # hip_1 / ankle_1  — front right  (-45°)
    ( 2.356, 0.28, 0.28),   # hip_2 / ankle_2  — back  left  (135°)
    (-2.356, 0.28, 0.28),   # hip_3 / ankle_3  — back  right (-135°)
    ( 0.785, 0.28, 0.28),   # hip_4 / ankle_4  — front left   (45°)
]
_TORSO_RADIUS = 0.15


def _draw_ant_pose(ax, hip_angles, ankle_angles, cx=0.0, cy=0.0, heading=0.0):
    """Draw a top-down stick-figure ant at (cx, cy) with given joint angles."""
    # Torso
    torso = plt.Circle((cx, cy), _TORSO_RADIUS, color="steelblue", zorder=3)
    ax.add_patch(torso)
    # Heading indicator
    ax.annotate("", xy=(cx + 0.25 * np.cos(heading), cy + 0.25 * np.sin(heading)),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5), zorder=4)

    colors = ["firebrick", "seagreen", "darkorange", "mediumpurple"]
    for i, (base_ang, upper_len, lower_len) in enumerate(_ANT_LEGS):
        world_base = heading + base_ang
        hip_ang = hip_angles[i] if i < len(hip_angles) else 0.0
        ankle_ang = ankle_angles[i] if i < len(ankle_angles) else 1.0

        # Upper leg direction (hip angle rotates around world_base)
        upper_dir = world_base + hip_ang
        knee_x = cx + upper_len * np.cos(upper_dir)
        knee_y = cy + upper_len * np.sin(upper_dir)

        # Lower leg bends inward relative to upper
        lower_dir = upper_dir + (ankle_ang - 1.0)  # ankle ~1 rad = roughly straight
        foot_x = knee_x + lower_len * np.cos(lower_dir)
        foot_y = knee_y + lower_len * np.sin(lower_dir)

        ax.plot([cx, knee_x], [cy, knee_y], color=colors[i], lw=2.5, zorder=2)
        ax.plot([knee_x, foot_x], [knee_y, foot_y], color=colors[i], lw=1.5,
                alpha=0.8, zorder=2)
        ax.plot(foot_x, foot_y, "o", color=colors[i], markersize=3, zorder=3)


def render_ant_gif(
    traj_data: dict,
    output_path: str,
    title: str = "",
    fps: int = 20,
) -> str:
    jp = traj_data["joint_pos_seq"].numpy()    # [T, 8]
    T = jp.shape[0]

    # Root position: use saved root_pos_seq if available, else zeros
    if "root_pos_seq" in traj_data:
        root_pos = traj_data["root_pos_seq"].numpy()   # [T, 3]
    else:
        root_pos = np.zeros((T, 3))

    xs = root_pos[:, 0]
    ys = root_pos[:, 1]

    # Pad view so ant is never right at the edge
    x_range = max(xs.max() - xs.min(), 1.0)
    y_range = max(ys.max() - ys.min(), 1.0)
    pad = 0.5
    x_mid = (xs.max() + xs.min()) / 2
    y_mid = (ys.max() + ys.min()) / 2

    fig, (ax_path, ax_pose) = plt.subplots(
        1, 2, figsize=(8, 4),
        gridspec_kw={"width_ratios": [1.6, 1]}
    )
    fig.suptitle(title, fontsize=7)

    # ── Left panel: top-down path ──────────────────────────────────────────
    ax_path.set_xlim(x_mid - x_range / 2 - pad, x_mid + x_range / 2 + pad)
    ax_path.set_ylim(y_mid - y_range / 2 - pad, y_mid + y_range / 2 + pad)
    ax_path.set_aspect("equal")
    ax_path.set_xlabel("x (m)", fontsize=8)
    ax_path.set_ylabel("y (m)", fontsize=8)
    ax_path.set_title("Top-down path", fontsize=8)
    ax_path.tick_params(labelsize=7)

    # Full path as faint grey trail
    ax_path.plot(xs, ys, color="lightgray", lw=1, zorder=1)

    (trail_line,) = ax_path.plot([], [], color="steelblue", lw=1.5,
                                  alpha=0.6, zorder=2)
    (pos_dot,) = ax_path.plot([], [], "o", color="firebrick",
                               markersize=6, zorder=3)
    step_text = ax_path.text(0.02, 0.97, "", transform=ax_path.transAxes,
                              fontsize=7, va="top")

    # ── Right panel: stick figure pose ────────────────────────────────────
    pose_half = 0.8
    ax_pose.set_xlim(-pose_half, pose_half)
    ax_pose.set_ylim(-pose_half, pose_half)
    ax_pose.set_aspect("equal")
    ax_pose.set_title("Ant pose (top-down)", fontsize=8)
    ax_pose.axis("off")

    # Hip joints: indices 0, 2, 4, 6  |  Ankle joints: 1, 3, 5, 7
    hip_idx   = [0, 2, 4, 6]
    ankle_idx = [1, 3, 5, 7]

    pose_artists = []

    def _update(frame: int):
        # Update path trail
        trail_line.set_data(xs[:frame + 1], ys[:frame + 1])
        pos_dot.set_data([xs[frame]], [ys[frame]])
        step_text.set_text(f"t={frame}/{T - 1}")

        # Redraw pose panel
        for artist in pose_artists:
            artist.remove()
        pose_artists.clear()

        hip_angs   = [float(jp[frame, i]) for i in hip_idx]
        ankle_angs = [float(jp[frame, i]) for i in ankle_idx]

        # Approximate heading from motion direction
        if frame > 0:
            dx = xs[frame] - xs[frame - 1]
            dy = ys[frame] - ys[frame - 1]
            heading = np.arctan2(dy, dx) if (dx**2 + dy**2) > 1e-6 else 0.0
        else:
            heading = 0.0

        # Torso circle
        torso = plt.Circle((0, 0), _TORSO_RADIUS, color="steelblue", zorder=3)
        ax_pose.add_patch(torso)
        pose_artists.append(torso)

        # Heading arrow
        arr = ax_pose.annotate(
            "", xy=(0.25 * np.cos(heading), 0.25 * np.sin(heading)),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="white", lw=1.5), zorder=4
        )
        pose_artists.append(arr)

        leg_colors = ["firebrick", "seagreen", "darkorange", "mediumpurple"]
        for i, (base_ang, upper_len, lower_len) in enumerate(_ANT_LEGS):
            world_base = heading + base_ang
            upper_dir  = world_base + hip_angs[i]
            knee_x = upper_len * np.cos(upper_dir)
            knee_y = upper_len * np.sin(upper_dir)
            lower_dir = upper_dir + (ankle_angs[i] - 1.0)
            foot_x = knee_x + lower_len * np.cos(lower_dir)
            foot_y = knee_y + lower_len * np.sin(lower_dir)

            (ul,) = ax_pose.plot([0, knee_x], [0, knee_y],
                                  color=leg_colors[i], lw=2.5, zorder=2)
            (ll,) = ax_pose.plot([knee_x, foot_x], [knee_y, foot_y],
                                  color=leg_colors[i], lw=1.5, alpha=0.8, zorder=2)
            (fd,) = ax_pose.plot(foot_x, foot_y, "o",
                                  color=leg_colors[i], markersize=3, zorder=3)
            pose_artists.extend([ul, ll, fd])

        return [trail_line, pos_dot, step_text] + pose_artists

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=int(1000 / fps), blit=False
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
    os.makedirs(output_dir, exist_ok=True)

    matched: Dict[str, tuple] = {}

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
    # Use A, B, C... naming instead of 0, 1, 2...
    labels = [chr(ord("A") + i) for i in range(max_gifs)]

    for i, (traj_key, (record, traj_data)) in enumerate(list(matched.items())[:max_gifs]):
        sigma_val = record.sigma if record is not None else mean_sigma
        label = labels[i]
        title = f"Episode {label}  |  σ={sigma_val:.3f}  |  {feedback_type}"
        out_path = os.path.join(output_dir, f"episode_{label}.gif")

        # Detect env from saved metadata, fall back to cartpole
        env_name = traj_data.get("env_name", "cartpole")
        try:
            if env_name == "ant":
                render_ant_gif(traj_data, out_path, title=title, fps=fps)
            else:
                render_cartpole_gif(traj_data, out_path, title=title, fps=fps)
            gif_paths.append(out_path)
            logging.info(f"[renderer] Rendered episode_{label}.gif ({env_name})")
        except Exception as exc:
            logging.warning(f"[renderer] Failed to render traj_key={traj_key}: {exc}")

    return gif_paths