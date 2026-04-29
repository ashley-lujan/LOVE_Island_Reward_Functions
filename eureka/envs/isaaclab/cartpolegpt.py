"""Isaac Lab DirectRLEnv implementation of Cartpole for Eureka reward generation.

Exposes the same self.* attributes as the IsaacGym version so that LLM-generated
reward functions are portable between the two backends.

Eureka reward injection contract:
  - Eureka writes a compute_reward() function to the bottom of this file
  - compute_reward() receives self.joint_pos and self.joint_vel as positional args
  - Returns Tuple[torch.Tensor, Dict[str, torch.Tensor]]
  - _get_rewards() calls compute_reward() and stores results in self.rew_buf / self.rew_dict

USD path: The cartpole.usda is stored in the local assets dir, bind-mounted to
/local-assets inside the container by isaaclab_train.py / slurm_love_island.sh.
Generate it once with: python scripts/create_cartpole_usd.py --output <path>
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import Tuple, Dict

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


# Bind-mounted inside the container by eureka.py / slurm_love_island.sh:
#   --bind /scratch/general/vast/${USER}/isaac-assets:/local-assets
CARTPOLE_USD_PATH = (
    "/local-assets/Isaac/IsaacLab/Robots/Classic/Cartpole/cartpole.usda"
)

CARTPOLE_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=CARTPOLE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0},
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2)
    decimation: int = 2

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=4.0, replicate_physics=True
    )

    # Robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG
    cart_dof_name: str = "slider_to_cart"
    pole_dof_name: str = "cart_to_pole"

    action_scale: float = 100.0  # [N]

    # Task parameters
    episode_length_s: float = 5.0
    max_cart_pos: float = 3.0
    initial_pole_angle_range = [-0.25, 0.25]

    # RL dimensions
    observation_space: int = 4
    action_space: int = 1
    num_states: int = 0

    # Terminal state saving (set by love_island.py / isaaclab_train.py --states_dir)
    states_dir: str = ""
    # Episode trajectory saving for GIF rendering (set via --trajectories_dir)
    trajectories_dir: str = ""
    max_traj_files: int = 100


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)

        self.action_scale = self.cfg.action_scale

        # Live references to joint state tensors (auto-update in-place each step).
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        # Buffers exposed to Eureka's reward functions (mirrors IsaacGym API)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.rew_dict: Dict[str, torch.Tensor] = {}
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.extras: Dict[str, torch.Tensor] = {}
        self.consecutive_successes = torch.zeros(1, device=self.device)

        # TensorBoard writer — set externally by isaaclab_train.py before training starts.
        # rl_games logs extras with an "episode/" prefix which Eureka's parser skips,
        # so we write gt_reward / gpt_reward / consecutive_successes here directly.
        self._tb_writer = None
        self._tb_step: int = 0

        # Cache for last gt_reward — populated in _get_rewards(), read in _get_dones()
        self._gt_reward_buf: torch.Tensor = torch.zeros(self.num_envs, device=self.device)

        # Trajectory ring buffer: env_idx -> ([jp_t0, jp_t1, ...], [jv_t0, jv_t1, ...])
        self._traj_buf: dict = {}
        self._traj_queue: deque = deque()  # (step, env_idx, filename) in save order

    @property
    def progress_buf(self) -> torch.Tensor:
        """Alias for episode_length_buf (IsaacGym API compatibility)."""
        return self.episode_length_buf

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["cartpole"] = self.cartpole
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pole_angle = self.joint_pos[:, self._pole_dof_idx[0]]
        pole_vel   = self.joint_vel[:, self._pole_dof_idx[0]]
        cart_pos   = self.joint_pos[:, self._cart_dof_idx[0]]
        cart_vel   = self.joint_vel[:, self._cart_dof_idx[0]]

        # Ground-truth reward (always computed, logged as gt_reward)
        gt_reward = _compute_gt_reward(
            pole_angle, pole_vel, cart_vel, cart_pos,
            self.cfg.max_cart_pos, self.episode_length_buf, self.max_episode_length,
        )
        self._gt_reward_buf[:] = gt_reward
        self.extras["gt_reward"] = gt_reward.mean()
        self.extras["consecutive_successes"] = self.consecutive_successes.mean()

        # Eureka-generated reward (injected at bottom of file by Eureka pipeline)
        if "compute_reward" in globals():
            self.rew_buf[:], self.rew_dict = compute_reward(  # noqa: F821
                self.joint_pos, self.joint_vel
            )
            self.extras["gpt_reward"] = self.rew_buf.mean()
            for k, v in self.rew_dict.items():
                self.extras[k] = v.mean()
        else:
            self.rew_buf[:] = gt_reward

        # Write directly to TensorBoard so keys appear without the "episode/" prefix
        # that rl_games adds, which Eureka's parser skips.
        self._tb_step += 1
        if self._tb_writer is not None and self._tb_step % 10 == 0:
            self._tb_writer.add_scalar("gt_reward", gt_reward.mean().item(), self._tb_step)
            self._tb_writer.add_scalar(
                "consecutive_successes",
                self.consecutive_successes.mean().item(), self._tb_step,
            )
            if "compute_reward" in globals():
                self._tb_writer.add_scalar("gpt_reward", self.rew_buf.mean().item(), self._tb_step)

        # Accumulate per-step joint state for trajectory saving
        if self.cfg.trajectories_dir:
            jp = self.joint_pos.detach().cpu()  # [N, J]
            jv = self.joint_vel.detach().cpu()  # [N, J]
            for env_idx in range(self.num_envs):
                if env_idx not in self._traj_buf:
                    self._traj_buf[env_idx] = ([], [])
                self._traj_buf[env_idx][0].append(jp[env_idx].clone())
                self._traj_buf[env_idx][1].append(jv[env_idx].clone())

        return self.rew_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Refresh live references
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(
            torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1
        )
        out_of_bounds = out_of_bounds | torch.any(
            torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1
        )

        done = out_of_bounds | time_out
        if done.sum() > 0:
            self.consecutive_successes[0] = (
                self.episode_length_buf.float() * done.float()
            ).sum() / done.float().sum()
            if self.cfg.states_dir:
                self._save_terminal_states(done)
            if self.cfg.trajectories_dir:
                done_indices = done.nonzero(as_tuple=False).squeeze(-1)
                for env_idx in done_indices.tolist():
                    self._flush_trajectory(int(env_idx))

        self.reset_buf[:] = done.long()
        return out_of_bounds, time_out

    def _save_terminal_states(self, done_mask: torch.Tensor) -> None:
        import os
        os.makedirs(self.cfg.states_dir, exist_ok=True)
        path = os.path.join(self.cfg.states_dir, f"terminal_{self._tb_step}.pt")
        torch.save(
            {
                "joint_pos": self.joint_pos[done_mask].cpu(),
                "joint_vel": self.joint_vel[done_mask].cpu(),
                "gt_reward": self._gt_reward_buf[done_mask].cpu(),
            },
            path,
        )

    def _flush_trajectory(self, env_idx: int) -> None:
        import json, os
        buf = self._traj_buf.pop(env_idx, None)
        if not buf or not buf[0]:
            return
        jp_seq = torch.stack(buf[0], dim=0)   # [T, J]
        jv_seq = torch.stack(buf[1], dim=0)   # [T, J]
        step = self._tb_step
        fname = f"traj_{step}_{env_idx}.pt"
        os.makedirs(self.cfg.trajectories_dir, exist_ok=True)
        fpath = os.path.join(self.cfg.trajectories_dir, fname)
        torch.save(
            {
                "joint_pos_seq": jp_seq,
                "joint_vel_seq": jv_seq,
                "episode_length": int(jp_seq.shape[0]),
                "cart_dof_idx": int(self._cart_dof_idx[0]),
                "pole_dof_idx": int(self._pole_dof_idx[0]),
            },
            fpath,
        )
        # Append to JSONL index (append-only, no lock needed in single-process training)
        idx_path = os.path.join(self.cfg.trajectories_dir, "index.json")
        with open(idx_path, "a") as f:
            f.write(json.dumps({"step": step, "env_idx": env_idx, "filename": fname}) + "\n")
        # Ring buffer: evict oldest file when over limit
        self._traj_queue.append((step, env_idx, fname))
        while len(self._traj_queue) > self.cfg.max_traj_files:
            _, _, old_fname = self._traj_queue.popleft()
            old_path = os.path.join(self.cfg.trajectories_dir, old_fname)
            if os.path.exists(old_path):
                os.remove(old_path)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.reset_buf[env_ids] = 0


# ---------------------------------------------------------------------------
# Ground-truth reward (always present, used for correlation metric)
# ---------------------------------------------------------------------------

@torch.jit.script
def _compute_gt_reward(
    pole_angle: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    max_cart_pos: float,
    progress_buf: torch.Tensor,
    max_episode_length: float,
) -> torch.Tensor:
    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
    reward = torch.where(torch.abs(cart_pos) > max_cart_pos, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > 1.5708, torch.ones_like(reward) * -2.0, reward)
    return reward


# ---------------------------------------------------------------------------
# Eureka injects compute_reward() here during reward generation
# ---------------------------------------------------------------------------

from typing import Tuple, Dict
import torch
@torch.jit.script
def compute_reward(joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward analysis of previous design:
    - task_score showed perfect max but low mean, indicating the policy can occasionally succeed
      but is not consistently learning a stable, smooth balancing strategy.
    - The prior reward used mostly mild exponentials with small regularizers. For Cartpole, the
      dominant signal should be the pole angle, with velocity damping to prevent oscillation.
    - Cart centering should be kept as a weak auxiliary term only.

    New design:
    - Stronger upright term with a tighter temperature so angle differences matter more.
    - Explicit velocity stabilization for pole angular velocity.
    - Small cart position and cart velocity penalties as weak regularizers.
    - Add a binary-like success shaping term to reinforce near-upright states.
    """

    cart_pos = joint_pos[:, 0]
    pole_angle = joint_pos[:, 1]
    cart_vel = joint_vel[:, 0]
    pole_ang_vel = joint_vel[:, 1]

    # Temperatures for shaping
    angle_temp = 0.04
    ang_vel_temp = 0.10
    cart_pos_temp = 2.00
    cart_vel_temp = 2.00
    success_angle_temp = 0.02

    # Main balance term
    upright_reward = torch.exp(-(pole_angle * pole_angle) / angle_temp)

    # Damping term to reduce pole oscillation
    angular_stability_reward = torch.exp(-(pole_ang_vel * pole_ang_vel) / ang_vel_temp)

    # Auxiliary regularizers
    cart_centering_reward = torch.exp(-(cart_pos * cart_pos) / cart_pos_temp)
    cart_smoothness_reward = torch.exp(-(cart_vel * cart_vel) / cart_vel_temp)

    # Extra reinforcement when the pole is very close to upright
    success_bonus = torch.exp(-(pole_angle * pole_angle) / success_angle_temp)

    # Weighted sum with balance as the dominant objective
    reward = (
        0.52 * upright_reward +
        0.20 * angular_stability_reward +
        0.10 * success_bonus +
        0.10 * cart_centering_reward +
        0.08 * cart_smoothness_reward
    )

    reward_components: Dict[str, torch.Tensor] = {
        "upright_reward": upright_reward,
        "angular_stability_reward": angular_stability_reward,
        "success_bonus": success_bonus,
        "cart_centering_reward": cart_centering_reward,
        "cart_smoothness_reward": cart_smoothness_reward,
    }

    return reward, reward_components
