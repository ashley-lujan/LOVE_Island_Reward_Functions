"""Isaac Lab DirectRLEnv implementation of Ant for Eureka reward generation.

Exposes the same self.* attributes as the IsaacGym version so that LLM-generated
reward functions are portable between the two backends.

Eureka reward injection contract:
  - Eureka writes a compute_reward() function to the bottom of this file
  - compute_reward() receives self.joint_pos and self.joint_vel as positional args
  - Returns Tuple[torch.Tensor, Dict[str, torch.Tensor]]
  - _get_rewards() calls compute_reward() and stores results in self.rew_buf / self.rew_dict

USD path: Run Step 0 from the plan (find ant USD in container) before using this env.
  Expected: /local-assets/Isaac/IsaacLab/Robots/Classic/Ant/ant.usd
  Fallback: convert eureka/envs/isaac/assets/mjcf/nv_ant.xml via Isaac Lab's mjcf_converter.
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
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import normalize


# Bind-mounted inside the container by eureka.py / slurm_love_island.sh:
#   --bind /scratch/general/vast/${USER}/isaac-assets:/local-assets
# Run Step 0 from the plan to verify this path inside the Apptainer container.
ANT_USD_PATH = "/local-assets/Isaac/IsaacLab/Robots/Classic/Ant/ant.usd"

ANT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ANT_USD_PATH,
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
        # z=0.55 matches the MuJoCo default init_qpos for nv_ant.xml
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "hip_.*": 0.0,
            "ankle_1": 1.0,
            "ankle_2": -1.0,
            "ankle_3": -1.0,
            "ankle_4": 1.0,
        },
    ),
    actuators={
        # Single group covers all 8 DOFs regardless of asset joint naming.
        # drive_mode: effort only (stiffness=0, damping=0 mirrors IsaacGym DOF_MODE_NONE)
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
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
class AntEnvCfg(DirectRLEnvCfg):
    # Simulation — 60 Hz physics, 2-step decimation → 30 Hz policy
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=2)
    decimation: int = 2

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=5.0, replicate_physics=True
    )

    # Robot
    robot_cfg: ArticulationCfg = ANT_CFG

    # Contact sensors on all foot bodies (prim_path uses regex to match ".*foot.*")
    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*foot",
        history_length=3,
        track_air_time=False,
    )

    # Action scale: mirrors IsaacGym's joint_gears(150) * power_scale(0.003) ≈ 0.45
    action_scale: float = 0.5

    # Task parameters
    episode_length_s: float = 16.0   # ~960 steps at 60 Hz / 2 decimation
    termination_height: float = 0.27  # torso falls below this → episode ends

    # RL dimensions
    observation_space: int = 60
    action_space: int = 8
    num_states: int = 0

    # GT reward weights (mirror IsaacGym ant config defaults)
    up_weight: float = 0.1
    heading_weight: float = 0.5
    actions_cost_scale: float = 0.005
    energy_cost_scale: float = 0.05
    joints_at_limit_cost_scale: float = 0.1
    death_cost: float = -1.0
    alive_reward_scale: float = 0.5

    # Observation scaling
    dof_vel_scale: float = 0.2
    contact_force_scale: float = 0.01

    # Terminal state / trajectory saving (set by isaaclab_train.py)
    states_dir: str = ""
    trajectories_dir: str = ""
    max_traj_files: int = 100


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AntEnv(DirectRLEnv):
    cfg: AntEnvCfg

    def __init__(self, cfg: AntEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Live references — Isaac Lab updates these tensors in-place each step.
        self.joint_pos = self.ant.data.joint_pos   # (num_envs, 8)
        self.joint_vel = self.ant.data.joint_vel   # (num_envs, 8)
        self.root_pos = self.ant.data.root_pos_w   # (num_envs, 3) world frame
        self.root_quat = self.ant.data.root_quat_w  # (num_envs, 4) w,x,y,z
        # Body-frame velocities are pre-transformed by Isaac Lab — no quat math needed.
        self.root_lin_vel = self.ant.data.root_lin_vel_b   # (num_envs, 3)
        self.root_ang_vel = self.ant.data.root_ang_vel_b   # (num_envs, 3)

        # Contact forces exposed to Eureka: (num_envs, 4, 6) — forces on 4 feet.
        # Torque channels ([:, :, 3:]) are zero-padded (Isaac Lab only exposes net forces).
        self.contact_forces = torch.zeros(self.num_envs, 4, 6, device=self.device)

        # Target far ahead in the +x direction; ant is rewarded for reaching it.
        self.targets = torch.tensor([1000.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Potential-based progress reward (mirrors IsaacGym ant)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.potentials = torch.full((self.num_envs,), -1000.0 / self.dt, device=self.device)
        self.prev_potentials = self.potentials.clone()

        # DOF limits for unscaling positions to [-1, 1]
        dof_limits = self.ant.data.soft_joint_pos_limits  # (num_envs, 8, 2) or (8, 2)
        if dof_limits.dim() == 3:
            self._dof_limits_lower = dof_limits[0, :, 0]
            self._dof_limits_upper = dof_limits[0, :, 1]
        else:
            self._dof_limits_lower = dof_limits[:, 0]
            self._dof_limits_upper = dof_limits[:, 1]

        # Last action buffer — included in obs at indices [52:60]
        self._last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Intermediate obs buffer shared between _get_observations → _get_rewards
        self.obs_buf = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)

        # Buffers mirroring IsaacGym API
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.rew_dict: Dict[str, torch.Tensor] = {}
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.extras: Dict[str, torch.Tensor] = {}
        self.consecutive_successes = torch.zeros(1, device=self.device)

        # TensorBoard writer set externally by isaaclab_train.py
        self._tb_writer = None
        self._tb_step: int = 0
        self._gt_reward_buf = torch.zeros(self.num_envs, device=self.device)

        self._traj_buf: dict = {}
        self._traj_queue: deque = deque()

    @property
    def progress_buf(self) -> torch.Tensor:
        """IsaacGym API alias for episode_length_buf."""
        return self.episode_length_buf

    def _setup_scene(self):
        self.ant = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["ant"] = self.ant
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        # TerrainImporter with terrain_type="plane" creates a physics-only flat plane
        # without loading any USD from ISAACLAB_NUCLEUS_DIR (which is unset on CHPC).
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )
        self.terrain = TerrainImporter(terrain_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._last_actions[:] = actions.clone()
        self.actions = self.cfg.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.ant.set_joint_effort_target(self.actions)

    def _get_observations(self) -> dict:
        # Refresh live references
        self.joint_pos = self.ant.data.joint_pos
        self.joint_vel = self.ant.data.joint_vel
        self.root_pos = self.ant.data.root_pos_w
        self.root_quat = self.ant.data.root_quat_w
        self.root_lin_vel = self.ant.data.root_lin_vel_b
        self.root_ang_vel = self.ant.data.root_ang_vel_b

        # Contact forces from sensor: (N, num_foot_bodies, 3) → (N, 4, 6)
        net_forces = self.contact_sensor.data.net_forces_w  # (N, K, 3)
        num_feet = min(net_forces.shape[1], 4)
        self.contact_forces[:, :num_feet, :3] = (
            net_forces[:, :num_feet, :] * self.cfg.contact_force_scale
        )

        # Potentials for progress reward
        to_target = self.targets - self.root_pos
        to_target[:, 2] = 0.0
        self.prev_potentials[:] = self.potentials.clone()
        self.potentials[:] = -torch.norm(to_target, p=2, dim=-1) / self.dt

        # Heading and up projections from quaternion (Isaac Lab convention: w,x,y,z)
        w = self.root_quat[:, 0]
        x = self.root_quat[:, 1]
        y = self.root_quat[:, 2]
        z = self.root_quat[:, 3]

        # Up projection: z-component of the ant's local-up axis in world frame.
        # R * [0,0,1] = [2(xz+wy), 2(yz-wx), 1-2(x²+y²)]
        up_proj = 1.0 - 2.0 * (x * x + y * y)

        # Heading: x-axis of ant body in world frame.
        # R * [1,0,0] = [1-2(y²+z²), 2(xy+wz), 2(xz-wy)]
        heading_x = 1.0 - 2.0 * (y * y + z * z)
        heading_y = 2.0 * (x * y + w * z)
        heading_z = 2.0 * (x * z - w * y)
        heading_vec = torch.stack([heading_x, heading_y, heading_z], dim=-1)

        target_dirs = normalize(to_target + 1e-8)  # avoid zero-norm
        heading_proj = (heading_vec * target_dirs).sum(dim=-1)

        # Euler angles from quaternion
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

        walk_target_angle = torch.atan2(
            self.targets[:, 1] - self.root_pos[:, 1],
            self.targets[:, 0] - self.root_pos[:, 0],
        )
        angle_to_target = walk_target_angle - yaw

        # DOF positions scaled to [-1, 1] using joint limits
        dof_pos_scaled = _unscale(self.joint_pos, self._dof_limits_lower, self._dof_limits_upper)

        # 60D observation vector — indices match IsaacGym ant obs_buf layout so that
        # the GT reward function (compute_success) indices are identical.
        obs = torch.cat(
            [
                self.root_pos[:, 2:3],                                # [0]    torso height
                self.root_lin_vel,                                     # [1:4]  body-frame linear vel
                self.root_ang_vel,                                     # [4:7]  body-frame angular vel
                yaw.unsqueeze(-1),                                     # [7]    yaw
                roll.unsqueeze(-1),                                    # [8]    roll
                angle_to_target.unsqueeze(-1),                        # [9]    angle to target
                up_proj.unsqueeze(-1),                                # [10]   up projection
                heading_proj.unsqueeze(-1),                           # [11]   heading projection
                dof_pos_scaled,                                        # [12:20] DOF positions scaled
                self.joint_vel * self.cfg.dof_vel_scale,              # [20:28] DOF velocities scaled
                self.contact_forces.reshape(self.num_envs, -1)[:, :24],  # [28:52] contact forces
                self._last_actions,                                    # [52:60] last actions
            ],
            dim=-1,
        )
        self.obs_buf = obs
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        gt_reward = _compute_gt_reward(
            obs_buf=self.obs_buf,
            actions=self.actions,
            potentials=self.potentials,
            prev_potentials=self.prev_potentials,
            up_weight=self.cfg.up_weight,
            heading_weight=self.cfg.heading_weight,
            actions_cost_scale=self.cfg.actions_cost_scale,
            energy_cost_scale=self.cfg.energy_cost_scale,
            joints_at_limit_cost_scale=self.cfg.joints_at_limit_cost_scale,
            termination_height=self.cfg.termination_height,
            death_cost=self.cfg.death_cost,
            alive_reward_scale=self.cfg.alive_reward_scale,
        )
        self._gt_reward_buf[:] = gt_reward
        self.extras["gt_reward"] = gt_reward.mean()
        self.extras["consecutive_successes"] = self.consecutive_successes.mean()

        if "compute_reward" in globals():
            self.rew_buf[:], self.rew_dict = compute_reward(  # noqa: F821
                self.joint_pos, self.joint_vel
            )
            self.extras["gpt_reward"] = self.rew_buf.mean()
            for k, v in self.rew_dict.items():
                self.extras[k] = v.mean()
        else:
            self.rew_buf[:] = gt_reward

        self._tb_step += 1
        if self._tb_writer is not None and self._tb_step % 10 == 0:
            self._tb_writer.add_scalar("gt_reward", gt_reward.mean().item(), self._tb_step)
            self._tb_writer.add_scalar(
                "consecutive_successes",
                self.consecutive_successes.mean().item(),
                self._tb_step,
            )
            if "compute_reward" in globals():
                self._tb_writer.add_scalar("gpt_reward", self.rew_buf.mean().item(), self._tb_step)

        if self.cfg.trajectories_dir:
            jp = self.joint_pos.detach().cpu()
            jv = self.joint_vel.detach().cpu()
            for env_idx in range(self.num_envs):
                if env_idx not in self._traj_buf:
                    self._traj_buf[env_idx] = ([], [])
                self._traj_buf[env_idx][0].append(jp[env_idx].clone())
                self._traj_buf[env_idx][1].append(jv[env_idx].clone())

        return self.rew_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Refresh live references in case _get_observations was not called this step
        self.joint_pos = self.ant.data.joint_pos
        self.joint_vel = self.ant.data.joint_vel
        self.root_pos = self.ant.data.root_pos_w

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen = self.root_pos[:, 2] < self.cfg.termination_height

        done = fallen | time_out
        if done.sum() > 0:
            # Track average episode length as "consecutive_successes" metric
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
        return fallen, time_out

    def _save_terminal_states(self, done_mask: torch.Tensor) -> None:
        import os
        os.makedirs(self.cfg.states_dir, exist_ok=True)
        path = os.path.join(self.cfg.states_dir, f"terminal_{self._tb_step}.pt")
        torch.save(
            {
                "joint_pos": self.joint_pos[done_mask].cpu(),
                "joint_vel": self.joint_vel[done_mask].cpu(),
                "root_pos": self.root_pos[done_mask].cpu(),
                "gt_reward": self._gt_reward_buf[done_mask].cpu(),
            },
            path,
        )

    def _flush_trajectory(self, env_idx: int) -> None:
        import json, os
        buf = self._traj_buf.pop(env_idx, None)
        if not buf or not buf[0]:
            return
        jp_seq = torch.stack(buf[0], dim=0)
        jv_seq = torch.stack(buf[1], dim=0)
        step = self._tb_step
        fname = f"traj_{step}_{env_idx}.pt"
        os.makedirs(self.cfg.trajectories_dir, exist_ok=True)
        fpath = os.path.join(self.cfg.trajectories_dir, fname)
        torch.save({"joint_pos_seq": jp_seq, "joint_vel_seq": jv_seq,
                    "episode_length": int(jp_seq.shape[0])}, fpath)
        idx_path = os.path.join(self.cfg.trajectories_dir, "index.json")
        with open(idx_path, "a") as f:
            f.write(json.dumps({"step": step, "env_idx": env_idx, "filename": fname}) + "\n")
        self._traj_queue.append((step, env_idx, fname))
        while len(self._traj_queue) > self.cfg.max_traj_files:
            _, _, old_fname = self._traj_queue.popleft()
            old_path = os.path.join(self.cfg.trajectories_dir, old_fname)
            if os.path.exists(old_path):
                os.remove(old_path)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.ant._ALL_INDICES
        super()._reset_idx(env_ids)

        # Small random perturbation around default pose (mirrors IsaacGym reset_idx)
        joint_pos = self.ant.data.default_joint_pos[env_ids]
        joint_pos += torch.rand_like(joint_pos) * 0.4 - 0.2  # ±0.2 rad
        joint_pos = torch.clamp(
            joint_pos,
            self._dof_limits_lower.unsqueeze(0),
            self._dof_limits_upper.unsqueeze(0),
        )
        joint_vel = torch.rand_like(joint_pos) * 0.2 - 0.1   # ±0.1 rad/s

        default_root_state = self.ant.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.ant.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.ant.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.ant.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # Reset potentials for progress reward
        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.prev_potentials[env_ids] = self.potentials[env_ids].clone()

        self.reset_buf[env_ids] = 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map x from [lower, upper] to [-1, 1], matching IsaacGym's unscale()."""
    return 2.0 * (x - lower) / (upper - lower + 1e-8) - 1.0


# ---------------------------------------------------------------------------
# Ground-truth reward (always present, used for correlation metric)
# Mirrors IsaacGym's compute_success() — obs indices are identical.
# ---------------------------------------------------------------------------

@torch.jit.script
def _compute_gt_reward(
    obs_buf: torch.Tensor,
    actions: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    actions_cost_scale: float,
    energy_cost_scale: float,
    joints_at_limit_cost_scale: float,
    termination_height: float,
    death_cost: float,
    alive_reward_scale: float,
) -> torch.Tensor:
    # obs_buf layout matches IsaacGym:
    #   [10] up_proj, [11] heading_proj, [12:20] dof_pos_scaled, [20:28] dof_vel_scaled

    heading_weight_t = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(
        obs_buf[:, 11] > 0.8,
        heading_weight_t,
        heading_weight * obs_buf[:, 11] / 0.8,
    )

    up_reward = torch.where(
        obs_buf[:, 10] > 0.93,
        torch.ones_like(heading_reward) * up_weight,
        torch.zeros_like(heading_reward),
    )

    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 20:28]), dim=-1)
    dof_at_limit_cost = torch.sum((obs_buf[:, 12:20] > 0.99).float(), dim=-1)

    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost * joints_at_limit_cost_scale
    )

    # Death penalty when torso falls below termination height
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height,
        torch.ones_like(total_reward) * death_cost,
        total_reward,
    )
    return total_reward


# ---------------------------------------------------------------------------
# Eureka injects compute_reward() here during reward generation
# ---------------------------------------------------------------------------

from typing import Tuple, Dict
import torch
@torch.jit.script
def compute_reward(joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # NOTE:
    # The provided contract only allows (joint_pos, joint_vel) as inputs, so this reward
    # uses those directly and keeps the structure TorchScript-friendly.
    #
    # Reward design:
    # - Encourage energetic joint motion for locomotion
    # - Penalize excessive joint velocities to reduce thrashing
    # - Add a smoothness/regularization term to avoid unstable oscillations

    vel_sq = joint_vel * joint_vel
    pos_sq = joint_pos * joint_pos

    # Stronger incentive for movement, but bounded via exp with temperature.
    motion_temp = 2.0
    motion_score = torch.exp(-vel_sq.mean(dim=1) / motion_temp)

    # Mild posture regularization to discourage extreme joint excursions.
    posture_temp = 1.0
    posture_score = torch.exp(-pos_sq.mean(dim=1) / posture_temp)

    # Penalize very large instantaneous joint speeds.
    vel_penalty = vel_sq.mean(dim=1)

    # Combine terms. The motion term is primary; posture is secondary.
    reward = 1.5 * motion_score + 0.5 * posture_score - 0.01 * vel_penalty

    reward_dict: Dict[str, torch.Tensor] = {
        "motion_score": motion_score,
        "posture_score": posture_score,
        "vel_penalty": vel_penalty,
    }
    return reward, reward_dict
