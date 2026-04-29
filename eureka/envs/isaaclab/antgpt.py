"""Isaac Lab DirectRLEnv implementation of Ant for Eureka reward generation.

Exposes the same self.* attributes as the IsaacGym version so that LLM-generated
reward functions are portable between the two backends.

Eureka reward injection contract:
  - Eureka writes a compute_reward() function to the bottom of this file
  - compute_reward() receives the named self.* tensors listed in ant_obs.py
  - Returns Tuple[torch.Tensor, Dict[str, torch.Tensor]]
  - _get_rewards() calls compute_reward() and stores results in self.rew_buf / self.rew_dict

Observation buffer layout (48-dim). KEY POINT: indices [0],[10],[11],[12:20],[20:28]
are IDENTICAL to the IsaacGym obs_buf so the gt_reward formula is unchanged.
  [0]       torso height (m)
  [1:4]     local linear velocity (m/s)
  [4:7]     local angular velocity (rad/s)
  [7]       yaw (rad)
  [8]       roll (rad)
  [9]       angle to target (rad)
  [10]      up_proj      <- gt_reward uses obs_buf[:, 10]
  [11]      heading_proj <- gt_reward uses obs_buf[:, 11]
  [12:20]   dof_pos_scaled [-1, 1]  <- gt_reward uses obs_buf[:, 12:20]
  [20:28]   dof_vel * dof_vel_scale <- gt_reward uses obs_buf[:, 20:28]
  [28:40]   contact forces, 4 feet x 3D (N), scaled  [was 4x6=24 in IsaacGym]
  [40:48]   last actions

NOTE on quaternion convention:
  Isaac Lab uses wxyz quaternions (root_state_w[:, 3:7] = [w, x, y, z]).
  IsaacGym used xyzw. All rotation math below is written for wxyz.

USD path: set ANT_USD_PATH to where your ant.usda lives inside the container.
  Quick check: find /isaac-lab -name "*.usd" 2>/dev/null | grep -i ant | head -5
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import Dict, Tuple

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    normalize,
)


# Bind-mounted inside the container by slurm_love_island.sh (same convention as cartpole).
# Adjust if your container layout differs.
ANT_USD_PATH = (
    "/local-assets/Isaac/IsaacLab/Robots/Classic/Ant/ant.usd"
)

# nv_ant.xml actuator gear = 150 for all 8 joints; replicate here.
_JOINT_GEAR = 150.0

ANT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ANT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,   # ant needs self-collision
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.44),
        joint_pos={
            "hip_1":    0.0,
            "ankle_1":  1.0,   # must be in [0.524, 1.745]
            "hip_2":    0.0,
            "ankle_2": -1.0,   # must be in [-1.745, -0.524]
            "hip_3":    0.0,
            "ankle_3": -1.0,   # must be in [-1.745, -0.524]
            "hip_4":    0.0,
            "ankle_4":  1.0,   # must be in [0.524, 1.745]
        },
    ),
    actuators={
        "ant_actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=500.0,  # generous cap; true limit = gear * power_scale
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
    # Simulation — original IsaacGym used dt=1/60 with no sub-stepping
    
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=2)
    decimation: int = 2

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=True
    )

    # Robot
    robot_cfg: ArticulationCfg = ANT_CFG

    # Contact sensor — prim_path selects bodies whose USD prim name contains "foot".
    # Matches 4 feet (front_left, front_right, back_left, back_right in nv_ant).
    # Isaac Lab ContactSensor gives 3D net forces only (not 6D force+torque),
    # so obs[28:40] = 4 feet x 3D = 12 dims (was 24 in IsaacGym with torques).
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/torso/.*foot",  # was /Robot/.*foot
        history_length=3,
        track_air_time=False,
    )

    # Task hyper-parameters (mirror IsaacGym yaml keys)
    episode_length_s: float = 16.67       # ~1000 steps at effective dt=1/60
    dof_vel_scale: float = 0.2
    contact_force_scale: float = 0.01
    power_scale: float = 0.5
    heading_weight: float = 0.5
    up_weight: float = 0.1
    actions_cost_scale: float = 0.005
    energy_cost_scale: float = 0.05
    joints_at_limit_cost_scale: float = 0.1
    death_cost: float = -2.0
    termination_height: float = 0.31      # episode ends when torso z < this

    # RL dims — observation_space changed from 60 to 48 because ContactSensor
    # provides 3D forces only. gt_reward critical indices are preserved.
    observation_space: int = 48
    action_space: int = 8
    num_states: int = 0

    # Eureka / LOVE Island pipeline hooks (set at runtime by isaaclab_train.py)
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

        # Gear ratios (all 150, matching nv_ant.xml actuator gear)
        self.joint_gears = torch.full(
            (self.cfg.action_space,), _JOINT_GEAR, device=self.device
        )

        # Target waypoint: far along +x axis (same as IsaacGym)
        self.targets = torch.tensor([1000.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )

        # World-frame basis vectors (z-up, x-forward)
        self.basis_vec0 = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )  # forward / heading
        self.basis_vec1 = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            self.num_envs, 1
        )  # world up

        # inv_start_rot: conjugate of the spawn rotation.
        # Spawn rotation is identity (wxyz=[1,0,0,0]), so inv_start_rot is also identity.
        # Kept as a tensor so heading math generalises if spawn rotation changes.
        self.inv_start_rot = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)  # wxyz

        # DOF joint limits for position scaling — shape [num_joints]
        # soft_joint_pos_limits is [num_envs, num_joints, 2]; all envs identical.
        self.dof_limits_lower = self.ant.data.soft_joint_pos_limits[0, :, 0]
        self.dof_limits_upper = self.ant.data.soft_joint_pos_limits[0, :, 1]

        # Effective timestep (used to normalise potentials)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Potential reward signal — initialised to -distance/dt at spawn
        potentials_init = -1000.0 / self.dt
        self.potentials = torch.full(
            (self.num_envs,), potentials_init, device=self.device
        )
        self.prev_potentials = self.potentials.clone()

        # ------------------------------------------------------------------ #
        # Named observation tensors exposed to Eureka's reward functions.     #
        # Every attribute here is documented in ant_obs.py.                   #
        # ------------------------------------------------------------------ #
        self.torso_height = torch.zeros(self.num_envs, device=self.device)
        self.vel_loc = torch.zeros(self.num_envs, 3, device=self.device)
        self.angvel_loc = torch.zeros(self.num_envs, 3, device=self.device)
        self.yaw = torch.zeros(self.num_envs, device=self.device)
        self.roll = torch.zeros(self.num_envs, device=self.device)
        self.pitch = torch.zeros(self.num_envs, device=self.device)
        self.angle_to_target = torch.zeros(self.num_envs, device=self.device)
        self.up_proj = torch.zeros(self.num_envs, device=self.device)
        self.heading_proj = torch.zeros(self.num_envs, device=self.device)
        self.up_vec = self.basis_vec1.clone()
        self.heading_vec = self.basis_vec0.clone()
        self.dof_pos_scaled = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self.contact_forces_feet = torch.zeros(
            self.num_envs, 4, 3, device=self.device
        )  # 4 feet x 3D net force (N)

        # Live references to joint state tensors (updated in _get_observations)
        self.joint_pos = self.ant.data.joint_pos   # [N, 8]
        self.joint_vel = self.ant.data.joint_vel   # [N, 8]

        # Actions stored after _pre_physics_step so reward fns can read them
        self.actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )

        # Buffers exposed to Eureka (mirrors IsaacGym API)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.rew_dict: Dict[str, torch.Tensor] = {}
        self.reset_buf = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.extras: Dict[str, torch.Tensor] = {}
        self.consecutive_successes = torch.zeros(1, device=self.device)

        # TensorBoard writer — set externally by isaaclab_train.py
        self._tb_writer = None
        self._tb_step: int = 0
        self._gt_reward_buf = torch.zeros(self.num_envs, device=self.device)

        # Trajectory ring buffer (same pattern as cartpole)
        self._traj_buf: dict = {}
        self._traj_queue: deque = deque()

    @property
    def progress_buf(self) -> torch.Tensor:
        """Alias for episode_length_buf (IsaacGym API compatibility)."""
        return self.episode_length_buf

    # ---------------------------------------------------------------------- #
    # Scene                                                                    #
    # ---------------------------------------------------------------------- #

    def _setup_scene(self):
        self.ant = Articulation(self.cfg.robot_cfg)

        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        wb_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/worldBody")
        if wb_prim.IsValid() and wb_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            wb_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["ant"] = self.ant
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------------- #
    # Action pipeline                                                          #
    # ---------------------------------------------------------------------- #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Mirror IsaacGym: force = actions * joint_gears * power_scale
        forces = self.actions * self.joint_gears * self.cfg.power_scale
        self.ant.set_joint_effort_target(forces)

    # ---------------------------------------------------------------------- #
    # Observations                                                             #
    # ---------------------------------------------------------------------- #

    def _get_observations(self) -> dict:
        # Refresh live joint state references
        self.joint_pos = self.ant.data.joint_pos
        self.joint_vel = self.ant.data.joint_vel

        # Root state layout: [pos(3), quat_wxyz(4), lin_vel(3), ang_vel(3)]
        root_state = self.ant.data.root_state_w
        torso_position = root_state[:, 0:3]
        torso_rotation = root_state[:, 3:7]   # wxyz — Isaac Lab convention
        velocity = root_state[:, 7:10]
        ang_velocity = root_state[:, 10:13]

        # ---- Potential reward signal (distance-to-target) -------------------
        to_target = self.targets - torso_position
        to_target[:, 2] = 0.0  # project to horizontal plane

        self.prev_potentials[:] = self.potentials.clone()
        self.potentials[:] = -torch.norm(to_target, p=2, dim=-1) / self.dt

        # ---- Heading & up vectors -------------------------------------------
        # torso_quat_local: torso orientation relative to spawn pose.
        # Since inv_start_rot is identity, this equals torso_rotation —
        # kept general for non-identity spawn rotations.
        torso_quat_local = quat_mul(self.inv_start_rot, torso_rotation)

        # Rotate world basis vectors into the frame at spawn orientation
        self.up_vec[:] = quat_rotate(torso_quat_local, self.basis_vec1)
        self.heading_vec[:] = quat_rotate(torso_quat_local, self.basis_vec0)

        # Scalar projections
        self.up_proj[:] = self.up_vec[:, 2]                              # dot with (0,0,1)
        to_target_dir = normalize(to_target)
        self.heading_proj[:] = (self.heading_vec * to_target_dir).sum(dim=-1)

        # ---- Local velocity & angular velocity ------------------------------
        # quat_rotate_inverse rotates a world-frame vector into body frame.
        self.vel_loc[:] = quat_rotate_inverse(torso_rotation, velocity)
        self.angvel_loc[:] = quat_rotate_inverse(torso_rotation, ang_velocity)

        # ---- ZYX Euler angles from torso_quat_local (wxyz) -----------------
        w = torso_quat_local[:, 0]
        x = torso_quat_local[:, 1]
        y = torso_quat_local[:, 2]
        z = torso_quat_local[:, 3]

        self.roll[:] = torch.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
        self.pitch[:] = torch.asin(sinp)
        self.yaw[:] = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        # ---- Remaining scalar observations ----------------------------------
        self.angle_to_target[:] = (
            torch.atan2(to_target[:, 1], to_target[:, 0]) - self.yaw
        )
        self.torso_height[:] = torso_position[:, 2]

        # ---- DOF positions scaled to [-1, 1] --------------------------------
        self.dof_pos_scaled[:] = _unscale(
            self.joint_pos, self.dof_limits_lower, self.dof_limits_upper
        )

        # ---- Contact forces on feet -----------------------------------------
        # net_forces_w shape: [num_envs, num_sensor_bodies, 3]
        # ContactSensorCfg ".*foot" matches exactly 4 bodies.
        self.contact_forces_feet[:] = self.contact_sensor.data.net_forces_w[:, :4, :]

        # ---- Assemble obs_buf (48-dim) ---------------------------------------
        obs = torch.cat(
            [
                self.torso_height.unsqueeze(-1),                                    # [0]     1
                self.vel_loc,                                                        # [1:4]   3
                self.angvel_loc,                                                     # [4:7]   3
                self.yaw.unsqueeze(-1),                                              # [7]     1
                self.roll.unsqueeze(-1),                                             # [8]     1
                self.angle_to_target.unsqueeze(-1),                                  # [9]     1
                self.up_proj.unsqueeze(-1),                                          # [10]    1
                self.heading_proj.unsqueeze(-1),                                     # [11]    1
                self.dof_pos_scaled,                                                 # [12:20] 8
                self.joint_vel * self.cfg.dof_vel_scale,                             # [20:28] 8
                self.contact_forces_feet.reshape(self.num_envs, -1)                  # [28:40] 12
                * self.cfg.contact_force_scale,
                self.actions,                                                        # [40:48] 8
            ],
            dim=-1,
        )

        if self.cfg.trajectories_dir:
            self._accumulate_trajectory()

        return {"policy": obs}

    # ---------------------------------------------------------------------- #
    # Rewards                                                                  #
    # ---------------------------------------------------------------------- #

    def _get_rewards(self) -> torch.Tensor:
        # All self.* attributes populated by _get_observations() above.
        dof_vel_scaled = self.joint_vel * self.cfg.dof_vel_scale

        gt_reward = _compute_gt_reward(
            self.torso_height,
            self.up_proj,
            self.heading_proj,
            self.dof_pos_scaled,
            dof_vel_scaled,
            self.actions,
            self.potentials,
            self.prev_potentials,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.joints_at_limit_cost_scale,
            self.cfg.termination_height,
            self.cfg.death_cost,
        )

        # consecutive_successes for ant = mean progress reward (mirrors IsaacGym)
        self.consecutive_successes[0] = (self.potentials - self.prev_potentials).mean()

        self._gt_reward_buf[:] = gt_reward
        self.extras["gt_reward"] = gt_reward.mean()
        self.extras["consecutive_successes"] = self.consecutive_successes.mean()

        # Eureka-generated reward (injected at bottom of file by the pipeline)
        # Eureka-generated reward (injected at bottom of file by the pipeline)
        if "compute_reward" in globals():
            param_names = [arg.name for arg in compute_reward.schema.arguments]
            args = [getattr(self, p) for p in param_names]
            self.rew_buf[:], self.rew_dict = compute_reward(*args)
            self.extras["gpt_reward"] = self.rew_buf.mean()
            for k, v in self.rew_dict.items():
                self.extras[k] = v.mean()
        else:
            self.rew_buf[:] = gt_reward

        # TensorBoard (direct write — avoids rl_games "episode/" prefix)
        self._tb_step += 1
        if self._tb_writer is not None and self._tb_step % 10 == 0:
            self._tb_writer.add_scalar(
                "gt_reward", gt_reward.mean().item(), self._tb_step
            )
            self._tb_writer.add_scalar(
                "consecutive_successes",
                self.consecutive_successes.mean().item(),
                self._tb_step,
            )
            if "compute_reward" in globals():
                self._tb_writer.add_scalar(
                    "gpt_reward", self.rew_buf.mean().item(), self._tb_step
                )

        return self.rew_buf

    # ---------------------------------------------------------------------- #
    # Termination                                                              #
    # ---------------------------------------------------------------------- #

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: _get_dones is called BEFORE _get_observations in Isaac Lab's
        # step loop, so we read root state directly here rather than from self.*.
        torso_z = self.ant.data.root_state_w[:, 2]

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torso_z < self.cfg.termination_height

        done = terminated | time_out
        if done.sum() > 0:
            if self.cfg.states_dir:
                self._save_terminal_states(done)
            if self.cfg.trajectories_dir:
                for env_idx in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                    self._flush_trajectory(int(env_idx))

        self.reset_buf[:] = done.long()
        return terminated, time_out

    # ---------------------------------------------------------------------- #
    # Reset                                                                    #
    # ---------------------------------------------------------------------- #

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.ant._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # Random perturbations (mirrors IsaacGym reset_idx)
        pos_noise = 0.2 * (
            torch.rand((num_reset, self.cfg.action_space), device=self.device) - 0.5
        )  # uniform [-0.2, 0.2]
        vel_noise = 0.1 * (
            torch.rand((num_reset, self.cfg.action_space), device=self.device) - 0.5
        )  # uniform [-0.1, 0.1]

        joint_pos = self.ant.data.default_joint_pos[env_ids] + pos_noise
        joint_pos = joint_pos.clamp(
            self.dof_limits_lower.unsqueeze(0),
            self.dof_limits_upper.unsqueeze(0),
        )
        joint_vel = vel_noise

        # Root state offset by env origin
        default_root_state = self.ant.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.ant.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.ant.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.ant.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # Reset potentials to match new spawn position
        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.reset_buf[env_ids] = 0

    # ---------------------------------------------------------------------- #
    # Trajectory / state saving (identical pattern to cartpole)               #
    # ---------------------------------------------------------------------- #

    def _accumulate_trajectory(self) -> None:
        jp = self.joint_pos.detach().cpu()
        jv = self.joint_vel.detach().cpu()
        for env_idx in range(self.num_envs):
            if env_idx not in self._traj_buf:
                self._traj_buf[env_idx] = ([], [])
            self._traj_buf[env_idx][0].append(jp[env_idx].clone())
            self._traj_buf[env_idx][1].append(jv[env_idx].clone())

    def _save_terminal_states(self, done_mask: torch.Tensor) -> None:
        import os
        os.makedirs(self.cfg.states_dir, exist_ok=True)
        path = os.path.join(self.cfg.states_dir, f"terminal_{self._tb_step}.pt")
        torch.save(
            {
                # joint state (kept for backward compat)
                "joint_pos": self.joint_pos[done_mask].cpu(),
                "joint_vel": self.joint_vel[done_mask].cpu(),
                "gt_reward": self._gt_reward_buf[done_mask].cpu(),
                # all named obs tensors so sigma.py can call any ant reward fn
                "torso_height": self.torso_height[done_mask].cpu(),
                "vel_loc": self.vel_loc[done_mask].cpu(),
                "angvel_loc": self.angvel_loc[done_mask].cpu(),
                "yaw": self.yaw[done_mask].cpu(),
                "roll": self.roll[done_mask].cpu(),
                "pitch": self.pitch[done_mask].cpu(),
                "angle_to_target": self.angle_to_target[done_mask].cpu(),
                "up_proj": self.up_proj[done_mask].cpu(),
                "heading_proj": self.heading_proj[done_mask].cpu(),
                "dof_pos_scaled": self.dof_pos_scaled[done_mask].cpu(),
                "potentials": self.potentials[done_mask].cpu(),
                "prev_potentials": self.prev_potentials[done_mask].cpu(),
                "actions": self.actions[done_mask].cpu(),
                "contact_forces_feet": self.contact_forces_feet[done_mask].cpu(),
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
        torch.save(
            {
                "joint_pos_seq": jp_seq,
                "joint_vel_seq": jv_seq,
                "episode_length": int(jp_seq.shape[0]),
            },
            fpath,
        )
        idx_path = os.path.join(self.cfg.trajectories_dir, "index.json")
        with open(idx_path, "a") as f:
            f.write(
                json.dumps({"step": step, "env_idx": env_idx, "filename": fname})
                + "\n"
            )
        self._traj_queue.append((step, env_idx, fname))
        while len(self._traj_queue) > self.cfg.max_traj_files:
            _, _, old_fname = self._traj_queue.popleft()
            old_path = os.path.join(self.cfg.trajectories_dir, old_fname)
            if os.path.exists(old_path):
                os.remove(old_path)


# ---------------------------------------------------------------------------
# Helper: scale DOF positions from [lower, upper] → [-1, 1]
# Mirrors unscale() from isaacgymenvs.utils.torch_jit_utils.
# ---------------------------------------------------------------------------

@torch.jit.script
def _unscale(
    x: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    return (2.0 * x - upper - lower) / (upper - lower)


# ---------------------------------------------------------------------------
# Ground-truth reward (always present, used for Eureka correlation metric).
#
# Mirrors compute_success() from the IsaacGym version with three changes:
#   1. Takes named tensors instead of obs_buf slices — clearer and jit-safe.
#   2. Does NOT return reset / consecutive_successes (handled by _get_dones).
#   3. electricity_cost uses dof_vel_scaled directly (was obs_buf[:, 20:28]).
# ---------------------------------------------------------------------------

@torch.jit.script
def _compute_gt_reward(
    torso_height: torch.Tensor,
    up_proj: torch.Tensor,
    heading_proj: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    dof_vel_scaled: torch.Tensor,
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
) -> torch.Tensor:

    # Heading reward: full weight when well-aligned, linear ramp below 0.8
    heading_reward = torch.where(
        heading_proj > 0.8,
        torch.ones_like(heading_proj) * heading_weight,
        heading_weight * heading_proj / 0.8,
    )

    # Up reward: bonus for staying upright
    up_reward = torch.where(
        up_proj > 0.93,
        torch.ones_like(up_proj) * up_weight,
        torch.zeros_like(up_proj),
    )

    # Cost terms (directly mirror IsaacGym compute_success)
    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * dof_vel_scaled), dim=-1)
    dof_at_limit_cost = (dof_pos_scaled > 0.99).float().sum(dim=-1)

    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - joints_at_limit_cost_scale * dof_at_limit_cost
    )

    # Death penalty when torso drops too low
    total_reward = torch.where(
        torso_height < termination_height,
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
def compute_reward(
    vel_loc: torch.Tensor,
    angvel_loc: torch.Tensor,
    up_proj: torch.Tensor,
    heading_proj: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    contact_forces_feet: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Forward velocity in the ant's local frame
    forward_vel = vel_loc[:, 0]

    # Progress toward target direction (Isaac Lab potential-based term)
    progress_reward = potentials - prev_potentials

    # Encourage the body to face and stay aligned with the target direction
    heading_reward = torch.clamp(heading_proj, 0.0, 1.0)

    # Stay upright
    upright_reward = torch.clamp(up_proj, 0.0, 1.0)

    # Penalize excessive turning / wobbling
    ang_vel_penalty = torch.sum(angvel_loc * angvel_loc, dim=-1)

    # Penalize overly large actions for smoother locomotion
    action_penalty = torch.sum(actions * actions, dim=-1)

    # Penalize large joint velocities to reduce flailing
    joint_vel_penalty = torch.sum(joint_vel * joint_vel, dim=-1)

    # Light foot contact penalty to discourage dragging/slapping
    foot_force_mag = torch.norm(contact_forces_feet, dim=-1)  # (num_envs, 4)
    contact_penalty = torch.sum(torch.clamp(foot_force_mag - 1.0, min=0.0), dim=-1)

    # Reward shaping terms
    speed_temp = 2.0
    progress_temp = 1.0
    heading_temp = 1.0
    upright_temp = 1.0

    forward_speed_reward = torch.exp(forward_vel / speed_temp)
    progress_reward_shaped = torch.exp(progress_reward / progress_temp)
    heading_reward_shaped = torch.exp(heading_reward / heading_temp)
    upright_reward_shaped = torch.exp(upright_reward / upright_temp)

    reward = (
        1.5 * forward_speed_reward
        + 1.0 * progress_reward_shaped
        + 0.5 * heading_reward_shaped
        + 0.5 * upright_reward_shaped
        - 0.01 * ang_vel_penalty
        - 0.001 * action_penalty
        - 0.0005 * joint_vel_penalty
        - 0.0001 * contact_penalty
    )

    reward_dict: Dict[str, torch.Tensor] = {
        "forward_speed_reward": forward_speed_reward,
        "progress_reward_shaped": progress_reward_shaped,
        "heading_reward_shaped": heading_reward_shaped,
        "upright_reward_shaped": upright_reward_shaped,
        "ang_vel_penalty": ang_vel_penalty,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty,
        "contact_penalty": contact_penalty,
        "total_reward": reward,
    }
    return reward, reward_dict
