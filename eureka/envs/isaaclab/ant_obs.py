# Observation specification for the Isaac Lab Ant environment.
# This file is provided as context for LLM-generated reward functions.
#
# Available attributes on the environment (self.*):
#
#   self.joint_pos       (num_envs, 8)   — 8 DOF positions (radians)
#   self.joint_vel       (num_envs, 8)   — 8 DOF velocities (rad/s)
#
#   self.root_pos        (num_envs, 3)   — torso [x, y, z] in world frame (meters)
#   self.root_quat       (num_envs, 4)   — torso quaternion [w, x, y, z] in world frame
#   self.root_lin_vel    (num_envs, 3)   — torso linear velocity in body frame (m/s)
#   self.root_ang_vel    (num_envs, 3)   — torso angular velocity in body frame (rad/s)
#
#   self.contact_forces  (num_envs, 4, 6) — foot contact data: [:, :, :3] = net force (N),
#                                            [:, :, 3:] = zeros (torques not available)
#
#   self.potentials      (num_envs,)     — current negative distance to target / dt
#   self.prev_potentials (num_envs,)     — previous step's potentials
#   self.targets         (num_envs, 3)   — target position [1000, 0, 0] in world frame
#
# Task goal: make the ant run forward (in the +x direction) as fast as possible.
# Episode ends when torso height < 0.27 m (fallen) or time limit reached.
#
# Eureka reward injection contract:
#   compute_reward(joint_pos, joint_vel) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
#   The returned tensor is the per-environment reward signal used for RL training.
