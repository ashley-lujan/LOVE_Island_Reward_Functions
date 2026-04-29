# [USES ISAACLAB]
class AntEnv:
    """Observation stub — shows the self.* attributes available for reward generation.
    These are the ONLY variables a compute_reward function for this environment
    may use as inputs. Every parameter name must exactly match a self.* attribute
    (without the 'self.' prefix).

    Observation buffer obs_buf indices (for reference — prefer named attrs below):
      [0]       torso_height
      [1:4]     vel_loc
      [4:7]     angvel_loc
      [7]       yaw
      [8]       roll
      [9]       angle_to_target
      [10]      up_proj
      [11]      heading_proj
      [12:20]   dof_pos_scaled
      [20:28]   joint_vel * dof_vel_scale
      [28:40]   contact_forces_feet (4 feet x 3D), scaled
      [40:48]   actions
    """
    def compute_observations(self):
        # (num_envs,) Height of torso above ground plane (m)
        self.torso_height

        # (num_envs, 3) Linear velocity of torso in the LOCAL body frame (m/s)
        self.vel_loc

        # (num_envs, 3) Angular velocity of torso in the LOCAL body frame (rad/s)
        self.angvel_loc

        # (num_envs,) Yaw angle of torso in world frame (rad)
        self.yaw

        # (num_envs,) Roll angle of torso in world frame (rad)
        self.roll

        # (num_envs,) Pitch angle of torso in world frame (rad)
        self.pitch

        # (num_envs,) Angle from torso heading direction to target direction (rad)
        self.angle_to_target

        # (num_envs,) Dot product of torso up-vector with world Z axis
        #   1.0 = perfectly upright, 0.0 = sideways, -1.0 = upside-down
        self.up_proj

        # (num_envs,) Dot product of torso heading-vector with normalised direction to target
        #   1.0 = heading directly at target, -1.0 = heading directly away
        self.heading_proj

        # (num_envs, 8) Joint (DOF) positions scaled to [-1, 1] using joint limits
        self.dof_pos_scaled

        # (num_envs, 8) Raw joint positions (rad)
        self.joint_pos

        # (num_envs, 8) Raw joint velocities (rad/s)
        self.joint_vel

        # (num_envs,) Potential reward signal = -||torso_to_target||_2 / dt
        #   More negative = farther from target
        self.potentials

        # (num_envs,) Potentials from the previous timestep
        #   progress_reward = potentials - prev_potentials
        self.prev_potentials

        # (num_envs, 8) Actions applied at the previous timestep (in [-1, 1])
        self.actions

        # (num_envs, 4, 3) Net contact forces on the 4 feet in world frame (N)
        #   Axis 1: foot index (0=front_left, 1=front_right, 2=back_left, 3=back_right)
        #   Axis 2: force components [Fx, Fy, Fz]
        self.contact_forces_feet