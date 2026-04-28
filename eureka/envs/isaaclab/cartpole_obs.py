class CartpoleEnv:
    """Observation stub — shows the self.* attributes available for reward generation.

    These are the ONLY variables a compute_reward function for this environment
    may use as inputs. Every parameter name must exactly match a self.* attribute
    (without the 'self.' prefix).
    """
    def compute_observations(self):
        # (num_envs, 2) DOF positions: [:, 0]=cart position (m), [:, 1]=pole angle (rad)
        self.joint_pos
        # (num_envs, 2) DOF velocities: [:, 0]=cart velocity (m/s), [:, 1]=pole angular vel (rad/s)
        self.joint_vel
