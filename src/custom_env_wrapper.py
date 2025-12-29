import gymnasium as gym
from gymnasium import RewardWrapper
import numpy as np


class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg

    def reward(self, reward: float) -> float:
        state = self.env.unwrapped.state
        theta = state[0]  # Angle in radians
        theta_dot = state[1]  # Angular velocity

        # Normalize angle to [0, 1] where 0 is upright
        angle_cost = (theta ** 2) / (np.pi ** 2)

        # Velocity cost (normalized)
        velocity_cost = 0.1 * (theta_dot ** 2) / 64  # Max velocity ≈ 8

        # Action cost
        last_action = self.env.unwrapped.last_u if hasattr(self.env.unwrapped, 'last_u') else 0.0
        action_cost = 0.001 * (last_action ** 2) / 4  # Max action = 2


        upright_bonus = np.exp(-3 * angle_cost)

        custom_reward = -16 * (angle_cost + velocity_cost + action_cost) + 2 * upright_bonus

        return custom_reward
