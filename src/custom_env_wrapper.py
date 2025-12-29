import gymnasium as gym
from gymnasium import RewardWrapper
import numpy as np


class CustomRewardWrapper(RewardWrapper):
    """
    Custom reward wrapper for Pendulum-v1 environment.

    Original Reward: -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
    Range: approximately -16.0 to 0.0

    Custom Reward Design:
    - Emphasizes staying upright (theta close to 0)
    - Penalizes excessive angular velocity
    - Rewards smooth control (low action magnitude)
    - Provides bonus for maintaining stable upright position
    """

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg

        # Custom reward hyperparameters (can be configured via cfg)
        self.angle_weight = cfg.get('angle_weight', 1.5)
        self.velocity_weight = cfg.get('velocity_weight', 0.05)
        self.action_weight = cfg.get('action_weight', 0.001)
        self.stability_bonus = cfg.get('stability_bonus', 1.0)
        self.stability_threshold = cfg.get('stability_threshold', 0.1)

    def reward(self, reward: float) -> float:
        """
        Custom reward function for Pendulum-v1.

        Args:
            reward: Original environment reward

        Returns:
            Custom modified reward
        """
        # Get current state from environment's internal state
        # Pendulum's internal state: [theta, theta_dot]
        state = self.env.unwrapped.state
        theta = state[0]  # Angle (in radians, -pi to pi)
        theta_dot = state[1]  # Angular velocity

        # Get last action (torque applied)
        # Note: In Pendulum, action is in range [-2, 2]
        last_action = self.env.unwrapped.last_u if hasattr(self.env.unwrapped, 'last_u') else 0.0

        # Custom reward components
        # 1. Angle penalty: Penalize deviation from upright (theta=0)
        angle_penalty = -self.angle_weight * (theta ** 2)

        # 2. Velocity penalty: Penalize high angular velocity
        velocity_penalty = -self.velocity_weight * (theta_dot ** 2)

        # 3. Action penalty: Encourage smooth control
        action_penalty = -self.action_weight * (last_action ** 2)

        # 4. Stability bonus: Reward staying near upright with low velocity
        if abs(theta) < self.stability_threshold and abs(theta_dot) < 1.0:
            stability_bonus = self.stability_bonus
        else:
            stability_bonus = 0.0

        # Combine all components
        custom_reward = angle_penalty + velocity_penalty + action_penalty + stability_bonus

        return custom_reward
