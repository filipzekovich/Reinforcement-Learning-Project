from custom_env_wrapper import CustomRewardWrapper
from utils import load_config
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os


def train_custom(cfg_path="config/config_custom.yaml"):
    cfg = load_config(cfg_path)

    # Create directories
    os.makedirs(f"results/custom", exist_ok=True)
    os.makedirs(f"logs/custom", exist_ok=True)

    # Map algorithm names to classes
    algo_map = {
        "DQN": DQN,
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC
    }

    # Number of trials for robustness (at least 3)
    num_trials = cfg.get('num_trials', 3)

    for trial in range(num_trials):
        print(f"\n{'=' * 50}")
        print(f"Starting Custom Reward Trial {trial + 1}/{num_trials}")
        print(f"{'=' * 50}\n")

        # Create base environment
        base_env = gym.make(cfg["environment"])

        # Wrap with custom reward function
        env = CustomRewardWrapper(base_env, cfg)

        # Pick Algorithm
        algo = algo_map[cfg['algorithm']]

        # Create model with tensorboard logging
        model = algo(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"logs/custom/{cfg['algorithm']}_trial{trial + 1}/"
        )

        # Setup checkpoint callback
        checkpoint_dir = f"logs/custom/{cfg['algorithm']}_trial{trial + 1}/checkpoints/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        cb = CheckpointCallback(
            save_freq=cfg["checkpoint_freq"],
            save_path=checkpoint_dir,
            name_prefix=f"{cfg['algorithm']}_custom_model"
        )

        # Train the model
        model.learn(total_timesteps=cfg["timesteps"], callback=cb)

        # Save final model
        model.save(f"results/custom/{cfg['algorithm']}_trial{trial + 1}")

        env.close()
        print(f"\nCustom Reward Trial {trial + 1} completed!")


if __name__ == '__main__':
    train_custom()
