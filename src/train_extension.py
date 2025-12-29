from custom_env_wrapper import CustomRewardWrapper
from utils import load_config
import gymnasium as gym
from stable_baselines3 import PPO, A2C, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
import os


def train_extension(cfg_path="config/config_extension.yaml"):
    cfg = load_config(cfg_path)

    os.makedirs("results/extension", exist_ok=True)
    os.makedirs("logs/extension", exist_ok=True)

    algo_map = {"PPO": PPO, "A2C": A2C, "TD3": TD3}

    for trial in range(cfg['num_trials']):
        print(f"\nStarting Extension Trial {trial + 1}/{cfg['num_trials']}\n")

        # Use custom reward (same as Part 2)
        base_env = gym.make(cfg["environment"])
        env = CustomRewardWrapper(base_env, cfg)

        algo = algo_map[cfg['algorithm']]

        model = algo(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"logs/extension/{cfg['algorithm']}_trial{trial + 1}/"
        )

        checkpoint_dir = f"logs/extension/{cfg['algorithm']}_trial{trial + 1}/checkpoints/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        cb = CheckpointCallback(
            save_freq=cfg["checkpoint_freq"],
            save_path=checkpoint_dir,
            name_prefix=f"{cfg['algorithm']}_extension_model"
        )

        model.learn(total_timesteps=cfg["timesteps"], callback=cb)
        model.save(f"results/extension/{cfg['algorithm']}_trial{trial + 1}")
        env.close()


if __name__ == '__main__':
    train_extension()
