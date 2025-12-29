from stable_baselines3 import PPO, DQN, A2C, SAC
import gymnasium as gym
from utils import load_config
from stable_baselines3.common.callbacks import CheckpointCallback
import os


def train_baseline(cfg_path="config/config_baseline.yaml"):
    cfg = load_config(cfg_path)

    # Create directories
    os.makedirs(f"results/baseline", exist_ok=True)
    os.makedirs(f"logs/baseline", exist_ok=True)

    # Map algorithm names to classes
    algo_map = {
        "DQN": DQN,
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC
    }

    for trial in range(cfg['num_trials']):
        print(f"\n{'=' * 50}")
        print(f"Starting Trial {trial + 1}/{cfg['num_trials']}")
        print(f"{'=' * 50}\n")

        # Create environment
        env = gym.make(cfg["environment"])

        # Pick Algorithm
        algo = algo_map[cfg['algorithm']]

        # Create model with tensorboard logging
        model = algo(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"logs/baseline/{cfg['algorithm']}_trial{trial + 1}/"
        )

        # Setup checkpoint callback
        checkpoint_dir = f"logs/baseline/{cfg['algorithm']}_trial{trial + 1}/checkpoints/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        cb = CheckpointCallback(
            save_freq=cfg["checkpoint_freq"],
            save_path=checkpoint_dir,
            name_prefix=f"{cfg['algorithm']}_model"
        )

        # Train the model
        model.learn(total_timesteps=cfg["timesteps"], callback=cb)

        # Save final model
        model.save(f"results/baseline/{cfg['algorithm']}_trial{trial + 1}")

        env.close()
        print(f"\nTrial {trial + 1} completed!")


if __name__ == '__main__':
    train_baseline()
