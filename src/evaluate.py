import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC
import numpy as np


def evaluate(model_path, env_id, episodes=10, render=False):
    # Map algorithm names for loading
    algo_map = {
        "DQN": DQN,
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC
    }

    # Detect algorithm from path
    algo_name = None
    for name in algo_map.keys():
        if name in model_path:
            algo_name = name
            break

    if algo_name is None:
        raise ValueError("Could not detect algorithm from model path")

    # Load model
    model = algo_map[algo_name].load(model_path)

    # Create environment
    if render:
        env = gym.make(env_id, render_mode="human")
    else:
        env = gym.make(env_id)

    episode_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        while not (done or truncated):
            # Predict action (deterministic=True for no exploration)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {step_count}")

    env.close()

    # Print statistics
    print(f"\n{'=' * 50}")
    print(f"Evaluation Results over {episodes} episodes:")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"{'=' * 50}")

    return episode_rewards, episode_lengths


if __name__ == '__main__':
    # Example usage for Pendulum-v1
    evaluate("results/baseline/SAC_trial1.zip", "Pendulum-v1", episodes=10, render=True)
