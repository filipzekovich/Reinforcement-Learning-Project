import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os


def plot_training_curves(baseline_paths, custom_paths, save_path="results/comparison_plot.png"):
    """
    Plot training curves comparing baseline and custom reward approaches.

    Args:
        baseline_paths: List of paths to baseline training logs
        custom_paths: List of paths to custom reward training logs
        save_path: Where to save the comparison plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot baseline results
    baseline_rewards = []
    for path in baseline_paths:
        if os.path.exists(path):
            df = load_results(path)
            x, y = ts2xy(df, 'timesteps')
            ax1.plot(x, y, alpha=0.3, color='blue')
            baseline_rewards.append(y)

    if baseline_rewards:
        mean_baseline = np.mean(baseline_rewards, axis=0)
        std_baseline = np.std(baseline_rewards, axis=0)
        ax1.plot(x, mean_baseline, color='blue', linewidth=2, label='Baseline Mean')
        ax1.fill_between(x, mean_baseline - std_baseline, mean_baseline + std_baseline,
                         alpha=0.2, color='blue')

    # Plot custom reward results
    custom_rewards = []
    for path in custom_paths:
        if os.path.exists(path):
            df = load_results(path)
            x, y = ts2xy(df, 'timesteps')
            ax1.plot(x, y, alpha=0.3, color='red')
            custom_rewards.append(y)

    if custom_rewards:
        mean_custom = np.mean(custom_rewards, axis=0)
        std_custom = np.std(custom_rewards, axis=0)
        ax1.plot(x, mean_custom, color='red', linewidth=2, label='Custom Reward Mean')
        ax1.fill_between(x, mean_custom - std_custom, mean_custom + std_custom,
                         alpha=0.2, color='red')

    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot final performance comparison
    if baseline_rewards and custom_rewards:
        final_baseline = [r[-1000:].mean() for r in baseline_rewards]
        final_custom = [r[-1000:].mean() for r in custom_rewards]

        ax2.bar(['Baseline', 'Custom Reward'],
                [np.mean(final_baseline), np.mean(final_custom)],
                yerr=[np.std(final_baseline), np.std(final_custom)],
                color=['blue', 'red'], alpha=0.7, capsize=10)
        ax2.set_ylabel('Mean Final Episode Reward')
        ax2.set_title('Final Performance Comparison')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")


def generate_report_stats():
    """Generate statistical comparison for report."""
    baseline_paths = [f"logs/baseline/SAC_trial{i}/" for i in range(1, 4)]
    custom_paths = [f"logs/custom/SAC_trial{i}/" for i in range(1, 4)]

    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON REPORT")
    print("=" * 60)

    # Analyze baseline
    baseline_rewards = []
    for path in baseline_paths:
        if os.path.exists(path):
            df = load_results(path)
            baseline_rewards.append(df['r'].values[-1000:].mean())

    # Analyze custom
    custom_rewards = []
    for path in custom_paths:
        if os.path.exists(path):
            df = load_results(path)
            custom_rewards.append(df['r'].values[-1000:].mean())

    if baseline_rewards and custom_rewards:
        print(f"\nBaseline Results (n={len(baseline_rewards)} trials):")
        print(f"  Mean Final Reward: {np.mean(baseline_rewards):.2f}")
        print(f"  Std Dev: {np.std(baseline_rewards):.2f}")

        print(f"\nCustom Reward Results (n={len(custom_rewards)} trials):")
        print(f"  Mean Final Reward: {np.mean(custom_rewards):.2f}")
        print(f"  Std Dev: {np.std(custom_rewards):.2f}")

        improvement = ((np.mean(custom_rewards) - np.mean(baseline_rewards)) /
                       abs(np.mean(baseline_rewards)) * 100)
        print(f"\nImprovement: {improvement:+.2f}%")

    print("=" * 60 + "\n")


if __name__ == '__main__':
    # Generate comparison plots
    baseline_paths = [f"logs/baseline/SAC_trial{i}/" for i in range(1, 4)]
    custom_paths = [f"logs/custom/SAC_trial{i}/" for i in range(1, 4)]

    plot_training_curves(baseline_paths, custom_paths)
    generate_report_stats()
