# Reinforcement Learning Project: Pendulum-v1 DRL Analysis

A comprehensive Deep Reinforcement Learning project comparing algorithms, reward functions, and training strategies on the Pendulum-v1 environment using Stable Baselines3.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements three core experiments:

1. **Baseline**: Train SAC algorithm with default Pendulum-v1 reward
2. **Custom Reward**: Train SAC with engineered custom reward function
3. **Extension**: Train PPO algorithm with custom reward for comparison

Each experiment runs 3 independent trials for statistical robustness.

### Environment: Pendulum-v1
- **Task**: Balance inverted pendulum in upright position
- **State Space**: 3D continuous [cos(θ), sin(θ), angular velocity]
- **Action Space**: 1D continuous torque [-2.0, 2.0]
- **Episode Length**: 200 steps
- **Success**: Minimize angle deviation and angular velocity

---

## 📁 Project Structure

```
Reinforcement-Learning-Project/
├── config/
│   ├── config_baseline.yaml       # SAC + default reward
│   ├── config_custom.yaml         # SAC + custom reward  
│   └── config_extension.yaml      # PPO + custom reward
├── src/
│   ├── train_baseline.py          # Part 1: Baseline training
│   ├── train_custom_reward.py     # Part 2: Custom reward training
│   ├── train_extension.py         # Part 3: Algorithm comparison
│   ├── evaluate.py                # Model evaluation script
│   ├── compare_results.py         # Generate comparison plots
│   ├── custom_env_wrapper.py      # Custom reward wrapper
│   └── utils.py                   # Configuration utilities
├── logs/                          # TensorBoard logs (auto-generated)
│   ├── baseline/
│   ├── custom/
│   └── extension/
├── results/                       # Trained models (auto-generated)
│   ├── baseline/
│   ├── custom/
│   └── extension/
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Reinforcement-Learning-Project.git
cd Reinforcement-Learning-Project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import stable_baselines3, gymnasium; print('✓ Installation successful!')"
```

---

## ⚡ Quick Start

Run all experiments in sequence:

```bash
# 1. Train baseline (SAC + default reward) - 3 trials
python src/train_baseline.py

# 2. Train custom reward (SAC + custom reward) - 3 trials  
python src/train_custom_reward.py

# 3. Train extension (PPO + custom reward) - 3 trials
python src/train_extension.py

# 4. Monitor training with TensorBoard
tensorboard --logdir=logs

# 5. Evaluate trained models
python src/evaluate.py

# 6. Generate comparison plots
python src/compare_results.py
```

**Estimated Training Time**: 6-8 hours total (CPU), 2-3 hours (GPU)

---

## 🧪 Experiments

### Part 1: Baseline

**Goal**: Establish performance benchmark with default reward function.

```bash
python src/train_baseline.py
```

**Configuration**: `config/config_baseline.yaml`
```yaml
environment: "Pendulum-v1"
algorithm: "SAC"
timesteps: 100000
checkpoint_freq: 10000
num_trials: 3
```

**Output**:
- Models: `results/baseline/SAC_trial{1,2,3}.zip`
- Logs: `logs/baseline/SAC_trial{1,2,3}/`
- Checkpoints: Every 10,000 steps

**Default Reward Function**:
```python
r = -(θ² + 0.1*θ_dot² + 0.001*action²)
# Range: approximately -16.27 to 0.0
```

---

### Part 2: Custom Reward Engineering

**Goal**: Design improved reward function for faster convergence.

```bash
python src/train_custom_reward.py
```

**Configuration**: `config/config_custom.yaml`
```yaml
environment: "Pendulum-v1"
algorithm: "SAC"
timesteps: 100000
checkpoint_freq: 10000
num_trials: 3

# Custom reward hyperparameters
angle_weight: 1.5
velocity_weight: 0.05
action_weight: 0.001
stability_bonus: 1.0
stability_threshold: 0.1
```

**Custom Reward Design** (`src/custom_env_wrapper.py`):
```python
# Normalized cost components
angle_cost = (θ²) / (π²)
velocity_cost = 0.1 * (θ_dot²) / 64
action_cost = 0.001 * (action²) / 4

# Dense reward shaping with exponential bonus
upright_bonus = exp(-3 * angle_cost)

# Combined reward (scaled to match baseline range)
custom_reward = -16 * (angle_cost + velocity_cost + action_cost) + 2 * upright_bonus
# Range: approximately -16 to +2
```

**Key Improvements**:
- ✅ Dense reward signal (exponential shaping)
- ✅ Normalized components for stable learning
- ✅ Explicit upright bonus (positive reward)
- ✅ Scaled to match baseline range

**Output**:
- Models: `results/custom/SAC_trial{1,2,3}.zip`
- Logs: `logs/custom/SAC_trial{1,2,3}/`

---

### Part 3: Extension - Algorithm Comparison

**Goal**: Compare PPO (on-policy) vs SAC (off-policy) with custom reward.

```bash
python src/train_extension.py
```

**Configuration**: `config/config_extension.yaml`
```yaml
environment: "Pendulum-v1"
algorithm: "PPO"
timesteps: 100000
checkpoint_freq: 10000
num_trials: 3
```

**Comparison**:
- **SAC**: Off-policy, sample efficient, good for continuous control
- **PPO**: On-policy, stable, widely used benchmark

**Output**:
- Models: `results/extension/PPO_trial{1,2,3}.zip`
- Logs: `logs/extension/PPO_trial{1,2,3}/`

---

## 📊 Evaluation

### Evaluate Single Model

Edit `src/evaluate.py` (line 74) to set model path:

```python
if __name__ == '__main__':
    # Change model path here
    evaluate("results/baseline/SAC_trial1.zip", "Pendulum-v1", episodes=10, render=True)
```

Run:
```bash
python src/evaluate.py
```

**Output**:
```
Episode 1: Reward = -119.76, Length = 200
Episode 2: Reward = -115.88, Length = 200
...
==================================================
Evaluation Results over 10 episodes:
Mean Reward: -132.11 ± 64.45
Mean Length: 200.00 ± 0.00
==================================================
```

### Evaluate All Trials

```bash
# Baseline
python src/evaluate.py  # Set path: results/baseline/SAC_trial1.zip
python src/evaluate.py  # Set path: results/baseline/SAC_trial2.zip
python src/evaluate.py  # Set path: results/baseline/SAC_trial3.zip

# Custom
python src/evaluate.py  # Set path: results/custom/SAC_trial1.zip
python src/evaluate.py  # Set path: results/custom/SAC_trial2.zip
python src/evaluate.py  # Set path: results/custom/SAC_trial3.zip

# Extension
python src/evaluate.py  # Set path: results/extension/PPO_trial1.zip
python src/evaluate.py  # Set path: results/extension/PPO_trial2.zip
python src/evaluate.py  # Set path: results/extension/PPO_trial3.zip
```

### Evaluation Metrics

- **Episode Reward**: Total reward per episode (higher is better, range: -16 to 0)
- **Episode Length**: Steps per episode (fixed at 200 for Pendulum-v1)
- **Mean ± Std**: Statistical robustness across episodes
- **Consistency**: Lower std indicates more reliable policy

**Performance Benchmark**:
- ✅ **Excellent**: Mean reward > -150
- ⚠️ **Good**: Mean reward -150 to -300
- ❌ **Poor**: Mean reward < -300

---

## 📈 Visualization

### TensorBoard

Monitor training progress in real-time:

```bash
# View all experiments side-by-side
tensorboard --logdir=logs

# View specific experiment
tensorboard --logdir=logs/baseline
tensorboard --logdir=logs/custom
tensorboard --logdir=logs/extension
```

Open browser: **http://localhost:6006/**

**Key Metrics to Monitor**:
- `rollout/ep_rew_mean` - Average episode reward (should increase)
- `rollout/ep_len_mean` - Average episode length
- `train/loss` - Training loss (should stabilize)
- `train/learning_rate` - Learning rate schedule
- `train/entropy_loss` - Exploration vs exploitation balance

### Generate Comparison Plots

```bash
python src/compare_results.py
```

**Generates**:
- Training curves with mean ± std across trials
- Final performance bar chart
- Convergence speed analysis
- Saves to `results/comparison_plot.png`

---

## ⚙️ Configuration

All experiments are configured via YAML files in `config/`.

### Modify Hyperparameters

Edit configuration files to customize experiments:

```yaml
# Example: config/config_baseline.yaml
environment: "Pendulum-v1"      # Gymnasium environment
algorithm: "SAC"                 # DRL algorithm (SAC, PPO, A2C, TD3)
timesteps: 100000               # Total training timesteps
checkpoint_freq: 10000          # Save model every N steps
num_trials: 3                   # Number of independent trials
```

### Supported Algorithms

- **SAC** (Soft Actor-Critic): Off-policy, sample efficient
- **PPO** (Proximal Policy Optimization): On-policy, stable
- **A2C** (Advantage Actor-Critic): On-policy, fast
- **TD3** (Twin Delayed DDPG): Off-policy, continuous control

### Custom Reward Hyperparameters

Edit `config/config_custom.yaml`:

```yaml
angle_weight: 1.5              # Weight for angle penalty
velocity_weight: 0.05          # Weight for velocity penalty
action_weight: 0.001           # Weight for action penalty
stability_bonus: 1.0           # Bonus for stable upright position
stability_threshold: 0.1       # Angle threshold (radians) for bonus
```

### Modify Custom Reward Function

Edit `src/custom_env_wrapper.py`:

```python
def reward(self, reward: float) -> float:
    state = self.env.unwrapped.state
    theta = state[0]
    theta_dot = state[1]
    
    # Your custom reward logic here
    # ...
    
    return custom_reward
```

---

## 📊 Results

### Expected Performance

| Experiment | Algorithm | Reward | Mean Reward | Convergence |
|------------|-----------|--------|-------------|-------------|
| **Baseline** | SAC | Default | -150 to -100 | ~60k steps |
| **Custom** | SAC | Custom | -150 to -100 | ~40-50k steps |
| **Extension** | PPO | Custom | -200 to -150 | ~70-80k steps |

### Comparison Analysis

**Three Key Comparisons**:

1. **Baseline vs Custom** (Same algorithm, different reward)
   - Tests reward engineering effectiveness
   - Metrics: convergence speed, final performance, stability

2. **Custom SAC vs Custom PPO** (Same reward, different algorithm)
   - Tests algorithm choice impact
   - Metrics: sample efficiency, training stability, final performance

3. **Baseline SAC vs Extension PPO** (Overall comparison)
   - Combined effect of reward + algorithm
   - Metrics: best overall performance

### Statistical Robustness

Each experiment runs **3 independent trials** to ensure:
- Mean performance is representative
- Standard deviation shows consistency
- Results are reproducible
- Statistical significance can be assessed

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: `IndexError: index 2 is out of bounds`
**Cause**: Accessing observation instead of internal state in custom wrapper.

**Solution**: Use `self.env.unwrapped.state` in `custom_env_wrapper.py`:
```python
state = self.env.unwrapped.state  # ✓ Correct
theta = state[0]
theta_dot = state[1]
```

#### Issue: Poor performance after training
**Checklist**:
- [ ] All configs use same timesteps (100,000)
- [ ] Custom reward scaled properly (-16 to +2 range)
- [ ] Evaluated with `deterministic=True` in `evaluate.py`
- [ ] Checked TensorBoard for training curve issues
- [ ] Verified environment creates successfully

#### Issue: `FileNotFoundError` for config
**Cause**: Running scripts from wrong directory.

**Solution**: Always run from project root:
```bash
cd Reinforcement-Learning-Project
python src/train_baseline.py  # ✓ Correct
```

#### Issue: TensorBoard shows no data
**Cause**: Logs directory path incorrect or training incomplete.

**Solution**:
```bash
# Verify logs exist
ls logs/baseline/SAC_trial1/

# Use correct path
tensorboard --logdir=logs
```

#### Issue: Training is too slow
**Solutions**:
- Reduce timesteps: `timesteps: 50000`
- Use GPU if available (automatic with PyTorch)
- Reduce checkpoint frequency: `checkpoint_freq: 20000`
- Run fewer trials during development: `num_trials: 1`

---

## 📚 Dependencies

```txt
gymnasium>=0.29.1
stable-baselines3>=2.2.1
tensorboard>=2.15.1
numpy>=1.24.3
PyYAML>=6.0.1
torch>=2.1.0
matplotlib>=3.7.0 (for compare_results.py)
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔗 Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)

---

## 👤 Author

**Filip Zekovich** and **Ryan Matthew**

---

## 🙏 Acknowledgments

- **Stable Baselines3** team for excellent DRL library
- **Gymnasium** (formerly OpenAI Gym) for standardized environments
- Course instructors for project guidance

