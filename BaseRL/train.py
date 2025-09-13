import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import nle
from nle import nethack
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import torch

class NetHackWrapper(gym.Wrapper):
    """Wrapper to simplify NetHack observations and add reward shaping"""
    
    def __init__(self, env, render_mode='human'):
        super().__init__(env)
        self._render_mode = render_mode
        
        # Use glyphs observation (2D grid representation)
        self.observation_space = gym.spaces.Box(
            low=0, high=5976, shape=(21, 79), dtype=np.int32
        )
        
        # Initialize tracking variables for reward shaping
        self.prev_blstats = None
        self.visited_positions = set()
        self.step_count = 0
        self.max_episode_steps = 5000
        
    def observation(self, obs):
        # Extract glyphs (2D representation of the game state)
        return obs['glyphs']
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset tracking variables
        self.prev_blstats = obs['blstats'].copy() if 'blstats' in obs else None
        self.visited_positions = set()
        self.step_count = 0
        # Add initial position to visited set
        if 'blstats' in obs and len(obs['blstats']) >= 2:
            x, y = obs['blstats'][0], obs['blstats'][1]
            self.visited_positions.add((x, y))
        return self.observation(obs), info
    
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Calculate shaped reward
        shaped_reward = self.calculate_reward(obs, base_reward, terminated, truncated, info)
        
        # Limit episode length to prevent infinite episodes
        if self.step_count >= self.max_episode_steps:
            truncated = True
            
        return self.observation(obs), shaped_reward, terminated, truncated, info
    
    def calculate_reward(self, obs, base_reward, terminated, truncated, info):
        """Calculate shaped reward based on game state changes"""
        reward = base_reward
        
        if 'blstats' not in obs or self.prev_blstats is None:
            return reward
            
        current_blstats = obs['blstats']
        
        # Reward components
        # 1. Score increase (most important)
        score_diff = current_blstats[9] - self.prev_blstats[9]
        if score_diff > 0:
            reward += score_diff * 0.01  # Scale down score rewards
        
        # 2. Health considerations
        hp_current = current_blstats[10] if len(current_blstats) > 10 else 0
        hp_max = current_blstats[11] if len(current_blstats) > 11 else 1
        hp_prev = self.prev_blstats[10] if len(self.prev_blstats) > 10 else 0
        
        # Reward for maintaining health, penalize for losing health
        if hp_current > hp_prev:
            reward += 0.1  # Small reward for healing
        elif hp_current < hp_prev:
            reward -= 0.1  # Small penalty for damage
            
        # Large penalty for dying
        if terminated and hp_current <= 0:
            reward -= 10.0
            
        # 3. Exploration reward
        if len(current_blstats) >= 2:
            x, y = current_blstats[0], current_blstats[1]
            pos = (x, y)
            if pos not in self.visited_positions:
                self.visited_positions.add(pos)
                reward += 0.05  # Small reward for exploring new positions
                
        # 4. Depth progression reward
        depth_diff = current_blstats[12] - self.prev_blstats[12] if len(current_blstats) > 12 else 0
        if depth_diff > 0:
            reward += 5.0 * depth_diff  # Significant reward for going deeper
            
        # 5. Experience gain
        exp_diff = current_blstats[18] - self.prev_blstats[18] if len(current_blstats) > 18 else 0
        if exp_diff > 0:
            reward += exp_diff * 0.001  # Small reward for gaining experience
            
        # 6. Time penalty to encourage efficient play
        reward -= 0.001  # Very small penalty per step
        
        # 7. Survival bonus
        if not terminated and not truncated:
            reward += 0.002  # Small bonus for staying alive
            
        # Update previous stats
        self.prev_blstats = current_blstats.copy()
        
        return reward
    
    def render(self, mode='human'):
        if mode == 'human' and self._render_mode == 'human':
            # Clear screen and render
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Get current observation
            obs = self.env.unwrapped.last_observation
            if obs is not None:
                # Print the TTY chars (ASCII representation)
                if 'tty_chars' in obs:
                    chars = obs['tty_chars']
                    for row in chars:
                        print(''.join([chr(c) if 32 <= c <= 126 else ' ' for c in row]))
                print(f"Score: {obs.get('score', 0)}")
                print(f"Turn: {obs.get('turn', 0)}")
        
        return self.env.render(mode)

class BenchmarkCallback(BaseCallback):
    """Custom callback to track training metrics and performance"""
    
    def __init__(self, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_start_time = None
        self.step_times = []
        
    def _on_training_start(self):
        self.training_start_time = time.time()
        
    def _on_step(self):
        step_start = time.time()
        
        # Track episode completion
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    if 'episode' in self.locals['infos'][i]:
                        episode_reward = self.locals['infos'][i]['episode']['r']
                        episode_length = self.locals['infos'][i]['episode']['l']
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        
                        if self.verbose > 0:
                            print(f"Episode finished: Reward={episode_reward:.2f}, Length={episode_length}")
        
        # Track step timing
        step_end = time.time()
        self.step_times.append(step_end - step_start)
        
        # Keep only recent step times for performance calculation
        if len(self.step_times) > 1000:
            self.step_times = self.step_times[-1000:]
            
        return True
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        total_time = time.time() - self.training_start_time if self.training_start_time else 0
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0
        
        stats = {
            'total_training_time': total_time,
            'avg_step_time': avg_step_time,
            'steps_per_second': steps_per_second,
            'total_episodes': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'best_episode_reward': np.max(self.episode_rewards) if self.episode_rewards else 0,
        }
        return stats

def create_nethack_env(render_mode='human'):
    """Create a NetHack environment with our wrapper"""
    env = gym.make('NetHackScore-v0')  # NetHackScore provides better base rewards than NetHackChallenge
    env = NetHackWrapper(env, render_mode=render_mode)
    return env

def train_agent(
    algorithm='PPO',
    total_timesteps=100000,
    render_training=False,
    eval_freq=5000,
    save_model=True,
    model_name='nethack_agent'
):
    """Train an SB3 agent on NetHack"""
    
    print(f"Starting training with {algorithm} for {total_timesteps} timesteps")
    
    # Create training environment
    render_mode = 'human' if render_training else None
    env = make_vec_env(
        lambda: Monitor(create_nethack_env(render_mode)), 
        n_envs=1
    )
    
    # Create evaluation environment
    eval_env = Monitor(create_nethack_env('human'))
    
    # Initialize the agent
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1,
            tensorboard_log="./nethack_tensorboard/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    elif algorithm == 'A2C':
        model = A2C(
            'MlpPolicy', 
            env, 
            verbose=1,
            tensorboard_log="./nethack_tensorboard/",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    elif algorithm == 'DQN':
        model = DQN(
            'MlpPolicy', 
            env, 
            verbose=1,
            tensorboard_log="./nethack_tensorboard/",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Set up callbacks
    benchmark_callback = BenchmarkCallback(eval_freq=1000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}_best/",
        log_path=f"./logs/{model_name}/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    callbacks = [benchmark_callback, eval_callback]
    
    # Train the agent
    print("Training started...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False
    )
    
    # Save the final model
    if save_model:
        model.save(f"./models/{model_name}_final")
        print(f"Model saved as {model_name}_final")
    
    # Get training statistics
    stats = benchmark_callback.get_stats()
    return model, stats, benchmark_callback

def benchmark_agent(model, num_episodes=10, render=True):
    """Benchmark the trained agent"""
    print(f"\nBenchmarking agent for {num_episodes} episodes...")
    
    env = create_nethack_env('human' if render else None)
    episode_rewards = []
    episode_lengths = []
    episode_times = []
    
    for episode in range(num_episodes):
        result = env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        episode_reward = 0
        episode_length = 0
        start_time = time.time()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
                time.sleep(0.1)  # Slow down for viewing
        
        episode_time = time.time() - start_time
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_times.append(episode_time)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Length={episode_length}, Time={episode_time:.2f}s")
    
    # Calculate benchmark statistics
    benchmark_stats = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'avg_time_per_episode': np.mean(episode_times),
        'best_reward': np.max(episode_rewards),
        'worst_reward': np.min(episode_rewards)
    }
    
    return benchmark_stats, episode_rewards, episode_lengths

def plot_training_progress(callback):
    """Plot training progress"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    ax1.plot(callback.episode_rewards)
    ax1.set_title('Episode Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Episode lengths
    ax2.plot(callback.episode_lengths)
    ax2.set_title('Episode Lengths Over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    
    # Reward distribution
    if callback.episode_rewards:
        ax3.hist(callback.episode_rewards, bins=20, alpha=0.7)
        ax3.set_title('Episode Reward Distribution')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
    
    # Rolling average reward
    if len(callback.episode_rewards) > 10:
        window_size = min(50, len(callback.episode_rewards) // 4)
        rolling_avg = np.convolve(
            callback.episode_rewards, 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        ax4.plot(rolling_avg)
        ax4.set_title(f'Rolling Average Reward (window={window_size})')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Reward')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150)
    plt.show()

def list_available_envs():
    """List all available NetHack environments"""
    nethack_envs = []
    for env_id in gym.envs.registry.keys():
        if 'nethack' in env_id.lower() or 'nle' in env_id.lower():
            nethack_envs.append(env_id)
    return sorted(nethack_envs)

def test_env_setup():
    """Test the environment setup and print debug info"""
    print("=== Testing NetHack Environment Setup ===")
    
    # First, check if NLE is properly installed
    try:
        import nle
        print(f"✓ NLE imported successfully. Version: {getattr(nle, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"✗ NLE import failed: {e}")
        print("Install with: pip install nle")
        return False
    
    # Check available environments
    available_envs = list_available_envs()
    print(f"Available NetHack environments: {available_envs}")
    
    try:
        env = create_nethack_env(render_mode=None)
        print("✓ Environment created successfully")
        print(f"✓ Environment type: {type(env.env)}")
        
        # Test reset
        result = env.reset()
        if isinstance(result, tuple):
            obs, info = result
            print(f"✓ Reset successful - obs shape: {obs.shape}, obs dtype: {obs.dtype}")
            print(f"  Info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")
        else:
            obs = result
            print(f"✓ Reset successful - obs shape: {obs.shape}, obs dtype: {obs.dtype}")
        
        print(f"✓ Observation space: {env.observation_space}")
        print(f"✓ Action space: {env.action_space}")
        
        # Check the original observation structure
        if hasattr(env, 'last_observation') and env.last_observation is not None:
            last_obs = env.last_observation
            if isinstance(last_obs, dict):
                print(f"✓ Raw observation keys: {list(last_obs.keys())}")
                for key, value in last_obs.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: type={type(value)}")
        
        # Test a few steps
        print("\nTesting environment steps...")
        for i in range(3):
            action = env.action_space.sample()
            step_result = env.step(action)
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            print(f"  Step {i+1}: action={action}, reward={reward:.3f}, done={done}, obs_shape={obs.shape}")
            if done:
                print("  Episode ended, resetting...")
                env.reset()
                break
        
        print("✓ Environment test completed successfully\n")
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training and benchmarking pipeline"""
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Test environment setup first
    if not test_env_setup():
        print("Environment setup failed. Please check your NetHack Learning Environment installation.")
        return
    
    # Training configuration
    config = {
        'algorithm': 'PPO',  # Change to 'A2C' or 'DQN' if desired
        'total_timesteps': 20000,  # Reduced for faster testing
        'render_training': False,  # Set to True to watch training (slower)
        'eval_freq': 2000,  # More frequent evaluation
        'model_name': 'nethack_ppo_shaped'
    }
    
    print("=== NetHack SB3 Training Pipeline ===")
    print(f"Configuration: {config}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Train the agent
    model, training_stats, callback = train_agent(**config)
    
    # Print training statistics
    print("\n=== Training Statistics ===")
    for key, value in training_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Plot training progress
    plot_training_progress(callback)
    
    # Benchmark the trained agent
    print("\n=== Benchmarking Phase ===")
    benchmark_stats, rewards, lengths = benchmark_agent(
        model, 
        num_episodes=5, 
        render=True  # Set to False for faster benchmarking
    )
    
    print("\n=== Benchmark Results ===")
    for key, value in benchmark_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\n=== Training Complete ===")
    print("Files saved:")
    print("- Model: ./models/")
    print("- Logs: ./logs/")
    print("- Training progress plot: training_progress.png")
    print("\nRun 'tensorboard --logdir ./nethack_tensorboard' to view detailed training metrics")

if __name__ == "__main__":
    main()