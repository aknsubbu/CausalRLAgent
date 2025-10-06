import gymnasium as gym  # Changed from 'import gym'
import nle
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import time

# NetHack character mappings for visualization
CHARACTER_MAP = {
    ord(' '): ' ', ord('.'): '.', ord('#'): '#', ord('>'): '>', ord('<'): '<',
    ord('+'): '+', ord('-'): '-', ord('|'): '|', ord('A'): 'A', ord('B'): 'B',
    ord('C'): 'C', ord('D'): 'D', ord('E'): 'E', ord('F'): 'F', ord('G'): 'G',
    ord('H'): 'H', ord('I'): 'I', ord('J'): 'J', ord('K'): 'K', ord('L'): 'L',
    ord('M'): 'M', ord('N'): 'N', ord('O'): 'O', ord('P'): 'P', ord('Q'): 'Q',
    ord('R'): 'R', ord('S'): 'S', ord('T'): 'T', ord('U'): 'U', ord('V'): 'V',
    ord('W'): 'W', ord('X'): 'X', ord('Y'): 'Y', ord('Z'): 'Z', ord('@'): '@',
    ord('a'): 'a', ord('b'): 'b', ord('c'): 'c', ord('d'): 'd', ord('e'): 'e',
    ord('f'): 'f', ord('g'): 'g', ord('h'): 'h', ord('i'): 'i', ord('j'): 'j',
    ord('k'): 'k', ord('l'): 'l', ord('m'): 'm', ord('n'): 'n', ord('o'): 'o',
    ord('p'): 'p', ord('q'): 'q', ord('r'): 'r', ord('s'): 's', ord('t'): 't',
    ord('u'): 'u', ord('v'): 'v', ord('w'): 'w', ord('x'): 'x', ord('y'): 'y',
    ord('z'): 'z'
}

class NetHackCNN(BaseFeaturesExtractor):
    """
    Custom CNN for processing NetHack observations
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(NetHackCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class VisualizationCallback(BaseCallback):
    """
    Custom callback for visualizing NetHack gameplay
    """
    def __init__(self, verbose=0, display_freq=100):
        super(VisualizationCallback, self).__init__(verbose)
        self.display_freq = display_freq
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Visualize every N episodes
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            if self.episode_count % self.display_freq == 0:
                self._visualize_game()
        return True

    def _visualize_game(self):
        """Create human-readable visualization of current game state"""
        if hasattr(self.model.get_env(), 'envs'):
            # Access the underlying NLE environment
            env = self.model.get_env().envs[0]
            if hasattr(env.unwrapped, 'last_observation'):
                obs = env.unwrapped.last_observation
                if obs is not None:
                    self._render_observation(obs)

    def _render_observation(self, obs):
        """Render the observation in a human-readable format"""
        glyphs = obs['glyphs']
        chars = np.vectorize(CHARACTER_MAP.get)(glyphs, ' ')
        
        plt.figure(figsize=(12, 8))
        
        # Create text-based visualization
        ax = plt.gca()
        ax.set_xlim(0, chars.shape[1])
        ax.set_ylim(0, chars.shape[0])
        
        for i in range(chars.shape[0]):
            for j in range(chars.shape[1]):
                char = chars[i, j]
                color = 'white' if char in ['#', '+', '-'] else 'lightblue'
                ax.text(j, chars.shape[0]-i-1, char, 
                       ha='center', va='center', fontsize=8,
                       color='black' if char != ' ' else 'gray')
        
        # Add status information
        blstats = obs.get('blstats', np.zeros(25))
        status_text = (
            f"Score: {blstats[10]} | "
            f"HP: {blstats[1]}/{blstats[2]} | "
            f"Level: {blstats[12]} | "
            f"Gold: {blstats[13]}"
        )
        
        plt.title(f"NetHack Visualization - {status_text}", fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def preprocess_observation(obs):
    """Convert observation to model input format"""
    if isinstance(obs, dict):
        glyphs = obs['glyphs']
    else:
        glyphs = obs
    
    # Normalize to [0, 1] range
    normalized = glyphs.astype(np.float32) / 255.0
    # Add channel dimension
    return normalized[np.newaxis, :, :]

class NetHackWrapper(gym.Wrapper):
    """
    Custom wrapper for preprocessing NetHack observations
    """
    def __init__(self, env):
        super(NetHackWrapper, self).__init__(env)
        # Update observation space to match preprocessed observations
        # Get actual shape from the environment
        sample_obs, _ = self.env.reset()
        processed_obs = preprocess_observation(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=processed_obs.shape, dtype=np.float32
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Convert to new API format (terminated/truncated)
        done = terminated or truncated
        return preprocess_observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return preprocess_observation(obs), info

def create_model(env, algorithm='PPO'):
    """
    Create RL model with custom CNN
    """
    policy_kwargs = dict(
        features_extractor_class=NetHackCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    if algorithm == 'PPO':
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./nethack_ppo_tensorboard/"
        )
    elif algorithm == 'DQN':
        model = DQN(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tensorboard_log="./nethack_dqn_tensorboard/"
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    return model

def train_nethack_agent(algorithm='PPO', episodes=10000, save_freq=1000):
    """
    Train NetHack agent with Stable Baselines3
    """
    # Create and wrap environment - Updated for Gymnasium
    env = gym.make('NetHack-v0')  # Changed from 'NetHackScore-v0'
    env = NetHackWrapper(env)
    env = Monitor(env, "./logs/")
    
    # Create model
    model = create_model(env, algorithm)
    
    # Create visualization callback
    vis_callback = VisualizationCallback(display_freq=50)
    
    # Train the model
    print(f"Starting training with {algorithm}...")
    model.learn(
        total_timesteps=episodes,
        callback=vis_callback,
        log_interval=10
    )
    
    # Save the final model
    model.save(f"nethack_{algorithm.lower()}_final")
    print(f"Training completed. Model saved as nethack_{algorithm.lower()}_final")
    
    return model

def evaluate_model(model_path, episodes=10, algorithm='PPO'):
    """
    Evaluate trained model
    """
    # Load environment
    env = gym.make('NetHack-v0')  # Changed from 'NetHackScore-v0'
    env = NetHackWrapper(env)
    
    # Load model
    if algorithm == 'PPO':
        model = PPO.load(model_path, env=env)
    else:
        model = DQN.load(model_path, env=env)
    
    # Evaluate
    scores = []
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Visualize occasionally
            if step % 50 == 0:
                print(f"Evaluation Episode {episode}, Step {step}")
        
        scores.append(total_reward)
        print(f"Evaluation Episode {episode}: Score = {total_reward}")
    
    avg_score = np.mean(scores)
    print(f"Average Score over {episodes} episodes: {avg_score}")
    return scores

if __name__ == "__main__":
    # Create directories
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)
    
    # Choose algorithm: 'PPO' or 'DQN'
    ALGORITHM = 'PPO'  # PPO generally works better for this type of environment
    
    # Train the agent
    model = train_nethack_agent(
        algorithm=ALGORITHM,
        episodes=50000,  # Adjust based on your computational resources
        save_freq=5000
    )
    
    # Evaluate the trained model
    evaluate_model(f"nethack_{ALGORITHM.lower()}_final", episodes=5, algorithm=ALGORITHM)