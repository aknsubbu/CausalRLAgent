#!/usr/bin/env python3
"""
Fixed training script for NetHack PPO with improved reward shaping and training stability.

Key fixes:
- More conservative reward shaping
- Longer training duration
- Better exploration incentives
- Proper early stopping criteria
"""

import gymnasium as gym
import nle.env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

# Import your existing classes
from improved_reward_shaping import *

class FixedNetHackRewardShaper:
    """Fixed reward shaping with more conservative bonuses"""

    def __init__(self):
        self.previous_stats = None
        self.previous_glyphs = None
        self.visited_positions = set()
        self.last_position = None
        
        # MUCH more conservative reward weights
        self.exploration_reward = 0.001  # Reduced from 0.005
        self.health_reward = 0.002       # Reduced from 0.01
        self.level_reward = 1.0          # Reduced from 5.0
        self.experience_reward = 0.0001  # Reduced from 0.001
        self.death_penalty = -1.0        # Reduced penalty
        self.stuck_penalty = -0.001      # Very small
        self.item_pickup_reward = 0.02   # Reduced from 0.1
        self.time_penalty = -0.00001     # Minimal time penalty
        
        # Tracking variables
        self.initial_stats = None
        self.episode_start_time = None
        self.steps_taken = 0
        self.consecutive_stuck_steps = 0
        self.max_stuck = 50  # Allow more stuck steps

    def shape_reward(self, obs, raw_reward, done, info):
        """Apply conservative reward shaping"""
        
        # Start with raw reward (scaled down less aggressively)
        shaped_reward = raw_reward * 0.5  # Less scaling
        
        # Extract current stats
        if isinstance(obs, tuple):
            obs = obs[0]

        current_stats = obs.get('blstats', np.zeros(26))
        current_glyphs = obs.get('glyphs', np.zeros((21, 79)))

        # Initialize on first call
        if self.initial_stats is None:
            self.initial_stats = current_stats.copy()
            self.episode_start_time = time.time()

        self.steps_taken += 1

        # Very conservative rewards
        if self.previous_stats is not None:
            # Health changes
            health_diff = current_stats[0] - self.previous_stats[0]
            if health_diff > 0:
                shaped_reward += health_diff * self.health_reward
            elif health_diff < 0:
                shaped_reward += health_diff * self.health_reward * 0.1  # Small penalty

            # Level progression (rare but important)
            level_diff = current_stats[7] - self.previous_stats[7]
            if level_diff > 0:
                shaped_reward += level_diff * self.level_reward

            # Experience (very small bonus)
            exp_diff = current_stats[8] - self.previous_stats[8]
            if exp_diff > 0:
                shaped_reward += exp_diff * self.experience_reward

        # Exploration with diminishing returns
        current_pos = tuple(current_stats[:2]) if len(current_stats) > 1 else (0, 0)
        if current_pos not in self.visited_positions:
            self.visited_positions.add(current_pos)
            # Bonus decreases with exploration count
            exploration_bonus = self.exploration_reward / (1.0 + len(self.visited_positions) * 0.1)
            shaped_reward += exploration_bonus

        # Anti-stuck mechanism (very gentle)
        if current_pos == self.last_position:
            self.consecutive_stuck_steps += 1
            if self.consecutive_stuck_steps > self.max_stuck:
                shaped_reward += self.stuck_penalty
        else:
            self.consecutive_stuck_steps = 0

        # Death penalty (conservative)
        if done and len(current_stats) > 0 and current_stats[0] <= 0:
            shaped_reward += self.death_penalty

        # Update tracking
        self.previous_stats = current_stats.copy()
        self.previous_glyphs = current_glyphs.copy()  
        self.last_position = current_pos

        # Clamp to prevent extreme values
        shaped_reward = np.clip(shaped_reward, -5.0, 5.0)

        return shaped_reward

    def get_episode_stats(self):
        """Get episode statistics"""
        return {
            'exploration_count': len(self.visited_positions),
            'unique_positions': len(self.visited_positions),
            'steps_taken': self.steps_taken
        }

    def reset(self):
        """Reset for new episode"""
        self.previous_stats = None
        self.previous_glyphs = None
        self.visited_positions.clear()
        self.last_position = None
        self.initial_stats = None
        self.episode_start_time = None
        self.steps_taken = 0
        self.consecutive_stuck_steps = 0

def fixed_train_agent():
    """Train agent with fixed parameters"""
    
    print("🚀 Starting FIXED NetHack PPO Training...")
    
    # Create environment
    env = create_nethack_env()
    
    # Create agent with more conservative parameters
    agent = EnhancedNetHackPPOAgent(
        action_dim=env.action_space.n,
        learning_rate=5e-5,  # MUCH lower learning rate
        gamma=0.99,
        clip_ratio=0.1,      # More conservative clipping
        entropy_coef=0.05,   # Higher entropy for exploration
        value_coef=0.5,
        max_grad_norm=0.5,
        use_wandb=False
    )
    
    # Replace reward shaper with fixed version
    agent.reward_shaper = FixedNetHackRewardShaper()
    
    print("✅ Fixed agent created with conservative parameters!")
    
    # Test observation processing
    obs = env.reset()
    processed_obs = agent.process_observation(obs)
    print("✅ Observation processing working!")
    
    # Train with much stricter criteria
    print("\n🎯 Starting fixed training...")
    
    raw_rewards, shaped_rewards = agent.train(
        env,
        num_episodes=500,      # More episodes
        update_freq=2048,      # Same update frequency  
        eval_freq=50,          # More frequent evaluation
        early_stopping_patience=100,  # More patience
        target_reward=20       # MUCH lower, realistic target
    )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"FIXED_nethack_ppo_{timestamp}.pth"
    agent.save_model(model_path)
    
    env.close()
    print(f"✅ Fixed training complete! Model saved as: {model_path}")
    
    return model_path

if __name__ == "__main__":
    model_path = fixed_train_agent()
    
    print(f"\n🎯 To evaluate your fixed model, run:")
    print(f"python simple_evaluate.py --model_path {model_path} --episodes 10")