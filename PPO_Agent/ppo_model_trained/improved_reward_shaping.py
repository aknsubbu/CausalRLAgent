import gymnasium as gym
import nle.env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import logging
import time
import csv
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Optional imports for advanced features
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

class TrainingLogger:
    """Comprehensive training logger with CSV metrics tracking"""

    def __init__(self, log_dir: str = "logs", experiment_name: str = None, use_wandb: bool = False):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"nethack_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Setup CSV files for logging
        self.episode_csv_file = os.path.join(log_dir, f"{self.experiment_name}_episodes.csv")
        self.training_csv_file = os.path.join(log_dir, f"{self.experiment_name}_training.csv")
        self.extensive_csv_file = os.path.join(log_dir, f"{self.experiment_name}_extensive.csv")

        # Initialize CSV files with headers
        self._init_csv_files()

        # Setup file logging with immediate flush
        log_file = os.path.join(log_dir, f"{self.experiment_name}.log")

        # Create custom stream handler with immediate flush
        import sys
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[file_handler, stream_handler],
            force=True
        )

        self.logger = logging.getLogger(__name__)

        # Ensure immediate output
        for handler in self.logger.handlers:
            handler.flush = lambda: sys.stdout.flush()

        # Initialize metrics storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.training_metrics = defaultdict(list)

        # Initialize wandb if available
        if self.use_wandb:
            wandb.init(project="nethack-ppo", name=self.experiment_name)
            self.logger.info("WandB logging initialized")

    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Episode CSV headers
        episode_headers = [
            'timestamp', 'episode', 'raw_reward', 'shaped_reward', 'episode_length',
            'died', 'level_ups', 'max_health', 'items_collected', 'unique_positions',
            'exploration_reward', 'survival_time', 'actions_taken'
        ]
        
        # Training CSV headers
        training_headers = [
            'timestamp', 'step', 'actor_loss', 'critic_loss', 'policy_loss',
            'value_loss', 'entropy', 'clip_fraction', 'grad_norm', 'learning_rate'
        ]

        # Extensive CSV headers (every 10 episodes)
        extensive_headers = [
            'timestamp', 'episode_batch', 'avg_raw_reward', 'avg_shaped_reward',
            'avg_episode_length', 'success_rate', 'survival_rate', 'best_reward',
            'worst_reward', 'reward_std', 'exploration_efficiency', 'total_unique_positions',
            'avg_actor_loss', 'avg_critic_loss', 'avg_entropy', 'avg_clip_fraction',
            'reward_trend', 'learning_stability', 'policy_variance'
        ]

        # Create CSV files with headers
        with open(self.episode_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(episode_headers)

        with open(self.training_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(training_headers)

        with open(self.extensive_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(extensive_headers)

    def log_episode_csv(self, episode: int, metrics: Dict):
        """Log episode-level metrics to CSV"""
        timestamp = datetime.now().isoformat()
        
        # Prepare episode data
        episode_data = [
            timestamp,
            episode,
            metrics.get('raw_reward', 0),
            metrics.get('shaped_reward', 0),
            metrics.get('episode_length', 0),
            metrics.get('died', 0),
            metrics.get('level_ups', 0),
            metrics.get('max_health', 0),
            metrics.get('items_collected', 0),
            metrics.get('unique_positions', 0),
            metrics.get('exploration_reward', 0),
            metrics.get('survival_time', 0),
            metrics.get('actions_taken', 0)
        ]

        # Write to CSV
        with open(self.episode_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(episode_data)

        # Also log to console
        self.logger.info(f"Episode {episode}: Raw Reward: {metrics.get('raw_reward', 0):.3f}, "
                        f"Shaped Reward: {metrics.get('shaped_reward', 0):.3f}, "
                        f"Length: {metrics.get('episode_length', 0)}")

        # Store for extensive logging
        for key, value in metrics.items():
            self.episode_metrics[key].append(value)

        if self.use_wandb:
            wandb.log({f"episode/{k}": v for k, v in metrics.items()}, step=episode)

    def log_training_csv(self, step: int, metrics: Dict):
        """Log training-level metrics to CSV"""
        timestamp = datetime.now().isoformat()
        
        # Prepare training data
        training_data = [
            timestamp,
            step,
            metrics.get('actor_loss', 0),
            metrics.get('critic_loss', 0),
            metrics.get('policy_loss', 0),
            metrics.get('value_loss', 0),
            metrics.get('entropy', 0),
            metrics.get('clip_fraction', 0),
            metrics.get('grad_norm', 0),
            metrics.get('learning_rate', 0)
        ]

        # Write to CSV
        with open(self.training_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(training_data)

        self.logger.info(f"Training Step {step}: Actor Loss: {metrics.get('actor_loss', 0):.4f}, "
                        f"Critic Loss: {metrics.get('critic_loss', 0):.4f}, "
                        f"Entropy: {metrics.get('entropy', 0):.4f}")

        for key, value in metrics.items():
            self.training_metrics[key].append(value)

        if self.use_wandb:
            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

    def log_extensive_analysis(self, episode_batch: int, recent_episodes: List[Dict]):
        """Log extensive analysis every 10 episodes"""
        timestamp = datetime.now().isoformat()
        
        if not recent_episodes:
            return

        # Calculate comprehensive metrics
        raw_rewards = [ep.get('raw_reward', 0) for ep in recent_episodes]
        shaped_rewards = [ep.get('shaped_reward', 0) for ep in recent_episodes]
        episode_lengths = [ep.get('episode_length', 0) for ep in recent_episodes]
        deaths = [ep.get('died', 0) for ep in recent_episodes]
        unique_positions = [ep.get('unique_positions', 0) for ep in recent_episodes]

        # Advanced metrics
        avg_raw_reward = np.mean(raw_rewards)
        avg_shaped_reward = np.mean(shaped_rewards)
        avg_episode_length = np.mean(episode_lengths)
        success_rate = sum(1 for r in raw_rewards if r > 0) / len(raw_rewards)
        survival_rate = 1 - np.mean(deaths)
        best_reward = max(raw_rewards)
        worst_reward = min(raw_rewards)
        reward_std = np.std(raw_rewards)
        exploration_efficiency = np.mean(unique_positions) / max(avg_episode_length, 1)
        total_unique_positions = sum(unique_positions)

        # Training metrics
        recent_training = list(self.training_metrics.get('actor_loss', []))[-50:]
        avg_actor_loss = np.mean(recent_training) if recent_training else 0
        recent_critic = list(self.training_metrics.get('critic_loss', []))[-50:]
        avg_critic_loss = np.mean(recent_critic) if recent_critic else 0
        recent_entropy = list(self.training_metrics.get('entropy', []))[-50:]
        avg_entropy = np.mean(recent_entropy) if recent_entropy else 0
        recent_clip = list(self.training_metrics.get('clip_fraction', []))[-50:]
        avg_clip_fraction = np.mean(recent_clip) if recent_clip else 0

        # Trend analysis
        reward_trend = self._calculate_trend(raw_rewards)
        learning_stability = 1 / (1 + reward_std) if reward_std > 0 else 1
        policy_variance = np.var(recent_entropy) if recent_entropy else 0

        # Prepare extensive data
        extensive_data = [
            timestamp,
            episode_batch,
            avg_raw_reward,
            avg_shaped_reward,
            avg_episode_length,
            success_rate,
            survival_rate,
            best_reward,
            worst_reward,
            reward_std,
            exploration_efficiency,
            total_unique_positions,
            avg_actor_loss,
            avg_critic_loss,
            avg_entropy,
            avg_clip_fraction,
            reward_trend,
            learning_stability,
            policy_variance
        ]

        # Write to CSV
        with open(self.extensive_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(extensive_data)

        # Enhanced console logging
        self.logger.info("="*80)
        self.logger.info(f"📊 EXTENSIVE ANALYSIS - Episodes {episode_batch-9} to {episode_batch}")
        self.logger.info("="*80)
        self.logger.info(f"🎯 Performance Metrics:")
        self.logger.info(f"   Average Raw Reward: {avg_raw_reward:.3f}")
        self.logger.info(f"   Average Shaped Reward: {avg_shaped_reward:.3f}")
        self.logger.info(f"   Best Episode Reward: {best_reward:.3f}")
        self.logger.info(f"   Reward Standard Deviation: {reward_std:.3f}")
        self.logger.info(f"   Success Rate: {success_rate:.1%}")
        self.logger.info(f"   Survival Rate: {survival_rate:.1%}")
        self.logger.info("")
        self.logger.info(f"🚀 Exploration Metrics:")
        self.logger.info(f"   Average Episode Length: {avg_episode_length:.1f}")
        self.logger.info(f"   Exploration Efficiency: {exploration_efficiency:.4f}")
        self.logger.info(f"   Total Unique Positions: {total_unique_positions}")
        self.logger.info("")
        self.logger.info(f"🧠 Learning Metrics:")
        self.logger.info(f"   Average Actor Loss: {avg_actor_loss:.4f}")
        self.logger.info(f"   Average Critic Loss: {avg_critic_loss:.4f}")
        self.logger.info(f"   Average Entropy: {avg_entropy:.4f}")
        self.logger.info(f"   Average Clip Fraction: {avg_clip_fraction:.4f}")
        self.logger.info(f"   Reward Trend: {reward_trend:.6f}")
        self.logger.info(f"   Learning Stability: {learning_stability:.3f}")
        self.logger.info("="*80)

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (positive = improving, negative = degrading)"""
        if len(values) < 3:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Simple linear regression slope
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        return slope

    def log_episode(self, episode: int, metrics: Dict):
        """Legacy method - redirect to CSV logging"""
        self.log_episode_csv(episode, metrics)

    def log_training(self, step: int, metrics: Dict):
        """Legacy method - redirect to CSV logging"""
        self.log_training_csv(step, metrics)

    def log_info(self, message: str):
        """Log general information"""
        self.logger.info(message)

    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        all_metrics = {
            'episode_metrics': convert_numpy_types(dict(self.episode_metrics)),
            'training_metrics': convert_numpy_types(dict(self.training_metrics))
        }

        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_file}")
        self.logger.info(f"CSV logs saved to:")
        self.logger.info(f"  Episodes: {self.episode_csv_file}")
        self.logger.info(f"  Training: {self.training_csv_file}")
        self.logger.info(f"  Extensive: {self.extensive_csv_file}")

class PerformanceEvaluator:
    """Advanced performance evaluation and metrics calculation"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.episode_rewards = deque(maxlen=self.window_size)
        self.episode_lengths = deque(maxlen=self.window_size)
        self.shaped_rewards = deque(maxlen=self.window_size)
        self.exploration_rewards = deque(maxlen=self.window_size)
        self.death_counts = deque(maxlen=self.window_size)
        self.level_ups = deque(maxlen=self.window_size)
        self.max_health_achieved = deque(maxlen=self.window_size)
        self.items_collected = deque(maxlen=self.window_size)
        self.unique_positions = deque(maxlen=self.window_size)

        # Training metrics
        self.actor_losses = deque(maxlen=1000)
        self.critic_losses = deque(maxlen=1000)
        self.entropies = deque(maxlen=1000)
        self.grad_norms = deque(maxlen=1000)
        self.learning_rates = deque(maxlen=1000)
        self.clip_fractions = deque(maxlen=1000)
        self.value_losses = deque(maxlen=1000)
        self.policy_losses = deque(maxlen=1000)

    def add_episode_metrics(self, episode_data: Dict):
        """Add episode-level metrics"""
        self.episode_rewards.append(episode_data.get('raw_reward', 0))
        self.episode_lengths.append(episode_data.get('length', 0))
        self.shaped_rewards.append(episode_data.get('shaped_reward', 0))
        self.exploration_rewards.append(episode_data.get('exploration_reward', 0))
        self.death_counts.append(episode_data.get('died', 0))
        self.level_ups.append(episode_data.get('level_ups', 0))
        self.max_health_achieved.append(episode_data.get('max_health', 0))
        self.items_collected.append(episode_data.get('items_collected', 0))
        self.unique_positions.append(episode_data.get('unique_positions', 0))

    def add_training_metrics(self, training_data: Dict):
        """Add training-level metrics"""
        self.actor_losses.append(training_data.get('actor_loss', 0))
        self.critic_losses.append(training_data.get('critic_loss', 0))
        self.entropies.append(training_data.get('entropy', 0))
        self.grad_norms.append(training_data.get('grad_norm', 0))
        self.learning_rates.append(training_data.get('learning_rate', 0))
        self.clip_fractions.append(training_data.get('clip_fraction', 0))
        self.value_losses.append(training_data.get('value_loss', 0))
        self.policy_losses.append(training_data.get('policy_loss', 0))

    def get_episode_metrics(self) -> Dict:
        """Calculate comprehensive episode metrics"""
        if not self.episode_rewards:
            return {}

        metrics = {
            # Basic metrics
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'mean_shaped_reward': np.mean(self.shaped_rewards),

            # Performance indicators
            'reward_trend': self._calculate_trend(self.episode_rewards),
            'length_trend': self._calculate_trend(self.episode_lengths),
            'success_rate': self._calculate_success_rate(),
            'survival_rate': 1 - np.mean(self.death_counts),

            # Exploration metrics
            'mean_exploration': np.mean(self.exploration_rewards),
            'mean_unique_positions': np.mean(self.unique_positions),
            'exploration_efficiency': self._calculate_exploration_efficiency(),

            # Progress metrics
            'mean_level_ups': np.mean(self.level_ups),
            'mean_items_collected': np.mean(self.items_collected),
            'mean_max_health': np.mean(self.max_health_achieved),

            # Stability metrics
            'reward_stability': 1 / (1 + np.std(self.episode_rewards)),
            'length_stability': 1 / (1 + np.std(self.episode_lengths)),
        }

        return metrics

    def get_training_metrics(self) -> Dict:
        """Calculate training-specific metrics"""
        if not self.actor_losses:
            return {}

        metrics = {
            'mean_actor_loss': np.mean(self.actor_losses),
            'mean_critic_loss': np.mean(self.critic_losses),
            'mean_entropy': np.mean(self.entropies),
            'mean_grad_norm': np.mean(self.grad_norms),
            'mean_clip_fraction': np.mean(self.clip_fractions),
            'mean_value_loss': np.mean(self.value_losses),
            'mean_policy_loss': np.mean(self.policy_losses),

            # Learning stability
            'loss_stability': 1 / (1 + np.std(self.actor_losses)),
            'entropy_trend': self._calculate_trend(self.entropies),
            'grad_norm_trend': self._calculate_trend(self.grad_norms),
        }

        return metrics

    def _calculate_trend(self, values: deque) -> float:
        """Calculate trend (positive = improving, negative = degrading)"""
        if len(values) < 10:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Simple linear regression slope
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        return slope

    def _calculate_success_rate(self) -> float:
        """Calculate success rate based on reward thresholds"""
        if not self.episode_rewards:
            return 0.0

        # Define success as episodes with positive reward
        successful_episodes = sum(1 for r in self.episode_rewards if r > 0)
        return successful_episodes / len(self.episode_rewards)

    def _calculate_exploration_efficiency(self) -> float:
        """Calculate exploration efficiency"""
        if not self.unique_positions or not self.episode_lengths:
            return 0.0

        # Exploration efficiency = unique positions per step
        total_positions = sum(self.unique_positions)
        total_steps = sum(self.episode_lengths)

        return total_positions / max(total_steps, 1)

class NetHackRewardShaper:
    """Advanced reward shaping for NetHack"""
    
    def __init__(self):
        self.previous_stats = None
        self.previous_glyphs = None
        self.visited_positions = set()
        self.last_position = None
        self.stuck_counter = 0
        self.max_stuck = 10
        
        # Reward weights
        self.exploration_reward = 0.01
        self.health_reward = 0.001
        self.level_reward = 1.0
        self.experience_reward = 0.0001
        self.death_penalty = -1.0
        self.stuck_penalty = -0.01
        self.item_pickup_reward = 0.05
        self.monster_kill_reward = 0.1
        
    def shape_reward(self, obs, raw_reward, done, info):
        """Apply reward shaping based on game state"""
        shaped_reward = raw_reward
        
        # Extract current stats
        if isinstance(obs, tuple):
            obs = obs[0]
            
        current_stats = obs.get('blstats', np.zeros(26))
        current_glyphs = obs.get('glyphs', np.zeros((21, 79)))
        
        if self.previous_stats is not None:
            # Health change reward/penalty
            health_diff = current_stats[0] - self.previous_stats[0]
            shaped_reward += health_diff * self.health_reward
            
            # Level up reward
            level_diff = current_stats[7] - self.previous_stats[7]
            shaped_reward += level_diff * self.level_reward
            
            # Experience gain reward
            exp_diff = current_stats[8] - self.previous_stats[8]
            shaped_reward += exp_diff * self.experience_reward
            
            # Item pickup detection (inventory count change)
            # This is a simplified version - you could make this more sophisticated
            inv_change = np.sum(current_glyphs > 0) - np.sum(self.previous_glyphs > 0)
            if inv_change > 0:
                shaped_reward += self.item_pickup_reward
        
        # Exploration reward
        current_pos = (current_stats[0], current_stats[1]) if len(current_stats) > 1 else (0, 0)
        if current_pos not in self.visited_positions:
            self.visited_positions.add(current_pos)
            shaped_reward += self.exploration_reward
        
        # Anti-stuck mechanism
        if current_pos == self.last_position:
            self.stuck_counter += 1
            if self.stuck_counter > self.max_stuck:
                shaped_reward += self.stuck_penalty
        else:
            self.stuck_counter = 0
        
        # Death penalty
        if done and current_stats[0] <= 0:  # Player died
            shaped_reward += self.death_penalty
        
        # Update tracking variables
        self.previous_stats = current_stats.copy()
        self.previous_glyphs = current_glyphs.copy()
        self.last_position = current_pos
        
        return shaped_reward
    
    def reset(self):
        """Reset reward shaper for new episode"""
        self.previous_stats = None
        self.previous_glyphs = None
        self.visited_positions.clear()
        self.last_position = None
        self.stuck_counter = 0

class NetHackObservationProcessor:
    """Enhanced observation processor with memory features"""
    
    def __init__(self):
        self.glyph_shape = (21, 79)
        self.stats_dim = 26
        self.message_dim = 256
        self.inventory_dim = 55
        
        # Memory features
        self.position_history = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        
    def process_observation(self, obs, last_action=None):
        """Process observation with memory features"""
        processed = {}
        
        if isinstance(obs, tuple):
            obs = obs[0]
        
        if not isinstance(obs, dict):
            raise ValueError(f"Expected dict observation, got {type(obs)}")
        
        # Process glyphs
        if 'glyphs' in obs:
            glyphs = np.array(obs['glyphs']).astype(np.float32) / 5976.0
            processed['glyphs'] = glyphs
        else:
            processed['glyphs'] = np.zeros(self.glyph_shape, dtype=np.float32)
        
        # Process stats with memory
        if 'blstats' in obs:
            stats = np.array(obs['blstats']).astype(np.float32)
            stats_normalized = stats.copy()
            
            if len(stats) > 1 and stats[1] > 0:
                stats_normalized[0] = stats[0] / stats[1]  # HP ratio
            if len(stats) > 7:
                stats_normalized[7] = min(stats[7] / 30.0, 1.0)  # Level
                
            # Add position to history
            if len(stats) > 1:
                current_pos = (stats[0], stats[1])
                self.position_history.append(current_pos)
            
            if len(stats_normalized) < self.stats_dim:
                padded_stats = np.zeros(self.stats_dim, dtype=np.float32)
                padded_stats[:len(stats_normalized)] = stats_normalized
                processed['stats'] = padded_stats
            else:
                processed['stats'] = stats_normalized[:self.stats_dim]
        else:
            processed['stats'] = np.zeros(self.stats_dim, dtype=np.float32)
        
        # Process message
        if 'message' in obs:
            message = np.array(obs['message']).astype(np.float32)
            if len(message) < self.message_dim:
                padded_message = np.zeros(self.message_dim, dtype=np.float32)
                padded_message[:len(message)] = message / 255.0
                processed['message'] = padded_message
            else:
                processed['message'] = message[:self.message_dim] / 255.0
        else:
            processed['message'] = np.zeros(self.message_dim, dtype=np.float32)
        
        # Process inventory
        if 'inv_strs' in obs:
            inventory = obs['inv_strs']
            inv_features = np.zeros(self.inventory_dim, dtype=np.float32)
            for i, item in enumerate(inventory):
                if i < len(inv_features):
                    try:
                        # FIX: Handle numpy arrays and bytes properly
                        if isinstance(item, np.ndarray):
                            if item.dtype.kind in ('U', 'S', 'O'):  # String-like types
                                item_str = str(item.item()) if item.size == 1 else ""
                            else:
                                item_str = ""
                        elif isinstance(item, bytes):
                            item_str = item.decode('ascii', errors='ignore')
                        elif item is not None:
                            item_str = str(item)
                        else:
                            item_str = ""
                        
                        if len(item_str.strip()) > 0 and item_str.strip() not in ["b''", ""]:
                            inv_features[i] = 1.0
                    except Exception as e:
                        # Silently skip problematic items
                        continue
            processed['inventory'] = inv_features
        else:
            processed['inventory'] = np.zeros(self.inventory_dim, dtype=np.float32)

        
        # Add action history
        if last_action is not None:
            self.action_history.append(last_action)
        
        # Create action history vector
        action_hist_vector = np.zeros(50, dtype=np.float32)
        for i, action in enumerate(list(self.action_history)[-50:]):
            action_hist_vector[i] = action / 23.0  # Normalize by action space size
        processed['action_history'] = action_hist_vector
        
        return processed

class RecurrentNetHackCNN(nn.Module):
    """CNN with LSTM for processing NetHack glyphs with memory"""
    
    def __init__(self, input_shape=(21, 79), cnn_output_dim=512, lstm_hidden_dim=256):
        super(RecurrentNetHackCNN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate CNN output size
        conv_out_size = self._get_conv_out_size(input_shape)
        self.cnn_fc = nn.Linear(conv_out_size, cnn_output_dim)
        
        # LSTM for temporal modeling
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(cnn_output_dim, lstm_hidden_dim, batch_first=True)
        
        # Hidden state initialization
        self.hidden_state = None
        
    def _get_conv_out_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *shape)
            dummy_output = self._forward_conv(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        return x
    
    def forward(self, x, reset_hidden=False):
        batch_size = x.size(0)
        
        # Reset hidden state if requested or if batch size changed
        if reset_hidden or self.hidden_state is None or self.hidden_state[0].size(1) != batch_size:
            self.hidden_state = (
                torch.zeros(1, batch_size, self.lstm_hidden_dim, device=x.device),
                torch.zeros(1, batch_size, self.lstm_hidden_dim, device=x.device)
            )
        
        # CNN forward pass
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        cnn_features = F.relu(self.cnn_fc(x))
        
        # LSTM forward pass
        cnn_features = cnn_features.unsqueeze(1)  # Add sequence dimension
        lstm_out, self.hidden_state = self.lstm(cnn_features, self.hidden_state)
        lstm_features = lstm_out.squeeze(1)  # Remove sequence dimension
        
        return lstm_features
    
    def reset_hidden_state(self):
        """Reset hidden state (call at episode start)"""
        self.hidden_state = None

class RecurrentPPOActor(nn.Module):
    """PPO Actor with LSTM memory"""
    
    def __init__(self, action_dim=23):
        super(RecurrentPPOActor, self).__init__()
        
        # Feature extractors
        self.glyph_cnn = RecurrentNetHackCNN(cnn_output_dim=512, lstm_hidden_dim=256)
        
        # Other feature processors
        self.stats_lstm = nn.LSTM(26, 64, batch_first=True)
        self.message_fc = nn.Linear(256, 128)
        self.inventory_fc = nn.Linear(55, 64)
        self.action_hist_fc = nn.Linear(50, 32)
        
        # Combined feature processing
        combined_dim = 256 + 64 + 128 + 64 + 32  # 544
        self.combined_fc1 = nn.Linear(combined_dim, 512)
        self.combined_fc2 = nn.Linear(512, 256)
        
        # Action head
        self.action_head = nn.Linear(256, action_dim)
        
        # Hidden states
        self.stats_hidden = None
        
    def forward(self, obs, reset_hidden=False):
        batch_size = obs['glyphs'].size(0)
        
        # Process glyphs with recurrent CNN
        glyph_features = self.glyph_cnn(obs['glyphs'], reset_hidden)
        
        # Process stats with LSTM
        if reset_hidden or self.stats_hidden is None or self.stats_hidden[0].size(1) != batch_size:
            self.stats_hidden = (
                torch.zeros(1, batch_size, 64, device=obs['stats'].device),
                torch.zeros(1, batch_size, 64, device=obs['stats'].device)
            )
        
        stats_input = obs['stats'].unsqueeze(1)  # Add sequence dimension
        stats_lstm_out, self.stats_hidden = self.stats_lstm(stats_input, self.stats_hidden)
        stats_features = stats_lstm_out.squeeze(1)  # Remove sequence dimension
        
        # Process other features
        message_features = F.relu(self.message_fc(obs['message']))
        inventory_features = F.relu(self.inventory_fc(obs['inventory']))
        action_hist_features = F.relu(self.action_hist_fc(obs['action_history']))
        
        # Combine all features
        combined = torch.cat([
            glyph_features, stats_features,
            message_features, inventory_features, action_hist_features
        ], dim=1)
        
        # Process combined features
        x = F.relu(self.combined_fc1(combined))
        x = F.relu(self.combined_fc2(x))
        
        # Output action logits
        action_logits = self.action_head(x)
        return action_logits
    
    def reset_hidden_states(self):
        """Reset all hidden states"""
        self.glyph_cnn.reset_hidden_state()
        self.stats_hidden = None

class RecurrentPPOCritic(nn.Module):
    """PPO Critic with LSTM memory"""
    
    def __init__(self):
        super(RecurrentPPOCritic, self).__init__()
        
        # Feature extractors (same architecture as actor)
        self.glyph_cnn = RecurrentNetHackCNN(cnn_output_dim=512, lstm_hidden_dim=256)
        self.stats_lstm = nn.LSTM(26, 64, batch_first=True)
        self.message_fc = nn.Linear(256, 128)
        self.inventory_fc = nn.Linear(55, 64)
        self.action_hist_fc = nn.Linear(50, 32)
        
        # Combined feature processing
        combined_dim = 256 + 64 + 128 + 64 + 32
        self.combined_fc1 = nn.Linear(combined_dim, 512)
        self.combined_fc2 = nn.Linear(512, 256)
        
        # Value head
        self.value_head = nn.Linear(256, 1)
        
        # Hidden states
        self.stats_hidden = None
        
    def forward(self, obs, reset_hidden=False):
        batch_size = obs['glyphs'].size(0)
        
        # Process glyphs with recurrent CNN
        glyph_features = self.glyph_cnn(obs['glyphs'], reset_hidden)
        
        # Process stats with LSTM
        if reset_hidden or self.stats_hidden is None or self.stats_hidden[0].size(1) != batch_size:
            self.stats_hidden = (
                torch.zeros(1, batch_size, 64, device=obs['stats'].device),
                torch.zeros(1, batch_size, 64, device=obs['stats'].device)
            )
        
        stats_input = obs['stats'].unsqueeze(1)
        stats_lstm_out, self.stats_hidden = self.stats_lstm(stats_input, self.stats_hidden)
        stats_features = stats_lstm_out.squeeze(1)
        
        # Process other features
        message_features = F.relu(self.message_fc(obs['message']))
        inventory_features = F.relu(self.inventory_fc(obs['inventory']))
        action_hist_features = F.relu(self.action_hist_fc(obs['action_history']))
        
        # Combine all features
        combined = torch.cat([
            glyph_features, stats_features,
            message_features, inventory_features, action_hist_features
        ], dim=1)
        
        x = F.relu(self.combined_fc1(combined))
        x = F.relu(self.combined_fc2(x))
        
        value = self.value_head(x)
        return value
    
    def reset_hidden_states(self):
        """Reset all hidden states"""
        self.glyph_cnn.reset_hidden_state()
        self.stats_hidden = None

class PPOBuffer:
    """Enhanced PPO Buffer"""
    
    def __init__(self, max_size=2048):
        self.max_size = max_size
        self.clear()
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages(self, gamma=0.99, lam=0.95):
        """Compute GAE advantages"""
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]
            
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + gamma * lam * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batch(self, batch_size):
        """Get random batch for training"""
        indices = np.random.choice(len(self.observations), batch_size, replace=False)
        
        batch_obs = {}
        for key in self.observations[0].keys():
            batch_obs[key] = torch.stack([self.observations[i][key] for i in indices])
        
        batch_actions = torch.tensor([self.actions[i] for i in indices], dtype=torch.long)
        batch_log_probs = torch.tensor([self.log_probs[i] for i in indices], dtype=torch.float32)
        batch_returns = torch.tensor([self.returns[i] for i in indices], dtype=torch.float32)
        batch_advantages = torch.tensor([self.advantages[i] for i in indices], dtype=torch.float32)
        
        return batch_obs, batch_actions, batch_log_probs, batch_returns, batch_advantages
    
    def __len__(self):
        return len(self.observations)

class EnhancedNetHackPPOAgent:
    """Enhanced PPO Agent with reward shaping and recurrent memory"""
    
    def __init__(self, action_dim=23, learning_rate=3e-4, gamma=0.99, clip_ratio=0.2,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, use_wandb=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks with recurrent layers
        self.actor = RecurrentPPOActor(action_dim=action_dim).to(self.device)
        self.critic = RecurrentPPOCritic().to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.buffer = PPOBuffer()
        
        # Enhanced observation processor and reward shaper
        self.obs_processor = NetHackObservationProcessor()
        self.reward_shaper = NetHackRewardShaper()
        
        # Performance evaluator for metrics tracking
        self.evaluator = PerformanceEvaluator()
        
        # Training stats
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.shaped_rewards = deque(maxlen=100)
        
        # Track last action for observation processing
        self.last_action = None
        
    def process_observation(self, obs):
        """Process observation with memory features"""
        processed = self.obs_processor.process_observation(obs, self.last_action)
        
        # Convert to tensors
        tensor_obs = {}
        for key, value in processed.items():
            tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
        
        return tensor_obs
    
    def select_action(self, obs, reset_hidden=False):
        """Select action using current policy"""
        with torch.no_grad():
            action_logits = self.actor(obs, reset_hidden)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.critic(obs, reset_hidden)
            
            return action.item(), log_prob.item(), value.item()
    
    def update(self, epochs=4, batch_size=64):
        """Enhanced PPO update with entropy regularization and proper loss computation"""
        if len(self.buffer) < batch_size:
            return {}
        
        # Compute advantages
        self.buffer.compute_advantages(self.gamma)
        
        # Normalize advantages
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.tolist()
        
        # Training metrics
        actor_losses = []
        critic_losses = []
        entropies = []
        clip_fractions = []
        
        for _ in range(epochs):
            batch_obs, batch_actions, old_log_probs, batch_returns, batch_advantages = \
                self.buffer.get_batch(min(batch_size, len(self.buffer)))
            
            # Move tensors to correct device and ensure correct dtype
            batch_obs = {k: v.to(self.device) for k, v in batch_obs.items()}
            batch_actions = batch_actions.to(self.device)
            old_log_probs = old_log_probs.to(self.device)
            batch_returns = batch_returns.to(self.device)
            batch_advantages = batch_advantages.to(self.device)
            
            # Reset hidden states for batch training
            self.actor.reset_hidden_states()
            self.critic.reset_hidden_states()
            
            # Actor update with entropy regularization
            action_logits = self.actor(batch_obs, reset_hidden=True)
            action_dist = Categorical(logits=action_logits)
            new_log_probs = action_dist.log_prob(batch_actions)
            entropy = action_dist.entropy().mean()
            
            # PPO loss computation
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
            
            # Calculate clip fraction for monitoring
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_ratio).float()).item()
            
            # Actor loss with entropy regularization
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.entropy_coef * entropy
            actor_loss = policy_loss + entropy_loss
            
            # Actor optimization
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Critic update
            values = self.critic(batch_obs, reset_hidden=True).squeeze()
            value_loss = F.mse_loss(values, batch_returns)
            critic_loss = self.value_coef * value_loss
            
            # Critic optimization
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            # Store metrics
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
            clip_fractions.append(clip_fraction)
        
        # Return training metrics
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'policy_loss': np.mean([l - self.entropy_coef * e for l, e in zip(actor_losses, entropies)]),
            'value_loss': np.mean(critic_losses) / self.value_coef,
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions),
        }
    
    def train(self, env, num_episodes=100, update_freq=2048, eval_freq=25, 
              early_stopping_patience=50, target_reward=50):
        """Train the enhanced PPO agent with CSV logging and extensive analysis"""
        step_count = 0
        best_reward = float('-inf')
        recent_episodes = []  # Store recent episode data for extensive logging
        
        # Initialize training logger
        logger = TrainingLogger(log_dir="training_logs", use_wandb=False)
        logger.log_info("Starting Enhanced NetHack PPO Training with CSV logging")
        logger.log_info(f"Training Configuration: {num_episodes} episodes, update every {update_freq} steps")
        logger.log_info("📊 Logging Schedule:")
        logger.log_info("   - Every episode: Episode metrics saved to CSV")
        logger.log_info("   - Every 2048 steps: Training metrics logged")
        logger.log_info("   - Every 10 episodes: Extensive analysis with detailed metrics")
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_shaped_reward = 0
            episode_length = 0
            episode_died = 0
            episode_level_ups = 0
            episode_max_health = 0
            episode_items_collected = 0
            episode_unique_positions = 0
            episode_exploration_reward = 0
            
            # Reset hidden states at episode start
            self.actor.reset_hidden_states()
            self.critic.reset_hidden_states()
            self.reward_shaper.reset()
            self.last_action = None
            
            reset_hidden = True
            visited_positions = set()
            
            while True:
                processed_obs = self.process_observation(obs)
                action, log_prob, value = self.select_action(processed_obs, reset_hidden)
                reset_hidden = False  # Only reset on first step
                
                # Store action for next observation processing
                self.last_action = action
                
                # Convert tensors back for buffer
                processed_obs_for_buffer = {}
                for key, tensor_val in processed_obs.items():
                    processed_obs_for_buffer[key] = tensor_val.squeeze(0).cpu()
                
                step_result = env.step(action)
                
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                else:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                # Apply reward shaping
                shaped_reward = self.reward_shaper.shape_reward(next_obs, reward, done, info)
                
                self.buffer.add(processed_obs_for_buffer, action, shaped_reward, value, log_prob, done)
                
                # Extract episode metrics
                if isinstance(next_obs, tuple):
                    next_obs_dict = next_obs[0] if isinstance(next_obs[0], dict) else {}
                else:
                    next_obs_dict = next_obs if isinstance(next_obs, dict) else {}
                
                # Track episode stats
                if 'blstats' in next_obs_dict:
                    stats = next_obs_dict['blstats']
                    if len(stats) > 0:
                        episode_max_health = max(episode_max_health, int(stats[0]) if stats[0] > 0 else 0)
                    if len(stats) > 7:
                        current_level = int(stats[7])
                        if current_level > episode_level_ups:
                            episode_level_ups = current_level
                    
                    # Track unique positions
                    if len(stats) > 1:
                        pos = (int(stats[0]), int(stats[1]))
                        if pos not in visited_positions:
                            visited_positions.add(pos)
                            episode_exploration_reward += 0.01
                
                # Check if died
                if done and 'blstats' in next_obs_dict:
                    stats = next_obs_dict['blstats']
                    if len(stats) > 0 and stats[0] <= 0:
                        episode_died = 1
                
                obs = next_obs
                episode_reward += reward
                episode_shaped_reward += shaped_reward
                episode_length += 1
                step_count += 1
                
                if done:
                    break
                
                # Log every 2048 steps - Enhanced logging
                if step_count % update_freq == 0:
                    training_metrics = self.update()
                    if training_metrics:
                        # Log training metrics to CSV
                        training_metrics['learning_rate'] = self.actor_optimizer.param_groups[0]['lr']
                        training_metrics['grad_norm'] = training_metrics.get('grad_norm', 0)
                        logger.log_training_csv(step_count, training_metrics)
                        
                        # Enhanced step logging
                        logger.log_info(f"🔄 Step {step_count} Training Update:")
                        logger.log_info(f"   Episode: {episode}, Step in Episode: {episode_length}")
                        logger.log_info(f"   Current Episode Reward: {episode_reward:.3f}")
                        logger.log_info(f"   Actor Loss: {training_metrics['actor_loss']:.4f}")
                        logger.log_info(f"   Critic Loss: {training_metrics['critic_loss']:.4f}")
                        logger.log_info(f"   Entropy: {training_metrics['entropy']:.4f}")
                        logger.log_info(f"   Clip Fraction: {training_metrics['clip_fraction']:.4f}")
                        
                        # Add to evaluator for tracking
                        self.evaluator.add_training_metrics(training_metrics)
                    self.buffer.clear()
            
            # Calculate final episode metrics
            episode_unique_positions = len(visited_positions)
            survival_time = episode_length if not episode_died else 0
            
            # Prepare episode metrics for logging
            episode_metrics = {
                'raw_reward': episode_reward,
                'shaped_reward': episode_shaped_reward,
                'episode_length': episode_length,
                'died': episode_died,
                'level_ups': episode_level_ups,
                'max_health': episode_max_health,
                'items_collected': episode_items_collected,
                'unique_positions': episode_unique_positions,
                'exploration_reward': episode_exploration_reward,
                'survival_time': survival_time,
                'actions_taken': episode_length
            }
            
            # Log episode to CSV (every episode)
            logger.log_episode_csv(episode, episode_metrics)
            
            # Add to evaluator for comprehensive tracking
            self.evaluator.add_episode_metrics(episode_metrics)
            
            # Store episode data for extensive logging
            recent_episodes.append(episode_metrics)
            
            # Store for agent's internal tracking
            self.episode_rewards.append(episode_reward)
            self.shaped_rewards.append(episode_shaped_reward)
            self.episode_lengths.append(episode_length)
            
            # Track best reward for model saving
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_model_path = f"best_model_episode_{episode}_reward_{episode_reward:.3f}.pth"
                self.save_model(best_model_path)
                logger.log_info(f"🏆 New best model saved! Episode {episode}, Reward: {episode_reward:.3f}")
            
            # Enhanced episode logging (every episode)
            logger.log_info(f"📋 Episode {episode} Complete:")
            logger.log_info(f"   Raw Reward: {episode_reward:.3f}")
            logger.log_info(f"   Shaped Reward: {episode_shaped_reward:.3f}")
            logger.log_info(f"   Episode Length: {episode_length}")
            logger.log_info(f"   Survived: {'Yes' if not episode_died else 'No'}")
            logger.log_info(f"   Unique Positions: {episode_unique_positions}")
            logger.log_info(f"   Level Ups: {episode_level_ups}")
            
            # Extensive logging every 10 episodes
            if (episode + 1) % 10 == 0:
                # Get last 10 episodes for analysis
                last_10_episodes = recent_episodes[-10:] if len(recent_episodes) >= 10 else recent_episodes
                logger.log_extensive_analysis(episode + 1, last_10_episodes)
                
                # Clear recent episodes to avoid memory buildup
                if len(recent_episodes) > 20:
                    recent_episodes = recent_episodes[-10:]
        
        # Save final model
        final_model_path = f"final_model_100_episodes.pth"
        self.save_model(final_model_path)
        logger.log_info(f"🎯 Final model saved: {final_model_path}")
        
        # Save all metrics
        logger.save_metrics()
        
        return list(self.episode_rewards), list(self.shaped_rewards)
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {path}")

def create_nethack_env():
    """Create and configure NetHack environment"""
    import nle.env

    try:
        env = gym.make("NetHackScore-v0")
    except:
        env = gym.make("NetHack-v0")

    return env

def main():
    """Enhanced main training function with comprehensive CSV logging"""
    print("🚀 Setting up Enhanced NetHack PPO Training with CSV Logging...")

    # Create environment
    env = create_nethack_env()
    print(f"Environment action space: {env.action_space.n}")

    # Enhanced agent with better hyperparameters
    agent = EnhancedNetHackPPOAgent(
        action_dim=env.action_space.n,
        learning_rate=1e-4,  # Reduced learning rate
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.02,  # Increased entropy for better exploration
        value_coef=0.5,
        max_grad_norm=0.5,
        use_wandb=False  # Set to True if you have wandb installed
    )

    # Test observation processing
    obs = env.reset()
    try:
        processed_obs = agent.process_observation(obs)
        print("✅ Enhanced observation processing successful!")
        for key, value in processed_obs.items():
            print(f"  {key}: {value.shape}")
    except Exception as e:
        print(f"❌ Error in observation processing: {e}")
        return

    # Train agent for exactly 100 episodes with CSV logging
    print("\n🎯 Starting 100-episode training with CSV logging and extensive analysis...")
    print("📊 Logs will be saved as CSV files in 'training_logs/' directory")
    print("🔍 Extensive analysis every 10 episodes")
    print("🏆 Best model will be saved automatically")

    start_time = time.time()
    raw_rewards, shaped_rewards = agent.train(
        env,
        num_episodes=100,    # Exactly 100 episodes as requested
        update_freq=1024,    # More frequent updates
        eval_freq=25,        # More frequent evaluation
        early_stopping_patience=50,
        target_reward=50     # Achievable target
    )

    training_time = time.time() - start_time
    print(f"\n⏱️ Training completed in {training_time:.2f} seconds")
    print(f"📈 Final average reward: {np.mean(raw_rewards[-10:]) if len(raw_rewards) >= 10 else np.mean(raw_rewards):.3f}")
    print(f"🏆 Best episode reward: {max(raw_rewards) if raw_rewards else 0:.3f}")

    # Create simple training summary plot
    create_simple_training_plot(raw_rewards, shaped_rewards)

    env.close()
    print("🎉 Training completed successfully!")
    print("📁 Check 'training_logs/' directory for CSV files:")
    print("   - episodes.csv: Per-episode metrics")
    print("   - training.csv: Training step metrics") 
    print("   - extensive.csv: Detailed analysis every 10 episodes")

def create_simple_training_plot(raw_rewards, shaped_rewards):
    """Create simple training visualization"""
    
    if not raw_rewards:
        print("No rewards to plot")
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(15, 10))
    
    # 1. Rewards over time
    plt.subplot(2, 2, 1)
    episodes = range(len(raw_rewards))
    plt.plot(episodes, raw_rewards, label='Raw Rewards', alpha=0.7, color='blue')
    plt.plot(episodes, shaped_rewards, label='Shaped Rewards', alpha=0.7, color='orange')
    plt.title('Training Progress - Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Moving averages
    plt.subplot(2, 2, 2)
    if len(raw_rewards) >= 10:
        window = 10
        raw_ma = np.convolve(raw_rewards, np.ones(window)/window, mode='valid')
        shaped_ma = np.convolve(shaped_rewards, np.ones(window)/window, mode='valid')
        
        plt.plot(range(window-1, len(raw_rewards)), raw_ma, label='Raw MA-10', linewidth=2)
        plt.plot(range(window-1, len(shaped_rewards)), shaped_ma, label='Shaped MA-10', linewidth=2)
        plt.title('10-Episode Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Reward distribution
    plt.subplot(2, 2, 3)
    plt.hist(raw_rewards, bins=20, alpha=0.7, color='blue', label='Raw Rewards')
    plt.hist(shaped_rewards, bins=20, alpha=0.7, color='orange', label='Shaped Rewards')
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Training summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    final_avg_reward = np.mean(raw_rewards[-10:]) if len(raw_rewards) >= 10 else np.mean(raw_rewards)
    best_reward = max(raw_rewards)
    worst_reward = min(raw_rewards)
    
    summary_text = f"""
Training Summary (100 Episodes):

Total Episodes: {len(raw_rewards)}
Final 10-Episode Average: {final_avg_reward:.3f}
Best Episode Reward: {best_reward:.3f}
Worst Episode Reward: {worst_reward:.3f}
Reward Standard Deviation: {np.std(raw_rewards):.3f}

Shaped Rewards:
Average: {np.mean(shaped_rewards):.3f}
Best: {max(shaped_rewards):.3f}

CSV logs saved in training_logs/
Best model automatically saved
    """
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plot_filename = f'training_summary_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training summary plot saved as '{plot_filename}'")

if __name__ == "__main__":
    main()