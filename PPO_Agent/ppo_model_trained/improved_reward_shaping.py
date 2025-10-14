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
    """Comprehensive training logger with metrics tracking"""

    def __init__(self, log_dir: str = "logs", experiment_name: str = None, use_wandb: bool = False):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"nethack_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

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

    def log_episode(self, episode: int, metrics: Dict):
        """Log episode-level metrics"""
        self.logger.info(f"Episode {episode}: " +
                        " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()]))

        for key, value in metrics.items():
            self.episode_metrics[key].append(value)

        if self.use_wandb:
            wandb.log({f"episode/{k}": v for k, v in metrics.items()}, step=episode)

    def log_training(self, step: int, metrics: Dict):
        """Log training-level metrics"""
        self.logger.info(f"Training Step {step}: " +
                        " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

        for key, value in metrics.items():
            self.training_metrics[key].append(value)

        if self.use_wandb:
            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

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
    """Advanced reward shaping for NetHack with improved balance"""

    def __init__(self):
        self.previous_stats = None
        self.previous_glyphs = None
        self.visited_positions = set()
        self.last_position = None
        self.stuck_counter = 0
        self.max_stuck = 15  # Increased tolerance
        self.episode_start_time = None

        # Improved reward weights (more balanced)
        self.exploration_reward = 0.005  # Reduced from 0.01
        self.health_reward = 0.01  # Increased from 0.001
        self.level_reward = 5.0  # Increased from 1.0
        self.experience_reward = 0.001  # Increased from 0.0001
        self.death_penalty = -2.0  # Increased penalty
        self.stuck_penalty = -0.005  # Reduced from -0.01
        self.item_pickup_reward = 0.1  # Increased from 0.05
        self.monster_kill_reward = 0.5  # Increased from 0.1
        self.time_penalty = -0.0001  # Small time penalty to encourage efficiency
        self.progress_bonus = 0.1  # Bonus for making progress

        # Tracking variables for better reward calculation
        self.initial_stats = None
        self.best_stats = None
        self.episode_exploration_count = 0
        self.episode_item_pickups = 0
        self.episode_level_ups = 0
        self.last_experience = 0
        self.last_level = 0
        self.last_health = 0
        self.consecutive_stuck_steps = 0

    def shape_reward(self, obs, raw_reward, done, info):
        """Apply improved reward shaping based on game state"""
        shaped_reward = raw_reward * 0.1  # Scale down raw reward

        # Extract current stats
        if isinstance(obs, tuple):
            obs = obs[0]

        current_stats = obs.get('blstats', np.zeros(26))
        current_glyphs = obs.get('glyphs', np.zeros((21, 79)))

        # Initialize tracking on first call
        if self.initial_stats is None:
            self.initial_stats = current_stats.copy()
            self.best_stats = current_stats.copy()
            self.last_experience = current_stats[8] if len(current_stats) > 8 else 0
            self.last_level = current_stats[7] if len(current_stats) > 7 else 0
            self.last_health = current_stats[0] if len(current_stats) > 0 else 0
            self.episode_start_time = time.time()

        if self.previous_stats is not None:
            # Health management rewards
            health_diff = current_stats[0] - self.previous_stats[0]
            if health_diff > 0:
                shaped_reward += health_diff * self.health_reward * 2  # Bonus for healing
            elif health_diff < 0:
                shaped_reward += health_diff * self.health_reward * 0.5  # Smaller penalty for damage

            # Level progression (major milestone)
            level_diff = current_stats[7] - self.previous_stats[7]
            if level_diff > 0:
                shaped_reward += level_diff * self.level_reward
                self.episode_level_ups += level_diff
                self.best_stats[7] = max(self.best_stats[7], current_stats[7])

            # Experience progression
            exp_diff = current_stats[8] - self.previous_stats[8]
            if exp_diff > 0:
                shaped_reward += exp_diff * self.experience_reward
                # Bonus for consistent experience gain
                if exp_diff > self.last_experience:
                    shaped_reward += self.progress_bonus

            # Improved item detection
            current_inv_count = np.sum(current_glyphs > 0)
            prev_inv_count = np.sum(self.previous_glyphs > 0)
            inv_change = current_inv_count - prev_inv_count

            if inv_change > 0:
                shaped_reward += inv_change * self.item_pickup_reward
                self.episode_item_pickups += inv_change

            # Monster killing detection (experience spikes)
            if exp_diff > 10:  # Likely killed a monster
                shaped_reward += self.monster_kill_reward

        # Exploration rewards with diminishing returns
        current_pos = tuple(current_stats[:2]) if len(current_stats) > 1 else (0, 0)
        if current_pos not in self.visited_positions:
            self.visited_positions.add(current_pos)
            self.episode_exploration_count += 1
            # Diminishing returns for exploration
            exploration_bonus = self.exploration_reward * (1.0 / (1.0 + self.episode_exploration_count * 0.01))
            shaped_reward += exploration_bonus

        # Improved anti-stuck mechanism
        if current_pos == self.last_position:
            self.consecutive_stuck_steps += 1
            if self.consecutive_stuck_steps > self.max_stuck:
                # Progressive penalty for being stuck
                stuck_multiplier = min(self.consecutive_stuck_steps / self.max_stuck, 5.0)
                shaped_reward += self.stuck_penalty * stuck_multiplier
        else:
            self.consecutive_stuck_steps = 0

        # Time-based penalty (encourage efficiency)
        if self.episode_start_time:
            episode_time = time.time() - self.episode_start_time
            shaped_reward += self.time_penalty * episode_time

        # Progress bonus (reward for improvement over initial state)
        if len(current_stats) > 8:
            progress_score = (
                (current_stats[7] - self.initial_stats[7]) * 10 +  # Level progress
                (current_stats[8] - self.initial_stats[8]) * 0.01 +  # XP progress
                len(self.visited_positions) * 0.01  # Exploration progress
            )
            shaped_reward += progress_score * 0.001

        # Death penalty with context
        if done:
            if len(current_stats) > 0 and current_stats[0] <= 0:  # Player died
                shaped_reward += self.death_penalty
                # Additional penalty for early death
                if len(self.visited_positions) < 10:
                    shaped_reward += self.death_penalty * 0.5
            else:
                # Small bonus for surviving (timeout/win)
                shaped_reward += 0.1

        # Update tracking variables
        self.previous_stats = current_stats.copy()
        self.previous_glyphs = current_glyphs.copy()
        self.last_position = current_pos
        self.last_experience = current_stats[8] if len(current_stats) > 8 else 0

        # Clamp shaped reward to prevent extreme values
        shaped_reward = np.clip(shaped_reward, -10.0, 10.0)

        return shaped_reward

    def get_episode_stats(self):
        """Get episode statistics for logging"""
        return {
            'exploration_count': self.episode_exploration_count,
            'unique_positions': len(self.visited_positions),
            'item_pickups': self.episode_item_pickups,
            'level_ups': self.episode_level_ups,
            'max_health': self.best_stats[0] if self.best_stats is not None else 0,
            'max_level': self.best_stats[7] if self.best_stats is not None else 0,
            'max_experience': self.best_stats[8] if self.best_stats is not None else 0,
        }

    def reset(self):
        """Reset reward shaper for new episode"""
        self.previous_stats = None
        self.previous_glyphs = None
        self.visited_positions.clear()
        self.last_position = None
        self.stuck_counter = 0
        self.consecutive_stuck_steps = 0
        self.initial_stats = None
        self.best_stats = None
        self.episode_exploration_count = 0
        self.episode_item_pickups = 0
        self.episode_level_ups = 0
        self.last_experience = 0
        self.last_level = 0
        self.last_health = 0
        self.episode_start_time = None

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
                    item_str = str(item) if item is not None else ""
                    if len(item_str.strip()) > 0 and item_str.strip() != "b''":
                        inv_features[i] = 1.0
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
        if len(self.observations) >= self.max_size:
            # Remove oldest entries if buffer is full
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.dones.pop(0)

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
    """Enhanced PPO Agent with comprehensive improvements"""

    def __init__(self, action_dim=23, learning_rate=3e-4, gamma=0.99, clip_ratio=0.2,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, use_wandb=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize networks with recurrent layers
        self.actor = RecurrentPPOActor(action_dim=action_dim).to(self.device)
        self.critic = RecurrentPPOCritic().to(self.device)

        # Improved optimizers with weight decay
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate,
                                        weight_decay=1e-5, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate,
                                         weight_decay=1e-5, eps=1e-5)

        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer,
                                                       step_size=1000, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                        step_size=1000, gamma=0.95)

        # Hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Enhanced buffer
        self.buffer = PPOBuffer()

        # Enhanced observation processor and reward shaper
        self.obs_processor = NetHackObservationProcessor()
        self.reward_shaper = NetHackRewardShaper()

        # Performance evaluator and logger
        self.evaluator = PerformanceEvaluator()
        self.logger = TrainingLogger(use_wandb=use_wandb)

        # Track last action for observation processing
        self.last_action = None

        # Training statistics
        self.total_steps = 0
        self.update_count = 0
        self.best_reward = float('-inf')
        self.episodes_since_improvement = 0

        # Adaptive parameters
        self.adaptive_entropy = True
        self.min_entropy_coef = 0.001
        self.max_entropy_coef = 0.05

    def process_observation(self, obs):
        """Process observation with memory features"""
        processed = self.obs_processor.process_observation(obs, self.last_action)

        # Convert to tensors
        tensor_obs = {}
        for key, value in processed.items():
            tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)

        return tensor_obs

    def select_action(self, obs, reset_hidden=False):
        """Select action using current policy with entropy tracking"""
        with torch.no_grad():
            action_logits = self.actor(obs, reset_hidden)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            value = self.critic(obs, reset_hidden)

            return action.item(), log_prob.item(), value.item(), entropy.item()

    def update(self, epochs=4, batch_size=64):
        """Enhanced PPO update with comprehensive metrics"""
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
        grad_norms = []

        for epoch in range(epochs):
            batch_obs, batch_actions, old_log_probs, batch_returns, batch_advantages = \
                self.buffer.get_batch(min(batch_size, len(self.buffer)))

            # Move tensors to correct device
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
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Critic update
            values = self.critic(batch_obs, reset_hidden=True).squeeze()
            value_loss = F.mse_loss(values, batch_returns)
            critic_loss = self.value_coef * value_loss

            # Critic optimization
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # Store metrics
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
            clip_fractions.append(clip_fraction)
            grad_norms.append(max(actor_grad_norm.item(), critic_grad_norm.item()))

        # Update learning rate schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # Adaptive entropy coefficient
        if self.adaptive_entropy:
            mean_entropy = np.mean(entropies)
            if mean_entropy < 0.5:  # Too low entropy
                self.entropy_coef = min(self.entropy_coef * 1.01, self.max_entropy_coef)
            elif mean_entropy > 2.0:  # Too high entropy
                self.entropy_coef = max(self.entropy_coef * 0.99, self.min_entropy_coef)

        self.update_count += 1

        # Return training metrics
        training_metrics = {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'policy_loss': np.mean([l - self.entropy_coef * e for l, e in zip(actor_losses, entropies)]),
            'value_loss': np.mean(critic_losses) / self.value_coef,
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions),
            'grad_norm': np.mean(grad_norms),
            'learning_rate': self.actor_optimizer.param_groups[0]['lr'],
            'entropy_coef': self.entropy_coef,
        }

        # Log training metrics
        self.evaluator.add_training_metrics(training_metrics)
        self.logger.log_training(self.total_steps, training_metrics)

        return training_metrics

    def train(self, env, num_episodes=1000, update_freq=2048, eval_freq=50,
              early_stopping_patience=100, target_reward=100):
        """Enhanced training loop with comprehensive monitoring"""
        step_count = 0
        best_avg_reward = float('-inf')
        patience_counter = 0

        self.logger.log_info(f"Starting training for {num_episodes} episodes")
        self.logger.log_info(f"Target reward: {target_reward}, Early stopping patience: {early_stopping_patience}")

        for episode in range(num_episodes):
            episode_start_time = time.time()
            obs = env.reset()
            episode_reward = 0
            episode_shaped_reward = 0
            episode_length = 0
            episode_entropy = 0

            # Reset hidden states at episode start
            self.actor.reset_hidden_states()
            self.critic.reset_hidden_states()
            self.reward_shaper.reset()
            self.last_action = None

            reset_hidden = True

            while True:
                processed_obs = self.process_observation(obs)
                action, log_prob, value, entropy = self.select_action(processed_obs, reset_hidden)
                reset_hidden = False

                # Store action for next observation processing
                self.last_action = action
                episode_entropy += entropy

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

                obs = next_obs
                episode_reward += reward
                episode_shaped_reward += shaped_reward
                episode_length += 1
                step_count += 1
                self.total_steps += 1

                if done:
                    break

                # Update networks periodically
                if step_count % update_freq == 0:
                    self.logger.log_info(f"Updating networks at step {step_count}")
                    training_metrics = self.update()
                    self.buffer.clear()

            # Episode completed - collect comprehensive metrics
            episode_time = time.time() - episode_start_time
            episode_stats = self.reward_shaper.get_episode_stats()

            # Determine if player died
            final_obs = obs if isinstance(obs, dict) else obs[0]
            final_stats = final_obs.get('blstats', np.zeros(26))
            died = len(final_stats) > 0 and final_stats[0] <= 0

            episode_data = {
                'raw_reward': episode_reward,
                'shaped_reward': episode_shaped_reward,
                'length': episode_length,
                'died': 1 if died else 0,
                'exploration_reward': episode_stats.get('exploration_count', 0) * 0.005,
                'level_ups': episode_stats.get('level_ups', 0),
                'max_health': episode_stats.get('max_health', 0),
                'items_collected': episode_stats.get('item_pickups', 0),
                'unique_positions': episode_stats.get('unique_positions', 0),
                'episode_time': episode_time,
                'avg_entropy': episode_entropy / max(episode_length, 1),
                'final_level': final_stats[7] if len(final_stats) > 7 else 0,
                'final_experience': final_stats[8] if len(final_stats) > 8 else 0,
            }

            # Add to evaluator
            self.evaluator.add_episode_metrics(episode_data)

            # Log episode metrics
            if episode % 10 == 0 or episode < 10:
                episode_metrics = self.evaluator.get_episode_metrics()
                training_metrics = self.evaluator.get_training_metrics()

                # Combine all metrics
                all_metrics = {**episode_data, **episode_metrics}
                self.logger.log_episode(episode, all_metrics)

                # Check for improvement
                current_avg_reward = episode_metrics.get('mean_reward', float('-inf'))
                if current_avg_reward > best_avg_reward:
                    best_avg_reward = current_avg_reward
                    patience_counter = 0
                    self.best_reward = current_avg_reward

                    # Save best model
                    self.save_model(f"best_model_episode_{episode}_reward_{current_avg_reward:.2f}.pth")
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    self.logger.log_info(f"Early stopping triggered after {patience_counter} episodes without improvement")
                    break

                # Success check
                if current_avg_reward >= target_reward:
                    self.logger.log_info(f"Target reward {target_reward} achieved!")
                    break

            # Detailed evaluation every eval_freq episodes
            if episode % eval_freq == 0 and episode > 0:
                self.logger.log_info(f"\n=== Detailed Evaluation at Episode {episode} ===")
                episode_metrics = self.evaluator.get_episode_metrics()
                training_metrics = self.evaluator.get_training_metrics()

                # Log comprehensive metrics
                for metric_name, value in episode_metrics.items():
                    self.logger.log_info(f"Episode {metric_name}: {value:.4f}")

                for metric_name, value in training_metrics.items():
                    self.logger.log_info(f"Training {metric_name}: {value:.4f}")

                # Performance analysis
                self._analyze_performance(episode, episode_metrics, training_metrics)

        # Final evaluation and cleanup
        self.logger.log_info("Training completed!")
        final_metrics = self.evaluator.get_episode_metrics()
        self.logger.log_info(f"Final average reward: {final_metrics.get('mean_reward', 0):.3f}")
        self.logger.log_info(f"Best average reward: {self.best_reward:.3f}")

        # Save final metrics
        self.logger.save_metrics()

        return self.evaluator.episode_rewards, self.evaluator.shaped_rewards

    def _analyze_performance(self, episode: int, episode_metrics: Dict, training_metrics: Dict):
        """Analyze performance and provide insights"""

        # Performance trend analysis
        reward_trend = episode_metrics.get('reward_trend', 0)
        if reward_trend > 0:
            self.logger.log_info(f"✓ Reward trend is improving (+{reward_trend:.4f})")
        else:
            self.logger.log_info(f"✗ Reward trend is declining ({reward_trend:.4f})")

        # Learning stability
        entropy_trend = training_metrics.get('entropy_trend', 0)
        if entropy_trend < -0.001:
            self.logger.log_info("⚠ Entropy decreasing rapidly - policy may be becoming too deterministic")
        elif entropy_trend > 0.001:
            self.logger.log_info("⚠ Entropy increasing - policy may not be converging")

        # Exploration analysis
        exploration_eff = episode_metrics.get('exploration_efficiency', 0)
        if exploration_eff < 0.01:
            self.logger.log_info("⚠ Low exploration efficiency - agent may be stuck")

        # Success metrics
        success_rate = episode_metrics.get('success_rate', 0)
        survival_rate = episode_metrics.get('survival_rate', 0)

        self.logger.log_info(f"Success rate: {success_rate:.2%}, Survival rate: {survival_rate:.2%}")

        # Training stability
        loss_stability = training_metrics.get('loss_stability', 0)
        if loss_stability < 0.5:
            self.logger.log_info("⚠ Training losses are unstable")

        # Adaptive recommendations
        mean_entropy = training_metrics.get('mean_entropy', 0)
        if mean_entropy < 0.1:
            self.logger.log_info("💡 Consider increasing entropy coefficient")
        elif mean_entropy > 2.0:
            self.logger.log_info("💡 Consider decreasing entropy coefficient")

        clip_fraction = training_metrics.get('mean_clip_fraction', 0)
        if clip_fraction > 0.3:
            self.logger.log_info("💡 High clipping - consider decreasing learning rate")
        elif clip_fraction < 0.05:
            self.logger.log_info("💡 Low clipping - consider increasing learning rate")

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
    """Enhanced main training function with comprehensive analysis"""
    print("🚀 Setting up Enhanced NetHack PPO Training...")

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

    # Train agent with enhanced monitoring
    print("\n🎯 Starting enhanced training with comprehensive monitoring...")

    start_time = time.time()
    raw_rewards, shaped_rewards = agent.train(
        env,
        num_episodes=100,  # Reduced for better monitoring
        update_freq=1024,  # More frequent updates
        eval_freq=25,      # More frequent evaluation
        early_stopping_patience=50,
        target_reward=50   # Achievable target
    )

    training_time = time.time() - start_time
    print(f"\n⏱️ Training completed in {training_time:.2f} seconds")

    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"enhanced_nethack_ppo_{timestamp}.pth"
    agent.save_model(model_path)

    # Generate comprehensive plots
    create_comprehensive_plots(agent, raw_rewards, shaped_rewards, timestamp)

    # Generate training report
    generate_training_report(agent, training_time, timestamp)

    env.close()
    print("🎉 Enhanced training completed successfully!")

def create_comprehensive_plots(agent, raw_rewards, shaped_rewards, timestamp):
    """Create comprehensive training visualization"""

    # Convert deque to list for proper indexing
    raw_rewards = list(raw_rewards) if raw_rewards else []
    shaped_rewards = list(shaped_rewards) if shaped_rewards else []

    # Get metrics from evaluator
    episode_metrics = agent.evaluator.get_episode_metrics()
    training_metrics = agent.evaluator.get_training_metrics()

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # 1. Rewards comparison
    ax1 = plt.subplot(3, 3, 1)
    episodes = range(len(raw_rewards))
    plt.plot(episodes, raw_rewards, label='Raw Rewards', alpha=0.7, color='blue')
    plt.plot(episodes, shaped_rewards, label='Shaped Rewards', alpha=0.7, color='orange')
    plt.title('Training Progress - Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Moving averages
    ax2 = plt.subplot(3, 3, 2)
    if len(raw_rewards) >= 20:
        window = min(20, len(raw_rewards) // 4)
        raw_ma = np.convolve(raw_rewards, np.ones(window)/window, mode='valid')
        shaped_ma = np.convolve(shaped_rewards, np.ones(window)/window, mode='valid')

        plt.plot(range(window-1, len(raw_rewards)), raw_ma, label=f'Raw MA-{window}', linewidth=2)
        plt.plot(range(window-1, len(shaped_rewards)), shaped_ma, label=f'Shaped MA-{window}', linewidth=2)
        plt.title('Moving Average Progress')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 3. Episode lengths
    ax3 = plt.subplot(3, 3, 3)
    if agent.evaluator.episode_lengths:
        episode_lengths = list(agent.evaluator.episode_lengths)
        plt.plot(episode_lengths, color='green', alpha=0.7)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True, alpha=0.3)

    # 4. Training losses
    ax4 = plt.subplot(3, 3, 4)
    if agent.evaluator.actor_losses:
        plt.plot(list(agent.evaluator.actor_losses), label='Actor Loss', alpha=0.7)
        plt.plot(list(agent.evaluator.critic_losses), label='Critic Loss', alpha=0.7)
        plt.title('Training Losses')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

    # 5. Entropy over time
    ax5 = plt.subplot(3, 3, 5)
    if agent.evaluator.entropies:
        plt.plot(list(agent.evaluator.entropies), color='purple', alpha=0.7)
        plt.title('Policy Entropy')
        plt.xlabel('Update Step')
        plt.ylabel('Entropy')
        plt.grid(True, alpha=0.3)

    # 6. Gradient norms
    ax6 = plt.subplot(3, 3, 6)
    if agent.evaluator.grad_norms:
        plt.plot(list(agent.evaluator.grad_norms), color='red', alpha=0.7)
        plt.title('Gradient Norms')
        plt.xlabel('Update Step')
        plt.ylabel('Grad Norm')
        plt.grid(True, alpha=0.3)

    # 7. Exploration metrics
    ax7 = plt.subplot(3, 3, 7)
    if agent.evaluator.unique_positions:
        plt.plot(list(agent.evaluator.unique_positions), color='brown', alpha=0.7)
        plt.title('Unique Positions Visited')
        plt.xlabel('Episode')
        plt.ylabel('Positions')
        plt.grid(True, alpha=0.3)

    # 8. Success metrics
    ax8 = plt.subplot(3, 3, 8)
    if agent.evaluator.death_counts:
        survival_rate = [1 - d for d in agent.evaluator.death_counts]
        plt.plot(survival_rate, color='darkgreen', alpha=0.7)
        plt.title('Survival Rate')
        plt.xlabel('Episode')
        plt.ylabel('Survival Rate')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

    # 9. Performance summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Summary statistics
    final_avg_reward = np.mean(raw_rewards[-10:]) if len(raw_rewards) >= 10 else (np.mean(raw_rewards) if raw_rewards else 0)
    best_reward = max(raw_rewards) if raw_rewards else 0
    
    summary_text = f"""
    Training Summary:

    Episodes: {len(raw_rewards)}
    Final Avg Reward: {final_avg_reward:.2f}
    Best Reward: {best_reward:.2f}

    Final Metrics:
    Success Rate: {episode_metrics.get('success_rate', 0):.2%}
    Survival Rate: {episode_metrics.get('survival_rate', 0):.2%}
    Avg Length: {episode_metrics.get('mean_length', 0):.1f}

    Training Metrics:
    Avg Entropy: {training_metrics.get('mean_entropy', 0):.3f}
    Avg Clip Frac: {training_metrics.get('mean_clip_fraction', 0):.3f}
    Final LR: {training_metrics.get('mean_learning_rate', 0):.6f}
    """

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'comprehensive_training_analysis_{timestamp}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"📊 Comprehensive training plots saved as 'comprehensive_training_analysis_{timestamp}.png'")

def generate_training_report(agent, training_time, timestamp):
    """Generate detailed training report"""

    episode_metrics = agent.evaluator.get_episode_metrics()
    training_metrics = agent.evaluator.get_training_metrics()

    report = f"""
# NetHack PPO Training Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Training Time: {training_time:.2f} seconds

## Training Configuration
- Episodes: {len(list(agent.evaluator.episode_rewards))}
- Update Frequency: 1024 steps
- Learning Rate: {agent.actor_optimizer.param_groups[0]['lr']:.6f}
- Entropy Coefficient: {agent.entropy_coef:.4f}
- Clip Ratio: {agent.clip_ratio}

## Performance Summary
- **Final Average Reward**: {episode_metrics.get('mean_reward', 0):.3f}
- **Best Episode Reward**: {episode_metrics.get('max_reward', 0):.3f}
- **Success Rate**: {episode_metrics.get('success_rate', 0):.2%}
- **Survival Rate**: {episode_metrics.get('survival_rate', 0):.2%}
- **Average Episode Length**: {episode_metrics.get('mean_length', 0):.1f} steps

## Learning Progress
- **Reward Trend**: {episode_metrics.get('reward_trend', 0):.6f}
- **Exploration Efficiency**: {episode_metrics.get('exploration_efficiency', 0):.4f}
- **Reward Stability**: {episode_metrics.get('reward_stability', 0):.3f}

## Training Stability
- **Average Actor Loss**: {training_metrics.get('mean_actor_loss', 0):.4f}
- **Average Critic Loss**: {training_metrics.get('mean_critic_loss', 0):.4f}
- **Average Entropy**: {training_metrics.get('mean_entropy', 0):.4f}
- **Average Gradient Norm**: {training_metrics.get('mean_grad_norm', 0):.4f}
- **Average Clip Fraction**: {training_metrics.get('mean_clip_fraction', 0):.4f}

## Recommendations
"""

    # Add recommendations based on metrics
    if episode_metrics.get('reward_trend', 0) > 0:
        report += "✅ **Positive Learning**: Reward trend is improving\n"
    else:
        report += "⚠️ **Learning Issues**: Reward trend is declining - consider adjusting hyperparameters\n"

    if training_metrics.get('mean_entropy', 0) < 0.1:
        report += "⚠️ **Low Entropy**: Policy may be too deterministic - consider increasing entropy coefficient\n"
    elif training_metrics.get('mean_entropy', 0) > 2.0:
        report += "⚠️ **High Entropy**: Policy may not be converging - consider decreasing entropy coefficient\n"
    else:
        report += "✅ **Good Entropy**: Policy exploration/exploitation balance looks healthy\n"

    if training_metrics.get('mean_clip_fraction', 0) > 0.3:
        report += "⚠️ **High Clipping**: Consider reducing learning rate\n"
    elif training_metrics.get('mean_clip_fraction', 0) < 0.05:
        report += "💡 **Low Clipping**: Could potentially increase learning rate\n"
    else:
        report += "✅ **Good Clipping**: Learning rate appears appropriate\n"

    # Save report
    report_file = f'training_report_{timestamp}.md'
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"📝 Training report saved as '{report_file}'")
    print(report)

if __name__ == "__main__":
    main()