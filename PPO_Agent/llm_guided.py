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
    
    def __init__(self, action_dim=23, learning_rate=3e-4, gamma=0.99, clip_ratio=0.2):
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
        self.buffer = PPOBuffer()
        
        # Enhanced observation processor and reward shaper
        self.obs_processor = NetHackObservationProcessor()
        self.reward_shaper = NetHackRewardShaper()
        
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
        """Update actor and critic networks"""
        if len(self.buffer) < batch_size:
            return
        
        # Compute advantages
        self.buffer.compute_advantages(self.gamma)
        
        # Normalize advantages
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.tolist()
        
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
            
            # Actor update
            action_logits = self.actor(batch_obs, reset_hidden=True)
            action_dist = Categorical(logits=action_logits)
            new_log_probs = action_dist.log_prob(batch_actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Critic update
            values = self.critic(batch_obs, reset_hidden=True).squeeze()
            critic_loss = F.mse_loss(values, batch_returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
    
    def train(self, env, num_episodes=1000, update_freq=2048):
        """Train the enhanced PPO agent"""
        step_count = 0
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_shaped_reward = 0
            episode_length = 0
            
            # Reset hidden states at episode start
            self.actor.reset_hidden_states()
            self.critic.reset_hidden_states()
            self.reward_shaper.reset()
            self.last_action = None
            
            reset_hidden = True
            
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
                
                obs = next_obs
                episode_reward += reward
                episode_shaped_reward += shaped_reward
                episode_length += 1
                step_count += 1
                
                if done:
                    break
                
                # Update networks periodically
                if step_count % update_freq == 0:
                    print(f"Updating networks at step {step_count}")
                    self.update()
                    self.buffer.clear()
            
            self.episode_rewards.append(episode_reward)
            self.shaped_rewards.append(episode_shaped_reward)
            self.episode_lengths.append(episode_length)
            
            if episode % 10 == 0:
                avg_reward = np.mean(list(self.episode_rewards))
                avg_shaped_reward = np.mean(list(self.shaped_rewards))
                avg_length = np.mean(list(self.episode_lengths))
                print(f"Episode {episode}: Raw Reward: {avg_reward:.3f}, "
                      f"Shaped Reward: {avg_shaped_reward:.3f}, Length: {avg_length:.1f}")
        
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
import re

# Import your existing enhanced agent components
# [The existing classes would be imported here - NetHackRewardShaper, NetHackObservationProcessor, etc.]
# For brevity, I'm showing the new LLM components and integration

class NetHackSemanticDescriptor:
    """Converts NetHack game state into natural language descriptions"""
    
    def __init__(self):
        # NetHack glyph mappings (simplified - you'd expand this)
        self.glyph_to_symbol = {
            # Terrain
            2359: "wall", 2360: "door", 2361: "floor", 2362: "corridor",
            2363: "room", 2364: "stairs_down", 2365: "stairs_up",
            # Monsters  
            2378: "kobold", 2379: "goblin", 2380: "orc", 2381: "troll",
            2382: "dragon", 2383: "giant", 2384: "demon",
            # Items
            2395: "gold", 2396: "weapon", 2397: "armor", 2398: "food",
            2399: "potion", 2400: "scroll", 2401: "wand", 2402: "ring",
            # Player
            2413: "player"
        }
        
        # Action mappings
        self.action_meanings = {
            0: "move_north", 1: "move_south", 2: "move_east", 3: "move_west",
            4: "move_northeast", 5: "move_northwest", 6: "move_southeast", 7: "move_southwest",
            8: "wait", 9: "pickup", 10: "drop", 11: "search", 12: "open_door",
            13: "close_door", 14: "kick", 15: "eat", 16: "drink", 17: "read",
            18: "apply", 19: "throw", 20: "wear", 21: "take_off", 22: "wield"
        }
        
        # Game messages decoder (simplified)
        self.message_patterns = {
            b"You hit": "combat_success",
            b"You miss": "combat_miss", 
            b"You are hit": "taking_damage",
            b"You feel": "status_change",
            b"You see": "observation",
            b"The door": "door_interaction",
            b"You pick up": "item_pickup",
            b"You drop": "item_drop"
        }
    
    def describe_surroundings(self, glyphs, player_pos):
        """Describe the immediate area around the player"""
        h, w = glyphs.shape
        py, px = player_pos
        
        # Look in 3x3 area around player
        nearby_items = []
        nearby_monsters = []
        terrain_features = []
        
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w:
                    glyph = glyphs[ny, nx]
                    symbol = self.glyph_to_symbol.get(glyph, "unknown")
                    
                    if "monster" in symbol or symbol in ["kobold", "goblin", "orc", "troll", "dragon"]:
                        direction = self._get_direction(dy, dx)
                        nearby_monsters.append(f"{symbol} {direction}")
                    elif symbol in ["gold", "weapon", "armor", "food", "potion", "scroll"]:
                        direction = self._get_direction(dy, dx)
                        nearby_items.append(f"{symbol} {direction}")
                    elif symbol in ["door", "stairs_up", "stairs_down"]:
                        direction = self._get_direction(dy, dx)
                        terrain_features.append(f"{symbol} {direction}")
        
        description = []
        if terrain_features:
            description.append(f"Terrain: {', '.join(terrain_features)}")
        if nearby_monsters:
            description.append(f"Threats: {', '.join(nearby_monsters)}")
        if nearby_items:
            description.append(f"Items: {', '.join(nearby_items)}")
        
        return "; ".join(description) if description else "Empty area"
    
    def _get_direction(self, dy, dx):
        """Convert relative position to direction"""
        if dy == 0 and dx == 0:
            return "here"
        elif dy == -1 and dx == 0:
            return "north"
        elif dy == 1 and dx == 0:
            return "south"
        elif dy == 0 and dx == -1:
            return "west"
        elif dy == 0 and dx == 1:
            return "east"
        elif dy == -1 and dx == -1:
            return "northwest"
        elif dy == -1 and dx == 1:
            return "northeast"
        elif dy == 1 and dx == -1:
            return "southwest"
        elif dy == 1 and dx == 1:
            return "southeast"
        else:
            return "nearby"
    
    def describe_player_status(self, stats):
        """Describe player's current status"""
        if len(stats) < 10:
            return "Status unknown"
        
        hp = int(stats[0])
        max_hp = int(stats[1])
        level = int(stats[7]) if len(stats) > 7 else 1
        experience = int(stats[8]) if len(stats) > 8 else 0
        
        hp_ratio = hp / max_hp if max_hp > 0 else 0
        health_status = "critical" if hp_ratio < 0.3 else "low" if hp_ratio < 0.6 else "good"
        
        return f"Level {level}, Health: {hp}/{max_hp} ({health_status}), Experience: {experience}"
    
    def describe_recent_message(self, message):
        """Interpret recent game message"""
        if len(message) == 0:
            return "No recent messages"
        
        # Convert message to string
        message_str = bytes(message).decode('ascii', errors='ignore').strip()
        if not message_str:
            return "No recent messages"
        
        # Pattern matching
        for pattern, meaning in self.message_patterns.items():
            if pattern in message_str.encode():
                return f"{meaning}: {message_str}"
        
        return f"Message: {message_str}"
    
    def describe_inventory(self, inventory_features):
        """Describe inventory status"""
        item_count = int(np.sum(inventory_features))
        if item_count == 0:
            return "Inventory empty"
        elif item_count < 5:
            return f"Carrying {item_count} items (light load)"
        elif item_count < 15:
            return f"Carrying {item_count} items (moderate load)"
        else:
            return f"Carrying {item_count} items (heavy load)"
    
    def describe_recent_actions(self, action_history):
        """Describe recent action pattern"""
        if len(action_history) == 0:
            return "No recent actions"
        
        # Get last few non-zero actions
        recent_actions = [int(a * 23) for a in action_history if a > 0][-5:]
        if not recent_actions:
            return "No recent actions"
        
        action_names = [self.action_meanings.get(a, f"action_{a}") for a in recent_actions]
        return f"Recent actions: {' → '.join(action_names)}"
    
    def generate_full_description(self, obs, processed_obs):
        """Generate complete semantic description of game state"""
        if isinstance(obs, tuple):
            obs = obs[0]
        
        glyphs = obs.get('glyphs', np.zeros((21, 79)))
        stats = obs.get('blstats', np.zeros(26))
        message = obs.get('message', np.zeros(256))
        
        # Find player position (simplified - assumes player is at center)
        player_pos = (10, 39)  # Center of 21x79 view
        
        # Generate description components
        surroundings = self.describe_surroundings(glyphs, player_pos)
        status = self.describe_player_status(stats)
        recent_msg = self.describe_recent_message(message)
        inventory = self.describe_inventory(processed_obs['inventory'])
        actions = self.describe_recent_actions(processed_obs['action_history'])
        
        # Combine into full description
        description = f"""
NETHACK GAME STATE:
Status: {status}
Surroundings: {surroundings}
Recent Message: {recent_msg}
Inventory: {inventory}
Recent Actions: {actions}

Current Situation: You are exploring a dungeon. Your goal is to survive, gain experience, collect items, and progress deeper.
        """.strip()
        
        return description

class LLMStrategicAdvisor:
    """Provides strategic advice using LLM API calls"""
    
    def __init__(self, call_frequency=10):
        self.call_frequency = call_frequency  # Call LLM every N steps
        self.step_count = 0
        self.last_advice = None
        self.advice_history = deque(maxlen=5)
        
    def should_call_llm(self):
        """Determine if we should call the LLM for advice"""
        self.step_count += 1
        return self.step_count % self.call_frequency == 0
    
    async def get_strategic_advice(self, semantic_description, recent_performance):
        """Get strategic advice from Ollama Phi model with robust parsing"""
        try:
            # Simplified prompt optimized for Phi model's context window
            prompt = f"""You are a NetHack expert. Analyze this game state and provide strategic advice.

    GAME STATE:
    {semantic_description}

    PERFORMANCE:
    Avg reward: {recent_performance.get('avg_reward', 0):.1f}
    Survival: {recent_performance.get('avg_length', 0):.0f} steps

    CRITICAL: Respond ONLY with valid JSON. No text before or after. No explanations.

    {{
        "immediate_priority": "what to do now",
        "risk_assessment": "main danger",
        "opportunities": "beneficial action",
        "strategy": "overall approach", 
        "action_suggestions": ["action1", "action2", "action3"]
    }}"""
            
            # Call Ollama API
            response = await self._call_llm_api(prompt)
            
            # Clean and parse response
            response = response.strip()
            
            # Remove markdown code blocks
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()
            
            # Try to find JSON object in response
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            try:
                advice = json.loads(response)
                
                # Validate structure
                required_keys = ['immediate_priority', 'risk_assessment', 
                            'opportunities', 'strategy', 'action_suggestions']
                
                for key in required_keys:
                    if key not in advice:
                        advice[key] = "unknown" if key != 'action_suggestions' else []
                
                # Ensure action_suggestions is a list of strings
                if not isinstance(advice['action_suggestions'], list):
                    advice['action_suggestions'] = []
                
                # Clean action suggestions
                advice['action_suggestions'] = [
                    str(action) for action in advice['action_suggestions']
                    if action  # Filter out None/empty values
                ][:5]  # Limit to 5
                
                self.last_advice = advice
                self.advice_history.append(advice.get('strategy', 'Unknown strategy'))
                return advice
                
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                print(f"Raw response: {response[:300]}...")
                return self._get_fallback_advice()
                
        except Exception as e:
            print(f"Error getting LLM advice: {e}")
            return self._get_fallback_advice()
    
    async def _call_llm_api(self, prompt):
        """Make API call to Ollama with Phi model"""
        import aiohttp
        import asyncio
        
        try:
            async with aiohttp.ClientSession() as session:
                ollama_payload = {
                    "model": "phi",  # or "phi3" depending on your Ollama setup
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent JSON
                        "top_k": 10,
                        "top_p": 0.9,
                        "num_predict": 512   # Limit response length
                    }
                }
                
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=ollama_payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        print(f"Ollama API error: {response.status}")
                        return self._get_fallback_response()
                        
        except asyncio.TimeoutError:
            print("Ollama API timeout - using fallback")
            return self._get_fallback_response()
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self):
        """Fallback JSON response when Ollama is unavailable"""
        return json.dumps({
            "immediate_priority": "explore safely",
            "risk_assessment": "unknown dangers",
            "opportunities": "search for items",
            "strategy": "cautious exploration",
            "action_suggestions": ["search", "move", "pickup"]
        })
    
    def _get_fallback_advice(self):
        """Fallback advice when LLM is unavailable"""
        return {
            "immediate_priority": "explore safely",
            "risk_assessment": "unknown dangers",
            "opportunities": "search for items",
            "strategy": "cautious exploration",
            "action_suggestions": ["search", "move", "pickup"]
        }

class LLMGuidedPPOActor(nn.Module):
    """Enhanced PPO Actor that incorporates LLM guidance"""
    
    def __init__(self, action_dim=23, llm_guidance_weight=0.3):
        super(LLMGuidedPPOActor, self).__init__()
        
        # Base recurrent actor (same as before)
        self.glyph_cnn = RecurrentNetHackCNN(cnn_output_dim=512, lstm_hidden_dim=256)
        self.stats_lstm = nn.LSTM(26, 64, batch_first=True)
        self.message_fc = nn.Linear(256, 128)
        self.inventory_fc = nn.Linear(55, 64)
        self.action_hist_fc = nn.Linear(50, 32)
        
        # LLM guidance integration
        self.llm_guidance_weight = llm_guidance_weight
        self.guidance_fc = nn.Linear(32, 64)  # Process LLM suggestions
        
        # Combined feature processing (increased dim for LLM features)
        combined_dim = 256 + 64 + 128 + 64 + 32 + 64  # 608
        self.combined_fc1 = nn.Linear(combined_dim, 512)
        self.combined_fc2 = nn.Linear(512, 256)
        
        # Action head
        self.action_head = nn.Linear(256, action_dim)
        
        # Hidden states
        self.stats_hidden = None
        
        # Action mappings for LLM suggestions
        self.action_name_to_id = {
            "move_north": 0, "move_south": 1, "move_east": 2, "move_west": 3,
            "move_northeast": 4, "move_northwest": 5, "move_southeast": 6, "move_southwest": 7,
            "wait": 8, "pickup": 9, "drop": 10, "search": 11, "open_door": 12,
            "move": 2, "explore": 11  # Common aliases
        }
    
    def process_llm_guidance(self, llm_advice):
        """Convert LLM advice to feature vector"""
        guidance_vector = np.zeros(32, dtype=np.float32)
        
        if llm_advice:
            # Extract suggested actions
            suggestions = llm_advice.get('action_suggestions', [])
            
            # Create weighted action preferences
            for i, suggestion in enumerate(suggestions[:5]):  # Top 5 suggestions
                # FIX: Ensure suggestion is a string
                if not isinstance(suggestion, str):
                    # Skip non-string suggestions or convert dict to string
                    if isinstance(suggestion, dict):
                        # Try to extract a meaningful string from dict
                        suggestion = str(suggestion.get('action', suggestion.get('name', '')))
                    else:
                        suggestion = str(suggestion)
                
                # Now safely call lower()
                suggestion_lower = suggestion.lower().strip()
                
                if not suggestion_lower:
                    continue
                    
                action_id = self.action_name_to_id.get(suggestion_lower)
                if action_id is not None and action_id < 23:
                    # Weight decreases for later suggestions
                    weight = (5 - i) / 5.0
                    guidance_vector[action_id] = weight
            
            # Add strategic indicators
            priority = llm_advice.get('immediate_priority', '')
            if isinstance(priority, str):
                priority_lower = priority.lower()
                if 'combat' in priority_lower or 'fight' in priority_lower:
                    guidance_vector[30] = 1.0
                elif 'explore' in priority_lower or 'search' in priority_lower:
                    guidance_vector[31] = 1.0
        
        return guidance_vector
    
    def forward(self, obs, reset_hidden=False, llm_advice=None):
        batch_size = obs['glyphs'].size(0)
        
        # Process visual and stats features (same as before)
        glyph_features = self.glyph_cnn(obs['glyphs'], reset_hidden)
        
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
        
        # Process LLM guidance
        if llm_advice:
            guidance_vector = self.process_llm_guidance(llm_advice)
            guidance_tensor = torch.FloatTensor(guidance_vector).to(obs['glyphs'].device)
            guidance_tensor = guidance_tensor.unsqueeze(0).expand(batch_size, -1)
            guidance_features = F.relu(self.guidance_fc(guidance_tensor))
        else:
            guidance_features = torch.zeros(batch_size, 64, device=obs['glyphs'].device)
        
        # Combine all features
        combined = torch.cat([
            glyph_features, stats_features, message_features, 
            inventory_features, action_hist_features, guidance_features
        ], dim=1)
        
        # Process combined features
        x = F.relu(self.combined_fc1(combined))
        x = F.relu(self.combined_fc2(x))
        
        # Output action logits
        base_logits = self.action_head(x)
        
        # Apply LLM guidance as action bias
        if llm_advice and self.llm_guidance_weight > 0:
            guidance_vector = self.process_llm_guidance(llm_advice)
            guidance_bias = torch.FloatTensor(guidance_vector[:23]).to(base_logits.device)
            if batch_size > 1:
                guidance_bias = guidance_bias.unsqueeze(0).expand(batch_size, -1)
            
            # Combine RL policy with LLM guidance
            guided_logits = base_logits + self.llm_guidance_weight * guidance_bias
            return guided_logits
        
        return base_logits
    
    def reset_hidden_states(self):
        """Reset all hidden states"""
        self.glyph_cnn.reset_hidden_state()
        self.stats_hidden = None

class LLMGuidedNetHackAgent:
    """Complete LLM-Guided NetHack RL Agent"""
    
    def __init__(self, action_dim=23, learning_rate=3e-4, gamma=0.99, clip_ratio=0.2, 
                 llm_guidance_weight=0.3, llm_call_frequency=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize enhanced components
        self.obs_processor = NetHackObservationProcessor()
        self.reward_shaper = NetHackRewardShaper()
        self.semantic_descriptor = NetHackSemanticDescriptor()
        self.llm_advisor = LLMStrategicAdvisor(call_frequency=llm_call_frequency)
        
        # Initialize networks with LLM guidance
        self.actor = LLMGuidedPPOActor(
            action_dim=action_dim, 
            llm_guidance_weight=llm_guidance_weight
        ).to(self.device)
        self.critic = RecurrentPPOCritic().to(self.device)  # Reuse existing critic
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Training parameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.buffer = PPOBuffer()
        
        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.shaped_rewards = deque(maxlen=100)
        self.llm_advice_log = []
        
        self.last_action = None
        self.current_llm_advice = None
    
    def process_observation(self, obs):
        """Process observation with enhanced semantic understanding"""
        processed = self.obs_processor.process_observation(obs, self.last_action)
        
        # Convert to tensors
        tensor_obs = {}
        for key, value in processed.items():
            tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
        
        return tensor_obs, processed
    
    async def select_action(self, obs, processed_obs, reset_hidden=False):
        """Select action with LLM guidance and robust error handling"""
        try:
            # Check if we should get new LLM advice
            if self.llm_advisor.should_call_llm():
                try:
                    # Generate semantic description
                    semantic_desc = self.semantic_descriptor.generate_full_description(obs, processed_obs)
                    
                    # Calculate recent performance
                    recent_performance = {
                        'avg_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0,
                        'avg_length': np.mean(list(self.episode_lengths)) if self.episode_lengths else 0,
                        'death_rate': len([r for r in list(self.episode_rewards) if r < 5]) / max(len(self.episode_rewards), 1)
                    }
                    
                    # Get LLM advice
                    self.current_llm_advice = await self.llm_advisor.get_strategic_advice(
                        semantic_desc, recent_performance
                    )
                    
                    # Validate advice structure
                    if not isinstance(self.current_llm_advice, dict):
                        print(f"Warning: LLM advice is not a dict: {type(self.current_llm_advice)}")
                        self.current_llm_advice = self.llm_advisor._get_fallback_advice()
                    
                    # Log the advice
                    self.llm_advice_log.append({
                        'step': self.llm_advisor.step_count,
                        'advice': self.current_llm_advice,
                        'description': semantic_desc[:200] + "..."
                    })
                    
                    priority = self.current_llm_advice.get('immediate_priority', 'No advice')
                    print(f"LLM Advice: {priority}")
                    
                except Exception as e:
                    print(f"Error getting LLM advice: {e}")
                    self.current_llm_advice = self.llm_advisor._get_fallback_advice()
            
            # Select action with current advice
            with torch.no_grad():
                tensor_obs = {}
                for key, value in processed_obs.items():
                    if not isinstance(value, np.ndarray):
                        print(f"Warning: {key} is not numpy array: {type(value)}")
                        continue
                    tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                
                # Ensure we have all required observation keys
                required_keys = ['glyphs', 'stats', 'message', 'inventory', 'action_history']
                for key in required_keys:
                    if key not in tensor_obs:
                        print(f"Warning: Missing observation key: {key}")
                        # Add dummy tensor with appropriate shape
                        if key == 'glyphs':
                            tensor_obs[key] = torch.zeros(1, 21, 79).to(self.device)
                        elif key == 'stats':
                            tensor_obs[key] = torch.zeros(1, 26).to(self.device)
                        elif key == 'message':
                            tensor_obs[key] = torch.zeros(1, 256).to(self.device)
                        elif key == 'inventory':
                            tensor_obs[key] = torch.zeros(1, 55).to(self.device)
                        elif key == 'action_history':
                            tensor_obs[key] = torch.zeros(1, 50).to(self.device)
                
                action_logits = self.actor(tensor_obs, reset_hidden, self.current_llm_advice)
                action_dist = Categorical(logits=action_logits)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                value = self.critic(tensor_obs, reset_hidden)
                
                return action.item(), log_prob.item(), value.item()
                
        except Exception as e:
            print(f"Critical error in select_action: {e}")
            import traceback
            traceback.print_exc()
            # Return safe fallback action (wait/search)
            return 8, 0.0, 0.0  # wait action
    
    async def train_episode(self, env):
        """Train single episode with LLM guidance"""
        obs = env.reset()
        episode_reward = 0
        episode_shaped_reward = 0
        episode_length = 0
        
        # Reset states
        self.actor.reset_hidden_states()
        self.critic.reset_hidden_states()
        self.reward_shaper.reset()
        self.last_action = None
        
        reset_hidden = True
        
        while True:
            tensor_obs, processed_obs = self.process_observation(obs)
            action, log_prob, value = await self.select_action(obs, processed_obs, reset_hidden)
            reset_hidden = False
            
            # Store action
            self.last_action = action
            
            # Take environment step
            step_result = env.step(action)
            
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            # Apply reward shaping
            shaped_reward = self.reward_shaper.shape_reward(next_obs, reward, done, info)
            
            # Store experience
            processed_obs_for_buffer = {}
            for key, tensor_val in tensor_obs.items():
                processed_obs_for_buffer[key] = tensor_val.squeeze(0).cpu()
            
            self.buffer.add(processed_obs_for_buffer, action, shaped_reward, value, log_prob, done)
            
            obs = next_obs
            episode_reward += reward
            episode_shaped_reward += shaped_reward
            episode_length += 1
            
            if done:
                break
        
        return episode_reward, episode_shaped_reward, episode_length

def create_nethack_env():
    """Create and configure NetHack environment"""
    import nle.env
    
    try:
        env = gym.make("NetHackScore-v0")
    except:
        env = gym.make("NetHack-v0")
    
    return env


import sys
import time
from collections import defaultdict
from datetime import datetime
import json

class TrainingMonitor:
    """Rich terminal monitoring for training with comparison metrics"""
    
    def __init__(self, agent_name="LLM-Guided", baseline_metrics_path=None):
        self.agent_name = agent_name
        self.start_time = time.time()
        self.episode_data = []
        self.current_episode_actions = []
        self.baseline_metrics = self._load_baseline_metrics(baseline_metrics_path)
        
        # Terminal colors
        self.HEADER = '\033[95m'
        self.BLUE = '\033[94m'
        self.CYAN = '\033[96m'
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RED = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'
        self.UNDERLINE = '\033[4m'
        
    def _load_baseline_metrics(self, path):
        """Load baseline RL agent metrics for comparison"""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def print_header(self):
        """Print training session header"""
        print("\n" + "="*80)
        print(f"{self.BOLD}{self.CYAN}NetHack RL Training Monitor - {self.agent_name}{self.ENDC}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
    
    def print_episode_start(self, episode):
        """Print episode start"""
        print(f"\n{self.BOLD}{self.BLUE}{'─'*80}")
        print(f"Episode {episode} Starting...")
        print(f"{'─'*80}{self.ENDC}\n")
        self.current_episode_actions = []
    
    def print_llm_advice(self, advice, step):
        """Print LLM advice in a formatted box"""
        if not advice:
            return
            
        print(f"\n{self.YELLOW}╔{'═'*78}╗")
        print(f"║ {self.BOLD}LLM STRATEGIC ADVICE (Step {step}){' '*44}{self.ENDC}{self.YELLOW}║")
        print(f"╠{'═'*78}╣{self.ENDC}")
        
        # Priority
        priority = advice.get('immediate_priority', 'N/A')
        print(f"{self.YELLOW}║{self.ENDC} {self.BOLD}Priority:{self.ENDC} {priority:<65} {self.YELLOW}║{self.ENDC}")
        
        # Risk assessment
        risk = advice.get('risk_assessment', 'N/A')
        risk_display = risk[:65] if len(risk) > 65 else risk
        print(f"{self.YELLOW}║{self.ENDC} {self.BOLD}Risk:{self.ENDC} {risk_display:<68} {self.YELLOW}║{self.ENDC}")
        
        # Opportunities
        opps = advice.get('opportunities', 'N/A')
        opps_display = opps[:65] if len(opps) > 65 else opps
        print(f"{self.YELLOW}║{self.ENDC} {self.BOLD}Opportunity:{self.ENDC} {opps_display:<61} {self.YELLOW}║{self.ENDC}")
        
        # Action suggestions
        actions = advice.get('action_suggestions', [])
        if actions:
            actions_str = ", ".join(str(a) for a in actions[:5])
            actions_display = actions_str[:65] if len(actions_str) > 65 else actions_str
            print(f"{self.YELLOW}║{self.ENDC} {self.BOLD}Suggested Actions:{self.ENDC} {actions_display:<56} {self.YELLOW}║{self.ENDC}")
        
        print(f"{self.YELLOW}╚{'═'*78}╝{self.ENDC}\n")
    
    def print_step_info(self, step, action, action_name, reward, shaped_reward, health, level):
        """Print information about current step"""
        # Track action
        self.current_episode_actions.append(action_name)
        
        # Color code reward
        if reward > 0:
            reward_color = self.GREEN
        elif reward < 0:
            reward_color = self.RED
        else:
            reward_color = self.ENDC
        
        # Color code health
        if health < 0.3:
            health_color = self.RED
        elif health < 0.6:
            health_color = self.YELLOW
        else:
            health_color = self.GREEN
        
        print(f"Step {step:4d} | "
              f"Action: {action_name:15s} | "
              f"R: {reward_color}{reward:6.2f}{self.ENDC} | "
              f"SR: {shaped_reward:7.3f} | "
              f"HP: {health_color}{health*100:5.1f}%{self.ENDC} | "
              f"Lvl: {level}")
    
    def print_episode_summary(self, episode, metrics, llm_advice_count):
        """Print detailed episode summary with comparison"""
        print(f"\n{self.BOLD}{self.GREEN}{'─'*80}")
        print(f"Episode {episode} Complete")
        print(f"{'─'*80}{self.ENDC}\n")
        
        # Episode metrics
        print(f"{self.BOLD}Episode Metrics:{self.ENDC}")
        print(f"  Raw Reward:    {metrics['raw_reward']:8.2f}")
        print(f"  Shaped Reward: {metrics['shaped_reward']:8.2f}")
        print(f"  Length:        {metrics['length']:8d} steps")
        print(f"  Survival Time: {metrics['length']/60:8.1f} minutes (simulated)")
        print(f"  LLM Calls:     {llm_advice_count:8d}")
        
        # Action distribution
        print(f"\n{self.BOLD}Action Distribution:{self.ENDC}")
        action_counts = defaultdict(int)
        for action in self.current_episode_actions:
            action_counts[action] += 1
        
        # Show top 5 actions
        top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for action, count in top_actions:
            pct = (count / len(self.current_episode_actions)) * 100
            bar_length = int(pct / 2)  # Scale to 50 chars max
            bar = '█' * bar_length
            print(f"  {action:15s}: {bar} {count:4d} ({pct:5.1f}%)")
        
        # Store episode data
        self.episode_data.append(metrics)
    
    def print_training_progress(self, episode, window=10):
        """Print rolling average training progress"""
        if len(self.episode_data) < window:
            return
        
        recent_data = self.episode_data[-window:]
        
        avg_raw = sum(d['raw_reward'] for d in recent_data) / window
        avg_shaped = sum(d['shaped_reward'] for d in recent_data) / window
        avg_length = sum(d['length'] for d in recent_data) / window
        
        print(f"\n{self.BOLD}{self.CYAN}{'─'*80}")
        print(f"Training Progress (Last {window} Episodes)")
        print(f"{'─'*80}{self.ENDC}")
        
        print(f"  Avg Raw Reward:    {avg_raw:8.2f}")
        print(f"  Avg Shaped Reward: {avg_shaped:8.2f}")
        print(f"  Avg Length:        {avg_length:8.1f} steps")
        
        # Comparison with baseline
        if self.baseline_metrics:
            baseline_window = self.baseline_metrics.get('rolling_averages', {}).get(str(episode), {})
            if baseline_window:
                baseline_reward = baseline_window.get('avg_reward', 0)
                baseline_length = baseline_window.get('avg_length', 0)
                
                print(f"\n{self.BOLD}Comparison with Baseline RL Agent:{self.ENDC}")
                
                # Reward comparison
                reward_diff = avg_raw - baseline_reward
                reward_pct = (reward_diff / abs(baseline_reward)) * 100 if baseline_reward != 0 else 0
                reward_symbol = "↑" if reward_diff > 0 else "↓"
                reward_color = self.GREEN if reward_diff > 0 else self.RED
                
                print(f"  Reward:  {self.agent_name}: {avg_raw:8.2f} vs Baseline: {baseline_reward:8.2f} "
                      f"{reward_color}({reward_symbol} {abs(reward_pct):5.1f}%){self.ENDC}")
                
                # Length comparison
                length_diff = avg_length - baseline_length
                length_pct = (length_diff / baseline_length) * 100 if baseline_length != 0 else 0
                length_symbol = "↑" if length_diff > 0 else "↓"
                length_color = self.GREEN if length_diff > 0 else self.RED
                
                print(f"  Length:  {self.agent_name}: {avg_length:8.1f} vs Baseline: {baseline_length:8.1f} "
                      f"{length_color}({length_symbol} {abs(length_pct):5.1f}%){self.ENDC}")
        
        # Time stats
        elapsed = time.time() - self.start_time
        eps_per_hour = (episode / elapsed) * 3600 if elapsed > 0 else 0
        print(f"\n  Episodes Completed: {episode}")
        print(f"  Training Time:      {elapsed/60:6.1f} minutes")
        print(f"  Episodes/Hour:      {eps_per_hour:6.1f}")
    
    def print_final_summary(self):
        """Print final training summary"""
        print(f"\n\n{self.BOLD}{self.HEADER}{'='*80}")
        print(f"TRAINING COMPLETE - FINAL SUMMARY")
        print(f"{'='*80}{self.ENDC}\n")
        
        if not self.episode_data:
            print("No episode data collected.")
            return
        
        # Overall statistics
        total_episodes = len(self.episode_data)
        total_raw = sum(d['raw_reward'] for d in self.episode_data)
        total_shaped = sum(d['shaped_reward'] for d in self.episode_data)
        avg_raw = total_raw / total_episodes
        avg_shaped = total_shaped / total_episodes
        avg_length = sum(d['length'] for d in self.episode_data) / total_episodes
        
        print(f"{self.BOLD}Overall Performance:{self.ENDC}")
        print(f"  Total Episodes:     {total_episodes}")
        print(f"  Avg Raw Reward:     {avg_raw:8.2f}")
        print(f"  Avg Shaped Reward:  {avg_shaped:8.2f}")
        print(f"  Avg Length:         {avg_length:8.1f} steps")
        
        # Best episode
        best_episode = max(self.episode_data, key=lambda x: x['raw_reward'])
        best_idx = self.episode_data.index(best_episode)
        print(f"\n{self.BOLD}Best Episode: #{best_idx}{self.ENDC}")
        print(f"  Raw Reward:    {best_episode['raw_reward']:8.2f}")
        print(f"  Shaped Reward: {best_episode['shaped_reward']:8.2f}")
        print(f"  Length:        {best_episode['length']:8d} steps")
        
        # Training time
        total_time = time.time() - self.start_time
        print(f"\n{self.BOLD}Training Time:{self.ENDC}")
        print(f"  Total:         {total_time/60:8.1f} minutes")
        print(f"  Per Episode:   {total_time/total_episodes:8.1f} seconds")
        
        # Save metrics
        self.save_metrics()
    
    def save_metrics(self):
        """Save metrics to JSON for future comparison"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{self.agent_name.lower().replace(' ', '_')}_{timestamp}.json"
        
        # Calculate rolling averages for comparison
        rolling_averages = {}
        window = 10
        for i in range(window, len(self.episode_data) + 1):
            recent = self.episode_data[i-window:i]
            rolling_averages[i] = {
                'avg_reward': sum(d['raw_reward'] for d in recent) / window,
                'avg_shaped_reward': sum(d['shaped_reward'] for d in recent) / window,
                'avg_length': sum(d['length'] for d in recent) / window,
            }
        
        metrics = {
            'agent_name': self.agent_name,
            'timestamp': timestamp,
            'total_episodes': len(self.episode_data),
            'episode_data': self.episode_data,
            'rolling_averages': rolling_averages,
            'summary': {
                'avg_raw_reward': sum(d['raw_reward'] for d in self.episode_data) / len(self.episode_data),
                'avg_shaped_reward': sum(d['shaped_reward'] for d in self.episode_data) / len(self.episode_data),
                'avg_length': sum(d['length'] for d in self.episode_data) / len(self.episode_data),
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n{self.GREEN}Metrics saved to: {filename}{self.ENDC}")


# Enhanced agent class with monitoring integration
class MonitoredLLMGuidedNetHackAgent(LLMGuidedNetHackAgent):
    """LLM-Guided agent with integrated monitoring"""
    
    def __init__(self, *args, baseline_metrics_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = TrainingMonitor(
            agent_name="LLM-Guided PPO",
            baseline_metrics_path=baseline_metrics_path
        )
        self.action_meanings = {
            0: "move_north", 1: "move_south", 2: "move_east", 3: "move_west",
            4: "move_northeast", 5: "move_northwest", 6: "move_southeast", 7: "move_southwest",
            8: "wait", 9: "pickup", 10: "drop", 11: "search", 12: "open_door",
            13: "close_door", 14: "kick", 15: "eat", 16: "drink", 17: "read",
            18: "apply", 19: "throw", 20: "wear", 21: "take_off", 22: "wield"
        }
    
    async def train_episode_monitored(self, env, episode):
        """Train episode with monitoring"""
        self.monitor.print_episode_start(episode)
        
        obs = env.reset()
        episode_reward = 0
        episode_shaped_reward = 0
        episode_length = 0
        llm_call_count = 0
        
        # Reset states
        self.actor.reset_hidden_states()
        self.critic.reset_hidden_states()
        self.reward_shaper.reset()
        self.last_action = None
        
        reset_hidden = True
        
        while True:
            # Check for LLM advice
            should_get_advice = self.llm_advisor.should_call_llm()
            
            tensor_obs, processed_obs = self.process_observation(obs)
            action, log_prob, value = await self.select_action(obs, processed_obs, reset_hidden)
            reset_hidden = False
            
            # Display LLM advice if it was just received
            if should_get_advice and self.current_llm_advice:
                self.monitor.print_llm_advice(self.current_llm_advice, episode_length)
                llm_call_count += 1
            
            # Store action
            self.last_action = action
            action_name = self.action_meanings.get(action, f"action_{action}")
            
            # Take environment step
            step_result = env.step(action)
            
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            # Apply reward shaping
            shaped_reward = self.reward_shaper.shape_reward(next_obs, reward, done, info)
            
            # Get health for display
            if isinstance(next_obs, tuple):
                next_obs_dict = next_obs[0]
            else:
                next_obs_dict = next_obs
                
            stats = next_obs_dict.get('blstats', np.zeros(26))
            health_ratio = stats[0] / stats[1] if len(stats) > 1 and stats[1] > 0 else 0
            level = int(stats[7]) if len(stats) > 7 else 1
            
            # Print step info every N steps (to avoid clutter)
            if episode_length % 5 == 0:
                self.monitor.print_step_info(
                    episode_length, action, action_name, 
                    reward, shaped_reward, health_ratio, level
                )
            
            # Store experience
            processed_obs_for_buffer = {}
            for key, tensor_val in tensor_obs.items():
                processed_obs_for_buffer[key] = tensor_val.squeeze(0).cpu()
            
            self.buffer.add(processed_obs_for_buffer, action, shaped_reward, value, log_prob, done)
            
            obs = next_obs
            episode_reward += reward
            episode_shaped_reward += shaped_reward
            episode_length += 1
            
            if done:
                break
        
        # Episode metrics
        metrics = {
            'raw_reward': episode_reward,
            'shaped_reward': episode_shaped_reward,
            'length': episode_length,
            'llm_calls': llm_call_count
        }
        
        self.monitor.print_episode_summary(episode, metrics, llm_call_count)
        
        return episode_reward, episode_shaped_reward, episode_length


async def main_monitored():
    """Main training function with monitoring"""
    import os
    
    # Check for baseline metrics
    baseline_path = None
    if len(sys.argv) > 1:
        baseline_path = sys.argv[1]
        if not os.path.exists(baseline_path):
            print(f"Warning: Baseline metrics file not found: {baseline_path}")
            baseline_path = None
    
    print("Setting up LLM-Guided NetHack PPO Training with Monitoring...")
    
    # Create environment
    env = create_nethack_env()
    print(f"Environment action space: {env.action_space.n}")
    
    # Create monitored agent
    agent = MonitoredLLMGuidedNetHackAgent(
        action_dim=env.action_space.n,
        llm_guidance_weight=0.3,
        llm_call_frequency=20,
        baseline_metrics_path=baseline_path
    )
    
    agent.monitor.print_header()
    
    # Training loop
    num_episodes = 100
    update_frequency = 2048
    
    for episode in range(num_episodes):
        episode_reward, episode_shaped_reward, episode_length = \
            await agent.train_episode_monitored(env, episode)
        
        agent.episode_rewards.append(episode_reward)
        agent.shaped_rewards.append(episode_shaped_reward)
        agent.episode_lengths.append(episode_length)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            agent.monitor.print_training_progress(episode + 1, window=10)
        
        # Update networks periodically
        if len(agent.buffer) >= update_frequency:
            print(f"\n{agent.monitor.CYAN}Updating neural networks...{agent.monitor.ENDC}")
            # agent.update()  # Uncomment when you have the update method
            agent.buffer.clear()
    
    # Final summary
    agent.monitor.print_final_summary()
    
    # Save model and advice log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.save_llm_advice_log(f"llm_advice_log_{timestamp}.json")
    
    env.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_monitored())

# async def main():
#     """Main training function for LLM-guided agent"""
#     print("Setting up LLM-Guided NetHack PPO Training...")
    
#     # Create environment
#     env = create_nethack_env()
#     print(f"Environment action space: {env.action_space.n}")
    
#     # Create LLM-guided agent
#     agent = LLMGuidedNetHackAgent(
#         action_dim=env.action_space.n,
#         llm_guidance_weight=0.3,  # 30% LLM influence
#         llm_call_frequency=20     # Get advice every 20 steps
#     )
    
#     print("Starting LLM-guided training...")
    
#     # Training loop
#     for episode in range(100):  # Reduced for testing
#         episode_reward, episode_shaped_reward, episode_length = await agent.train_episode(env)
        
#         agent.episode_rewards.append(episode_reward)
#         agent.shaped_rewards.append(episode_shaped_reward)
#         agent.episode_lengths.append(episode_length)
        
#         if episode % 10 == 0:
#             avg_reward = np.mean(list(agent.episode_rewards))
#             avg_shaped_reward = np.mean(list(agent.shaped_rewards))
#             avg_length = np.mean(list(agent.episode_lengths))
#             print(f"Episode {episode}: Raw: {avg_reward:.3f}, "
#                   f"Shaped: {avg_shaped_reward:.3f}, Length: {avg_length:.1f}")
        
#         # Update networks periodically
#         if len(agent.buffer) >= 2048:
#             print(f"Updating networks after episode {episode}")
#             # agent.update()  # You'd implement this similar to the enhanced agent
#             agent.buffer.clear()
    
#     # Save results
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     agent.save_llm_advice_log(f"llm_advice_log_{timestamp}.json")
    
#     env.close()
#     print("LLM-guided training completed!")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())