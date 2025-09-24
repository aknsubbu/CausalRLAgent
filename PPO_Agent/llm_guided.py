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
        """Get strategic advice from Ollama Phi model"""
        try:
            # Simplified prompt optimized for Phi model's context window
            prompt = f"""You are a NetHack expert. Analyze this game state and provide strategic advice.

GAME STATE:
{semantic_description}

PERFORMANCE:
Avg reward: {recent_performance.get('avg_reward', 0):.1f}
Survival: {recent_performance.get('avg_length', 0):.0f} steps

TASK: Respond with valid JSON only. No other text.

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
            # Remove any markdown code blocks
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "").strip()
            elif response.startswith("```"):
                response = response.replace("```", "").strip()
            
            try:
                advice = json.loads(response)
                self.last_advice = advice
                self.advice_history.append(advice.get('strategy', 'Unknown strategy'))
                return advice
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                print(f"Raw response: {response[:200]}...")
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
                action_id = self.action_name_to_id.get(suggestion.lower())
                if action_id is not None and action_id < 23:
                    # Weight decreases for later suggestions
                    weight = (5 - i) / 5.0
                    guidance_vector[action_id] = weight
            
            # Add strategic indicators
            priority = llm_advice.get('immediate_priority', '').lower()
            if 'combat' in priority:
                guidance_vector[30] = 1.0
            elif 'explore' in priority:
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
        """Select action with LLM guidance"""
        # Check if we should get new LLM advice
        if self.llm_advisor.should_call_llm():
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
            
            # Log the advice
            self.llm_advice_log.append({
                'step': self.llm_advisor.step_count,
                'advice': self.current_llm_advice,
                'description': semantic_desc[:200] + "..."
            })
            
            print(f"LLM Advice: {self.current_llm_advice.get('immediate_priority', 'No advice')}")
        
        # Select action with current advice
        with torch.no_grad():
            tensor_obs = {}
            for key, value in processed_obs.items():
                tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
            
            action_logits = self.actor(tensor_obs, reset_hidden, self.current_llm_advice)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.critic(tensor_obs, reset_hidden)
            
            return action.item(), log_prob.item(), value.item()
    
    def save_llm_advice_log(self, filepath):
        """Save LLM advice log for analysis"""
        with open(filepath, 'w') as f:
            json.dump(self.llm_advice_log, f, indent=2)
    
    # [Rest of the training methods would be similar to the enhanced agent but with async action selection]
    
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

async def main():
    """Main training function for LLM-guided agent"""
    print("Setting up LLM-Guided NetHack PPO Training...")
    
    # Create environment
    env = create_nethack_env()
    print(f"Environment action space: {env.action_space.n}")
    
    # Create LLM-guided agent
    agent = LLMGuidedNetHackAgent(
        action_dim=env.action_space.n,
        llm_guidance_weight=0.3,  # 30% LLM influence
        llm_call_frequency=20     # Get advice every 20 steps
    )
    
    print("Starting LLM-guided training...")
    
    # Training loop
    for episode in range(100):  # Reduced for testing
        episode_reward, episode_shaped_reward, episode_length = await agent.train_episode(env)
        
        agent.episode_rewards.append(episode_reward)
        agent.shaped_rewards.append(episode_shaped_reward)
        agent.episode_lengths.append(episode_length)
        
        if episode % 10 == 0:
            avg_reward = np.mean(list(agent.episode_rewards))
            avg_shaped_reward = np.mean(list(agent.shaped_rewards))
            avg_length = np.mean(list(agent.episode_lengths))
            print(f"Episode {episode}: Raw: {avg_reward:.3f}, "
                  f"Shaped: {avg_shaped_reward:.3f}, Length: {avg_length:.1f}")
        
        # Update networks periodically
        if len(agent.buffer) >= 2048:
            print(f"Updating networks after episode {episode}")
            # agent.update()  # You'd implement this similar to the enhanced agent
            agent.buffer.clear()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.save_llm_advice_log(f"llm_advice_log_{timestamp}.json")
    
    env.close()
    print("LLM-guided training completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())