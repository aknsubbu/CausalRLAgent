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
import sys
import time

# %%
def create_nethack_env():
    """Create and configure NetHack environment"""
    import nle.env
    
    try:
        env = gym.make("NetHackScore-v0")
    except:
        env = gym.make("NetHack-v0")
    
    return env


# %% [markdown]
# ## NetHack Reward Shaper
# 

# %%
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


# %% [markdown]
# ## NetHack Semantic Descriptor
# 

# %%
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


# %% [markdown]
# ## NetHack Observation Processor
# 

# %%
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


# %% [markdown]
# ## Recurrent NetHack CNN
# 

# %%
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


# %% [markdown]
# ## Recurrent PPO Critic
# 

# %%
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



# %% [markdown]
# ## PPO Buffer
# 

# %%
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


# %% [markdown]
# ## LLM Guided PPO Actor
# 

# %%
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
        
        # Gating network to learn when to trust LLM
        self.trust_gate = nn.Sequential(
            nn.Linear(256 + 64, 128),  # combined features + guidance
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output 0-1 trust score
        )
        
        # Combined feature processing (increased dim for LLM features)
        combined_dim = 256 + 64 + 128 + 64 + 32 + 64  # 608
        self.combined_fc1 = nn.Linear(combined_dim, 512)
        self.combined_fc2 = nn.Linear(512, 256)
        
        # Action head
        self.action_head = nn.Linear(256, action_dim)
        
        # Hidden states
        self.stats_hidden = None
        
        # ✅ COMPREHENSIVE action mappings with aliases and fuzzy matching support
        self.action_name_to_id = {
            # Movement - Cardinal directions (0-3)
            "move_north": 0, "north": 0, "up": 0, "go_north": 0, "n": 0, "move up": 0,
            "move_south": 1, "south": 1, "down": 1, "go_south": 1, "s": 1, "move down": 1,
            "move_east": 2, "east": 2, "right": 2, "go_east": 2, "e": 2, "move right": 2,
            "move_west": 3, "west": 3, "left": 3, "go_west": 3, "w": 3, "move left": 3,
            
            # Movement - Diagonal directions (4-7)
            "move_northeast": 4, "northeast": 4, "ne": 4, "up right": 4, "go_northeast": 4,
            "move_northwest": 5, "northwest": 5, "nw": 5, "up left": 5, "go_northwest": 5,
            "move_southeast": 6, "southeast": 6, "se": 6, "down right": 6, "go_southeast": 6,
            "move_southwest": 7, "southwest": 7, "sw": 7, "down left": 7, "go_southwest": 7,
            
            # Generic movement
            "move": 2, "walk": 2, "go": 2, "travel": 2, "navigate": 2,
            
            # Wait (8)
            "wait": 8, "rest": 8, "pause": 8, "stay": 8, "remain": 8, "idle": 8, "do nothing": 8,
            
            # Pickup (9)
            "pickup": 9, "pick": 9, "take": 9, "grab": 9, "get": 9, "collect": 9, "pick up": 9,
            "get item": 9, "take item": 9, "grab item": 9,
            
            # Drop (10)
            "drop": 10, "throw_away": 10, "discard": 10, "release": 10, "drop item": 10,
            "put down": 10, "leave": 10,
            
            # Search (11)
            "search": 11, "look": 11, "explore": 11, "find": 11, "investigate": 11, 
            "examine": 11, "scout": 11, "look around": 11, "search area": 11,
            
            # Open door (12)
            "open_door": 12, "open": 12, "unlock": 12, "open door": 12,
            
            # Close door (13)
            "close_door": 13, "close": 13, "shut": 13, "close door": 13,
            
            # Kick/Attack (14)
            "kick": 14, "attack": 14, "strike": 14, "hit": 14, "fight": 14, "combat": 14,
            "melee": 14, "assault": 14,
            
            # Eat (15)
            "eat": 15, "consume": 15, "food": 15, "eat food": 15, "have food": 15,
            "consume food": 15, "bite": 15, "feed": 15,
            
            # Drink (16)
            "drink": 16, "quaff": 16, "potion": 16, "drink potion": 16, "consume potion": 16,
            "sip": 16, "gulp": 16,
            
            # Read (17)
            "read": 17, "scroll": 17, "read scroll": 17, "peruse": 17, "study": 17,
            
            # Apply (18)
            "apply": 18, "use": 18, "activate": 18, "utilize": 18, "use item": 18,
            "apply item": 18, "employ": 18,
            
            # Throw (19)
            "throw": 19, "toss": 19, "hurl": 19, "fling": 19, "cast": 19, "throw item": 19,
            
            # Wear (20)
            "wear": 20, "equip": 20, "armor": 20, "put on": 20, "wear armor": 20,
            "equip armor": 20, "don": 20, "dress": 20,
            
            # Take off (21)
            "take_off": 21, "remove": 21, "unequip": 21, "doff": 21, "take off armor": 21,
            "remove armor": 21, "strip": 21,
            
            # Wield (22)
            "wield": 22, "weapon": 22, "hold": 22, "equip weapon": 22, "wield weapon": 22,
            "arm": 22, "brandish": 22, "grasp": 22
        }
        
        # ✅ Add action category groups for semantic matching
        self.action_categories = {
            'movement': [0, 1, 2, 3, 4, 5, 6, 7],
            'exploration': [11, 8],  # search, wait
            'combat': [14, 22, 19],  # kick/attack, wield, throw
            'item_use': [15, 16, 17, 18],  # eat, drink, read, apply
            'inventory': [9, 10, 20, 21],  # pickup, drop, wear, take_off
            'doors': [12, 13],  # open, close
            'defensive': [8, 15, 16],  # wait, eat, drink (for healing)
            'offensive': [14, 22, 19]  # attack, wield, throw
        }
        
        # ✅ Category keywords for fuzzy semantic matching
        self.category_keywords = {
            'movement': ['move', 'go', 'walk', 'travel', 'navigate', 'head', 'proceed'],
            'exploration': ['search', 'look', 'explore', 'investigate', 'scout', 'examine', 'find'],
            'combat': ['attack', 'fight', 'kill', 'strike', 'combat', 'hit', 'battle', 'engage'],
            'item_use': ['eat', 'drink', 'use', 'consume', 'apply', 'activate', 'read'],
            'inventory': ['pick', 'take', 'grab', 'drop', 'equip', 'wear', 'collect', 'get'],
            'defensive': ['heal', 'rest', 'recover', 'restore', 'hide', 'retreat', 'flee', 'escape'],
            'offensive': ['attack', 'kill', 'destroy', 'eliminate', 'defeat']
        }
    
    def _fuzzy_match_action(self, suggestion: str):
        """✅ NEW: Fuzzy matching for action names with multiple strategies"""
        if not suggestion:
            return None
            
        suggestion_lower = suggestion.lower().strip()
        
        # Strategy 1: Check if suggestion contains any known action word
        for action_name, action_id in self.action_name_to_id.items():
            if action_name in suggestion_lower or suggestion_lower in action_name:
                return action_id
        
        # Strategy 2: Check for category keywords
        for category, keywords in self.category_keywords.items():
            if any(kw in suggestion_lower for kw in keywords):
                # Return first action from that category
                return self.action_categories.get(category, [None])[0]
        
        # Strategy 3: Partial word matching (e.g., "moving" matches "move")
        for action_name, action_id in self.action_name_to_id.items():
            if len(action_name) >= 4:  # Only for longer words
                # Check if first 4 characters match
                if suggestion_lower[:4] == action_name[:4]:
                    return action_id
        
        return None
    
    def process_llm_guidance(self, llm_advice):
        """✅ IMPROVED: Convert LLM advice to feature vector with fuzzy matching and category boosting"""
        guidance_vector = np.zeros(32, dtype=np.float32)
        
        if not llm_advice:
            return guidance_vector
        
        suggestions = llm_advice.get('action_suggestions', [])
        
        # ✅ Weight actions from suggestions with fuzzy matching
        for i, suggestion in enumerate(suggestions[:5]):  # Top 5 suggestions
            # Ensure suggestion is a string
            if not isinstance(suggestion, str):
                if isinstance(suggestion, dict):
                    suggestion = str(suggestion.get('action', suggestion.get('name', '')))
                else:
                    suggestion = str(suggestion)
            
            suggestion_lower = suggestion.lower().strip()
            
            if not suggestion_lower:
                continue
            
            # ✅ Try exact match first
            action_id = self.action_name_to_id.get(suggestion_lower)
            
            # ✅ Try fuzzy match if exact fails
            if action_id is None:
                action_id = self._fuzzy_match_action(suggestion_lower)
            
            if action_id is not None and action_id < 23:
                # Weight decreases for later suggestions
                weight = (5 - i) / 5.0
                guidance_vector[action_id] = max(guidance_vector[action_id], weight)
        
        # ✅ Boost action categories based on priority
        priority = llm_advice.get('immediate_priority', '')
        if isinstance(priority, str):
            priority_lower = priority.lower()
            
            # Combat priority
            if any(word in priority_lower for word in ['combat', 'fight', 'attack', 'kill', 'enemy', 'monster']):
                for action_id in self.action_categories['combat']:
                    guidance_vector[action_id] += 0.3
                guidance_vector[30] = 1.0  # Combat indicator
            
            # Exploration priority
            elif any(word in priority_lower for word in ['explore', 'search', 'look', 'find', 'investigate']):
                for action_id in self.action_categories['exploration']:
                    guidance_vector[action_id] += 0.3
                for action_id in self.action_categories['movement']:
                    guidance_vector[action_id] += 0.1  # Also boost movement
                guidance_vector[31] = 1.0  # Exploration indicator
            
            # Healing/Survival priority
            elif any(word in priority_lower for word in ['eat', 'drink', 'heal', 'food', 'health', 'restore', 'potion', 'starv', 'hungry', 'weak', 'faint']):
                for action_id in self.action_categories['item_use']:
                    guidance_vector[action_id] += 0.4
                for action_id in [9]:  # pickup to get items
                    guidance_vector[action_id] += 0.2
            
            # Danger/Flee priority
            elif any(word in priority_lower for word in ['flee', 'escape', 'run', 'retreat', 'danger', 'low health', 'critical']):
                # Boost movement away (generic movement boost)
                for action_id in self.action_categories['movement']:
                    guidance_vector[action_id] += 0.4
        
        # ✅ Check strategy for additional context
        strategy = llm_advice.get('strategy', '')
        if isinstance(strategy, str):
            strategy_lower = strategy.lower()
            
            if 'item' in strategy_lower or 'collect' in strategy_lower:
                for action_id in self.action_categories['inventory']:
                    guidance_vector[action_id] += 0.2
            
            if 'door' in strategy_lower:
                for action_id in self.action_categories['doors']:
                    guidance_vector[action_id] += 0.2
        
        # ✅ Normalize to prevent extreme bias
        if guidance_vector[:23].sum() > 0:
            guidance_vector[:23] = guidance_vector[:23] / (guidance_vector[:23].max() + 1e-8)
        
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
        if llm_advice and self.llm_guidance_weight > 0:
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
        
        # ✅ IMPROVED: Apply LLM guidance with learned trust gating
        if llm_advice and self.llm_guidance_weight > 0:
            guidance_vector = self.process_llm_guidance(llm_advice)
            
            # Prepare guidance tensor
            guidance_tensor = torch.FloatTensor(guidance_vector).to(base_logits.device)
            if batch_size > 1:
                guidance_tensor = guidance_tensor.unsqueeze(0).expand(batch_size, -1)
            else:
                guidance_tensor = guidance_tensor.unsqueeze(0)
            
            # ✅ Learn trust/confidence in LLM advice
            combined_for_trust = torch.cat([x, guidance_features], dim=1)
            trust_score = self.trust_gate(combined_for_trust)  # [batch, 1]
            
            # ✅ Apply trust-modulated guidance
            guidance_bias = guidance_tensor[:, :23]  # Only action dimensions
            
            # ✅ Use learned trust score combined with fixed weight
            adaptive_weight = trust_score * self.llm_guidance_weight
            guided_logits = base_logits + adaptive_weight * guidance_bias
            
            return guided_logits
        
        return base_logits
    
    def reset_hidden_states(self):
        """Reset all hidden states"""
        self.glyph_cnn.reset_hidden_state()
        self.stats_hidden = None


# %% [markdown]
# ## LLM Strategic Advisor
# 

# %%
class LLMStrategicAdvisor:
    """Provides strategic advice using LLM API calls"""
    
    def __init__(self, call_frequency=10):
        self.call_frequency = call_frequency  # Call LLM every N steps
        self.step_count = 0
        self.last_advice = None
        self.advice_history = deque(maxlen=5)
        
        # ✅ ADD: Define valid action vocabulary for LLM
        self.valid_actions = [
            "move_north", "move_south", "move_east", "move_west",
            "move_northeast", "move_northwest", "move_southeast", "move_southwest",
            "wait", "search", "pickup", "drop", "eat", "drink",
            "open_door", "close_door", "kick", "read", "apply",
            "wear", "take_off", "wield", "throw"
        ]
        
        # ✅ ADD: Track advice effectiveness
        self.advice_outcomes = defaultdict(lambda: {'reward_sum': 0.0, 'count': 0})
        
    def should_call_llm(self):
        """Determine if we should call the LLM for advice"""
        self.step_count += 1
        return self.step_count % self.call_frequency == 0
    
    async def get_strategic_advice(self, semantic_description, recent_performance):
        """✅ IMPROVED: Get strategic advice with better prompt engineering"""
        try:
            # ✅ Build context-aware prompt with valid actions
            prompt = f"""You are an expert NetHack player AI advisor. Analyze the game state and provide strategic advice.

GAME STATE:
{semantic_description}

RECENT PERFORMANCE:
- Average Reward: {recent_performance.get('avg_reward', 0):.2f}
- Average Survival: {recent_performance.get('avg_length', 0):.0f} steps
- Death Rate: {recent_performance.get('death_rate', 0)*100:.1f}%

VALID ACTIONS (you MUST choose from these ONLY):
{', '.join(self.valid_actions)}

INSTRUCTIONS:
1. Analyze the immediate danger level (health, nearby enemies)
2. Identify opportunities (items, stairs, resources)
3. Suggest 3-5 specific actions from the VALID ACTIONS list above
4. Prioritize survival if health is low
5. Use EXACT action names from the list (e.g., "move_north" not "go north")

Respond with EXACTLY this JSON format (no markdown, no extra text, no code blocks):
{{
  "immediate_priority": "one sentence describing most urgent need",
  "risk_assessment": "brief danger evaluation",
  "opportunities": "what good options are available",
  "strategy": "high-level approach",
  "action_suggestions": ["move_east", "search", "pickup"]
}}

Example valid response:
{{
  "immediate_priority": "health is critical, need to find food or healing",
  "risk_assessment": "low health with enemies nearby",
  "opportunities": "potion visible to the east",
  "strategy": "prioritize survival by moving to potion",
  "action_suggestions": ["move_east", "pickup", "drink"]
}}

IMPORTANT: 
- Use only action names from VALID ACTIONS list
- Return pure JSON without any markdown formatting
- Do not include explanations outside the JSON

JSON response:
"""
            
            # Call Ollama API
            response = await self._call_llm_api(prompt)
            
            # Debug: show raw response
            print(f"\n[DEBUG] Raw LLM response (first 300 chars):\n{response[:300]}\n")
            
            # Aggressive JSON extraction
            response = response.strip()
            
            # Remove markdown
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()
            
            # Find JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
                
            # Try to parse
            try:
                advice = json.loads(response)
                
                # ✅ Validate and filter action suggestions against valid actions
                valid_suggestions = []
                for action in advice.get('action_suggestions', []):
                    if isinstance(action, str):
                        action_lower = action.lower().strip()
                        # Check if it's valid or close to valid
                        if action_lower in [va.lower() for va in self.valid_actions]:
                            valid_suggestions.append(action_lower)
                        else:
                            # Try to find close match
                            for valid_action in self.valid_actions:
                                if action_lower in valid_action.lower() or valid_action.lower() in action_lower:
                                    valid_suggestions.append(valid_action.lower())
                                    break
                
                # ✅ Ensure at least one valid action
                if not valid_suggestions:
                    # Default based on priority
                    priority = advice.get('immediate_priority', '').lower()
                    if 'combat' in priority or 'attack' in priority:
                        valid_suggestions = ["kick", "wield", "move_north"]
                    elif 'health' in priority or 'heal' in priority or 'food' in priority:
                        valid_suggestions = ["eat", "drink", "search", "pickup"]
                    elif 'explore' in priority:
                        valid_suggestions = ["search", "move_east", "move_north"]
                    else:
                        valid_suggestions = ["search", "move_east", "wait"]
                
                advice['action_suggestions'] = valid_suggestions[:5]
                
                # ✅ Log advice quality
                self.last_advice = advice
                self.advice_history.append(advice.get('strategy', 'Unknown'))
                
                print(f"[DEBUG] Parsed advice successfully: {len(valid_suggestions)} actions")
                return advice
                
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parse error: {e}")
                print(f"[ERROR] Failed to parse: {response[:200]}")
                return self._get_fallback_advice()
                
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_advice()
    
    def _get_fallback_advice(self):
        """✅ IMPROVED: Smarter fallback based on history"""
        # If we have recent successful advice, reuse similar strategy
        if self.advice_history:
            strategy = self.advice_history[-1]
        else:
            strategy = "cautious exploration"
        
        return {
            "immediate_priority": "continue safe exploration",
            "risk_assessment": "unknown situation, proceed carefully",
            "opportunities": "search for items and map layout",
            "strategy": strategy,
            "action_suggestions": ["search", "move_east", "move_north", "pickup", "wait"]
        }
    
    def update_advice_outcome(self, advice_id, reward):
        """✅ NEW: Track which advice patterns work well"""
        if self.last_advice:
            strategy = self.last_advice.get('strategy', 'unknown')
            self.advice_outcomes[strategy]['reward_sum'] += reward
            self.advice_outcomes[strategy]['count'] += 1
    
    def get_advice_statistics(self):
        """✅ NEW: Get statistics on advice effectiveness"""
        stats = {}
        for strategy, outcome in self.advice_outcomes.items():
            if outcome['count'] > 0:
                stats[strategy] = {
                    'avg_reward': outcome['reward_sum'] / outcome['count'],
                    'count': outcome['count']
                }
        return stats
    
    async def _call_llm_api(self, prompt):
        """Call Ollama API with timeout and error handling"""
        import aiohttp
        import asyncio
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': 'phi3:mini',
                        'prompt': prompt,
                        'stream': False,
                        'options': {
                            'temperature': 0.3,
                            'top_p': 0.9,
                            'num_predict': 300
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        print(f"[ERROR] API returned status {response.status}")
                        return ""
        except asyncio.TimeoutError:
            print("[ERROR] LLM API timeout")
            return ""
        except Exception as e:
            print(f"[ERROR] LLM API call failed: {e}")
            return ""


# %% [markdown]
# ## LLM Guided NetHack Agent
# 

# %%
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
    
    def save_llm_advice_log(self, filename):
        """Save LLM advice log to file"""
        with open(filename, 'w') as f:
            json.dump({
                'llm_advice_log': self.llm_advice_log,
                'total_calls': len(self.llm_advice_log),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"LLM advice log saved to: {filename}")


# %% [markdown]
# ## Causal Model Logger
# 

# %%
class CausalModelLogger:
    """
    Comprehensive logging system for building causal models of NetHack RL agent.
    Captures state transitions, actions, rewards, and LLM interventions.
    """
    
    def __init__(self, log_file_prefix="causal_log"):
        self.log_file_prefix = log_file_prefix
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Transition logs: (state_t, action_t, state_t+1, reward_t+1)
        self.transitions = []
        
        # Episode-level logs
        self.episodes = []
        self.current_episode = None
        
        # LLM intervention logs
        self.llm_interventions = []
        
        # Causal feature tracking
        self.feature_correlations = defaultdict(list)
        
        # Action outcome tracking for causal discovery
        self.action_outcomes = defaultdict(lambda: {
            'count': 0,
            'total_reward': 0.0,
            'total_shaped_reward': 0.0,
            'health_changes': [],
            'level_changes': [],
            'death_count': 0
        })
        
    def start_episode(self, episode_id, initial_obs):
        """Start logging a new episode"""
        self.current_episode = {
            'episode_id': episode_id,
            'start_time': datetime.now().isoformat(),
            'initial_state': self._extract_state_features(initial_obs),
            'steps': [],
            'llm_calls': [],
            'final_reward': 0.0,
            'final_shaped_reward': 0.0,
            'length': 0,
            'death': False
        }
    
    def log_step(self, step_data):
        """
        Log a single step with all relevant information.
        
        step_data should contain:
        - obs: current observation
        - action: action taken
        - action_name: human-readable action name
        - next_obs: next observation
        - reward: raw reward
        - shaped_reward: shaped reward
        - value_estimate: critic's value estimate
        - action_probs: full action probability distribution
        - llm_advice_active: whether LLM advice influenced this action
        - done: whether episode ended
        """
        if self.current_episode is None:
            raise ValueError("Must call start_episode() before logging steps")
        
        # Extract state features
        state_t = self._extract_state_features(step_data['obs'])
        state_t1 = self._extract_state_features(step_data['next_obs'])
        
        # Calculate state differences (causal effects)
        state_diff = self._calculate_state_diff(state_t, state_t1)
        
        # Build comprehensive step record
        step_record = {
            'step': len(self.current_episode['steps']),
            'timestamp': datetime.now().isoformat(),
            
            # State information
            'state_t': state_t,
            'state_t1': state_t1,
            'state_diff': state_diff,
            
            # Action information
            'action': step_data['action'],
            'action_name': step_data['action_name'],
            'action_probs': step_data.get('action_probs', []),
            'action_entropy': self._calculate_entropy(step_data.get('action_probs', [])),
            
            # Reward information
            'reward': float(step_data['reward']),
            'shaped_reward': float(step_data['shaped_reward']),
            'reward_shaping_delta': float(step_data['shaped_reward'] - step_data['reward']),
            
            # Value estimation
            'value_estimate': float(step_data.get('value_estimate', 0.0)),
            
            # LLM influence
            'llm_advice_active': step_data.get('llm_advice_active', False),
            'llm_guidance_weight': step_data.get('llm_guidance_weight', 0.0),
            
            # Episode status
            'done': step_data['done'],
            'death': state_t1.get('health', 1.0) <= 0
        }
        
        self.current_episode['steps'].append(step_record)
        
        # Add to transitions for causal analysis
        self.transitions.append({
            'state_t': state_t,
            'action': step_data['action'],
            'action_name': step_data['action_name'],
            'state_t1': state_t1,
            'reward': step_data['reward'],
            'shaped_reward': step_data['shaped_reward'],
            'llm_active': step_data.get('llm_advice_active', False)
        })
        
        # Update action outcome statistics
        self._update_action_outcomes(step_data['action_name'], step_record)
        
        # Track feature correlations
        self._track_correlations(step_record)
    
    def log_llm_intervention(self, step, advice, state_before, state_after=None):
        """Log when LLM provides strategic advice"""
        intervention = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'advice': advice,
            'state_before': self._extract_state_features(state_before),
            'immediate_priority': advice.get('immediate_priority', ''),
            'risk_assessment': advice.get('risk_assessment', ''),
            'suggested_actions': advice.get('action_suggestions', [])
        }
        
        if state_after:
            intervention['state_after'] = self._extract_state_features(state_after)
        
        self.llm_interventions.append(intervention)
        
        if self.current_episode:
            self.current_episode['llm_calls'].append(intervention)
    
    def end_episode(self, final_reward, final_shaped_reward, death=False):
        """Finalize episode logging"""
        if self.current_episode is None:
            return
        
        self.current_episode['end_time'] = datetime.now().isoformat()
        self.current_episode['final_reward'] = float(final_reward)
        self.current_episode['final_shaped_reward'] = float(final_shaped_reward)
        self.current_episode['length'] = len(self.current_episode['steps'])
        self.current_episode['death'] = death
        
        # Calculate episode-level causal metrics
        self.current_episode['causal_metrics'] = self._calculate_episode_causal_metrics()
        
        self.episodes.append(self.current_episode)
        self.current_episode = None
    
    def _extract_state_features(self, obs):
        """Extract key state features for causal analysis"""
        if isinstance(obs, tuple):
            obs = obs[0]
        
        if not isinstance(obs, dict):
            return {}
        
        stats = obs.get('blstats', np.zeros(26))
        glyphs = obs.get('glyphs', np.zeros((21, 79)))
        
        features = {
            # Health status
            'health': float(stats[0]) if len(stats) > 0 else 0.0,
            'max_health': float(stats[1]) if len(stats) > 1 else 1.0,
            'health_ratio': float(stats[0] / stats[1]) if len(stats) > 1 and stats[1] > 0 else 0.0,
            
            # Position
            'pos_x': float(stats[0]) if len(stats) > 0 else 0.0,
            'pos_y': float(stats[1]) if len(stats) > 1 else 0.0,
            
            # Character stats
            'level': int(stats[7]) if len(stats) > 7 else 1,
            'experience': int(stats[8]) if len(stats) > 8 else 0,
            'strength': int(stats[2]) if len(stats) > 2 else 10,
            'dexterity': int(stats[3]) if len(stats) > 3 else 10,
            
            # Dungeon depth
            'dungeon_level': int(stats[12]) if len(stats) > 12 else 1,
            
            # Environment complexity
            'unique_glyphs': int(len(np.unique(glyphs))),
            'empty_tiles': int(np.sum(glyphs == 0)),
            'wall_tiles': int(np.sum(glyphs == 2359)),
            
            # Spatial awareness (simplified)
            'nearby_entities': int(np.sum(glyphs > 2370)),  # Likely monsters/items
        }
        
        return features
    
    def _calculate_state_diff(self, state_t, state_t1):
        """Calculate differences between consecutive states"""
        diff = {}
        for key in state_t.keys():
            if key in state_t1 and isinstance(state_t[key], (int, float)):
                diff[f'delta_{key}'] = state_t1[key] - state_t[key]
        return diff
    
    def _calculate_entropy(self, probs):
        """Calculate entropy of action probability distribution"""
        if probs is None:
            return 0.0
        probs = np.array(probs)
        if probs.size == 0:
            return 0.0
        probs = probs[probs > 0]  # Remove zeros
        if probs.size == 0:
            return 0.0
        return float(-np.sum(probs * np.log(probs + 1e-10)))
    
    def _update_action_outcomes(self, action_name, step_record):
        """Update statistics for action outcomes"""
        outcome = self.action_outcomes[action_name]
        outcome['count'] += 1
        outcome['total_reward'] += step_record['reward']
        outcome['total_shaped_reward'] += step_record['shaped_reward']
        
        if 'delta_health' in step_record['state_diff']:
            outcome['health_changes'].append(step_record['state_diff']['delta_health'])
        
        if 'delta_level' in step_record['state_diff']:
            outcome['level_changes'].append(step_record['state_diff']['delta_level'])
        
        if step_record['death']:
            outcome['death_count'] += 1
    
    def _track_correlations(self, step_record):
        """Track correlations between features for causal discovery"""
        # Track reward correlations with state changes
        for key, value in step_record['state_diff'].items():
            self.feature_correlations[key].append({
                'reward': step_record['reward'],
                'value': value,
                'action': step_record['action_name']
            })
    
    def _calculate_episode_causal_metrics(self):
        """Calculate causal metrics for the episode"""
        if not self.current_episode or not self.current_episode['steps']:
            return {}
        
        steps = self.current_episode['steps']
        
        # Action distribution
        action_counts = defaultdict(int)
        for step in steps:
            action_counts[step['action_name']] += 1
        
        # LLM impact analysis
        llm_steps = [s for s in steps if s['llm_advice_active']]
        non_llm_steps = [s for s in steps if not s['llm_advice_active']]
        
        llm_avg_reward = np.mean([s['reward'] for s in llm_steps]) if llm_steps else 0.0
        non_llm_avg_reward = np.mean([s['reward'] for s in non_llm_steps]) if non_llm_steps else 0.0
        
        # Health trajectory
        health_trajectory = [s['state_t']['health_ratio'] for s in steps]
        health_volatility = np.std(health_trajectory) if len(health_trajectory) > 1 else 0.0
        
        return {
            'total_steps': len(steps),
            'unique_actions': len(action_counts),
            'action_distribution': dict(action_counts),
            'llm_step_count': len(llm_steps),
            'llm_step_ratio': len(llm_steps) / len(steps) if steps else 0.0,
            'llm_avg_reward': float(llm_avg_reward),
            'non_llm_avg_reward': float(non_llm_avg_reward),
            'llm_reward_advantage': float(llm_avg_reward - non_llm_avg_reward),
            'health_volatility': float(health_volatility),
            'final_health_ratio': float(steps[-1]['state_t1']['health_ratio']) if steps else 0.0,
            'avg_action_entropy': float(np.mean([s['action_entropy'] for s in steps])),
        }
    
    def save_logs(self):
        """Save all logs to disk"""
        base_filename = f"{self.log_file_prefix}_{self.timestamp}"
        
        # Save episodes
        with open(f"{base_filename}_episodes.json", 'w') as f:
            json.dump({
                'episodes': self.episodes,
                'total_episodes': len(self.episodes),
                'timestamp': self.timestamp
            }, f, indent=2)
        
        # Save transitions (for causal discovery algorithms)
        with open(f"{base_filename}_transitions.json", 'w') as f:
            json.dump({
                'transitions': self.transitions,
                'total_transitions': len(self.transitions)
            }, f, indent=2)
        
        # Save LLM interventions
        with open(f"{base_filename}_llm_interventions.json", 'w') as f:
            json.dump({
                'interventions': self.llm_interventions,
                'total_interventions': len(self.llm_interventions)
            }, f, indent=2)
        
        # Save action outcome statistics
        action_stats = {}
        for action, stats in self.action_outcomes.items():
            action_stats[action] = {
                'count': stats['count'],
                'avg_reward': stats['total_reward'] / stats['count'] if stats['count'] > 0 else 0.0,
                'avg_shaped_reward': stats['total_shaped_reward'] / stats['count'] if stats['count'] > 0 else 0.0,
                'avg_health_change': float(np.mean(stats['health_changes'])) if stats['health_changes'] else 0.0,
                'death_rate': stats['death_count'] / stats['count'] if stats['count'] > 0 else 0.0
            }
        
        with open(f"{base_filename}_action_statistics.json", 'w') as f:
            json.dump(action_stats, f, indent=2)
        
        # Save feature correlations for causal discovery
        correlation_summary = {}
        for feature, data_points in self.feature_correlations.items():
            if len(data_points) > 1:
                rewards = [d['reward'] for d in data_points]
                values = [d['value'] for d in data_points]
                correlation = np.corrcoef(rewards, values)[0, 1] if len(rewards) > 1 else 0.0
                correlation_summary[feature] = {
                    'correlation_with_reward': float(correlation),
                    'sample_size': len(data_points),
                    'mean_value': float(np.mean(values)),
                    'std_value': float(np.std(values))
                }
        
        with open(f"{base_filename}_correlations.json", 'w') as f:
            json.dump(correlation_summary, f, indent=2)
        
        print(f"\nCausal logs saved with prefix: {base_filename}")
        print(f"  - Episodes: {len(self.episodes)}")
        print(f"  - Transitions: {len(self.transitions)}")
        print(f"  - LLM Interventions: {len(self.llm_interventions)}")
        print(f"  - Actions tracked: {len(self.action_outcomes)}")
    
    def generate_causal_graph_data(self):
        """
        Generate data structure suitable for causal graph construction.
        Returns nodes and edges that can be used with causal discovery algorithms.
        """
        # Define causal variables (nodes)
        nodes = [
            # State variables
            'health_ratio', 'level', 'experience', 'dungeon_level',
            'unique_glyphs', 'nearby_entities',
            
            # Action variable
            'action',
            
            # LLM intervention
            'llm_active',
            
            # Outcomes
            'reward', 'shaped_reward', 'health_change', 'death'
        ]
        
        # Collect data for each variable across all transitions
        data_matrix = []
        for trans in self.transitions:
            row = [
                trans['state_t'].get('health_ratio', 0.0),
                trans['state_t'].get('level', 0),
                trans['state_t'].get('experience', 0),
                trans['state_t'].get('dungeon_level', 0),
                trans['state_t'].get('unique_glyphs', 0),
                trans['state_t'].get('nearby_entities', 0),
                trans['action'],
                1 if trans['llm_active'] else 0,
                trans['reward'],
                trans['shaped_reward'],
                trans['state_t1'].get('health', 0) - trans['state_t'].get('health', 0),
                1 if trans['state_t1'].get('health', 1) <= 0 else 0
            ]
            data_matrix.append(row)
        
        return {
            'nodes': nodes,
            'data_matrix': data_matrix,
            'variable_descriptions': {
                'health_ratio': 'Current health as ratio of maximum',
                'level': 'Character level',
                'experience': 'Experience points',
                'dungeon_level': 'Depth in dungeon',
                'unique_glyphs': 'Environmental complexity',
                'nearby_entities': 'Number of nearby entities',
                'action': 'Action taken (encoded)',
                'llm_active': 'Whether LLM advice was active',
                'reward': 'Raw environment reward',
                'shaped_reward': 'Reward after shaping',
                'health_change': 'Change in health',
                'death': 'Whether agent died'
            }
        }




# %% [markdown]
# ## Training Monitor
# 

# %%
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
    
    def print_step(self, step, action_name, reward, shaped_reward, current_hp, max_hp):
        """Print step information in the format expected by the training loop"""
        # Track action
        self.current_episode_actions.append(action_name)
        
        # Color code reward
        if reward > 0:
            reward_color = self.GREEN
        elif reward < 0:
            reward_color = self.RED
        else:
            reward_color = self.ENDC
        
        # Calculate health ratio for color coding
        health_ratio = current_hp / max_hp if max_hp > 0 else 0.0
        
        # Color code health
        if health_ratio < 0.3:
            health_color = self.RED
        elif health_ratio < 0.6:
            health_color = self.YELLOW
        else:
            health_color = self.GREEN
        
        print(f"Step {step:4d} | "
              f"Action: {action_name:15s} | "
              f"R: {reward_color}{reward:6.2f}{self.ENDC} | "
              f"SR: {shaped_reward:7.3f} | "
              f"HP: {health_color}{current_hp:3d}/{max_hp:3d}{self.ENDC} | "
              f"Ratio: {health_color}{health_ratio*100:5.1f}%{self.ENDC}")
    
    def print_episode_end(self, episode_reward, episode_shaped_reward, episode_length, llm_call_count):
        """Print episode end summary"""
        print(f"\n{self.BOLD}{self.GREEN}{'─'*80}")
        print(f"Episode Complete")
        print(f"{'─'*80}{self.ENDC}\n")
        
        # Episode metrics
        print(f"{self.BOLD}Episode Results:{self.ENDC}")
        print(f"  Raw Reward:       {episode_reward:8.2f}")
        print(f"  Shaped Reward:    {episode_shaped_reward:8.2f}")
        print(f"  Length:           {episode_length:8d} steps")
        print(f"  LLM Calls:        {llm_call_count:8d}")
        
        # Action distribution for this episode
        if self.current_episode_actions:
            action_counts = defaultdict(int)
            for action in self.current_episode_actions:
                action_counts[action] += 1
            
            # Show top 3 actions for this episode
            print(f"\n{self.BOLD}Top Actions This Episode:{self.ENDC}")
            top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for action, count in top_actions:
                pct = (count / len(self.current_episode_actions)) * 100
                print(f"  {action:15s}: {count:4d} times ({pct:5.1f}%)")
        
        # Store episode data for tracking
        episode_data = {
            'raw_reward': episode_reward,
            'shaped_reward': episode_shaped_reward,
            'length': episode_length,
            'llm_calls': llm_call_count
        }
        self.episode_data.append(episode_data)
    
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


# %% [markdown]
# ## Monitored LLM Guided NetHack Agent
# 

# %%
class MonitoredLLMGuidedNetHackAgent(LLMGuidedNetHackAgent):
    """LLM-Guided agent with integrated monitoring"""
    
    def __init__(self, *args, baseline_metrics_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = TrainingMonitor(
            agent_name="LLM-Guided PPO",
            baseline_metrics_path=baseline_metrics_path
        )
        # ADD THIS:
        self.causal_logger = CausalModelLogger(log_file_prefix="llm_guided_causal")
        
        self.action_meanings = {
            0: "move_north", 1: "move_south", 2: "move_east", 3: "move_west",
            4: "move_northeast", 5: "move_northwest", 6: "move_southeast", 7: "move_southwest",
            8: "wait", 9: "pickup", 10: "drop", 11: "search", 12: "open_door",
            13: "close_door", 14: "kick", 15: "eat", 16: "drink", 17: "read",
            18: "apply", 19: "throw", 20: "wear", 21: "take_off", 22: "wield"
        }
        
        # ✅ ADD: Track advice effectiveness
        self.advice_tracker = {
            'total_advice': 0,
            'advice_followed': 0,
            'advice_rewards': [],
            'no_advice_rewards': [],
            'advice_actions': defaultdict(int),
            'advice_outcomes': defaultdict(list)
        }
    
    async def train_episode_monitored(self, env, episode):
        """Train episode with monitoring"""
        self.monitor.print_episode_start(episode)
        
        # FIX: Get observation BEFORE trying to use it
        obs = env.reset()
        self.causal_logger.start_episode(episode, obs)  # NOW START LOGGING
        
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
            llm_calls_before = len(self.llm_advisor.advice_history)
            
            tensor_obs, processed_obs = self.process_observation(obs)
            action, log_prob, value = await self.select_action(obs, processed_obs, reset_hidden)
            reset_hidden = False
            
            # Check if LLM was called during select_action
            llm_was_called = len(self.llm_advisor.advice_history) > llm_calls_before
            
            # Store action
            self.last_action = action
            action_name = self.action_meanings.get(action, f"action_{action}")
            
            # ✅ ADD: Track if action matched LLM advice
            if self.current_llm_advice:
                suggestions = self.current_llm_advice.get('action_suggestions', [])
                # Check if action matches any suggestion (exact or partial)
                action_followed = action_name in suggestions or \
                                 any(sug in action_name or action_name in sug for sug in suggestions)
                
                if action_followed:
                    self.advice_tracker['advice_followed'] += 1
                
                self.advice_tracker['total_advice'] += 1
                self.advice_tracker['advice_actions'][action_name] += 1
            
            # Display and log LLM advice if it was just received
            if llm_was_called and self.current_llm_advice:
                self.monitor.print_llm_advice(self.current_llm_advice, episode_length)
                self.causal_logger.log_llm_intervention(
                    episode_length, 
                    self.current_llm_advice, 
                    obs
                )
                llm_call_count += 1
            
            # Take environment step
            step_result = env.step(action)
            
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            # Apply reward shaping
            shaped_reward = self.reward_shaper.shape_reward(next_obs, reward, done, info)
            
            # ✅ ADD: Track reward by advice status
            if self.current_llm_advice:
                suggestions = self.current_llm_advice.get('action_suggestions', [])
                action_followed = action_name in suggestions or \
                                 any(sug in action_name or action_name in sug for sug in suggestions)
                
                if action_followed:
                    self.advice_tracker['advice_rewards'].append(shaped_reward)
                    self.advice_tracker['advice_outcomes'][action_name].append(shaped_reward)
                else:
                    self.advice_tracker['no_advice_rewards'].append(shaped_reward)
            
            # Get health for display
            if isinstance(next_obs, tuple):
                next_obs_dict = next_obs[0]
            else:
                next_obs_dict = next_obs
            
            stats = next_obs_dict.get('blstats', np.zeros(26))
            current_hp = int(stats[0]) if len(stats) > 0 else 0
            max_hp = int(stats[1]) if len(stats) > 1 else 1
            
            # Display step
            self.monitor.print_step(
                episode_length, action_name, reward, shaped_reward, current_hp, max_hp
            )
            
            # LOG CAUSAL DATA
            action_probs = F.softmax(self.actor(tensor_obs, llm_advice=self.current_llm_advice), dim=-1)
            self.causal_logger.log_step({
                'obs': obs,
                'action': action,
                'action_name': action_name,
                'next_obs': next_obs,
                'reward': reward,
                'shaped_reward': shaped_reward,
                'value_estimate': value,
                'action_probs': action_probs.detach().cpu().numpy()[0],
                'llm_advice_active': self.current_llm_advice is not None,
                'llm_guidance_weight': self.actor.llm_guidance_weight if self.current_llm_advice else 0.0,
                'done': done
            })
            
            # Store in buffer
            processed_obs_for_buffer = {}
            for key, tensor_val in tensor_obs.items():
                processed_obs_for_buffer[key] = tensor_val.squeeze(0).cpu()
            
            self.buffer.add(
                processed_obs_for_buffer,
                action,
                shaped_reward,
                value,
                log_prob,
                done
            )
            
            episode_reward += reward
            episode_shaped_reward += shaped_reward
            episode_length += 1
            
            # Update observation
            obs = next_obs
            
            if done:
                break
        
        # END EPISODE LOGGING
        self.causal_logger.end_episode(episode_reward, episode_shaped_reward, episode_length)
        
        self.monitor.print_episode_end(
            episode_reward, episode_shaped_reward, episode_length, llm_call_count
        )
        
        # ✅ ADD: Print advice effectiveness every 20 episodes
        if (episode + 1) % 20 == 0:
            self.print_advice_effectiveness()
        
        return episode_reward, episode_shaped_reward, episode_length
    
    def print_advice_effectiveness(self):
        """✅ NEW: Print how well LLM advice is working"""
        if self.advice_tracker['total_advice'] == 0:
            print(f"\n{self.monitor.YELLOW}No LLM advice given yet{self.monitor.ENDC}")
            return
        
        follow_rate = self.advice_tracker['advice_followed'] / self.advice_tracker['total_advice']
        
        avg_reward_followed = np.mean(self.advice_tracker['advice_rewards']) \
            if self.advice_tracker['advice_rewards'] else 0.0
        avg_reward_not_followed = np.mean(self.advice_tracker['no_advice_rewards']) \
            if self.advice_tracker['no_advice_rewards'] else 0.0
        
        print(f"\n{self.monitor.BOLD}{self.monitor.CYAN}╔══════════════════════════════════════════════════╗{self.monitor.ENDC}")
        print(f"{self.monitor.BOLD}{self.monitor.CYAN}║       LLM ADVICE EFFECTIVENESS REPORT          ║{self.monitor.ENDC}")
        print(f"{self.monitor.BOLD}{self.monitor.CYAN}╚══════════════════════════════════════════════════╝{self.monitor.ENDC}")
        
        print(f"\n{self.monitor.BOLD}Advice Usage:{self.monitor.ENDC}")
        print(f"  Total Steps with Advice:  {self.advice_tracker['total_advice']}")
        print(f"  Actions Followed:         {self.advice_tracker['advice_followed']}")
        print(f"  Follow Rate:              {follow_rate*100:.1f}%")
        
        print(f"\n{self.monitor.BOLD}Reward Comparison:{self.monitor.ENDC}")
        print(f"  Avg Reward (followed):    {avg_reward_followed:8.3f}")
        print(f"  Avg Reward (ignored):     {avg_reward_not_followed:8.3f}")
        
        advantage = avg_reward_followed - avg_reward_not_followed
        if advantage > 0:
            print(f"  {self.monitor.GREEN}Advantage: +{advantage:.3f} (LLM advice helps!){self.monitor.ENDC}")
        elif advantage < 0:
            print(f"  {self.monitor.YELLOW}Advantage: {advantage:.3f} (agent knows better){self.monitor.ENDC}")
        else:
            print(f"  Advantage: {advantage:.3f} (neutral)")
        
        # Top actions taken with advice
        if self.advice_tracker['advice_actions']:
            print(f"\n{self.monitor.BOLD}Most Common Advised Actions:{self.monitor.ENDC}")
            sorted_actions = sorted(
                self.advice_tracker['advice_actions'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            for action_name, count in sorted_actions:
                avg_outcome = np.mean(self.advice_tracker['advice_outcomes'][action_name]) \
                    if self.advice_tracker['advice_outcomes'][action_name] else 0.0
                print(f"  {action_name:15s}: {count:4d} times (avg reward: {avg_outcome:6.3f})")
        
        # LLM advisor statistics
        advisor_stats = self.llm_advisor.get_advice_statistics()
        if advisor_stats:
            print(f"\n{self.monitor.BOLD}Strategy Effectiveness:{self.monitor.ENDC}")
            sorted_strategies = sorted(
                advisor_stats.items(), 
                key=lambda x: x[1]['avg_reward'], 
                reverse=True
            )[:3]
            for strategy, stats in sorted_strategies:
                print(f"  '{strategy[:40]}': avg {stats['avg_reward']:.3f} ({stats['count']} uses)")
        
        print(f"{self.monitor.CYAN}{'─' * 50}{self.monitor.ENDC}\n")


# %% [markdown]
# # Main Training Loop
# 

# %%
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
        llm_guidance_weight=0.5,
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
    
    # Save model and advice log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.save_llm_advice_log(f"llm_advice_log_{timestamp}.json")

    agent.monitor.print_final_summary()
    agent.causal_logger.save_logs()  # SAVE CAUSAL LOGS
    
    # Generate causal graph data
    causal_graph_data = agent.causal_logger.generate_causal_graph_data()
    with open(f"causal_graph_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(causal_graph_data, f, indent=2)
    
    
    env.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_monitored())


