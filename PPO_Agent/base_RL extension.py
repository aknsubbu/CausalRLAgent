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
import time

# ========================================
# BASE RL COMPONENTS (FROM YOUR WORKING AGENT)
# ========================================

class NetHackRewardShaper:
    """Use EXACT reward shaping from your working base RL"""
    
    def __init__(self):
        self.previous_stats = None
        self.previous_glyphs = None
        self.visited_positions = set()
        self.last_position = None
        self.stuck_counter = 0
        self.max_stuck = 10
        
        # EXACT weights from base RL
        self.exploration_reward = 0.01
        self.health_reward = 0.001
        self.level_reward = 1.0
        self.experience_reward = 0.0001
        self.death_penalty = -1.0
        self.stuck_penalty = -0.01
        self.item_pickup_reward = 0.05
        self.monster_kill_reward = 0.1
        
    def shape_reward(self, obs, raw_reward, done, info):
        """EXACT reward shaping from base RL"""
        shaped_reward = raw_reward
        
        if isinstance(obs, tuple):
            obs = obs[0]
            
        current_stats = obs.get('blstats', np.zeros(26))
        current_glyphs = obs.get('glyphs', np.zeros((21, 79)))
        
        if self.previous_stats is not None:
            # Health change
            health_diff = current_stats[0] - self.previous_stats[0]
            shaped_reward += health_diff * self.health_reward
            
            # Level up
            level_diff = current_stats[7] - self.previous_stats[7]
            shaped_reward += level_diff * self.level_reward
            
            # Experience gain
            exp_diff = current_stats[8] - self.previous_stats[8]
            shaped_reward += exp_diff * self.experience_reward
            
            # Item pickup
            inv_change = np.sum(current_glyphs > 0) - np.sum(self.previous_glyphs > 0)
            if inv_change > 0:
                shaped_reward += self.item_pickup_reward
        
        # Exploration
        current_pos = (current_stats[0], current_stats[1]) if len(current_stats) > 1 else (0, 0)
        if current_pos not in self.visited_positions:
            self.visited_positions.add(current_pos)
            shaped_reward += self.exploration_reward
        
        # Anti-stuck
        if current_pos == self.last_position:
            self.stuck_counter += 1
            if self.stuck_counter > self.max_stuck:
                shaped_reward += self.stuck_penalty
        else:
            self.stuck_counter = 0
        
        # Death penalty
        if done and current_stats[0] <= 0:
            shaped_reward += self.death_penalty
        
        self.previous_stats = current_stats.copy()
        self.previous_glyphs = current_glyphs.copy()
        self.last_position = current_pos
        
        return shaped_reward
    
    def reset(self):
        self.previous_stats = None
        self.previous_glyphs = None
        self.visited_positions.clear()
        self.last_position = None
        self.stuck_counter = 0


class NetHackObservationProcessor:
    """Use EXACT observation processing from base RL"""
    
    def __init__(self):
        self.glyph_shape = (21, 79)
        self.stats_dim = 26
        self.message_dim = 256
        self.inventory_dim = 55
        
        self.position_history = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        
    def process_observation(self, obs, last_action=None):
        """EXACT processing from base RL"""
        processed = {}
        
        if isinstance(obs, tuple):
            obs = obs[0]
        
        if not isinstance(obs, dict):
            raise ValueError(f"Expected dict observation, got {type(obs)}")
        
        # Glyphs
        if 'glyphs' in obs:
            glyphs = np.array(obs['glyphs']).astype(np.float32) / 5976.0
            processed['glyphs'] = glyphs
        else:
            processed['glyphs'] = np.zeros(self.glyph_shape, dtype=np.float32)
        
        # Stats
        if 'blstats' in obs:
            stats = np.array(obs['blstats']).astype(np.float32)
            stats_normalized = stats.copy()
            
            if len(stats) > 1 and stats[1] > 0:
                stats_normalized[0] = stats[0] / stats[1]  # HP ratio
            if len(stats) > 7:
                stats_normalized[7] = min(stats[7] / 30.0, 1.0)  # Level
                
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
        
        # Message
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
        
        # Inventory
        if 'inv_strs' in obs:
            inventory = obs['inv_strs']
            inv_features = np.zeros(self.inventory_dim, dtype=np.float32)
            for i, item in enumerate(inventory):
                if i < len(inv_features):
                    try:
                        if isinstance(item, np.ndarray):
                            if item.dtype.kind in ('U', 'S', 'O'):
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
                    except:
                        continue
            processed['inventory'] = inv_features
        else:
            processed['inventory'] = np.zeros(self.inventory_dim, dtype=np.float32)
        
        # Action history
        if last_action is not None:
            self.action_history.append(last_action)
        
        action_hist_vector = np.zeros(50, dtype=np.float32)
        for i, action in enumerate(list(self.action_history)[-50:]):
            action_hist_vector[i] = action / 23.0
        processed['action_history'] = action_hist_vector
        
        return processed


class RecurrentNetHackCNN(nn.Module):
    """EXACT CNN from base RL"""
    
    def __init__(self, input_shape=(21, 79), cnn_output_dim=512, lstm_hidden_dim=256):
        super(RecurrentNetHackCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        conv_out_size = self._get_conv_out_size(input_shape)
        self.cnn_fc = nn.Linear(conv_out_size, cnn_output_dim)
        
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(cnn_output_dim, lstm_hidden_dim, batch_first=True)
        
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
        
        if reset_hidden or self.hidden_state is None or self.hidden_state[0].size(1) != batch_size:
            self.hidden_state = (
                torch.zeros(1, batch_size, self.lstm_hidden_dim, device=x.device),
                torch.zeros(1, batch_size, self.lstm_hidden_dim, device=x.device)
            )
        
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        cnn_features = F.relu(self.cnn_fc(x))
        
        cnn_features = cnn_features.unsqueeze(1)
        lstm_out, self.hidden_state = self.lstm(cnn_features, self.hidden_state)
        lstm_features = lstm_out.squeeze(1)
        
        return lstm_features
    
    def reset_hidden_state(self):
        self.hidden_state = None


# ========================================
# LLM-ENHANCED ACTOR (MINIMAL MODIFICATION)
# ========================================

class LLMEnhancedPPOActor(nn.Module):
    """Base RL Actor with MINIMAL LLM guidance layer"""
    
    def __init__(self, action_dim=23):
        super(LLMEnhancedPPOActor, self).__init__()
        
        # EXACT architecture from base RL
        self.glyph_cnn = RecurrentNetHackCNN(cnn_output_dim=512, lstm_hidden_dim=256)
        self.stats_lstm = nn.LSTM(26, 64, batch_first=True)
        self.message_fc = nn.Linear(256, 128)
        self.inventory_fc = nn.Linear(55, 64)
        self.action_hist_fc = nn.Linear(50, 32)
        
        # Combined processing - EXACT from base RL
        combined_dim = 256 + 64 + 128 + 64 + 32  # 544
        self.combined_fc1 = nn.Linear(combined_dim, 512)
        self.combined_fc2 = nn.Linear(512, 256)
        
        # Action head - EXACT from base RL
        self.action_head = nn.Linear(256, action_dim)
        
        # NEW: Tiny LLM guidance layer (optional)
        self.llm_guidance_fc = nn.Linear(action_dim, action_dim)  # Learn to use LLM hints
        
        # Hidden states
        self.stats_hidden = None
        
        # LLM guidance settings - START MINIMAL
        self.llm_guidance_weight = 0.0  # Start with NO guidance
        self.use_llm = False
        
    def forward(self, obs, reset_hidden=False, llm_hints=None):
        batch_size = obs['glyphs'].size(0)
        
        # EXACT forward pass from base RL
        glyph_features = self.glyph_cnn(obs['glyphs'], reset_hidden)
        
        if reset_hidden or self.stats_hidden is None or self.stats_hidden[0].size(1) != batch_size:
            self.stats_hidden = (
                torch.zeros(1, batch_size, 64, device=obs['stats'].device),
                torch.zeros(1, batch_size, 64, device=obs['stats'].device)
            )
        
        stats_input = obs['stats'].unsqueeze(1)
        stats_lstm_out, self.stats_hidden = self.stats_lstm(stats_input, self.stats_hidden)
        stats_features = stats_lstm_out.squeeze(1)
        
        message_features = F.relu(self.message_fc(obs['message']))
        inventory_features = F.relu(self.inventory_fc(obs['inventory']))
        action_hist_features = F.relu(self.action_hist_fc(obs['action_history']))
        
        combined = torch.cat([
            glyph_features, stats_features,
            message_features, inventory_features, action_hist_features
        ], dim=1)
        
        x = F.relu(self.combined_fc1(combined))
        x = F.relu(self.combined_fc2(x))
        
        # Base policy logits
        base_logits = self.action_head(x)
        
        # OPTIONAL: Add minimal LLM guidance if enabled
        if self.use_llm and llm_hints is not None and self.llm_guidance_weight > 0:
            # LLM hints are soft suggestions, not hard constraints
            llm_bias = torch.FloatTensor(llm_hints).to(base_logits.device)
            if batch_size > 1:
                llm_bias = llm_bias.unsqueeze(0).expand(batch_size, -1)
            
            # Learn to use LLM hints through trainable layer
            learned_guidance = self.llm_guidance_fc(llm_bias)
            
            # Very small guidance weight
            guided_logits = base_logits + self.llm_guidance_weight * learned_guidance
            return guided_logits
        
        return base_logits
    
    def reset_hidden_states(self):
        self.glyph_cnn.reset_hidden_state()
        self.stats_hidden = None


class RecurrentPPOCritic(nn.Module):
    """EXACT critic from base RL - NO CHANGES"""
    
    def __init__(self):
        super(RecurrentPPOCritic, self).__init__()
        
        self.glyph_cnn = RecurrentNetHackCNN(cnn_output_dim=512, lstm_hidden_dim=256)
        self.stats_lstm = nn.LSTM(26, 64, batch_first=True)
        self.message_fc = nn.Linear(256, 128)
        self.inventory_fc = nn.Linear(55, 64)
        self.action_hist_fc = nn.Linear(50, 32)
        
        combined_dim = 256 + 64 + 128 + 64 + 32
        self.combined_fc1 = nn.Linear(combined_dim, 512)
        self.combined_fc2 = nn.Linear(512, 256)
        
        self.value_head = nn.Linear(256, 1)
        
        self.stats_hidden = None
        
    def forward(self, obs, reset_hidden=False):
        batch_size = obs['glyphs'].size(0)
        
        glyph_features = self.glyph_cnn(obs['glyphs'], reset_hidden)
        
        if reset_hidden or self.stats_hidden is None or self.stats_hidden[0].size(1) != batch_size:
            self.stats_hidden = (
                torch.zeros(1, batch_size, 64, device=obs['stats'].device),
                torch.zeros(1, batch_size, 64, device=obs['stats'].device)
            )
        
        stats_input = obs['stats'].unsqueeze(1)
        stats_lstm_out, self.stats_hidden = self.stats_lstm(stats_input, self.stats_hidden)
        stats_features = stats_lstm_out.squeeze(1)
        
        message_features = F.relu(self.message_fc(obs['message']))
        inventory_features = F.relu(self.inventory_fc(obs['inventory']))
        action_hist_features = F.relu(self.action_hist_fc(obs['action_history']))
        
        combined = torch.cat([
            glyph_features, stats_features,
            message_features, inventory_features, action_hist_features
        ], dim=1)
        
        x = F.relu(self.combined_fc1(combined))
        x = F.relu(self.combined_fc2(x))
        
        value = self.value_head(x)
        return value
    
    def reset_hidden_states(self):
        self.glyph_cnn.reset_hidden_state()
        self.stats_hidden = None


class PPOBuffer:
    """EXACT buffer from base RL - NO CHANGES"""
    
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


# ========================================
# LLM ADVISOR WITH REAL API CALLS
# ========================================

# ========================================
# ENHANCED LLM ADVISOR WITH YOUR DESCRIPTOR
# ========================================

class EnhancedLLMAdvisor:
    """LLM advisor using comprehensive NetHack semantic description"""
    
    def __init__(self, call_frequency=50):
        self.call_frequency = call_frequency
        self.step_count = 0
        self.last_advice = None
        self.semantic_descriptor = NetHackSemanticDescriptor()
        
        # Action categories for hints
        self.action_categories = {
            'explore': [0, 1, 2, 3, 4, 5, 6, 7, 11],  # moves + search
            'combat': [14],  # kick
            'collect': [9],  # pickup
            'retreat': [0, 1, 2, 3],  # move away
            'wait': [8],  # wait/rest
        }
        
        # Track recent actions to detect loops
        self.recent_actions = deque(maxlen=20)
        
    def should_call_llm(self, performance_metrics):
        """Call LLM when struggling OR every N steps"""
        self.step_count += 1
        
        if self.step_count % self.call_frequency != 0:
            return False
        
        # Call if performance is poor OR periodically for guidance
        avg_reward = performance_metrics.get('avg_reward', 0)
        return avg_reward < 5.0 or self.step_count % (self.call_frequency * 2) == 0
    
    async def get_strategic_advice(self, raw_obs, processed_obs, performance):
        """Get strategic advice using full semantic description"""
        try:
            # Generate comprehensive description
            full_description = self.semantic_descriptor.generate_full_description(
                raw_obs, processed_obs, self.recent_actions
            )
            
            # Call LLM with rich context
            llm_response = await self._call_ollama_api(full_description, performance)
            
            # Parse response
            strategy = self._parse_strategic_response(llm_response, raw_obs)
            
            # Convert to action hints
            hints = self._strategy_to_hints(strategy, raw_obs)
            
            print(f"🤖 LLM Strategy: {strategy}")
            
            return hints
            
        except Exception as e:
            print(f"⚠️ LLM advisor error: {e}")
            return self._fallback_advice(raw_obs)
    
    async def _call_ollama_api(self, semantic_description, performance):
        """Call Ollama with rich semantic description"""
        import aiohttp
        import asyncio
        
        prompt = f"""You are an expert NetHack strategic advisor.

{semantic_description}

RECENT PERFORMANCE:
- Average Reward: {performance.get('avg_reward', 0):.2f}
- Average Survival: {performance.get('avg_length', 0):.0f} steps

STRATEGIC ANALYSIS:
Based on the game state above, choose ONE primary strategy:

1. "explore" - No immediate threats, safe to move and search for stairs/items
2. "combat" - Monster nearby AND health is good (>60%), engage in combat
3. "retreat" - Monster nearby BUT health is low (<40%), move away to safety
4. "collect" - Items nearby and safe, pick them up
5. "wait" - Critical health (<30%) or need to recover

CRITICAL RULES:
- If "CLOSEST THREAT" shows distance 1-2 AND health < 40%: Choose "retreat"
- If "CLOSEST THREAT" shows distance 1-2 AND health > 60%: Choose "combat"
- If "NO IMMEDIATE THREATS": Choose "explore" or "collect"
- If stuck warning present: Choose strategy that breaks the loop
- If health is "critical": Choose "wait" or "retreat"

Respond with ONLY ONE WORD from: explore, combat, retreat, collect, wait

Your strategic choice:"""

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower for more consistent decisions
                        "num_predict": 50
                    }
                }
                
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "").strip().lower()
                    else:
                        return ""
                        
        except Exception as e:
            print(f"⚠️ Ollama error: {e}")
            return ""
    
    def _parse_strategic_response(self, response, raw_obs):
        """Parse LLM response to extract strategy"""
        response_lower = response.lower()
        
        # Extract health for decision validation
        stats = self._get_stats(raw_obs)
        health_ratio = self._get_health_ratio(stats)
        
        # Look for strategy keywords
        if 'retreat' in response_lower or 'flee' in response_lower or 'escape' in response_lower:
            return 'retreat'
        
        elif 'combat' in response_lower or 'attack' in response_lower or 'fight' in response_lower:
            # Validate: only combat if healthy
            if health_ratio > 0.5:
                return 'combat'
            else:
                return 'retreat'  # Override bad advice
        
        elif 'collect' in response_lower or 'pickup' in response_lower or 'gather' in response_lower:
            return 'collect'
        
        elif 'wait' in response_lower or 'rest' in response_lower:
            return 'wait'
        
        elif 'explore' in response_lower or 'search' in response_lower or 'move' in response_lower:
            return 'explore'
        
        else:
            # Fallback: use game state
            return self._rule_based_strategy(stats, raw_obs)
    
    def _strategy_to_hints(self, strategy, raw_obs):
        """Convert strategy to action probability hints"""
        hints = np.zeros(23, dtype=np.float32)
        
        if strategy in self.action_categories:
            for action_id in self.action_categories[strategy]:
                hints[action_id] = 0.2  # 20% boost
        
        # Special case: if retreating, boost opposite directions
        if strategy == 'retreat':
            stats = self._get_stats(raw_obs)
            if len(stats) > 1:
                # Boost all movement actions equally for now
                for action_id in range(8):  # All 8 directions
                    hints[action_id] = 0.25
        
        return hints
    
    def _rule_based_strategy(self, stats, raw_obs):
        """Fallback rule-based strategy"""
        health_ratio = self._get_health_ratio(stats)
        
        if health_ratio < 0.3:
            return 'wait'
        elif health_ratio < 0.5:
            return 'retreat'
        elif health_ratio > 0.7:
            return 'explore'
        else:
            return 'collect'
    
    def _get_stats(self, raw_obs):
        """Extract stats from observation"""
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        return raw_obs.get('blstats', np.zeros(26)) if isinstance(raw_obs, dict) else np.zeros(26)
    
    def _get_health_ratio(self, stats):
        """Get health ratio from stats"""
        if len(stats) > 11:
            hp = float(stats[10])
            max_hp = float(stats[11])
            return hp / max_hp if max_hp > 0 else 0.5
        return 0.5
    
    def _fallback_advice(self, raw_obs):
        """Rule-based fallback"""
        stats = self._get_stats(raw_obs)
        strategy = self._rule_based_strategy(stats, raw_obs)
        return self._strategy_to_hints(strategy, raw_obs)
    
    def update_action_history(self, action):
        """Track recent actions for loop detection"""
        self.recent_actions.append(action)


# ========================================
# UPDATE LLM-ENHANCED AGENT TO USE ENHANCED ADVISOR
# ========================================

class LLMEnhancedNetHackAgent:
    """Base RL agent with enhanced LLM guidance"""
    
    def __init__(self, action_dim=23, learning_rate=1e-4, gamma=0.99, clip_ratio=0.2,
                 entropy_coef=0.02, value_coef=0.5, max_grad_norm=0.5, 
                 enable_llm=False, llm_guidance_weight=0.05):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.actor = LLMEnhancedPPOActor(action_dim=action_dim).to(self.device)
        self.critic = RecurrentPPOCritic().to(self.device)
        
        # LLM settings
        self.actor.use_llm = enable_llm
        self.actor.llm_guidance_weight = llm_guidance_weight if enable_llm else 0.0
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Components
        self.buffer = PPOBuffer()
        self.obs_processor = NetHackObservationProcessor()
        self.reward_shaper = NetHackRewardShaper()
        
        # NEW: Enhanced LLM advisor
        self.llm_advisor = EnhancedLLMAdvisor(call_frequency=50) if enable_llm else None
        self.current_llm_hints = None
        self.llm_call_count = 0
        
        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.shaped_rewards = deque(maxlen=100)
        self.last_action = None
        
    def process_observation(self, obs):
        """Process observation - returns both processed and raw"""
        processed = self.obs_processor.process_observation(obs, self.last_action)
        
        tensor_obs = {}
        for key, value in processed.items():
            tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
        
        # Also return processed dict for semantic descriptor
        return tensor_obs, processed, obs
    
    async def select_action(self, tensor_obs, processed_obs, raw_obs, 
                           reset_hidden=False, performance_metrics=None):
        """Action selection with enhanced LLM guidance"""
        
        # Get LLM hints if enabled
        if self.llm_advisor and performance_metrics:
            if self.llm_advisor.should_call_llm(performance_metrics):
                print(f"\n🤖 Calling LLM for strategic advice...")
                
                # Get comprehensive advice
                self.current_llm_hints = await self.llm_advisor.get_strategic_advice(
                    raw_obs, processed_obs, performance_metrics
                )
                self.llm_call_count += 1
                
                # Show which actions got boosted
                boosted_actions = np.where(self.current_llm_hints > 0)[0]
                if len(boosted_actions) > 0:
                    action_meanings = self.llm_advisor.semantic_descriptor.action_meanings
                    action_names = [action_meanings.get(a, str(a)) for a in boosted_actions[:5]]
                    print(f"   Suggested actions: {', '.join(action_names)}")
        
        # Update action history for loop detection
        if self.llm_advisor and self.last_action is not None:
            self.llm_advisor.update_action_history(self.last_action)
        
        # Select action
        with torch.no_grad():
            action_logits = self.actor(tensor_obs, reset_hidden, self.current_llm_hints)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.critic(tensor_obs, reset_hidden)
            
            return action.item(), log_prob.item(), value.item()
    
    def update(self, epochs=4, batch_size=64):
        """EXACT update from base RL"""
        if len(self.buffer) < batch_size:
            return {}
        
        self.buffer.compute_advantages(self.gamma)
        
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.tolist()
        
        actor_losses = []
        critic_losses = []
        entropies = []
        clip_fractions = []
        
        for _ in range(epochs):
            batch_obs, batch_actions, old_log_probs, batch_returns, batch_advantages = \
                self.buffer.get_batch(min(batch_size, len(self.buffer)))
            
            batch_obs = {k: v.to(self.device) for k, v in batch_obs.items()}
            batch_actions = batch_actions.to(self.device)
            old_log_probs = old_log_probs.to(self.device)
            batch_returns = batch_returns.to(self.device)
            batch_advantages = batch_advantages.to(self.device)
            
            self.actor.reset_hidden_states()
            self.critic.reset_hidden_states()
            
            # Actor update
            action_logits = self.actor(batch_obs, reset_hidden=True, llm_hints=None)
            action_dist = Categorical(logits=action_logits)
            new_log_probs = action_dist.log_prob(batch_actions)
            entropy = action_dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
            
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_ratio).float()).item()
            
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.entropy_coef * entropy
            actor_loss = policy_loss + entropy_loss
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Critic update
            values = self.critic(batch_obs, reset_hidden=True).squeeze()
            value_loss = F.mse_loss(values, batch_returns)
            critic_loss = self.value_coef * value_loss
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
            clip_fractions.append(clip_fraction)
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions),
        }
    
    async def train(self, env, num_episodes=100, update_freq=1024, print_freq=10):
        """Training loop with enhanced LLM"""
        step_count = 0
        
        print(f"🚀 Training for {num_episodes} episodes")
        print(f"   LLM Guidance: {'ENABLED (Enhanced Descriptor)' if self.actor.use_llm else 'DISABLED'}")
        if self.actor.use_llm:
            print(f"   LLM Guidance Weight: {self.actor.llm_guidance_weight:.3f}")
            print(f"   LLM Call Frequency: Every {self.llm_advisor.call_frequency} steps")
        print(f"   Update Frequency: {update_freq} steps")
        print()
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_shaped_reward = 0
            episode_length = 0
            
            # Reset states
            self.actor.reset_hidden_states()
            self.critic.reset_hidden_states()
            self.reward_shaper.reset()
            self.last_action = None
            self.current_llm_hints = None
            
            reset_hidden = True
            
            # Performance metrics
            performance_metrics = {
                'avg_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0,
                'avg_length': np.mean(list(self.episode_lengths)) if self.episode_lengths else 0,
            }
            
            while True:
                tensor_obs, processed_obs, raw_obs = self.process_observation(obs)
                
                action, log_prob, value = await self.select_action(
                    tensor_obs, processed_obs, raw_obs, reset_hidden, performance_metrics
                )
                reset_hidden = False
                
                self.last_action = action
                
                # Store for buffer
                processed_obs_for_buffer = {}
                for key, tensor_val in tensor_obs.items():
                    processed_obs_for_buffer[key] = tensor_val.squeeze(0).cpu()
                
                # Environment step
                step_result = env.step(action)
                
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                else:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                # Reward shaping
                shaped_reward = self.reward_shaper.shape_reward(next_obs, reward, done, info)
                
                self.buffer.add(processed_obs_for_buffer, action, shaped_reward, value, log_prob, done)
                
                obs = next_obs
                episode_reward += reward
                episode_shaped_reward += shaped_reward
                episode_length += 1
                step_count += 1
                
                # Update
                if step_count % update_freq == 0:
                    training_metrics = self.update()
                    if training_metrics:
                        print(f"  📊 Step {step_count}: "
                              f"Actor Loss: {training_metrics['actor_loss']:.4f}, "
                              f"Critic Loss: {training_metrics['critic_loss']:.4f}")
                    self.buffer.clear()
                
                if done:
                    break
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.shaped_rewards.append(episode_shaped_reward)
            self.episode_lengths.append(episode_length)
            
            # Print progress
            if episode % print_freq == 0:
                avg_reward = np.mean(list(self.episode_rewards)[-10:]) if len(self.episode_rewards) >= 10 else np.mean(list(self.episode_rewards))
                avg_length = np.mean(list(self.episode_lengths)[-10:]) if len(self.episode_lengths) >= 10 else np.mean(list(self.episode_lengths))
                
                llm_info = f", LLM Calls: {self.llm_call_count}" if self.actor.use_llm else ""
                print(f"📈 Episode {episode}: "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Avg Length: {avg_length:.1f}"
                      f"{llm_info}")
        
        if self.actor.use_llm:
            print(f"\n🤖 Total LLM Calls: {self.llm_call_count}")
        
        return list(self.episode_rewards), list(self.shaped_rewards)
    
    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)
        print(f"💾 Model saved to {path}")



# ========================================
# ENHANCED GAME STATE EXTRACTION
# ========================================

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
        
        # NetHack blstats structure (25 indices total):
        # Reference: https://nethackwiki.com/wiki/Blstats
        self.BLSTATS_INDEX = {
            'x': 0,           # X position
            'y': 1,           # Y position
            'strength': 2,    # Strength (can be > 18)
            'dexterity': 4,   # Dexterity
            'constitution': 5, # Constitution
            'intelligence': 6, # Intelligence
            'wisdom': 7,      # Wisdom
            'charisma': 8,    # Charisma
            'score': 9,       # Current score
            'hitpoints': 10,  # Current HP
            'max_hitpoints': 11, # Maximum HP
            'depth': 12,      # Dungeon depth/level
            'gold': 13,       # Gold
            'energy': 14,     # Current energy (for spells)
            'max_energy': 15, # Maximum energy
            'armor_class': 16, # Armor class (lower is better)
            'monster_level': 17, # Not used for player
            'experience_level': 18, # Character level
            'experience_points': 19, # Experience points
            'time': 20,       # Turn count
            'hunger': 21,     # Hunger status (0=satiated, 1=not hungry, 2=hungry, etc.)
            'encumbrance': 22, # Encumbrance level
            'dungeon_number': 23, # Which dungeon (Dungeons of Doom, Mines, etc.)
            'level_number': 24,   # Level within that dungeon
        }
    
    def describe_surroundings(self, glyphs, player_pos):
            """Enhanced description with distances"""
            h, w = glyphs.shape
            py, px = player_pos
            
            nearby_monsters = []
            nearby_items = []
            
            # Check 5x5 area (not just 3x3)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        glyph = glyphs[ny, nx]
                        symbol = self.glyph_to_symbol.get(glyph, "unknown")
                        distance = abs(dy) + abs(dx)  # Manhattan distance
                        
                        if symbol in ["kobold", "goblin", "orc", "troll"]:
                            direction = self._get_direction(dy, dx)
                            # ADD DISTANCE INFO
                            nearby_monsters.append(f"{symbol} {direction} (dist:{distance})")
                        elif symbol in ["gold", "weapon", "armor", "food", "potion"]:
                            direction = self._get_direction(dy, dx)
                            nearby_items.append(f"{symbol} {direction} (dist:{distance})")
            
            # Sort by distance
            nearby_monsters.sort(key=lambda x: int(x.split('dist:')[1].split(')')[0]))
            
            description = []
            if nearby_monsters:
                closest_monster = nearby_monsters[0]
                description.append(f"CLOSEST THREAT: {closest_monster}")
                if len(nearby_monsters) > 1:
                    description.append(f"Other threats: {', '.join(nearby_monsters[1:])}")
            else:
                description.append("NO IMMEDIATE THREATS - safe to explore")
                
            if nearby_items:
                description.append(f"Items nearby: {', '.join(nearby_items)}")
            
            return "; ".join(description) if description else "Empty area - keep exploring"
    
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
        """Describe player's current status using CORRECT blstats indices"""
        if len(stats) < 20:
            return "Status unknown"
        
        # CORRECT indices for NetHack blstats
        hp = int(stats[self.BLSTATS_INDEX['hitpoints']])          # Index 10
        max_hp = int(stats[self.BLSTATS_INDEX['max_hitpoints']])  # Index 11
        level = int(stats[self.BLSTATS_INDEX['experience_level']]) # Index 18
        experience = int(stats[self.BLSTATS_INDEX['experience_points']]) # Index 19
        depth = int(stats[self.BLSTATS_INDEX['depth']])           # Index 12
        gold = int(stats[self.BLSTATS_INDEX['gold']])             # Index 13
        
        hp_ratio = hp / max_hp if max_hp > 0 else 0
        health_status = "critical" if hp_ratio < 0.3 else "low" if hp_ratio < 0.6 else "good"
        
        return f"Level {level}, Health: {hp}/{max_hp} ({health_status}), XP: {experience}, Depth: {depth}, Gold: {gold}"
    
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
    
    def get_player_position(self, stats):
        """Get correct player position from blstats"""
        if len(stats) < 2:
            return (10, 39)  # Default fallback to center
        
        # CORRECT: X and Y are at indices 0 and 1
        x = int(stats[self.BLSTATS_INDEX['x']])  # Index 0
        y = int(stats[self.BLSTATS_INDEX['y']])  # Index 1
        
        return (y, x)  # Return as (row, col) for array indexing
    
    def generate_full_description(self, obs, processed_obs,recent_actions):
        """Generate complete semantic description of game state"""
        if isinstance(obs, tuple):
            obs = obs[0]
        
        glyphs = obs.get('glyphs', np.zeros((21, 79)))
        stats = obs.get('blstats', np.zeros(26))
        message = obs.get('message', np.zeros(256))

        stuck_warning = ""
        if recent_actions and len(recent_actions) >= 5:
            from collections import Counter
            
            # FIX: Convert to list explicitly before slicing
            recent_actions_list = list(recent_actions)[-10:]  # Get last 10 actions
            action_counts = Counter(recent_actions_list)
            most_common_action, count = action_counts.most_common(1)[0]
            
            if count >= 7:
                stuck_warning = f"\n\n🚨 CRITICAL WARNING: Agent is STUCK in a loop! Action '{most_common_action}' repeated {count}/10 times.\n💡 MUST suggest DIFFERENT actions (avoid '{most_common_action}')!"
        # Get CORRECT player position from blstats
        player_pos = self.get_player_position(stats)
        
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
{stuck_warning}

Current Situation: You are exploring a dungeon. Your goal is to survive, gain experience, collect items, and progress deeper.
        """.strip()
        
        return description


# ========================================
# UPDATE LLM-ENHANCED AGENT
# ========================================

class LLMEnhancedNetHackAgent:
    """Base RL agent with optional minimal LLM guidance"""
    
    def __init__(self, action_dim=23, learning_rate=1e-4, gamma=0.99, clip_ratio=0.2,
                 entropy_coef=0.02, value_coef=0.5, max_grad_norm=0.5, 
                 enable_llm=False, llm_guidance_weight=0.05):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.actor = LLMEnhancedPPOActor(action_dim=action_dim).to(self.device)
        self.critic = RecurrentPPOCritic().to(self.device)
        
        # LLM settings
        self.actor.use_llm = enable_llm
        self.actor.llm_guidance_weight = llm_guidance_weight if enable_llm else 0.0
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Components
        self.buffer = PPOBuffer()
        self.obs_processor = NetHackObservationProcessor()
        self.reward_shaper = NetHackRewardShaper()
        
        # NEW: LLM components
        self.llm_advisor = EnhancedLLMAdvisor(call_frequency=50) if enable_llm else None
        self.state_extractor = NetHackSemanticDescriptor() if enable_llm else None
        self.current_llm_hints = None
        self.llm_call_count = 0
        
        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.shaped_rewards = deque(maxlen=100)
        self.last_action = None
        
    def process_observation(self, obs):
        """EXACT same as base RL"""
        processed = self.obs_processor.process_observation(obs, self.last_action)
        
        tensor_obs = {}
        for key, value in processed.items():
            tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
        
        return tensor_obs, obs  # Return both processed and raw obs
    
    async def select_action(self, processed_obs, raw_obs, reset_hidden=False, performance_metrics=None):
        """Action selection with optional LLM hints"""
        
        # Get LLM hints if enabled and needed
        if self.llm_advisor and self.state_extractor and performance_metrics:
            if self.llm_advisor.should_call_llm(performance_metrics):
                # Extract game state
                game_state = self.state_extractor.extract_state(raw_obs)
                
                # Get LLM advice
                print(f"🤖 Calling LLM... (Health: {game_state['health_ratio']*100:.0f}%, "
                      f"Level: {game_state['level']}, "
                      f"Monsters: {'Yes' if game_state['nearby_monsters'] else 'No'})")
                
                self.current_llm_hints = await self.llm_advisor.get_simple_advice(
                    game_state, performance_metrics
                )
                self.llm_call_count += 1
                
                # Show which actions got boosted
                boosted_actions = np.where(self.current_llm_hints > 0)[0]
                if len(boosted_actions) > 0:
                    action_names = [self.llm_advisor.action_names.get(a, str(a)) 
                                  for a in boosted_actions if a in self.llm_advisor.action_names]
                    print(f"   LLM suggests: {', '.join(action_names)}")
        
        # Select action (with or without LLM hints)
        with torch.no_grad():
            action_logits = self.actor(processed_obs, reset_hidden, self.current_llm_hints)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.critic(processed_obs, reset_hidden)
            
            return action.item(), log_prob.item(), value.item()
    
    def update(self, epochs=4, batch_size=64):
        """EXACT update from base RL - NO CHANGES"""
        if len(self.buffer) < batch_size:
            return {}
        
        self.buffer.compute_advantages(self.gamma)
        
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.tolist()
        
        actor_losses = []
        critic_losses = []
        entropies = []
        clip_fractions = []
        
        for _ in range(epochs):
            batch_obs, batch_actions, old_log_probs, batch_returns, batch_advantages = \
                self.buffer.get_batch(min(batch_size, len(self.buffer)))
            
            batch_obs = {k: v.to(self.device) for k, v in batch_obs.items()}
            batch_actions = batch_actions.to(self.device)
            old_log_probs = old_log_probs.to(self.device)
            batch_returns = batch_returns.to(self.device)
            batch_advantages = batch_advantages.to(self.device)
            
            self.actor.reset_hidden_states()
            self.critic.reset_hidden_states()
            
            # Actor update (no LLM hints in training)
            action_logits = self.actor(batch_obs, reset_hidden=True, llm_hints=None)
            action_dist = Categorical(logits=action_logits)
            new_log_probs = action_dist.log_prob(batch_actions)
            entropy = action_dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
            
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_ratio).float()).item()
            
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.entropy_coef * entropy
            actor_loss = policy_loss + entropy_loss
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Critic update
            values = self.critic(batch_obs, reset_hidden=True).squeeze()
            value_loss = F.mse_loss(values, batch_returns)
            critic_loss = self.value_coef * value_loss
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
            clip_fractions.append(clip_fraction)
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions),
        }
    
    async def train(self, env, num_episodes=100, update_freq=1024, print_freq=10):
        """Training loop with LLM integration"""
        step_count = 0
        
        print(f"🚀 Training for {num_episodes} episodes")
        print(f"   LLM Guidance: {'ENABLED' if self.actor.use_llm else 'DISABLED'}")
        if self.actor.use_llm:
            print(f"   LLM Guidance Weight: {self.actor.llm_guidance_weight:.3f}")
            print(f"   LLM Call Frequency: Every {self.llm_advisor.call_frequency} steps (when struggling)")
        print(f"   Update Frequency: {update_freq} steps")
        print()
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_shaped_reward = 0
            episode_length = 0
            
            # Reset states
            self.actor.reset_hidden_states()
            self.critic.reset_hidden_states()
            self.reward_shaper.reset()
            self.last_action = None
            self.current_llm_hints = None
            
            reset_hidden = True
            
            # Performance metrics for LLM
            performance_metrics = {
                'avg_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0,
                'avg_length': np.mean(list(self.episode_lengths)) if self.episode_lengths else 0,
            }
            
            while True:
                processed_obs, raw_obs = self.process_observation(obs)
                action, log_prob, value = await self.select_action(
                    processed_obs, raw_obs, reset_hidden, performance_metrics
                )
                reset_hidden = False
                
                self.last_action = action
                
                # Store for buffer
                processed_obs_for_buffer = {}
                for key, tensor_val in processed_obs.items():
                    processed_obs_for_buffer[key] = tensor_val.squeeze(0).cpu()
                
                # Environment step
                step_result = env.step(action)
                
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                else:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                # Reward shaping
                shaped_reward = self.reward_shaper.shape_reward(next_obs, reward, done, info)
                
                self.buffer.add(processed_obs_for_buffer, action, shaped_reward, value, log_prob, done)
                
                obs = next_obs
                episode_reward += reward
                episode_shaped_reward += shaped_reward
                episode_length += 1
                step_count += 1
                
                # Update - same schedule as base RL
                if step_count % update_freq == 0:
                    training_metrics = self.update()
                    if training_metrics:
                        print(f"  📊 Step {step_count}: "
                              f"Actor Loss: {training_metrics['actor_loss']:.4f}, "
                              f"Critic Loss: {training_metrics['critic_loss']:.4f}, "
                              f"Entropy: {training_metrics['entropy']:.4f}")
                    self.buffer.clear()
                
                if done:
                    break
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.shaped_rewards.append(episode_shaped_reward)
            self.episode_lengths.append(episode_length)
            
            # Print progress
            if episode % print_freq == 0:
                avg_reward = np.mean(list(self.episode_rewards)[-10:]) if len(self.episode_rewards) >= 10 else np.mean(list(self.episode_rewards))
                avg_length = np.mean(list(self.episode_lengths)[-10:]) if len(self.episode_lengths) >= 10 else np.mean(list(self.episode_lengths))
                
                llm_info = f", LLM Calls: {self.llm_call_count}" if self.actor.use_llm else ""
                print(f"📈 Episode {episode}: "
                      f"Avg Reward (last 10): {avg_reward:.3f}, "
                      f"Avg Length: {avg_length:.1f}"
                      f"{llm_info}")
        
        if self.actor.use_llm:
            print(f"\n🤖 Total LLM Calls: {self.llm_call_count}")
        
        return list(self.episode_rewards), list(self.shaped_rewards)
    
    def save_model(self, path):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"📂 Model loaded from {path}")

# ========================================
# MAIN TRAINING FUNCTION
# ========================================

def create_nethack_env():
    """Create NetHack environment"""
    import nle.env
    
    try:
        env = gym.make("NetHackScore-v0")
    except:
        env = gym.make("NetHack-v0")
    
    return env


async def main():
    """Main training function with comparison"""
    print("="*80)
    print("🎮 NetHack RL Training: Base RL vs LLM-Enhanced")
    print("="*80)
    
    env = create_nethack_env()
    print(f"✅ Environment created: {env.action_space.n} actions")
    
    # ========================================
    # EXPERIMENT 1: Pure Base RL (Baseline)
    # ========================================
    print("\n" + "="*80)
    print("📊 EXPERIMENT 1: Pure Base RL (No LLM)")
    print("="*80)
    
    base_agent = LLMEnhancedNetHackAgent(
        action_dim=env.action_space.n,
        learning_rate=1e-4,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.02,
        value_coef=0.5,
        max_grad_norm=0.5,
        enable_llm=False,  # NO LLM
        llm_guidance_weight=0.0
    )
    
    print("🚀 Training Base RL for 100 episodes...")
    start_time = time.time()
    base_rewards, base_shaped = await base_agent.train(
        env, 
        num_episodes=100, 
        update_freq=1024,
        print_freq=10
    )
    base_time = time.time() - start_time
    
    base_agent.save_model("models/base_rl_model.pth")
    
    print(f"\n✅ Base RL Training Complete!")
    print(f"   Time: {base_time:.1f}s")
    print(f"   Final Avg Reward (last 10): {np.mean(base_rewards[-10:]):.3f}")
    print(f"   Best Episode: {max(base_rewards):.3f}")
    
    # ========================================
    # EXPERIMENT 2: LLM-Enhanced RL (Minimal Guidance)
    # ========================================
    print("\n" + "="*80)
    print("📊 EXPERIMENT 2: LLM-Enhanced RL (Minimal Guidance)")
    print("="*80)
    
    # Reset environment
    env = create_nethack_env()
    
    llm_agent = LLMEnhancedNetHackAgent(
        action_dim=env.action_space.n,
        learning_rate=1e-4,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.02,
        value_coef=0.5,
        max_grad_norm=0.5,
        enable_llm=True,  # ENABLE LLM
        llm_guidance_weight=0.05  # Very small guidance
    )
    
    print("🚀 Training LLM-Enhanced RL for 100 episodes...")
    start_time = time.time()
    llm_rewards, llm_shaped = await llm_agent.train(
        env,
        num_episodes=100,
        update_freq=1024,
        print_freq=10
    )
    llm_time = time.time() - start_time
    
    llm_agent.save_model("models/llm_enhanced_model.pth")
    
    print(f"\n✅ LLM-Enhanced RL Training Complete!")
    print(f"   Time: {llm_time:.1f}s")
    print(f"   Final Avg Reward (last 10): {np.mean(llm_rewards[-10:]):.3f}")
    print(f"   Best Episode: {max(llm_rewards):.3f}")
    
    # ========================================
    # COMPARISON & ANALYSIS
    # ========================================
    print("\n" + "="*80)
    print("📈 COMPARISON RESULTS")
    print("="*80)
    
    base_final_avg = np.mean(base_rewards[-10:])
    llm_final_avg = np.mean(llm_rewards[-10:])
    
    base_best = max(base_rewards)
    llm_best = max(llm_rewards)
    
    improvement = ((llm_final_avg - base_final_avg) / abs(base_final_avg)) * 100 if base_final_avg != 0 else 0
    
    print(f"\n🏆 Final Performance (Last 10 Episodes):")
    print(f"   Base RL:        {base_final_avg:.3f}")
    print(f"   LLM-Enhanced:   {llm_final_avg:.3f}")
    print(f"   Improvement:    {improvement:+.1f}%")
    
    print(f"\n🎯 Best Episode Performance:")
    print(f"   Base RL:        {base_best:.3f}")
    print(f"   LLM-Enhanced:   {llm_best:.3f}")
    
    print(f"\n⏱️ Training Time:")
    print(f"   Base RL:        {base_time:.1f}s")
    print(f"   LLM-Enhanced:   {llm_time:.1f}s")
    print(f"   Overhead:       {(llm_time - base_time):.1f}s ({((llm_time/base_time - 1) * 100):.1f}%)")
    
    # Save comparison data
    comparison_data = {
        'base_rl': {
            'rewards': base_rewards,
            'shaped_rewards': base_shaped,
            'final_avg': float(base_final_avg),
            'best': float(base_best),
            'time': base_time
        },
        'llm_enhanced': {
            'rewards': llm_rewards,
            'shaped_rewards': llm_shaped,
            'final_avg': float(llm_final_avg),
            'best': float(llm_best),
            'time': llm_time
        },
        'improvement': float(improvement)
    }
    
    with open('comparison_results.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n💾 Results saved to 'comparison_results.json'")
    
    # ========================================
    # VISUALIZATION
    # ========================================
    plot_comparison(base_rewards, llm_rewards, base_shaped, llm_shaped)
    
    env.close()
    print("\n🎉 All experiments complete!")


def plot_comparison(base_rewards, llm_rewards, base_shaped, llm_shaped):
    """Create comparison plots"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(16, 10))
    
    # 1. Raw rewards comparison
    plt.subplot(2, 3, 1)
    episodes = range(len(base_rewards))
    plt.plot(episodes, base_rewards, alpha=0.6, label='Base RL', color='blue')
    plt.plot(episodes, llm_rewards, alpha=0.6, label='LLM-Enhanced', color='red')
    plt.title('Raw Rewards Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Moving average (smoothed)
    plt.subplot(2, 3, 2)
    if len(base_rewards) >= 10:
        window = 10
        base_ma = np.convolve(base_rewards, np.ones(window)/window, mode='valid')
        llm_ma = np.convolve(llm_rewards, np.ones(window)/window, mode='valid')
        
        plt.plot(range(window-1, len(base_rewards)), base_ma, 
                label='Base RL (MA-10)', linewidth=2, color='blue')
        plt.plot(range(window-1, len(llm_rewards)), llm_ma, 
                label='LLM-Enhanced (MA-10)', linewidth=2, color='red')
        plt.title('10-Episode Moving Average', fontsize=12, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Shaped rewards comparison
    plt.subplot(2, 3, 3)
    plt.plot(episodes, base_shaped, alpha=0.6, label='Base RL', color='blue')
    plt.plot(episodes, llm_shaped, alpha=0.6, label='LLM-Enhanced', color='red')
    plt.title('Shaped Rewards Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Shaped Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Reward distribution
    plt.subplot(2, 3, 4)
    plt.hist(base_rewards, bins=20, alpha=0.6, color='blue', label='Base RL')
    plt.hist(llm_rewards, bins=20, alpha=0.6, color='red', label='LLM-Enhanced')
    plt.title('Reward Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Cumulative rewards
    plt.subplot(2, 3, 5)
    base_cumsum = np.cumsum(base_rewards)
    llm_cumsum = np.cumsum(llm_rewards)
    plt.plot(episodes, base_cumsum, label='Base RL', color='blue', linewidth=2)
    plt.plot(episodes, llm_cumsum, label='LLM-Enhanced', color='red', linewidth=2)
    plt.title('Cumulative Rewards', fontsize=12, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Statistical summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    base_mean = np.mean(base_rewards)
    llm_mean = np.mean(llm_rewards)
    base_std = np.std(base_rewards)
    llm_std = np.std(llm_rewards)
    
    base_final = np.mean(base_rewards[-10:])
    llm_final = np.mean(llm_rewards[-10:])
    
    improvement = ((llm_final - base_final) / abs(base_final)) * 100 if base_final != 0 else 0
    
    summary_text = f"""
Statistical Summary:

Overall Performance:
  Base RL Mean:       {base_mean:.3f} ± {base_std:.3f}
  LLM-Enhanced Mean:  {llm_mean:.3f} ± {llm_std:.3f}

Final Performance (Last 10):
  Base RL:            {base_final:.3f}
  LLM-Enhanced:       {llm_final:.3f}
  Improvement:        {improvement:+.1f}%

Best Episodes:
  Base RL:            {max(base_rewards):.3f}
  LLM-Enhanced:       {max(llm_rewards):.3f}

Worst Episodes:
  Base RL:            {min(base_rewards):.3f}
  LLM-Enhanced:       {min(llm_rewards):.3f}
"""
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plot_filename = f'comparison_plot_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved as '{plot_filename}'")
    plt.close()


# ========================================
# ENTRY POINT
# ========================================

if __name__ == "__main__":
    import asyncio
    
    # Create models directory
    os.makedirs("models_vx", exist_ok=True)
    
    print("🚀 Starting NetHack RL Experiments...")
    print("   This will train two agents for 100 episodes each:")
    print("   1. Pure Base RL (baseline)")
    print("   2. LLM-Enhanced RL (minimal guidance)")
    print()
    
    asyncio.run(main())


