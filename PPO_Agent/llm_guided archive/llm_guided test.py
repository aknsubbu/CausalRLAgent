"""
COMPLETE FIXED LLM-GUIDED NETHACK RL AGENT
============================================

Key improvements:
1. Proper probability-based LLM guidance blending
2. Comprehensive action mapping with fuzzy matching
3. Anti-repetition mechanisms
4. Improved reward shaping with bigger signals
5. Better LLM prompt engineering
6. Diagnostic tools to verify LLM influence

Usage:
    python this_file.py
"""

import gymnasium as gym
import nle.env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque, defaultdict
import json
import re
import time
from datetime import datetime
import asyncio
import aiohttp


# ============================================================================
# FIXED ACTION MAPPER
# ============================================================================

class ImprovedActionMapper:
    """Robust action mapping with fuzzy matching"""
    
    def __init__(self):
        self.action_name_to_id = {
            # Movement - all variants
            "move_north": 0, "north": 0, "go_north": 0, "head_north": 0, "n": 0, "up": 0,
            "move_south": 1, "south": 1, "go_south": 1, "head_south": 1, "s": 1, "down": 1,
            "move_east": 2, "east": 2, "go_east": 2, "head_east": 2, "e": 2, "right": 2,
            "move_west": 3, "west": 3, "go_west": 3, "head_west": 3, "w": 3, "left": 3,
            "move_northeast": 4, "northeast": 4, "ne": 4,
            "move_northwest": 5, "northwest": 5, "nw": 5,
            "move_southeast": 6, "southeast": 6, "se": 6,
            "move_southwest": 7, "southwest": 7, "sw": 7,
            
            # Combat - all variants including "kick_monster"
            "kick": 14, "attack": 14, "fight": 14, "hit": 14, "strike": 14, "combat": 14,
            "kick_kobold": 14, "kick_orc": 14, "kick_goblin": 14, "kick_monster": 14,
            "attack_kobold": 14, "attack_orc": 14, "fight_monster": 14, "engage": 14,
            
            # Items
            "pickup": 9, "pick": 9, "take": 9, "grab": 9, "collect": 9, "get": 9,
            "search": 11, "look": 11, "explore": 11, "investigate": 11, "scout": 11,
            "eat": 15, "consume": 15, "food": 15,
            "drink": 16, "quaff": 16, "potion": 16,
            "read": 17, "scroll": 17,
            "apply": 18, "use": 18,
            "wear": 20, "equip": 20, "armor": 20,
            "wield": 22, "weapon": 22, "arm": 22,
            "drop": 10, "discard": 10,
            
            # Utility
            "wait": 8, "rest": 8, "pause": 8, "stay": 8,
            "open_door": 12, "open": 12,
            "close_door": 13, "close": 13,
            "throw": 19, "toss": 19,
            "take_off": 21, "remove": 21,
        }
        
        # Movement keywords for fuzzy matching
        self.direction_keywords = {
            'north': 0, 'south': 1, 'east': 2, 'west': 3,
            'up': 0, 'down': 1, 'right': 2, 'left': 3,
        }
        
        # Action keywords for fuzzy matching
        self.action_keywords = {
            'kick': 14, 'attack': 14, 'fight': 14, 'combat': 14,
            'search': 11, 'look': 11, 'explore': 11,
            'pickup': 9, 'take': 9, 'grab': 9, 'get': 9,
            'wait': 8, 'rest': 8,
        }
    
    def map_action(self, action_str):
        """Map action string to ID with fuzzy matching"""
        if not isinstance(action_str, str):
            return None
        
        action_lower = action_str.lower().strip()
        
        # Remove underscores and extra spaces
        action_clean = action_lower.replace('_', ' ').replace('  ', ' ')
        
        # Try exact match
        if action_lower in self.action_name_to_id:
            return self.action_name_to_id[action_lower]
        
        if action_clean in self.action_name_to_id:
            return self.action_name_to_id[action_clean]
        
        # Try substring matching on keys
        for key, action_id in self.action_name_to_id.items():
            if key in action_lower or action_lower in key:
                return action_id
        
        # Try direction keywords
        for keyword, action_id in self.direction_keywords.items():
            if keyword in action_lower:
                return action_id
        
        # Try action keywords
        for keyword, action_id in self.action_keywords.items():
            if keyword in action_lower:
                return action_id
        
        return None
    
    def get_action_distribution(self, llm_suggestions, temperature=1.5):
        """
        Convert LLM suggestions to probability distribution
        
        Args:
            llm_suggestions: List of action strings from LLM
            temperature: Higher = more uniform, lower = more peaked
        
        Returns:
            action_probs: np.array of shape (23,) with probabilities
        """
        action_scores = np.zeros(23, dtype=np.float32)
        
        for i, suggestion in enumerate(llm_suggestions[:5]):
            action_id = self.map_action(suggestion)
            if action_id is not None and 0 <= action_id < 23:
                # Exponential decay for priority
                weight = np.exp(-(i / temperature))
                action_scores[action_id] += weight
        
        # Convert to probabilities
        if action_scores.sum() > 0:
            action_probs = action_scores / action_scores.sum()
        else:
            # If no valid actions mapped, return uniform
            action_probs = np.ones(23, dtype=np.float32) / 23
        
        return action_probs



# ============================================================================
# IMPROVED REWARD SHAPER
# ============================================================================

class ImprovedRewardShaper:
    """Better reward shaping with clearer signals"""
    
    def __init__(self):
        self.previous_stats = None
        self.visited_positions = set()
        self.last_position = None
        self.last_action = None
        self.action_repeat_count = 0
        self.consecutive_stuck = 0
        
        # BIGGER reward signals
        self.exploration_reward = 0.05  # Up from 0.005
        self.level_up_reward = 20.0     # Up from 5.0
        self.experience_reward = 0.02   # Up from 0.001
        self.combat_kill_reward = 2.0   # Up from 0.5
        self.health_loss_penalty = -0.1 # Stronger
        self.health_gain_reward = 0.2   # Reward healing
        self.death_penalty = -10.0      # Up from -2.0
        self.stuck_penalty = -0.05      # Up from -0.005
        self.repetition_penalty = -0.02 # NEW
        self.gold_reward = 0.01         # Per gold
        
    def shape_reward(self, obs, raw_reward, done, info, action=None):
        """Apply improved reward shaping"""
        shaped_reward = raw_reward
        
        if isinstance(obs, tuple):
            obs = obs[0]
        
        current_stats = obs.get('blstats', np.zeros(26))
        
        if self.previous_stats is not None:
            # Health changes (index 10)
            health_diff = current_stats[10] - self.previous_stats[10]
            if health_diff > 0:
                shaped_reward += health_diff * self.health_gain_reward
            elif health_diff < 0:
                shaped_reward += health_diff * self.health_loss_penalty
            
            # LEVEL UP (index 18) - BIG REWARD
            level_diff = current_stats[18] - self.previous_stats[18]
            if level_diff > 0:
                shaped_reward += self.level_up_reward
                print(f"🎉 LEVEL UP! +{self.level_up_reward}")
            
            # Experience (index 19)
            exp_diff = current_stats[19] - self.previous_stats[19]
            if exp_diff > 0:
                shaped_reward += exp_diff * self.experience_reward
                
                # Monster kill detection (big exp spike)
                if exp_diff > 10:
                    shaped_reward += self.combat_kill_reward
                    print(f"⚔️ Kill! +{self.combat_kill_reward}")
            
            # Gold (index 13)
            gold_diff = current_stats[13] - self.previous_stats[13]
            if gold_diff > 0:
                shaped_reward += gold_diff * self.gold_reward
        
        # Exploration (indices 0, 1 for position)
        current_pos = tuple(current_stats[:2]) if len(current_stats) > 1 else (0, 0)
        if current_pos not in self.visited_positions:
            self.visited_positions.add(current_pos)
            shaped_reward += self.exploration_reward
        
        # Anti-stuck
        if current_pos == self.last_position:
            self.consecutive_stuck += 1
            if self.consecutive_stuck > 15:
                shaped_reward += self.stuck_penalty
        else:
            self.consecutive_stuck = 0
        
        # Anti-repetition
        if action == self.last_action:
            self.action_repeat_count += 1
            if self.action_repeat_count > 5:
                shaped_reward += self.repetition_penalty
        else:
            self.action_repeat_count = 0
        
        # Death penalty
        if done and current_stats[10] <= 0:
            shaped_reward += self.death_penalty
        
        # Update state
        self.previous_stats = current_stats.copy()
        self.last_position = current_pos
        self.last_action = action
        
        return shaped_reward
    
    def reset(self):
        """Reset for new episode"""
        self.previous_stats = None
        self.visited_positions.clear()
        self.last_position = None
        self.last_action = None
        self.action_repeat_count = 0
        self.consecutive_stuck = 0


# ============================================================================
# IMPROVED LLM ADVISOR
# ============================================================================

class ImprovedLLMAdvisor:
    """Better LLM advisor with clearer prompts"""
    
    def __init__(self, call_frequency=20):
        self.call_frequency = call_frequency
        self.step_count = 0
        self.last_advice = None
    
    def should_call_llm(self):
        self.step_count += 1
        return self.step_count % self.call_frequency == 0
    
    async def get_strategic_advice(self, semantic_description, recent_performance):
        """Get advice with improved prompt"""
        try:
            # IMPROVED PROMPT with exact action names
            prompt = f"""You are a NetHack expert advisor. Give SPECIFIC action recommendations.

CURRENT GAME STATE:
{semantic_description}

RECENT PERFORMANCE:
- Avg Reward: {recent_performance.get('avg_reward', 0):.2f}
- Avg Survival: {recent_performance.get('avg_length', 0):.0f} steps

VALID ACTIONS (use EXACTLY these names):
- Movement: move_north, move_south, move_east, move_west, move_northeast, move_northwest, move_southeast, move_southwest
- Combat: kick (attacks adjacent enemy)
- Items: pickup, search, eat, drink
- Utility: wait

STRATEGY GUIDELINES:
1. If "CLOSEST THREAT: monster X (dist:1)" → health > 60%? suggest "kick" | health < 40%? suggest movement away
2. If "NO IMMEDIATE THREATS" → suggest "search", "pickup", or movement to explore
3. If stuck (negative rewards) → suggest different movement direction
4. ALWAYS suggest 3-5 actions in priority order
5. VARY movement directions (don't always suggest same direction)

CRITICAL: Use EXACT action names from valid list above!

Respond with JSON ONLY (no markdown):
{{
  "action_suggestions": ["action1", "action2", "action3"],
  "immediate_priority": "one clear goal",
  "risk_assessment": "low/medium/high",
  "opportunities": "available rewards",
  "strategy": "brief 1-sentence plan"
}}

JSON:"""

            response = await self._call_llm_api(prompt)
            
            # Parse response
            response = response.strip()
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            try:
                advice = json.loads(response)
                
                # Validate structure
                if 'action_suggestions' not in advice or not isinstance(advice['action_suggestions'], list):
                    advice['action_suggestions'] = ["search", "move_north", "pickup"]
                
                self.last_advice = advice
                return advice
                
            except json.JSONDecodeError:
                return self._get_fallback_advice()
                
        except Exception as e:
            print(f"LLM error: {e}")
            return self._get_fallback_advice()
    
    async def _call_llm_api(self, prompt):
        """Call Ollama API"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 512
                    }
                }
                
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        return self._get_fallback_response()
                        
        except Exception as e:
            print(f"Ollama API error: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self):
        """Fallback response"""
        return json.dumps({
            "action_suggestions": ["search", "move_north", "pickup"],
            "immediate_priority": "explore safely",
            "risk_assessment": "unknown",
            "opportunities": "search for items",
            "strategy": "cautious exploration"
        })
    
    def _get_fallback_advice(self):
        """Fallback advice"""
        return {
            "action_suggestions": ["search", "move_north", "pickup"],
            "immediate_priority": "explore safely",
            "risk_assessment": "unknown",
            "opportunities": "search for items",
            "strategy": "cautious exploration"
        }


# ============================================================================
# FIXED LLM-GUIDED ACTOR WITH STRONG GUIDANCE (CORRECTED)
# ============================================================================

class FixedLLMGuidedActor(nn.Module):
    """Actor with proper probability-based LLM guidance"""
    
    def __init__(self, base_actor, llm_guidance_strength=0.8):
        super().__init__()
        self.base_actor = base_actor
        self.llm_guidance_strength = llm_guidance_strength
        self.action_mapper = ImprovedActionMapper()
        
        # Statistics tracking
        self.guidance_stats = {
            'total_steps': 0,
            'llm_influenced_steps': 0,
            'top_action_followed': 0,
            'any_suggested_followed': 0,
            'llm_success_rewards': [],
            'rl_success_rewards': []
        }
    
    def forward(self, obs, reset_hidden=False, llm_advice=None):
        """
        Forward with probability blending instead of logit addition
        
        This is the KEY FIX: We blend probability distributions, not logits
        """
        # Get base RL policy logits - WITHOUT llm_advice parameter
        base_logits = self.base_actor(obs, reset_hidden)
        base_probs = F.softmax(base_logits, dim=-1)
        
        # If no LLM advice, return base policy
        if llm_advice is None or self.llm_guidance_strength == 0:
            return base_logits
        
        # Get LLM guidance as probability distribution
        llm_suggestions = llm_advice.get('action_suggestions', [])
        if not llm_suggestions:
            return base_logits
        
        llm_probs = self.action_mapper.get_action_distribution(llm_suggestions)
        llm_probs_tensor = torch.FloatTensor(llm_probs).to(base_logits.device)
        
        # Expand for batch size
        batch_size = base_probs.size(0)
        if batch_size > 1:
            llm_probs_tensor = llm_probs_tensor.unsqueeze(0).expand(batch_size, -1)
        else:
            llm_probs_tensor = llm_probs_tensor.unsqueeze(0)
        
        # CRITICAL: Blend probability distributions (not logits!)
        # This ensures LLM has real influence
        alpha = self.llm_guidance_strength
        blended_probs = alpha * llm_probs_tensor + (1 - alpha) * base_probs
        
        # Normalize (should already be normalized, but be safe)
        blended_probs = blended_probs / (blended_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Convert back to logits for Categorical
        blended_logits = torch.log(blended_probs + 1e-10)
        
        return blended_logits
    
    def reset_hidden_states(self, batch_size=1):
        """Pass through to base actor's reset method"""
        if hasattr(self.base_actor, 'reset_hidden_states'):
            return self.base_actor.reset_hidden_states(batch_size)
    
    def update_stats(self, action_taken, llm_advice, reward):
        """Track guidance effectiveness"""
        self.guidance_stats['total_steps'] += 1
        
        if llm_advice is None:
            self.guidance_stats['rl_success_rewards'].append(reward)
            return
        
        llm_suggestions = llm_advice.get('action_suggestions', [])
        if not llm_suggestions:
            return
        
        self.guidance_stats['llm_influenced_steps'] += 1
        
        # Check if top suggestion was followed
        top_suggestion = self.action_mapper.map_action(llm_suggestions[0])
        if top_suggestion == action_taken:
            self.guidance_stats['top_action_followed'] += 1
        
        # Check if any suggestion was followed
        suggested_ids = [self.action_mapper.map_action(s) for s in llm_suggestions[:3]]
        suggested_ids = [a for a in suggested_ids if a is not None]
        if action_taken in suggested_ids:
            self.guidance_stats['any_suggested_followed'] += 1
            self.guidance_stats['llm_success_rewards'].append(reward)
    
    def print_stats(self):
        """Print guidance statistics"""
        stats = self.guidance_stats
        total = stats['total_steps']
        
        if total == 0:
            return
        
        llm_influence = stats['llm_influenced_steps'] / total
        top_follow_rate = (stats['top_action_followed'] / 
                          max(stats['llm_influenced_steps'], 1))
        any_follow_rate = (stats['any_suggested_followed'] / 
                          max(stats['llm_influenced_steps'], 1))
        
        print(f"\n{'='*60}")
        print(f"LLM GUIDANCE STATISTICS")
        print(f"{'='*60}")
        print(f"Total steps: {total}")
        print(f"Steps with LLM advice: {stats['llm_influenced_steps']} ({llm_influence:.1%})")
        print(f"Top suggestion followed: {top_follow_rate:.1%}")
        print(f"Any suggestion followed: {any_follow_rate:.1%}")
        
        if stats['llm_success_rewards']:
            avg_llm = np.mean(stats['llm_success_rewards'])
            print(f"Avg reward when following LLM: {avg_llm:.3f}")
        
        if stats['rl_success_rewards']:
            avg_rl = np.mean(stats['rl_success_rewards'])
            print(f"Avg reward without LLM: {avg_rl:.3f}")
        
        print(f"{'='*60}\n")


# ============================================================================
# DIAGNOSTIC TOOL (CORRECTED)
# ============================================================================

async def diagnose_llm_influence(agent, env, num_steps=50):
    """Test if LLM actually influences actions"""
    
    print("\n" + "="*70)
    print("DIAGNOSTIC: Testing LLM Guidance Influence")
    print("="*70)
    
    obs = env.reset()
    actions_with_llm = []
    actions_without_llm = []
    
    # Create fake LLM advice that suggests move_north
    fake_advice = {
        'action_suggestions': ['move_north', 'search', 'pickup'],
        'immediate_priority': 'explore north'
    }
    
    for step in range(num_steps):
        # Process observation - handle variable return values
        obs_result = agent.process_observation(obs)
        if isinstance(obs_result, tuple):
            if len(obs_result) == 2:
                tensor_obs, processed_obs = obs_result
            elif len(obs_result) == 3:
                tensor_obs, processed_obs, _ = obs_result
            else:
                tensor_obs = obs_result[0]
                processed_obs = obs_result[1] if len(obs_result) > 1 else None
        else:
            tensor_obs = obs_result
            processed_obs = None
        
        # Get action WITH LLM (suggesting move_north)
        with torch.no_grad():
            # Call with llm_advice parameter for FixedLLMGuidedActor
            action_logits_llm = agent.actor(tensor_obs, False, llm_advice=fake_advice)
            action_dist_llm = Categorical(logits=action_logits_llm)
            action_llm = action_dist_llm.sample().item()
            actions_with_llm.append(action_llm)
            
            # Get action WITHOUT LLM (llm_advice=None)
            action_logits_base = agent.actor(tensor_obs, False, llm_advice=None)
            action_dist_base = Categorical(logits=action_logits_base)
            action_base = action_dist_base.sample().item()
            actions_without_llm.append(action_base)
        
        # Step environment
        step_result = env.step(action_llm)
        if len(step_result) == 4:
            obs, _, done, _ = step_result
        else:
            obs, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        
        if done:
            obs = env.reset()
    
    # Analysis
    different = sum(1 for a, b in zip(actions_with_llm, actions_without_llm) if a != b)
    diff_rate = different / num_steps
    
    # Count how often move_north (action 0) was chosen with LLM
    move_north_count = sum(1 for a in actions_with_llm if a == 0)
    move_north_rate = move_north_count / num_steps
    
    print(f"\nResults:")
    print(f"  Actions changed by LLM: {different}/{num_steps} ({diff_rate:.1%})")
    print(f"  Move north (suggested): {move_north_count}/{num_steps} ({move_north_rate:.1%})")
    
    if diff_rate < 0.3:
        print(f"\n❌ PROBLEM: LLM influence too weak ({diff_rate:.1%})")
        print(f"   Action: Increase llm_guidance_strength to 0.8-0.9")
    elif diff_rate > 0.6:
        print(f"\n✓ GOOD: LLM has strong influence ({diff_rate:.1%})")
    else:
        print(f"\n~ MODERATE: LLM has some influence ({diff_rate:.1%})")
    
    if move_north_rate > 0.3:
        print(f"✓ LLM suggestions are being followed ({move_north_rate:.1%} north)")
    else:
        print(f"⚠ LLM suggestions may not be followed well ({move_north_rate:.1%} north)")
    
    print("="*70 + "\n")


# ============================================================================
# MAIN TRAINING FUNCTION (FIXED)
# ============================================================================

async def train_fixed_agent():
    """Train with all fixes applied"""
    
    print("\n" + "="*70)
    print("FIXED LLM-GUIDED NETHACK RL TRAINING")
    print("="*70 + "\n")
    
    # Import necessary components from your existing code
    # (You'll need to include the base agent classes from your original file)
    
    from llm_guided import (
        create_nethack_env,
        EnhancedNetHackPPOAgent,
        NetHackSemanticDescriptor,
        NetHackObservationProcessor
    )
    
    # Create environment
    env = create_nethack_env()
    print(f"Environment created with {env.action_space.n} actions")
    
    # Create base agent
    base_agent = EnhancedNetHackPPOAgent(
        action_dim=env.action_space.n,
        learning_rate=3e-4
    )
    
    # APPLY FIXES
    print("\nApplying fixes...")
    
    # 1. Wrap actor with fixed LLM guidance
    base_agent.actor = FixedLLMGuidedActor(
        base_agent.actor,
        llm_guidance_strength=0.8  # Strong LLM influence
    )
    print("✓ Fixed LLM guidance (strength=0.8)")
    
    # 2. Replace reward shaper
    base_agent.reward_shaper = ImprovedRewardShaper()
    print("✓ Improved reward shaper")
    
    # 3. Replace LLM advisor
    base_agent.llm_advisor = ImprovedLLMAdvisor(call_frequency=20)
    print("✓ Improved LLM advisor")
    
    # 4. Add semantic descriptor if not present
    if not hasattr(base_agent, 'semantic_descriptor'):
        base_agent.semantic_descriptor = NetHackSemanticDescriptor()
    
    print("\n" + "="*70)
    
    # Run diagnostic first
    print("\nRunning diagnostic...")
    await diagnose_llm_influence(base_agent, env)
    
    # Training loop
    num_episodes = 100
    
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode}")
        print(f"{'='*70}")
        
        obs = env.reset()
        episode_reward = 0
        episode_shaped_reward = 0
        episode_length = 0
        
        # Reset
        base_agent.actor.base_actor.reset_hidden_states()
        base_agent.critic.reset_hidden_states()
        base_agent.reward_shaper.reset()
        base_agent.last_action = None
        
        while True:
            # Check if we should get LLM advice
            if base_agent.llm_advisor.should_call_llm():
                # Generate semantic description - handle variable return values
                obs_result = base_agent.process_observation(obs)
                if isinstance(obs_result, tuple):
                    if len(obs_result) >= 2:
                        tensor_obs, processed_obs = obs_result[0], obs_result[1]
                    else:
                        tensor_obs = obs_result[0]
                        processed_obs = None
                else:
                    tensor_obs = obs_result
                    processed_obs = None
                
                semantic_desc = base_agent.semantic_descriptor.generate_full_description(
                    obs, processed_obs
                )
                
                recent_performance = {
                    'avg_reward': np.mean(list(base_agent.episode_rewards)) if base_agent.episode_rewards else 0,
                    'avg_length': np.mean(list(base_agent.episode_lengths)) if base_agent.episode_lengths else 0
                }
                
                llm_advice = await base_agent.llm_advisor.get_strategic_advice(
                    semantic_desc, recent_performance
                )
                
                print(f"\n[Step {episode_length}] LLM Advice: {llm_advice.get('immediate_priority', 'N/A')}")
                print(f"  Suggestions: {llm_advice.get('action_suggestions', [])}")
            else:
                llm_advice = base_agent.llm_advisor.last_advice
            
            # Select action - handle variable return values
            obs_result = base_agent.process_observation(obs)
            if isinstance(obs_result, tuple):
                if len(obs_result) >= 2:
                    tensor_obs, processed_obs = obs_result[0], obs_result[1]
                else:
                    tensor_obs = obs_result[0]
                    processed_obs = None
            else:
                tensor_obs = obs_result
                processed_obs = None
            
            action, log_prob, value = await base_agent.select_action(obs, processed_obs, False)
            
            # Update LLM stats
            base_agent.actor.update_stats(action, llm_advice, 0)  # Will update reward later
            
            # Environment step
            step_result = env.step(action)
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            # Shape reward
            shaped_reward = base_agent.reward_shaper.shape_reward(
                next_obs, reward, done, info, action
            )
            
            # Update LLM stats with actual reward
            base_agent.actor.update_stats(action, llm_advice, shaped_reward)
            
            # Store in buffer
            processed_obs_buffer = {k: v.squeeze(0).cpu() for k, v in tensor_obs.items()}
            base_agent.buffer.add(processed_obs_buffer, action, shaped_reward, value, log_prob, done)
            
            obs = next_obs
            episode_reward += reward
            episode_shaped_reward += shaped_reward
            episode_length += 1
            
            if episode_length % 10 == 0:
                print(f"  Step {episode_length}: R={episode_reward:.2f}, SR={episode_shaped_reward:.2f}")
            
            if done or episode_length >= 1000:
                break
        
        # Episode summary
        base_agent.episode_rewards.append(episode_reward)
        base_agent.shaped_rewards.append(episode_shaped_reward)
        base_agent.episode_lengths.append(episode_length)
        
        print(f"\nEpisode {episode} completed: Total R={episode_reward:.2f}, Shaped R={episode_shaped_reward:.2f}, Length={episode_length}")
        print(f"  Average R={np.mean(base_agent.episode_rewards):.2f}, Average SR={np.mean(base_agent.shaped_rewards):.2f}, Average Length={np.mean(base_agent.episode_lengths):.2f}")
        
        # Update policy
        if len(base_agent.buffer.rewards) > 0:
            print("\nUpdating policy...")
            await base_agent.update_policy()
        
        # Print LLM guidance stats every 10 episodes
        if (episode + 1) % 10 == 0:
            base_agent.actor.print_stats()
            
            # Print running averages
            recent_episodes = min(10, len(base_agent.episode_rewards))
            recent_rewards = list(base_agent.episode_rewards)[-recent_episodes:]
            recent_shaped = list(base_agent.shaped_rewards)[-recent_episodes:]
            recent_lengths = list(base_agent.episode_lengths)[-recent_episodes:]
            
            print(f"\n{'='*70}")
            print(f"PROGRESS REPORT (Last {recent_episodes} episodes)")
            print(f"{'='*70}")
            print(f"Average Raw Reward: {np.mean(recent_rewards):.2f}")
            print(f"Average Shaped Reward: {np.mean(recent_shaped):.2f}")
            print(f"Average Episode Length: {np.mean(recent_lengths):.1f}")
            print(f"Total Episodes: {episode + 1}")
            print(f"{'='*70}\n")
        
        # Save checkpoint every 25 episodes
        if (episode + 1) % 25 == 0:
            checkpoint_path = f"checkpoint_ep{episode+1}.pt"
            torch.save({
                'episode': episode,
                'actor_state_dict': base_agent.actor.state_dict(),
                'critic_state_dict': base_agent.critic.state_dict(),
                'optimizer_state_dict': base_agent.optimizer.state_dict(),
                'episode_rewards': list(base_agent.episode_rewards),
                'shaped_rewards': list(base_agent.shaped_rewards),
                'episode_lengths': list(base_agent.episode_lengths),
                'guidance_stats': base_agent.actor.guidance_stats
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Final statistics
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    # Overall performance
    print(f"\nOverall Statistics:")
    print(f"  Total Episodes: {num_episodes}")
    print(f"  Average Raw Reward: {np.mean(list(base_agent.episode_rewards)):.2f}")
    print(f"  Average Shaped Reward: {np.mean(list(base_agent.shaped_rewards)):.2f}")
    print(f"  Average Episode Length: {np.mean(list(base_agent.episode_lengths)):.1f}")
    print(f"  Max Raw Reward: {max(base_agent.episode_rewards):.2f}")
    print(f"  Max Episode Length: {max(base_agent.episode_lengths)}")
    
    # Final LLM stats
    base_agent.actor.print_stats()
    
    # Save final model
    final_model_path = "final_model.pt"
    torch.save({
        'actor_state_dict': base_agent.actor.state_dict(),
        'critic_state_dict': base_agent.critic.state_dict(),
        'optimizer_state_dict': base_agent.optimizer.state_dict(),
        'episode_rewards': list(base_agent.episode_rewards),
        'shaped_rewards': list(base_agent.shaped_rewards),
        'episode_lengths': list(base_agent.episode_lengths),
        'guidance_stats': base_agent.actor.guidance_stats,
        'training_complete': True
    }, final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    # Close environment
    env.close()
    
    print("\n" + "="*70)
    print("Training complete! Check the saved models and statistics.")
    print("="*70 + "\n")
    
    return base_agent




# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING LLM-GUIDED NETHACK TRAINING")
    print("="*70)
    print("\nThis script will:")
    print("  1. Create NetHack environment")
    print("  2. Initialize PPO agent with LLM guidance")
    print("  3. Run diagnostic to verify LLM influence")
    print("  4. Train for specified number of episodes")
    print("  5. Save checkpoints and final model")
    print("\nMake sure Ollama is running on localhost:11434")
    print("="*70 + "\n")
    
    try:
        # Run training
        asyncio.run(train_fixed_agent())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving current state...")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutdown complete.")