# NetHack Causal Model Data Logging Guide

## Overview

This document outlines the data collection strategy for building a Graphical Causal Model (GCM) from NetHack RL agent trajectories. The goal is to capture sufficient information to learn causal relationships between agent actions, environment states, and outcomes.

## 1. Core Data Structure

### Episode-Level Data
```python
{
  "episode_id": str,
  "seed": int,
  "start_timestamp": datetime,
  "end_timestamp": datetime,
  "outcome": str,  # "death", "ascension", "quit"
  "final_score": int,
  "timesteps": List[Timestep]
}
```

### Timestep-Level Data
```python
{
  "timestep": int,
  "agent_state": AgentState,
  "environment_state": EnvironmentState,
  "action": Action,
  "consequences": Consequences,
  "metadata": Metadata
}
```

## 2. Agent State Variables

### Critical Variables (Always Log)
```python
AgentState = {
  # Core vitals
  "hp_current": int,
  "hp_max": int,
  "power_current": int,  # Mana
  "power_max": int,
  
  # Character stats
  "experience_level": int,
  "experience_points": int,
  "strength": int,
  "dexterity": int,
  "constitution": int,
  "intelligence": int,
  "wisdom": int,
  "charisma": int,
  
  # Position
  "position_x": int,
  "position_y": int,
  "dungeon_level": int,
  "dungeon_branch": str,  # "dungeons", "mines", "sokoban", etc.
  
  # Nutrition
  "hunger_state": str,  # "satiated", "not hungry", "hungry", "weak", "fainting"
  "nutrition_value": int,
  
  # Status effects
  "is_blind": bool,
  "is_confused": bool,
  "is_stunned": bool,
  "is_hallucinating": bool,
  "is_sick": bool,
  "is_slimed": bool,
  
  # Equipment (what's currently equipped)
  "weapon_wielded": Optional[Item],
  "armor_worn": List[Item],
  "ring_left": Optional[Item],
  "ring_right": Optional[Item],
  "amulet": Optional[Item],
  
  # Inventory summary
  "inventory_size": int,
  "carrying_capacity": int,
  "gold_amount": int,
}
```

### Item Representation
```python
Item = {
  "item_id": str,
  "item_class": str,  # "weapon", "armor", "scroll", "potion", etc.
  "item_name": str,
  "enchantment": int,
  "blessed_cursed_status": str,  # "blessed", "uncursed", "cursed", "unknown"
  "erosion_level": int,
  "quantity": int,
}
```

## 3. Environment State Variables

### Immediate Surroundings (Critical)
```python
EnvironmentState = {
  # Visible monsters (8x8 or configurable radius)
  "monsters_visible": List[Monster],
  
  # Visible items
  "items_visible": List[ItemLocation],
  
  # Terrain features
  "terrain_type": str,  # "corridor", "room", "altar", "fountain", etc.
  "adjacent_terrain": Dict[str, str],  # {direction: terrain_type}
  
  # Hazards
  "traps_known": List[Trap],
  "is_in_shop": bool,
  "is_near_altar": bool,
  
  # Level information
  "level_depth": int,
  "level_explored_percentage": float,
  "stairs_up_known": bool,
  "stairs_down_known": bool,
}
```

### Monster Representation
```python
Monster = {
  "monster_id": str,
  "monster_type": str,
  "position_x": int,
  "position_y": int,
  "distance": int,
  "is_hostile": bool,
  "is_peaceful": bool,
  "estimated_hp": str,  # "healthy", "wounded", "severely wounded"
  "is_asleep": bool,
}
```

## 4. Action Logging

```python
Action = {
  "action_type": str,  # "move", "attack", "use_item", "search", "rest", etc.
  "action_code": int,  # NetHack action code
  "direction": Optional[str],  # For directional actions
  "target_object": Optional[str],  # Item or monster targeted
  "action_embedding": Optional[np.ndarray],  # If using learned representations
}
```

## 5. Consequences (Causal Effects)

Track what changed immediately after the action:

```python
Consequences = {
  # State changes
  "hp_delta": int,
  "power_delta": int,
  "experience_delta": int,
  "gold_delta": int,
  "nutrition_delta": int,
  
  # Position changes
  "position_changed": bool,
  "new_level_entered": bool,
  
  # Combat outcomes
  "damage_dealt": Optional[int],
  "damage_received": Optional[int],
  "monster_killed": Optional[str],
  "monster_id_killed": Optional[str],
  
  # Item changes
  "item_acquired": Optional[Item],
  "item_used": Optional[Item],
  "item_lost": Optional[Item],
  
  # Status changes
  "status_effects_gained": List[str],
  "status_effects_lost": List[str],
  
  # Environmental
  "trap_triggered": Optional[str],
  "special_event": Optional[str],  # "levelup", "prayer_answered", etc.
  
  # Messages (for unstructured events)
  "game_messages": List[str],
}
```

## 6. Metadata for Analysis

```python
Metadata = {
  # Policy information
  "policy_version": str,
  "action_probabilities": Dict[int, float],  # Distribution over actions
  "value_estimate": float,  # V(s) if available
  "q_values": Optional[Dict[int, float]],  # Q(s,a) for each action
  
  # Exploration
  "epsilon": float,  # If using epsilon-greedy
  "is_exploratory": bool,
  
  # Training phase
  "training_step": int,
  "episode_number": int,
}
```

## 7. Implementation Strategy

### Option 1: Wrapper-Based Logging

Create a logging wrapper around your NetHack environment:

```python
class CausalLoggingWrapper(gym.Wrapper):
    def __init__(self, env, log_file_path):
        super().__init__(env)
        self.log_file_path = log_file_path
        self.episode_data = []
        self.current_episode = None
        
    def reset(self):
        # Save previous episode if exists
        if self.current_episode:
            self._save_episode()
        
        # Start new episode
        obs = self.env.reset()
        self.current_episode = {
            "episode_id": str(uuid.uuid4()),
            "seed": self.env.get_seeds()[0],
            "start_timestamp": datetime.now(),
            "timesteps": []
        }
        
        # Log initial state
        self._log_timestep(obs, action=None, reward=0, done=False, info={})
        return obs
    
    def step(self, action):
        # Capture pre-action state
        pre_state = self._extract_agent_state()
        
        # Execute action
        obs, reward, done, info = self.env.step(action)
        
        # Capture post-action state and consequences
        post_state = self._extract_agent_state()
        consequences = self._compute_consequences(pre_state, post_state, info)
        
        # Log the timestep
        self._log_timestep(obs, action, reward, done, info, 
                          pre_state, post_state, consequences)
        
        if done:
            self.current_episode["end_timestamp"] = datetime.now()
            self.current_episode["outcome"] = info.get("end_status", "unknown")
            self._save_episode()
        
        return obs, reward, done, info
```

### Option 2: Callback-Based Logging

If using a framework like Stable-Baselines3:

```python
from stable_baselines3.common.callbacks import BaseCallback

class CausalModelCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_buffer = []
        
    def _on_step(self) -> bool:
        # Extract and log data at each step
        timestep_data = self._extract_timestep_data()
        self.episode_buffer.append(timestep_data)
        
        # Check if episode ended
        if self.locals["dones"][0]:
            self._save_episode()
            self.episode_buffer = []
        
        return True
```

### Option 3: Post-Processing from Replay Buffer

If you already have a replay buffer:

```python
class CausalDataExtractor:
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer
        
    def extract_causal_data(self):
        # Process replay buffer to extract causal variables
        # This is useful if you want to retroactively create the dataset
        pass
```

## 8. Data Storage Format

### Recommended: HDF5 with Hierarchical Structure

```python
import h5py

# Structure:
# episodes/
#   episode_0000/
#     metadata (attrs)
#     timesteps/
#       agent_state (dataset)
#       environment_state (dataset)
#       actions (dataset)
#       consequences (dataset)
#   episode_0001/
#     ...
```

### Alternative: Parquet for Analytics

```python
import pandas as pd
import pyarrow.parquet as pq

# Flatten the hierarchical structure for each timestep
df = pd.DataFrame(flattened_timesteps)
df.to_parquet("nethack_causal_data.parquet", compression="snappy")
```

### Lightweight: JSONL for Small Datasets

```python
import jsonlines

with jsonlines.open("episodes.jsonl", mode="w") as writer:
    for episode in episodes:
        writer.write(episode)
```

## 9. Data Validation Checklist

Before using data for causal modeling:

- [ ] **Temporal consistency**: t+1 states come after t actions
- [ ] **No missing critical variables**: HP, position, level always logged
- [ ] **Action-consequence linkage**: Can trace action → outcome
- [ ] **Sufficient coverage**: Multiple examples of each causal pathway
- [ ] **Episode boundaries**: Clear start/end markers
- [ ] **State transitions**: Can reconstruct s_t, a_t, s_{t+1} triples
- [ ] **Confounders captured**: Variables that affect multiple outcomes logged

## 10. Privacy and Ethics Considerations

- **No PII**: Ensure no personally identifiable information
- **Seed diversity**: Collect from various random seeds
- **Policy diversity**: Include data from different training stages
- **Failure cases**: Don't just log successful runs

## 11. Usage Example

```python
# Training loop with causal logging
env = CausalLoggingWrapper(
    gym.make("NetHackChallenge-v0"),
    log_file_path="causal_data/run_001.h5"
)

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        agent.train_step(obs, reward, done)

# Later: Load data for causal analysis
from causal_analysis import CausalDataset
dataset = CausalDataset.from_hdf5("causal_data/run_001.h5")
causal_graph = learn_causal_structure(dataset)
```

## 12. Next Steps

1. **Start minimal**: Log only critical variables first (HP, position, actions, consequences)
2. **Iterate**: Add more variables as you identify important causal relationships
3. **Validate**: Regularly check that logged data makes sense
4. **Analyze**: Use the data to learn causal graphs and validate them against known game mechanics
5. **Scale**: Once validated, scale to larger datasets for robust causal discovery