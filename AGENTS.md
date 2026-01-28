# AGENTS.md - AI Agent Guidelines for CausalRLAgent

This document provides guidelines for AI coding agents working on this codebase.

## Project Overview

CausalRLAgent is a Python reinforcement learning research project combining:
- **PPO (Proximal Policy Optimization)** for NetHack game playing
- **LLM strategic guidance** via Ollama API for high-level decisions
- **Causal inference** using DoWhy to analyze LLM advice effectiveness

**Target Environment**: NetHack Learning Environment (NLE) - a complex roguelike game.

## Build/Install Commands

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import gymnasium, nle, torch, dowhy; print('All dependencies installed!')"
```

## Running the Project

```bash
# Main LLM-guided training (recommended entry point)
cd PPO_Agent
python "base_RL extension with causal.py"

# Baseline training with Stable Baselines3
cd BaseRL
python trainv2.py --timesteps 50000

# Model evaluation
python PPO_Agent/ppo_model_trained/evaluate_model.py --model_path model.pth --episodes 10
python PPO_Agent/ppo_model_trained/simple_evaluate.py --model_path model.pth
```

## Test Commands

This is a research codebase without a formal test framework. Tests are standalone scripts:

```bash
# Run LLM guidance unit tests
python PPO_Agent/tests/quick_test.py

# Verify LLM guidance impact
python PPO_Agent/tests/verify_impact.py

# Run a single test file
python <path/to/test_file.py>
```

## Code Style Guidelines

### Imports

Organize imports in this order with blank lines between groups:
1. Standard library (os, json, time, datetime, re, collections)
2. Third-party packages (numpy, pandas, torch, gymnasium)
3. Local modules

```python
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym

from causal_logger import ImprovedCausalLogger
```

### Type Hints

Type hints are used selectively, primarily in parsing/utility modules:

```python
from typing import Dict, List, Tuple, Optional, Set

def parse_logs(self, log_file: str, alignment_window: int = 10) -> pd.DataFrame:
    ...

def _semantically_matches(self, action: str, suggestion: str, debug: bool = False) -> bool:
    ...
```

Core RL/neural network code typically omits type hints for brevity.

### Naming Conventions

- **Classes**: PascalCase (`NetHackPPOAgent`, `DoublyRobustEstimator`)
- **Functions/Methods**: snake_case (`process_observation`, `compute_advantages`)
- **Constants**: UPPER_SNAKE_CASE (`BLSTATS_IDX`, `DEFAULT_REWARD_CONFIG`)
- **Private methods**: prefix with underscore (`_extract_health_ratio`, `_forward_conv`)
- **File names**: snake_case with optional version suffix (`log_parser_v2.py`, `llm_guided_v1.py`)

### Class Structure

Follow this pattern for major classes:

```python
class ClassName:
    """
    Brief description of class purpose.
    
    More details if needed about:
    - Key features
    - Integration with other components
    """
    
    def __init__(self, param1, param2=default):
        # Initialize attributes
        self.param1 = param1
        self.param2 = param2
        
        # Print initialization info (common pattern)
        print(f"ClassName initialized")
        print(f"  param1: {self.param1}")
    
    # Public methods first
    def public_method(self):
        ...
    
    # Private methods after
    def _private_helper(self):
        ...
```

### Error Handling

Use try/except with fallback values for robustness:

```python
# Pattern 1: Fallback on any exception
try:
    env = gym.make("NetHackScore-v0")
except:
    env = gym.make("NetHack-v0")

# Pattern 2: Type checking before processing
if isinstance(obs, tuple):
    obs = obs[0]

if not isinstance(obs, dict):
    raise ValueError(f"Expected dict observation, got {type(obs)}")

# Pattern 3: Safe extraction with defaults
def safe_get_blstat(blstats: Optional[np.ndarray], name: str, default: int = 0) -> int:
    idx = BLSTATS_IDX.get(name)
    if idx is None or blstats is None:
        return default
    try:
        if len(blstats) <= idx:
            return default
        return int(blstats[idx])
    except Exception:
        return default
```

### Logging and Output

Use print statements with emoji prefixes for visibility:

```python
print(f"Setting up LLM-Guided NetHack PPO Training...")
print(f"Using device: {self.device}")
print(f"  Steps log: {self.steps_file}")
print(f"  Interventions log: {self.interventions_file}")
print("=" * 80)
print("DOUBLY ROBUST CAUSAL ANALYSIS")
print("=" * 80)
```

### Docstrings

Use triple-quoted docstrings for classes and important functions:

```python
def compute_advantages(self, gamma=0.99, lam=0.95):
    """Compute GAE advantages"""
    ...

class DoublyRobustEstimator:
    """
    Doubly Robust Estimator for Causal Effect of LLM Advice
    
    Combines:
    1. Propensity score weighting (inverse probability weighting)
    2. Outcome regression (predicted counterfactuals)
    
    More robust than either method alone - if either model is correct, 
    the estimate is consistent.
    """
```

## Directory Structure

Key directories to know:
- `PPO_Agent/` - Main RL agent implementations (primary code)
- `PPO_Agent/tests/` - Unit tests
- `PPO_Agent/ppo_model_trained/` - Pre-trained models and evaluation scripts
- `BaseRL/` - Baseline RL with Stable Baselines3
- `log_parsers/` - Log parsing utilities
- `archive/` - Experimental and older versions
- `docs/` - Documentation
- `causal_analysis/` - Causal inference output

## Key Patterns

### PyTorch Neural Networks

```python
class NetworkModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NetworkModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

### Gymnasium Environment Handling

Handle both old (4-tuple) and new (5-tuple) return formats:

```python
step_result = env.step(action)
if len(step_result) == 4:
    next_obs, reward, done, info = step_result
    terminated = truncated = done
else:
    next_obs, reward, terminated, truncated, info = step_result
    done = terminated or truncated
```

### Model Save/Load

```python
def save_model(self, path):
    torch.save({
        'actor_state_dict': self.actor.state_dict(),
        'critic_state_dict': self.critic.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
    }, path)

def load_model(self, path):
    checkpoint = torch.load(path, map_location=self.device)
    self.actor.load_state_dict(checkpoint['actor_state_dict'])
```

## Dependencies

Key packages (see requirements.txt for full list):
- `torch` - Neural networks
- `gymnasium` - RL environment interface
- `nle` - NetHack Learning Environment
- `stable-baselines3` - Baseline RL algorithms
- `dowhy` - Causal inference
- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `requests`, `aiohttp` - LLM API calls

## Common Tasks

### Adding a New Reward Component
Edit `NetHackRewardShaper` in `PPO_Agent/base_RL extension with causal.py`

### Modifying LLM Prompts
Look for prompt construction in LLM-guided agent files

### Adding Causal Analysis Features
Extend `DoublyRobustEstimator` in `PPO_Agent/causal_analysis.py`

### Parsing Training Logs
Use `log_parsers/log_parser_v4.py` (latest version)
