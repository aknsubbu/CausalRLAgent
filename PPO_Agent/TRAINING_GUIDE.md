# Training Guide

Step-by-step instructions for training CausalRLAgent on NetHack.

---

## Prerequisites

1. **Python 3.8+** with dependencies installed
2. **Ollama** running locally with `llama3:8b` model
3. **NetHack Learning Environment** (`nle` package)

```bash
# Verify Ollama is running
curl http://localhost:11434/api/version

# Expected output: {"version":"0.x.x"}
```

---

## Quick Start

### 1. Basic Training (No LLM)

```bash
cd PPO_Agent
python -c "
import asyncio
from causal_filtered import LLMEnhancedNetHackAgent, create_nethack_env

async def train():
    env = create_nethack_env()
    agent = LLMEnhancedNetHackAgent(
        action_dim=23,
        enable_llm=False  # No LLM guidance
    )
    await agent.train(env, num_episodes=50)
    env.close()

asyncio.run(train())
"
```

### 2. LLM-Guided Training

```bash
cd PPO_Agent
python -c "
import asyncio
from causal_filtered import LLMEnhancedNetHackAgent, create_nethack_env

async def train():
    env = create_nethack_env()
    agent = LLMEnhancedNetHackAgent(
        action_dim=23,
        enable_llm=True,
        llm_guidance_weight=0.05
    )
    await agent.train(env, num_episodes=100)
    agent.save_model('llm_agent.pth')
    env.close()

asyncio.run(train())
"
```

### 3. Adversarial Experiments

```bash
cd PPO_Agent
python -c "
import asyncio
from causal_filtered import (
    AdversarialLLMEnhancedAgent,
    AdversarialAttacker,
    AdversarialAttackType,
    create_nethack_env
)

async def train():
    env = create_nethack_env()

    attacker = AdversarialAttacker(
        attack_type=AdversarialAttackType.STRATEGIC_POISONING,
        attack_strength=0.5
    )

    agent = AdversarialLLMEnhancedAgent(
        action_dim=23,
        enable_llm=True,
        attacker=attacker
    )

    await agent.train(env, num_episodes=50)
    env.close()

asyncio.run(train())
"
```

---

## Configuration Reference

### Agent Parameters

| Parameter             | Default | Description                     |
| --------------------- | ------- | ------------------------------- |
| `action_dim`          | 23      | NetHack action space size       |
| `learning_rate`       | 1e-4    | Adam optimizer learning rate    |
| `gamma`               | 0.99    | Reward discount factor          |
| `clip_ratio`          | 0.2     | PPO clipping parameter          |
| `entropy_coef`        | 0.02    | Entropy bonus coefficient       |
| `value_coef`          | 0.5     | Value loss coefficient          |
| `max_grad_norm`       | 0.5     | Gradient clipping threshold     |
| `enable_llm`          | False   | Enable LLM guidance             |
| `llm_guidance_weight` | 0.05    | How much LLM influences actions |

### Training Parameters

| Parameter      | Default | Description                     |
| -------------- | ------- | ------------------------------- |
| `num_episodes` | 100     | Number of training episodes     |
| `update_freq`  | 1024    | PPO update frequency (steps)    |
| `print_freq`   | 10      | Print progress every N episodes |

### LLM Advisor Parameters

| Parameter        | Default     | Description              |
| ---------------- | ----------- | ------------------------ |
| `call_frequency` | 50          | Call LLM every N steps   |
| Model            | `llama3:8b` | Ollama model name        |
| Temperature      | 0.2         | LLM sampling temperature |

---

## Output Files

Training produces:

| File                                   | Description                         |
| -------------------------------------- | ----------------------------------- |
| `*.pth`                                | PyTorch model checkpoint            |
| `*_monitor.json`                       | Training metrics (rewards, lengths) |
| `causal_logs_v2/steps_*.jsonl`         | Step-level data for causal analysis |
| `causal_logs_v2/interventions_*.jsonl` | LLM call data                       |

---

## Monitoring Training

### Console Output

```
🚀 Training for 100 episodes
   LLM Guidance: ENABLED (Enhanced Descriptor)
   LLM Guidance Weight: 0.050
   LLM Call Frequency: Every 50 steps

📈 Episode 0: Avg Reward: 12.500, Avg Length: 150.0, LLM Calls: 3
  📊 Step 1024: Actor Loss: 0.0234, Critic Loss: 0.1567
📈 Episode 10: Avg Reward: 25.300, Avg Length: 280.5, LLM Calls: 35

⚠️  ADVERSARIAL ATTACK APPLIED: strategic_poisoning
   Attack strength: 0.50
   Total attacks so far: 1
🤖 LLM Strategy (after attack): explore
```

### Key Metrics

| Metric     | Good Sign             | Bad Sign                |
| ---------- | --------------------- | ----------------------- |
| Avg Reward | Increasing over time  | Stuck negative          |
| Avg Length | 200+ steps            | \<50 steps (dying fast) |
| Actor Loss | Decreasing            | Oscillating wildly      |
| LLM Calls  | Proportional to steps | Too few (LLM disabled)  |

---

## Troubleshooting

### "Connection refused" to Ollama

```bash
# Start Ollama server
ollama serve

# In another terminal, verify model exists
ollama list
# Should show: llama3:8b
```

### Agent Gets Stuck

Symptoms: Episode lengths hit 5000, rewards negative

Solutions:

1. Increase `entropy_coef` (e.g., 0.05)
2. Reduce `llm_guidance_weight` (e.g., 0.01)
3. Check for adversarial attacks

### GPU Out of Memory

```python
# Reduce batch processing
agent = LLMEnhancedNetHackAgent(
    action_dim=23,
    # Hidden size is 256 by default in RecurrentNetHackCNN
)

# Or force CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

---

## Running Full Adversarial Experiments

Use the built-in experiment runner:

```python
import asyncio
from causal_filtered import run_adversarial_experiments

# This runs all 8 attack configurations
results = asyncio.run(run_adversarial_experiments())
```

Output:

- `adversarial_results.json` - All experiment results
- `adversarial_comparison_*.png` - Comparison plots
