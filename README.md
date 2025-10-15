# CausalRLAgent: NetHack RL Agent with LLM Guidance and Causal Analysis

A comprehensive reinforcement learning framework for NetHack that integrates Large Language Model (LLM) strategic guidance with causal inference analysis to understand and improve agent decision-making.

## 📋 Table of Contents

- Overview
- Key Features
- Project Structure
- Installation
- Quick Start
- Components
- Training
- Evaluation
- Causal Analysis
- Logging and Monitoring
- Advanced Usage
- Architecture
- Contributing
- Troubleshooting
- Citation

## 🎯 Overview

CausalRLAgent is a research framework that combines:

- **Deep Reinforcement Learning** using Proximal Policy Optimization (PPO)
- **LLM Strategic Guidance** via Ollama API for high-level decision support
- **Causal Inference** using DoWhy to analyze the impact of LLM advice
- **Comprehensive Logging** for building causal models of agent behavior

The framework targets NetHack, a complex roguelike game with:

- Partial observability
- Large action space (23 actions)
- Long-horizon dependencies
- Procedurally generated environments
- Sparse rewards

## ✨ Key Features

### 1. **LLM-Guided RL Agent**

- Semantic state interpretation converting raw observations to natural language
- Strategic advice from LLM (default: Ollama Phi model) every N steps
- Learned trust mechanism that adapts LLM influence based on performance
- Action mapping with fuzzy matching supporting 100+ action aliases

### 2. **Advanced PPO Implementation**

- Recurrent CNN + LSTM architecture for temporal modeling
- Sophisticated reward shaping with 10+ reward components
- Memory-augmented observation processing
- Proper handling of partial observability

### 3. **Causal Analysis Pipeline**

- Comprehensive logging of state transitions, actions, and LLM interventions
- DoWhy integration for causal effect estimation
- Multiple estimation strategies (IPW, Doubly Robust, PSM, Stratification)
- Semantic action alignment detection
- Episode-level causal metrics

### 4. **Rich Monitoring & Visualization**

- Real-time terminal output with colored formatting
- Episode summaries with reward breakdowns
- Training progress tracking with baseline comparisons
- Action distribution analysis
- LLM advice effectiveness metrics

## 📁 Project Structure

```
CausalRLAgent/
├── PPO_Agent/                      # Main RL agent implementations
│   ├── llm_guided.py              # 🔥 LLM-guided PPO agent (RECOMMENDED)
│   ├── agent.py                   # Base PPO agent
│   ├── fixed_training.py          # Training utilities
│   └── ppo_model_trained/         # Pre-trained models and evaluation
│       ├── improved_reward_shaping.py
│       ├── evaluate_model.py
│       ├── simple_evaluate.py
│       └── EVALUATION_GUIDE.md
│
├── BaseRL/                         # Baseline RL implementations
│   ├── trainv2.py                 # Baseline training with SB3
│   └── train.py                   # Alternative training script
│
├── archive/                        # Development versions and experiments
│   ├── llm_guided_v2.py          # Previous LLM-guided version
│   ├── llm_guided.ipynb          # Jupyter notebook experiments
│   ├── nethack_demo.py           # Environment demonstrations
│   ├── state_parser.py           # State interpretation utilities
│   └── llm_advisor.py            # LLM API wrappers
│
├── causal_analysis/                # Causal inference components
│   ├── causal_estimator.py       # 🔥 DoWhy-based causal analysis
│   ├── log_parser_v2.py          # 🔥 Advanced log parsing
│   └── log_parser.py             # Basic log parsing
│
├── docs/                           # Documentation
│   ├── nethack_causal_logging.md # Logging specification
│   └── Pre_reqs for DoWhy.md     # DoWhy setup guide
│
├── logs/                           # Training logs
├── models/                         # Saved model checkpoints
├── processed_data/                 # Parsed log data
├── processed_data_v2/              # 🔥 Latest parsed data with alignment
│   ├── processed_data.csv
│   ├── episode_summary.csv
│   ├── alignment_report.txt
│   └── metadata.json
│
├── tests/                          # Test data and utilities
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- Ollama (for LLM guidance) - [Installation Guide](https://ollama.ai/)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/CausalRLAgent.git
cd CausalRLAgent
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

- `torch` - PyTorch for neural networks
- `gymnasium` - RL environment interface
- `nle` - NetHack Learning Environment
- `dowhy` - Causal inference
- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `requests`, `aiohttp` - LLM API calls

### Step 4: Install Ollama (for LLM guidance)

```bash
# macOS/Linux
curl https://ollama.ai/install.sh | sh

# Pull the Phi model
ollama pull phi
```

### Step 5: Verify Installation

```bash
python -c "import gymnasium, nle, torch, dowhy; print('✅ All dependencies installed!')"
```

## 🏃 Quick Start

### 1. Train LLM-Guided Agent (Recommended)

```bash
cd PPO_Agent
python llm_guided.py
```

This will:

- Initialize an LLM-guided PPO agent
- Train for 100 episodes with real-time monitoring
- Save model, logs, and causal data
- Display comprehensive training statistics

Expected output:

```
🚀 Setting up LLM-Guided NetHack PPO Training with Monitoring...
Using device: cuda
Environment action space: 23

================================================================================
NetHack RL Training Monitor - LLM-Guided PPO
Started: 2025-01-13 14:30:00
================================================================================

Episode 0 Starting...
────────────────────────────────────────────────────────────────────────────────

╔══════════════════════════════════════════════════════════════════════════════╗
║ LLM STRATEGIC ADVICE (Step 10)                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Priority: Explore safely while maintaining health                            ║
║ Risk: Low health with unknown enemies nearby                                 ║
║ Opportunity: Items visible to the east                                       ║
║ Actions: move_east, search, pickup                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 2. Parse Training Logs

```bash
python log_parser_v2.py --log_file logs/nethack_logs.txt --output_dir processed_data_v2
```

This generates:

- `processed_data.csv` - Step-level data
- `episode_summary.csv` - Episode-level metrics
- alignment_report.txt - LLM advice alignment analysis
- `metadata.json` - Dataset metadata

### 3. Run Causal Analysis

```bash
python causal_estimator.py --data processed_data_v2/episode_summary.csv --output causal_analysis
```

This estimates:

- **Treatment**: Following LLM advice vs. not following
- **Outcomes**: Episode reward, length, survival rate
- **Methods**: Inverse Propensity Weighting, Doubly Robust, PSM
- **Refutations**: Placebo treatment, data subset tests

Output:

```
================================================================================
CAUSAL ANALYSIS: episode_reward
================================================================================

[1/4] Creating causal model...
✓ Causal graph created
  Treatment: followed_advice_lenient
  Outcome: episode_reward
  Confounders: hp_mean, level_mean, episode_length

[2/4] Identifying causal effect...
✓ Identified estimand using backdoor criterion

[3/4] Estimating causal effect...
backdoor.propensity_score_weighting    ATE =   2.3456
backdoor.propensity_score_matching     ATE =   2.1234
backdoor.propensity_score_stratification ATE =   2.4567

--- Doubly Robust Estimate ---
ATE = 2.3456 (SE = 0.1234)
95% CI = [2.1037, 2.5875]
p-value = 0.0123
✓ Statistically significant at α=0.05
```

### 4. Evaluate Trained Model

```bash
cd PPO_Agent/ppo_model_trained
python evaluate_model.py --model_path enhanced_nethack_ppo_20251013_145217.pth --episodes 10
```

See EVALUATION_GUIDE.md for detailed evaluation instructions.

## 🧩 Components

### Core Agents

#### 1. **LLM-Guided PPO Agent** (`llm_guided.py`)

The main agent integrating LLM strategic guidance with PPO.

**Key Classes:**

- `MonitoredLLMGuidedNetHackAgent` - Complete agent with monitoring
- `ImprovedLLMGuidedPPOActor` - Actor network with LLM integration
- `RecurrentPPOCritic` - Value network with LSTM
- `ImprovedLLMStrategicAdvisor` - LLM API wrapper

**Usage:**

```python
agent = MonitoredLLMGuidedNetHackAgent(
    action_dim=23,
    llm_guidance_weight=1.0,      # LLM influence (0-1)
    llm_call_frequency=20,        # Call LLM every N steps
    baseline_metrics_path=None    # Optional baseline for comparison
)

# Train
for episode in range(100):
    await agent.train_episode_monitored(env, episode)
```

#### 2. **Enhanced PPO Agent** (`improved_reward_shaping.py`)

Baseline PPO agent with advanced reward shaping.

**Reward Components:**

```python
NetHackRewardShaper:
  - Exploration: +0.01 per new position
  - Health: +0.0001 per HP gained
  - Level up: +5.0
  - Experience: +0.00001 per XP
  - Gold: +0.001 per gold piece
  - Monster kill: +1.0
  - Item pickup: +0.1
  - Stairs: +2.0
  - Death penalty: -5.0
  - Stuck penalty: -0.005
```

### Causal Analysis

#### 1. **Causal Effect Estimator** (`causal_estimator.py`)

Estimates causal effects using DoWhy.

**Methods:**

- **Inverse Propensity Weighting (IPW)**: Weights observations by inverse of propensity score
- **Doubly Robust (DR)**: Robust to misspecification of either propensity or outcome model
- **Propensity Score Matching (PSM)**: Matches treated/control units by propensity score
- **Stratification**: Stratifies by propensity score and averages within strata

**Example:**

```python
estimator = CausalEffectEstimator(
    data_path='processed_data_v2/episode_summary.csv',
    treatment_col='followed_advice_lenient',
    outcome_cols=['episode_reward', 'episode_length', 'survival_rate'],
    confounder_cols=['hp_mean', 'level_mean', 'steps_count']
)

# Run complete analysis
estimator.run_all_analyses()

# Generate report
summary_df = estimator.generate_report('causal_analysis')
```

#### 2. **Log Parser** (`log_parser_v2.py`)

Parses training logs with semantic action alignment detection.

**Features:**

- Semantic matching of LLM suggestions to actual actions
- Configurable alignment window (default: 10 steps)
- Episode-level alignment metrics
- Detailed alignment reports

**Usage:**

```bash
python log_parser_v2.py \
  --log_file logs/training.log \
  --output_dir processed_data_v2 \
  --window 10 \
  --debug
```

### Monitoring & Logging

#### 1. **Training Monitor** (`llm_guided.py`)

Rich terminal output during training.

**Features:**

- Colored episode progress display
- Real-time LLM advice visualization
- Reward breakdown per step
- Episode summaries with action distributions
- Rolling average statistics
- Baseline comparisons

#### 2. **Causal Model Logger** (`llm_guided.py`)

Comprehensive data logging for causal analysis.

**Captures:**

- State transitions: (s*t, a_t, s*{t+1}, r\_{t+1})
- LLM interventions: advice content, timing, state context
- Action outcomes: reward, health changes, level changes
- Episode metrics: survival, exploration, LLM effectiveness

**Output Files:**

```
causal_log_20251013_143000_episodes.json        # Episode-level data
causal_log_20251013_143000_transitions.json     # Step-level transitions
causal_log_20251013_143000_llm_interventions.json # LLM advice log
causal_log_20251013_143000_action_statistics.json # Action outcome stats
```

## 🎓 Training

### Basic Training

```bash
cd PPO_Agent
python llm_guided.py
```

### Advanced Training Options

**Modify hyperparameters:**

```python
agent = MonitoredLLMGuidedNetHackAgent(
    action_dim=23,
    learning_rate=3e-4,           # Learning rate
    gamma=0.99,                   # Discount factor
    clip_ratio=0.2,               # PPO clip ratio
    llm_guidance_weight=1.0,      # LLM influence
    llm_call_frequency=20         # LLM call frequency
)
```

**Training loop control:**

```python
num_episodes = 100                # Total episodes
update_frequency = 2048           # Update every N steps
MAX_STEPS = 1000                  # Max steps per episode
NO_PROGRESS_THRESHOLD = 200       # Early stopping threshold
```

### Training with Baseline Comparison

If you have baseline metrics from a pure RL agent:

```bash
python llm_guided.py metrics_baseline_20251012_120000.json
```

The monitor will display comparisons:

```
Comparison with Baseline:
  Reward Improvement:  +15.3% (+3.4 points)
  Length Improvement:  +22.1% (+45 steps)
  Death Rate:          -8.2% (fewer deaths)
```

### Baseline Training (No LLM)

To train a baseline PPO agent without LLM guidance:

```bash
cd BaseRL
python trainv2.py --timesteps 50000
```

Or using Stable-Baselines3:

```bash
cd BaseRL
python train.py --algo PPO --timesteps 100000
```

## 📊 Evaluation

### Quick Evaluation

```bash
cd PPO_Agent/ppo_model_trained
python simple_evaluate.py --model_path model.pth --episodes 10
```

### Comprehensive Evaluation

```bash
python evaluate_model.py \
  --model_path enhanced_nethack_ppo_20251013_145217.pth \
  --episodes 20 \
  --max_steps 5000 \
  --render \
  --save_trajectories \
  --output_dir evaluation_results
```

**Options:**

- `--render`: Show game visually (slower)
- `--save_trajectories`: Save detailed step-by-step data
- `--episodes N`: Number of episodes to evaluate
- `--max_steps N`: Maximum steps per episode

**Output:**

- Comprehensive dashboard visualization
- JSON results file with detailed metrics
- Action frequency analysis
- Survival statistics
- Performance recommendations

See EVALUATION_GUIDE.md for complete evaluation guide.

## 🔬 Causal Analysis

### Step 1: Parse Logs

```bash
python log_parser_v2.py \
  --log_file logs/llm_guided_causal_20251013_143000.txt \
  --output_dir processed_data_v2 \
  --window 10
```

### Step 2: Inspect Alignment

```bash
cat processed_data_v2/alignment_report.txt
```

Look for:

- **Lenient alignment**: ≥1 semantic match within window
- **Strict alignment**: Immediate match (step 0)
- **Common suggestion-action patterns**
- **Sample matched/non-matched episodes**

### Step 3: Estimate Causal Effects

```bash
python causal_estimator.py \
  --data processed_data_v2/episode_summary.csv \
  --output causal_analysis \
  --treatment followed_advice_lenient
```

**Treatment Options:**

- `followed_advice_lenient` - Any semantic match within window
- `followed_advice_strict` - Immediate action match
- `episode_matched_any` - Any match during episode

**Outcome Options:**

- `episode_reward` - Total episode reward
- `episode_shaped_reward` - Shaped reward
- `episode_length` - Episode length in steps
- `survival_rate` - Did agent survive? (0/1)

### Step 4: Interpret Results

**Average Treatment Effect (ATE):**

- Positive ATE: Following LLM advice improves outcome
- Negative ATE: Following LLM advice hurts outcome
- Near-zero ATE: No causal effect

**Statistical Significance:**

- p-value < 0.05: Statistically significant
- 95% CI excludes zero: Strong evidence of effect

**Sensitivity Tests:**

- Placebo treatment: Should show no effect
- Random common cause: Effect should be robust
- Data subset: Effect should persist in subsamples

### Advanced Causal Analysis

**Custom confounders:**

```python
estimator = CausalEffectEstimator(
    data_path='processed_data_v2/episode_summary.csv',
    treatment_col='followed_advice_lenient',
    outcome_cols=['episode_reward'],
    confounder_cols=[
        'hp_mean',              # Average health
        'level_mean',           # Average dungeon level
        'steps_count',          # Episode length
        'llm_call_count',       # Number of LLM calls
        'exploration_ratio'     # Exploration rate
    ]
)
```

**Custom DoWhy analysis:**

```python
# Build causal model
model = CausalModel(
    data=df,
    treatment='followed_advice_lenient',
    outcome='episode_reward',
    common_causes=['hp_mean', 'level_mean'],
    effect_modifiers=[]
)

# Identify effect
identified_estimand = model.identify_effect()

# Estimate effect
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting"
)

print(f"ATE: {estimate.value}")
```

## 📝 Logging and Monitoring

### Log File Formats

#### Training Logs

```
Episode 5 Starting...
────────────────────────────────────────────────────────────────────────────────

Step    0 | Action: search          | R:   0.00 | SR:   0.010 | HP:  80.0% | Lvl: 1
Step    1 | Action: move_east       | R:   0.00 | SR:   0.020 | HP:  80.0% | Lvl: 1

╔══════════════════════════════════════════════════════════════════════════════╗
║ LLM STRATEGIC ADVICE (Step 10)                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Priority: Find food to avoid starvation                                      ║
║ Risk: Hunger level critical                                                  ║
║ Opportunity: Corridor to the south may have items                            ║
║ Actions: search, move_south, eat                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Step   10 | Action: move_south      | R:   0.00 | SR:   0.025 | HP:  78.5% | Lvl: 1
```

#### Causal Logs (JSON)

```json
{
  "episode_id": 5,
  "start_time": "2025-01-13T14:30:00",
  "steps": [
    {
      "step": 0,
      "state_t": {
        "health_ratio": 0.8,
        "level": 1,
        "depth": 1,
        "gold": 0
      },
      "action": 11,
      "action_name": "search",
      "reward": 0.0,
      "shaped_reward": 0.01,
      "llm_advice_active": false
    }
  ],
  "llm_calls": [
    {
      "step": 10,
      "advice": {
        "immediate_priority": "Find food",
        "action_suggestions": ["search", "move_south", "eat"]
      }
    }
  ]
}
```

### Monitoring Outputs

**Episode Summary:**

```
Episode 5 Complete
────────────────────────────────────────────────────────────────────────────────

Episode Metrics:
  Raw Reward:        12.50
  Shaped Reward:     25.30
  Length:           150 steps
  Survival Time:      2.5 minutes (simulated)
  LLM Calls:          7

Action Distribution:
  search:           25 (16.7%)
  move_east:        20 (13.3%)
  move_south:       18 (12.0%)
  pickup:           12 ( 8.0%)
  eat:               8 ( 5.3%)
```

**Training Progress:**

```
Training Progress (Last 10 Episodes)
────────────────────────────────────────────────────────────────────────────────
  Avg Raw Reward:       10.35
  Avg Shaped Reward:    22.47
  Avg Length:          145.2 steps

  Episodes Completed: 50
  Training Time:       42.3 minutes
  Episodes/Hour:       71.0
```

## 🔧 Advanced Usage

### Custom Reward Shaping

Modify `NetHackRewardShaper`:

```python
class CustomRewardShaper(NetHackRewardShaper):
    def __init__(self):
        super().__init__()

        # Adjust reward weights
        self.exploration_reward = 0.02        # Encourage exploration
        self.health_reward = 0.0005           # Prioritize health
        self.level_reward = 10.0              # Emphasize level ups
        self.death_penalty = -10.0            # Harsher death penalty

    def shape_reward(self, obs, raw_reward, done, info):
        shaped_reward = super().shape_reward(obs, raw_reward, done, info)

        # Add custom shaping
        if self._detected_treasure(obs):
            shaped_reward += 1.0

        return shaped_reward
```

### Custom LLM Prompts

Modify `ImprovedLLMStrategicAdvisor.get_strategic_advice()`:

```python
prompt = f"""You are an expert NetHack player. Analyze the situation:

{semantic_description}

CUSTOM RULES:
1. Prioritize survival above all
2. Avoid unnecessary combat
3. Focus on finding food when hunger < 500

Respond in JSON:
{{
  "immediate_priority": "...",
  "risk_assessment": "...",
  "action_suggestions": ["action1", "action2", "action3"]
}}
"""
```

### Custom Observation Processing

Extend `NetHackObservationProcessor`:

```python
class CustomObsProcessor(NetHackObservationProcessor):
    def process_observation(self, obs, last_action=None):
        processed = super().process_observation(obs, last_action)

        # Add custom features
        processed['nearby_monsters'] = self._count_nearby_monsters(obs)
        processed['items_in_inventory'] = self._count_items(obs)
        processed['hunger_level'] = self._extract_hunger(obs)

        return processed
```

### Distributed Training

For multi-GPU training:

```python
import torch.distributed as dist

# Initialize
dist.init_process_group(backend='nccl')

# Create agent on specific GPU
device = torch.device(f"cuda:{dist.get_rank()}")
agent = MonitoredLLMGuidedNetHackAgent(device=device)

# Train
for episode in range(num_episodes):
    await agent.train_episode_monitored(env, episode)
```

## 🏗️ Architecture

### Network Architecture

```
INPUT: NetHack Observation
├─ Glyphs (21x79)
│  └─ RecurrentNetHackCNN
│     ├─ Conv2d(1→32, k=3)
│     ├─ Conv2d(32→64, k=3)
│     ├─ Conv2d(64→128, k=3)
│     └─ LSTM(512→256)
│
├─ Stats (26 features)
│  └─ LSTM(26→64)
│
├─ Message (256 features)
│  └─ Linear(256→128)
│
├─ Inventory (55 features)
│  └─ Linear(55→64)
│
├─ Action History (50 features)
│  └─ Linear(50→32)
│
└─ LLM Guidance (32 features)
   └─ Linear(32→64)

COMBINED FEATURES (608)
├─ Linear(608→512) + ReLU
└─ Linear(512→256) + ReLU

OUTPUT
├─ Actor Head: Linear(256→23) → Action Logits
└─ Critic Head: Linear(256→1) → Value Estimate
```

### Data Flow

```
Environment → Observation
    ↓
Observation Processor → Processed Features
    ↓
Semantic Descriptor → Natural Language
    ↓
LLM Advisor (every N steps) → Strategic Advice
    ↓
Actor Network (with LLM guidance) → Action Probabilities
    ↓
Action Selection → Environment Step
    ↓
Reward Shaper → Shaped Reward
    ↓
Causal Logger → Log Transitions
    ↓
PPO Buffer → Store Experience
    ↓
PPO Update (every M steps) → Update Networks
```

### Causal Analysis Pipeline

```
Training Logs
    ↓
Log Parser → Extract episodes, detect alignment
    ↓
processed_data.csv + episode_summary.csv
    ↓
Causal Estimator
├─ Propensity Score Estimation (Logistic Regression)
├─ Outcome Model (Random Forest)
├─ IPW Estimation
├─ Doubly Robust Estimation
├─ PSM
├─ Stratification
└─ Refutation Tests
    ↓
Causal Effect Estimates + Sensitivity Analysis
    ↓
causal_report.txt + visualizations
```

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Improved LLM prompting strategies
- Alternative causal inference methods
- Better reward shaping functions
- Visualization tools
- Documentation improvements

## ❗ Troubleshooting

### Common Issues

**1. Ollama not responding**

```
Error: Ollama API call failed: Connection refused
```

**Solution:**

```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Verify model is available
ollama pull phi
```

**2. CUDA out of memory**

```
RuntimeError: CUDA out of memory
```

**Solution:**

```python
# Reduce batch size
batch_size = 32  # Down from 64

# Reduce buffer size
self.buffer = PPOBuffer(max_size=1024)  # Down from 2048

# Use CPU
agent = MonitoredLLMGuidedNetHackAgent(device=torch.device('cpu'))
```

**3. NetHack environment errors**

```
ImportError: No module named 'nle'
```

**Solution:**

```bash
pip install nle

# If that fails, install dependencies:
# macOS
brew install cmake

# Ubuntu
sudo apt-get install cmake libncurses5-dev flex bison

# Then retry
pip install nle
```

**4. DoWhy estimation errors**

```
ValueError: Not enough data points for estimation
```

**Solution:**

- Train for more episodes (need >50 episodes for reliable estimates)
- Reduce confounder count
- Check data quality in `episode_summary.csv`

**5. Log parsing issues**

```
No episodes found in log file
```

**Solution:**

```bash
# Check log file format
head -20 logs/training.log

# Verify log contains episode markers
grep "Episode.*Starting" logs/training.log

# Use debug mode
python log_parser_v2.py --log_file logs/training.log --debug
```

### Performance Tips

**Training:**

- Use GPU for 5-10x speedup
- Reduce LLM call frequency (20-50 steps) for faster training
- Use early stopping if no progress after N steps
- Save checkpoints regularly

**Evaluation:**

- Start with 5-10 episodes for quick analysis
- Use `--render` only for debugging
- Save trajectories only when needed (large files)
- Run 20+ episodes for statistical significance

**Causal Analysis:**

- Ensure balanced treatment groups (30%+ in each)
- Include relevant confounders (health, level, episode stage)
- Use lenient alignment for more data points
- Validate with multiple estimation methods

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{causalrlagent2025,
  author = {Your Name},
  title = {CausalRLAgent: NetHack RL with LLM Guidance and Causal Analysis},
  year = {2025},
  url = {https://github.com/yourusername/CausalRLAgent}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NetHack Learning Environment (NLE) team
- DoWhy causal inference library
- Ollama for local LLM inference
- NetHack DevTeam for the game

## 📞 Support

For questions or issues:

1. Check Troubleshooting
2. Review documentation in docs
3. Open an issue on GitHub
4. Contact: your.email@example.com

---

**Happy Training! 🎮🤖🔬**
