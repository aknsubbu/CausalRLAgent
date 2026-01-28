# Causal Defense Against Adversarial Attacks on LLM-Guided RL
## Comprehensive Research Documentation

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Motivation & Problem Statement](#motivation--problem-statement)
3. [Causal Inference Framework](#causal-inference-framework)
4. [System Architecture](#system-architecture)
5. [Implementation Details](#implementation-details)
6. [Experimental Design](#experimental-design)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Key Findings & Analysis](#key-findings--analysis)
9. [Flowcharts](#flowcharts)
10. [Theoretical Foundations](#theoretical-foundations)

---

## Executive Summary

### Research Objective
Develop and evaluate a **causal inference-based defense mechanism** to protect LLM-guided reinforcement learning agents from adversarial attacks on semantic input descriptions.

### Core Innovation
**Doubly Robust Causal Filter**: A real-time, online learning system that:
- Predicts whether LLM advice will help or harm the agent
- Uses causal inference (propensity scoring + outcome modeling) to estimate treatment effects
- Automatically rejects harmful advice before it influences the agent's policy
- Requires no prior knowledge of attack types or patterns

### Main Research Question
**Can causal inference techniques automatically detect and filter adversarially poisoned LLM advice in real-time, protecting RL agent performance without manual intervention?**

### Hypothesis
By estimating the causal treatment effect of LLM advice on agent performance, we can:
1. Distinguish helpful from harmful advice with high accuracy
2. Recover 50-80% of performance loss caused by adversarial attacks
3. Adapt online without requiring pre-labeled attack examples

---

## Motivation & Problem Statement

### The Vulnerability Problem

**LLM-Guided RL Agents are Vulnerable:**
```
Raw Observations → Semantic Description → [ATTACK POINT] → LLM Reasoning → Action Hints → Agent Policy
```

Adversarial attacks on semantic descriptions can:
- Invert critical information (safe → dangerous)
- Inject misleading strategic advice
- Remove critical decision-making information
- Cause 20-70% performance degradation

### Why Traditional Defenses Fail

1. **Input Sanitization**: Cannot distinguish semantically valid but strategically harmful advice
2. **Adversarial Training**: Requires knowing attack patterns in advance
3. **Ensemble Methods**: All LLMs may be fooled by well-crafted semantic manipulations
4. **Output Filtering**: Cannot detect subtle strategic errors without understanding causality

### The Causal Solution

**Key Insight**: Instead of detecting attacks directly, estimate the **causal effect** of following LLM advice:
- **Treatment (T)**: Using LLM advice vs. not using it
- **Outcome (Y)**: Agent's reward/performance
- **Covariates (X)**: Game state features

If the estimated causal effect is negative → **Reject the advice**

---

## Causal Inference Framework

### Causal DAG (Directed Acyclic Graph)

```
State Features (X)
       │
       ├──────────────┐
       │              │
       ▼              ▼
LLM Treatment (T)  Outcome (Y)
       │              ▲
       └──────────────┘
     (Confounding)
```

**Variables:**
- **X** = State features (health, level, depth, recent rewards, episode progress)
- **T** = Binary treatment (1 = LLM advice used, 0 = baseline policy only)
- **Y** = Outcome (shaped reward for current step)

**Confounding**: Game state affects BOTH whether LLM is called AND the expected reward
- Example: Low health → More likely to call LLM for help → Also more likely to die
- Solution: Use doubly robust estimation to adjust for confounding

### Potential Outcomes Framework

**Notation:**
- Y(1) = Potential outcome if LLM advice is used
- Y(0) = Potential outcome if LLM advice is not used
- **ITE** (Individual Treatment Effect) = Y(1) - Y(0)

**Goal**: Estimate ITE for current state to predict if LLM advice will help

**Challenge**: We only observe ONE outcome (either Y(1) or Y(0)), not both

**Solution**: Use statistical models to estimate the unobserved counterfactual

### Doubly Robust Estimation

**Why "Doubly Robust"?**
The estimator is consistent if EITHER:
1. The propensity model is correct, OR
2. The outcome models are correct

This provides protection against model misspecification.

**Mathematical Formulation:**

For individual i, the doubly robust estimator is:

```
ITE_i = μ₁(X_i) - μ₀(X_i) + (T_i/e(X_i)) * (Y_i - μ₁(X_i)) - ((1-T_i)/(1-e(X_i))) * (Y_i - μ₀(X_i))
```

Where:
- **e(X_i)** = Propensity score P(T=1|X=X_i)
- **μ₁(X_i)** = Expected outcome with treatment E[Y|T=1, X=X_i]
- **μ₀(X_i)** = Expected outcome without treatment E[Y|T=0, X=X_i]

**Intuition:**
1. **Model-based part**: μ₁(X) - μ₀(X) predicts effect from outcome models
2. **Weighting correction**: IPW terms adjust for any model errors using propensity scores
3. **Double protection**: If one model is wrong, the other can compensate

### Advantages Over Alternatives

| Method | Pros | Cons |
|--------|------|------|
| **Simple Difference** | Easy to compute | Biased by confounding |
| **Propensity Weighting** | Unconfounded estimates | Unstable with extreme weights |
| **Outcome Regression** | Uses all data efficiently | Biased if model wrong |
| **Doubly Robust** | Robust to one model failure | Requires fitting 3 models |

---

## System Architecture

### High-Level Defense System

```
┌─────────────────────────────────────────────────────────────┐
│                    NetHack Environment                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ Raw Observations
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Semantic Description Module                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ Clean Description
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            ⚠️  ADVERSARIAL ATTACKER ⚠️                       │
│        (Strategic Poisoning, State Inversion, etc.)         │
└──────────────────────┬──────────────────────────────────────┘
                       │ Attacked Description
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 LLM Advisor (Ollama)                         │
│          (Generates advice from attacked input)              │
└──────────────────────┬──────────────────────────────────────┘
                       │ LLM Advice (Potentially Harmful)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           🛡️  ONLINE CAUSAL FILTER 🛡️                       │
│                                                              │
│  Step 1: Extract State Features (X)                         │
│  Step 2: Predict Propensity Score e(X)                      │
│  Step 3: Predict Outcomes μ₁(X), μ₀(X)                      │
│  Step 4: Calculate ITE = μ₁(X) - μ₀(X)                      │
│  Step 5: Decision: Accept if ITE ≥ threshold               │
│                                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │ Filtered Advice (or None)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PPO Agent (Actor-Critic)                    │
│         (Uses advice only if filter approves)                │
└──────────────────────┬──────────────────────────────────────┘
                       │ Action
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Environment Step & Reward                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ Reward Feedback
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Data Collection for Causal Models                 │
│   (Store: state, treatment, outcome for model updates)      │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Online Learning**: Models update continuously during training (no pre-training required)
2. **Lightweight**: Uses Random Forests (50 trees, depth 6) for fast inference
3. **Warmup Period**: Collects data for 500 steps before making predictions
4. **Adaptive**: Updates every 100 steps to adapt to changing game dynamics
5. **Conservative**: Default threshold = -0.01 (reject only if clearly harmful)

---

## Implementation Details

### Component Breakdown

#### 1. OnlineCausalFilter Class

**Core Responsibilities:**
- Collect data during agent-environment interaction
- Train and update causal models periodically
- Predict treatment effects in real-time
- Make accept/reject decisions

**Key Methods:**

```python
class OnlineCausalFilter:
    def __init__(self, warmup_steps=500, update_frequency=100):
        # Data collection
        self.step_history = []           # Stores (X, T, Y) tuples
        self.current_step = 0
        
        # Causal models
        self.propensity_model = None     # P(T=1|X)
        self.outcome_model_with_llm = None    # E[Y|T=1,X]
        self.outcome_model_without_llm = None # E[Y|T=0,X]
        self.scaler = StandardScaler()   # Feature normalization
        
        # Tracking
        self.rejections = 0
        self.acceptances = 0
        self.predictions_total = 0
```

**Feature Engineering:**

```python
feature_cols = [
    'health_ratio',      # HP/MaxHP (0.0 to 1.0)
    'level_norm',        # Level/30 (normalized)
    'depth_norm',        # Dungeon depth/50 (normalized)
    'reward_lag1',       # Previous step's reward
    'reward_ma5',        # 5-step moving average reward
    'episode_progress'   # Current step / episode length
]
```

**Why These Features?**
- **Health**: Strong predictor of both LLM calls and survival
- **Level/Depth**: Proxy for game difficulty and agent skill
- **Reward history**: Captures recent performance trends
- **Episode progress**: Early vs. late game have different dynamics

#### 2. Data Collection

**What Gets Collected:**

```python
def collect_data(self, obs, processed_obs, action, reward, 
                 shaped_reward, llm_advice_given, episode_step, episode_length):
    
    # Extract state features
    stats = obs.get('blstats', np.zeros(26))
    health_ratio = hp / max_hp if max_hp > 0 else 0.5
    level_norm = level / 30.0
    depth_norm = depth / 50.0
    
    # Lagged features
    reward_lag1 = self.step_history[-1]['shaped_reward'] if self.step_history else 0
    reward_ma5 = np.mean([s['shaped_reward'] for s in self.step_history[-5:]])
    
    # Episode context
    episode_progress = episode_step / episode_length
    
    data_point = {
        'step': self.current_step,
        'health_ratio': health_ratio,
        'level_norm': level_norm,
        'depth_norm': depth_norm,
        'reward': reward,
        'shaped_reward': shaped_reward,      # This is Y (outcome)
        'reward_lag1': reward_lag1,
        'reward_ma5': reward_ma5,
        'episode_progress': episode_progress,
        'llm_advice_given': llm_advice_given,  # This is T (treatment)
        'action': action
    }
    
    self.step_history.append(data_point)
```

**Data Flow:**
```
Every Step → Extract Features → Store (X, T, Y) → Buffer (last 2000 steps)
Every 100 Steps → Update Models from Buffer
```

#### 3. Model Training

**Propensity Model (Classification):**

```python
# Predict P(T=1|X) - probability of using LLM advice
self.propensity_model = RandomForestClassifier(
    n_estimators=50,      # 50 trees (fast but accurate)
    max_depth=6,          # Prevent overfitting
    min_samples_leaf=20,  # Require sufficient data
    random_state=42
)

X_scaled = self.scaler.fit_transform(X)  # Standardize features
self.propensity_model.fit(X_scaled, T)
```

**Outcome Models (Regression):**

```python
# Model 1: E[Y|T=1, X] - expected reward WITH LLM advice
with_llm_mask = (T == 1)
if np.sum(with_llm_mask) > 50:  # Need sufficient data
    self.outcome_model_with_llm = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42
    )
    self.outcome_model_with_llm.fit(X_scaled[with_llm_mask], 
                                     Y[with_llm_mask])

# Model 2: E[Y|T=0, X] - expected reward WITHOUT LLM advice
without_llm_mask = (T == 0)
if np.sum(without_llm_mask) > 50:
    self.outcome_model_without_llm = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42
    )
    self.outcome_model_without_llm.fit(X_scaled[without_llm_mask], 
                                        Y[without_llm_mask])
```

**Why Random Forests?**
- Fast training and inference (critical for online learning)
- Handle non-linear relationships naturally
- Robust to feature scaling issues
- Provide implicit feature selection
- Good with small-to-medium datasets (2000 samples)

**Update Schedule:**
```python
if current_step >= warmup_steps and current_step % update_frequency == 0:
    update_causal_models()
```

#### 4. Real-Time Prediction

**Decision-Making Process:**

```python
def should_accept_llm_advice(self, obs, processed_obs, episode_step, episode_length):
    
    # Phase 1: Warmup (always accept)
    if self.current_step < self.warmup_steps:
        return True, 0.0, 0.0
    
    # Phase 2: Extract current state features
    features = extract_features(obs, episode_step, episode_length)
    features_scaled = self.scaler.transform(features)
    
    # Phase 3: Predict counterfactual outcomes
    Y1_pred = self.outcome_model_with_llm.predict(features_scaled)[0]
    Y0_pred = self.outcome_model_without_llm.predict(features_scaled)[0]
    
    # Phase 4: Estimate treatment effect
    predicted_effect = Y1_pred - Y0_pred
    
    # Phase 5: Decision with threshold
    THRESHOLD = -0.01
    should_accept = (predicted_effect >= THRESHOLD)
    
    if should_accept:
        self.acceptances += 1
    else:
        self.rejections += 1
        print(f"🚫 REJECTED (predicted effect: {predicted_effect:.4f})")
    
    return should_accept, predicted_effect, confidence
```

**Decision Logic:**

| Predicted Effect | Interpretation | Decision |
|-----------------|----------------|----------|
| > +0.05 | Strongly helpful | ✅ Accept |
| 0.0 to +0.05 | Slightly helpful | ✅ Accept |
| -0.01 to 0.0 | Negligible harm | ✅ Accept |
| < -0.01 | Harmful | 🚫 Reject |

**Threshold Tuning:**
- **Conservative (-0.01)**: Reject only clearly harmful advice (default)
- **Aggressive (-0.05)**: Accept only clearly helpful advice
- **Optimistic (0.0)**: Reject anything not beneficial

---

## Experimental Design

### Experimental Configurations

#### Configuration 1: Baseline (No LLM)
```python
{
    'name': 'Baseline (No LLM)',
    'use_llm': False,
    'attack_type': AdversarialAttackType.NONE,
    'attack_strength': 0.0,
    'use_causal': False
}
```
**Purpose**: Establish upper bound performance (no LLM, no attacks)

#### Configuration 2: LLM with Attack (Unprotected)
```python
{
    'name': 'Strategic Poisoning (Unprotected)',
    'use_llm': True,
    'attack_type': AdversarialAttackType.STRATEGIC_POISONING,
    'attack_strength': 0.8,
    'use_causal': False
}
```
**Purpose**: Demonstrate vulnerability (how much attack degrades performance)

#### Configuration 3: LLM with Attack + Causal Defense
```python
{
    'name': 'Strategic Poisoning (Causal Protected)',
    'use_llm': True,
    'attack_type': AdversarialAttackType.STRATEGIC_POISONING,
    'attack_strength': 0.8,
    'use_causal': True
}
```
**Purpose**: Evaluate defense effectiveness (how much performance is recovered)

### Attack Types Tested

| Attack | Strength | Why Chosen |
|--------|----------|------------|
| **Strategic Poisoning** | 0.8 | Most sophisticated - context-aware harmful advice |
| **State Inversion** | 0.8 | Tests if filter detects semantically flipped states |
| **Misleading Context** | 0.7 | Tests filtering of authoritative-sounding misinformation |
| **Critical Info Removal** | 0.6 | Tests decision-making under uncertainty |

### Hyperparameters

#### Causal Filter Parameters
```python
warmup_steps = 500          # Steps to collect before predicting
update_frequency = 100      # Steps between model updates
rejection_threshold = -0.01 # Minimum acceptable effect
buffer_size = 2000          # Historical data retained
```

#### Model Parameters
```python
# Random Forest (all models)
n_estimators = 50           # Number of trees
max_depth = 6               # Maximum tree depth
min_samples_leaf = 20       # Minimum samples per leaf
```

#### Training Parameters
```python
num_episodes = 50           # Episodes per experiment
update_freq = 512           # RL update frequency
learning_rate = 1e-4        # PPO learning rate
llm_guidance_weight = 0.05  # Weight of LLM hints
```

### Experimental Procedure

```
FOR EACH attack configuration:
    1. Initialize environment
    2. Create adversarial attacker with specified type and strength
    3. Create causal-protected agent (if applicable)
    4. Initialize monitoring
    
    5. TRAINING LOOP (50 episodes):
        FOR EACH episode:
            Reset environment
            WHILE not done:
                - Observe state
                - If LLM called:
                    * Generate semantic description
                    * Apply adversarial attack
                    * Get LLM advice
                    * [CAUSAL FILTER] Predict effect and decide
                - Select action (with or without LLM advice)
                - Execute action, observe reward
                - Collect data for causal models
                - Update causal models every 100 steps
                - Update RL policy every 512 steps
            Log episode metrics
    
    6. Save results and generate visualizations
    7. Calculate recovery metrics vs baseline and unprotected
```

---

## Evaluation Metrics

### Primary Metrics

#### 1. Performance Recovery Rate

**Definition:**
```
Recovery Rate = (Perf_Protected - Perf_Unprotected) / (Perf_Baseline - Perf_Unprotected) × 100%
```

**Interpretation:**
- **100%**: Complete protection (recovered all performance)
- **50-80%**: Strong protection (most damage mitigated)
- **0-50%**: Partial protection (some benefit)
- **<0%**: Defense actually harmed performance

**Example:**
```
Baseline:     10.0 reward
Unprotected:   4.0 reward (6.0 loss)
Protected:     8.0 reward (4.0 recovered)

Recovery Rate = (8.0 - 4.0) / (10.0 - 4.0) × 100% = 66.7%
```

#### 2. Rejection Rate

**Definition:**
```
Rejection Rate = Rejections / Total_LLM_Calls × 100%
```

**Interpretation:**
- **High rejection (>50%)**: Filter is aggressive or attack is severe
- **Medium rejection (20-50%)**: Selective filtering
- **Low rejection (<20%)**: Conservative filtering or weak attack

**Trade-off:**
- Too high → May reject helpful advice (false positives)
- Too low → May accept harmful advice (false negatives)

#### 3. Prediction Accuracy (Post-hoc)

**Requires labeled data:**
```
Accuracy = (True_Positives + True_Negatives) / Total_Predictions

Where:
- True Positive: Rejected harmful advice correctly
- True Negative: Accepted helpful advice correctly
- False Positive: Rejected helpful advice (Type I error)
- False Negative: Accepted harmful advice (Type II error)
```

### Secondary Metrics

#### 4. Average Treatment Effect Estimate

**Definition:**
```
ATE = (1/N) Σ(ITE_i)

Where ITE_i is the estimated individual treatment effect at step i
```

**Interpretation:**
- Positive ATE → LLM advice is helping on average
- Negative ATE → LLM advice is hurting on average
- Near-zero ATE → LLM advice has no net effect

#### 5. Temporal Analysis

**Early vs. Late Performance:**
```
Early Phase (Episodes 0-33%):
- Higher rejection rate expected (models still learning)

Mid Phase (Episodes 33-66%):
- Rejection rate stabilizes

Late Phase (Episodes 66-100%):
- Optimal rejection rate reached
```

#### 6. Degradation Metrics

**Absolute Degradation:**
```
Absolute Degradation = Baseline_Reward - Attack_Reward
```

**Relative Degradation:**
```
Relative Degradation = (Baseline_Reward - Attack_Reward) / Baseline_Reward × 100%
```

**Residual Degradation (After Defense):**
```
Residual = (Baseline_Reward - Protected_Reward) / Baseline_Reward × 100%
```

---

## Key Findings & Analysis

### Expected Results

#### Hypothesis 1: Causal Filter Effectiveness

**Prediction:**
- Recovery Rate: **50-80%** for most attacks
- Strategic Poisoning: **60-75%** recovery
- State Inversion: **55-70%** recovery
- Misleading Context: **40-60%** recovery (harder to detect)

**Reasoning:**
- Causal models can detect performance impact even when attack semantics seem valid
- Outcome models learn that certain "advice patterns" correlate with poor rewards
- Propensity weighting corrects for confounding game states

#### Hypothesis 2: Rejection Patterns

**Prediction:**
- Rejection rate will **increase during early episodes** (models learning)
- Rejection rate will **stabilize at 30-50%** for strong attacks
- Rejection rate will **correlate with attack strength**

**Temporal Pattern:**
```
Episodes:  0-10  | 10-20 | 20-30 | 30-40 | 40-50
Rejection: 20%   | 35%   | 42%   | 45%   | 47%
           (Warmup) (Learning) (Stabilizing) (Optimal)
```

#### Hypothesis 3: Model Accuracy Evolution

**Prediction:**
- Propensity model accuracy: **70-80%** (easier task)
- Outcome model R²: **0.5-0.7** (harder task, noisy rewards)
- Treatment effect RMSE: **Will decrease over time**

**Learning Curve:**
```
      Model Quality
         ▲
    1.0 │                    ╱─────
        │                  ╱
    0.8 │               ╱
        │            ╱
    0.6 │        ╱
        │     ╱
    0.4 │  ╱
        │╱
    0.0 └──────────────────────────►
        0   500   1000  1500  2000
               Training Steps
```

#### Hypothesis 4: Attack-Specific Performance

**Predicted Recovery by Attack Type:**

| Attack Type | Unprotected Loss | Protected Loss | Recovery Rate |
|-------------|------------------|----------------|---------------|
| Strategic Poisoning | 60% | 15% | 75% |
| State Inversion | 55% | 18% | 67% |
| Misleading Context | 45% | 22% | 51% |
| Info Removal | 40% | 15% | 63% |

**Why Strategic Poisoning is Most Recoverable:**
- Context-specific harm creates clear performance patterns
- Outcome models quickly learn "when health is low, combat advice → death"
- Strong causal signal in the data

**Why Misleading Context is Hardest:**
- Advice may seem reasonable in isolation
- Harm is subtle and delayed
- Requires more data to detect pattern

### Analytical Questions

#### Q1: Does the filter adapt to evolving attack strategies?
**Analysis Method:**
- Compare rejection rates in early vs. late episodes
- Track prediction accuracy over time
- Test with time-varying attack strengths

**Expected Finding:**
- Filter adapts within 500-1000 steps
- Performance improves as more data is collected

#### Q2: What is the computational overhead?
**Metrics:**
- Model training time per update
- Inference time per prediction
- Memory usage for data buffer

**Expected Overhead:**
- Training: ~1-2 seconds per update (every 100 steps)
- Inference: <10ms per prediction
- Memory: ~50MB for 2000-step buffer
- **Total impact**: <5% slowdown in training

#### Q3: Can the filter distinguish subtle attacks?
**Test:**
- Compare performance on mild (strength 0.3) vs severe (strength 0.8) attacks
- Measure false positive/negative rates

**Expected Finding:**
- High-accuracy detection for severe attacks (>80%)
- Lower accuracy for mild attacks (60-70%)
- Threshold tuning can adjust sensitivity

#### Q4: Does the filter hurt performance without attacks?
**Control Experiment:**
- Run with causal filter but NO adversarial attacks
- Compare to baseline (no filter, no LLM)

**Expected Finding:**
- **Slight benefit** (2-5% improvement)
- Filter rejects genuinely bad advice even without attacks
- Acts as general quality control for LLM

#### Q5: How does buffer size affect performance?
**Ablation Study:**
- Test with buffer sizes: 500, 1000, 2000, 5000 steps

**Expected Finding:**
- Too small (500): Unstable models, poor predictions
- Optimal (2000): Good balance of recency and data
- Too large (5000): Stale data, slower updates

---

## Flowcharts

### 1. Overall Causal Defense System Flow

```
START TRAINING
     │
     ▼
┌─────────────────┐
│ Initialize:     │
│ - Environment   │
│ - Agent         │
│ - Causal Filter │
│ - Attacker      │
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│  FOR EACH EPISODE  │
└────────┬───────────┘
         │
         ▼
    ┌────────────┐
    │ Reset Env  │
    └──────┬─────┘
           │
           ▼
    ┌──────────────────┐
    │ WHILE not done:  │
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │ Observe State    │
    └──────┬───────────┘
           │
           ▼
    ┌─────────────────────┐
    │ Should Call LLM?    │
    └──────┬──────────────┘
           │
      Yes  │  No
           │   └───────────┐
           ▼               │
    ┌─────────────────┐   │
    │ Generate        │   │
    │ Semantic Desc.  │   │
    └──────┬──────────┘   │
           │               │
           ▼               │
    ┌─────────────────┐   │
    │ ⚠️ APPLY        │   │
    │    ATTACK       │   │
    └──────┬──────────┘   │
           │               │
           ▼               │
    ┌─────────────────┐   │
    │ Get LLM Advice  │   │
    └──────┬──────────┘   │
           │               │
           ▼               │
    ┌─────────────────────────────┐
    │ 🛡️ CAUSAL FILTER            │
    │                              │
    │ 1. Extract Features (X)     │
    │ 2. Predict e(X)             │
    │ 3. Predict μ₁(X), μ₀(X)     │
    │ 4. Compute ITE              │
    │ 5. Accept if ITE ≥ -0.01   │
    └──────┬──────────────────────┘
           │
      ┌────┴────┐
   Accept    Reject
      │         │
      ▼         ▼
    ┌─────┐  ┌─────┐
    │Use  │  │Don't│
    │Hints│  │Use  │
    └──┬──┘  └──┬──┘
       └────┬───┘
            │
            ▼
    ┌────────────────┐
    │ Select Action  │
    │ (PPO Policy)   │
    └───────┬────────┘
            │
            ▼
    ┌────────────────┐
    │ Execute Action │
    └───────┬────────┘
            │
            ▼
    ┌────────────────┐
    │ Observe Reward │
    └───