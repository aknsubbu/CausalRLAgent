# API Reference

Technical reference for CausalRLAgent classes and methods.

---

## Agents

### LLMEnhancedNetHackAgent

Main agent class with LLM guidance support.

```python
from causal_filtered import LLMEnhancedNetHackAgent

agent = LLMEnhancedNetHackAgent(
    action_dim: int = 23,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.02,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    enable_llm: bool = False,
    llm_guidance_weight: float = 0.05
)
```

**Methods:**

| Method                                                      | Returns | Description                              |
| ----------------------------------------------------------- | ------- | ---------------------------------------- |
| `process_observation(obs)`                                  | tuple   | Process raw obs → tensor, processed, raw |
| `select_action(tensor_obs, processed, raw, reset, metrics)` | tuple   | Async; returns (action, log_prob, value) |
| `update(batch_size, epochs)`                                | dict    | PPO update; returns losses               |
| `train(env, num_episodes, update_freq, print_freq)`         | tuple   | Async; returns (rewards, shaped_rewards) |
| `save_model(path)`                                          | None    | Save actor/critic to file                |
| `load_model(path)`                                          | None    | Load actor/critic from file              |

---

### AdversarialLLMEnhancedAgent

Agent with adversarial attack injection.

```python
from causal_filtered import AdversarialLLMEnhancedAgent, AdversarialAttacker

agent = AdversarialLLMEnhancedAgent(
    action_dim: int = 23,
    # ... same as LLMEnhancedNetHackAgent ...
    attacker: AdversarialAttacker = None
)
```

Inherits all methods from `LLMEnhancedNetHackAgent`. Attacks are automatically applied during LLM calls.

---

## Neural Networks

### LLMEnhancedPPOActor

Policy network with LLM hint integration.

```python
from causal_filtered import LLMEnhancedPPOActor

actor = LLMEnhancedPPOActor(
    action_dim: int = 23,
    use_llm: bool = False,
    llm_guidance_weight: float = 0.05
)
```

**Forward signature:**

```python
def forward(
    self,
    obs: dict,              # {glyphs, stats, message, inventory, action_history}
    reset_hidden: bool,     # Reset LSTM state
    llm_hints: torch.Tensor # Shape (batch, action_dim) or None
) -> torch.Tensor:          # Action logits
```

### RecurrentPPOCritic

Value network with LSTM.

```python
from causal_filtered import RecurrentPPOCritic

critic = RecurrentPPOCritic()
```

**Forward signature:**

```python
def forward(
    self,
    obs: dict,
    reset_hidden: bool
) -> torch.Tensor:  # Value estimate
```

---

## Adversarial Attacks

### AdversarialAttackType

Enum of attack types.

```python
from causal_filtered import AdversarialAttackType

AdversarialAttackType.NONE                  # No attack
AdversarialAttackType.NOISE_INJECTION       # Random noise
AdversarialAttackType.STATE_INVERSION       # Flip meanings
AdversarialAttackType.MISLEADING_CONTEXT    # Bad advice
AdversarialAttackType.CONTRADICTORY_INFO    # Conflicts
AdversarialAttackType.CRITICAL_INFO_REMOVAL # Redact info
AdversarialAttackType.STRATEGIC_POISONING   # Context-aware
AdversarialAttackType.RANDOM_CORRUPTION     # Character noise
```

### AdversarialAttacker

Attack execution class.

```python
from causal_filtered import AdversarialAttacker

attacker = AdversarialAttacker(
    attack_type: AdversarialAttackType = AdversarialAttackType.NONE,
    attack_strength: float = 0.5  # 0.0 to 1.0
)
```

**Methods:**

| Method                                     | Returns | Description                                            |
| ------------------------------------------ | ------- | ------------------------------------------------------ |
| `attack_description(description, raw_obs)` | str     | Apply attack to description                            |
| `get_attack_stats()`                       | dict    | Return `{total_attacks, attack_type, attack_strength}` |

---

## Semantic Processing

### NetHackSemanticDescriptor

Converts game state to natural language.

```python
from causal_filtered import NetHackSemanticDescriptor

descriptor = NetHackSemanticDescriptor()
```

**Methods:**

| Method                                                          | Returns | Description                |
| --------------------------------------------------------------- | ------- | -------------------------- |
| `generate_full_description(obs, processed_obs, recent_actions)` | str     | Complete state description |
| `describe_surroundings(glyphs, player_pos)`                     | str     | Nearby monsters/items      |
| `describe_player_status(stats)`                                 | str     | HP, level, gold            |
| `get_player_position(stats)`                                    | tuple   | (row, col) from blstats    |

---

## Causal Logging

### ImprovedCausalLogger

Treatment/control observation logger.

```python
from causal_logger import ImprovedCausalLogger

logger = ImprovedCausalLogger(
    log_dir: str = "causal_logs_v2",
    window_size: int = 50
)
```

**Methods:**

| Method                                    | Returns | Description                           |
| ----------------------------------------- | ------- | ------------------------------------- |
| `log_step(step_data)`                     | None    | Log single step                       |
| `log_llm_intervention(intervention_data)` | int     | Log LLM call; returns intervention ID |
| `start_episode(episode_num)`              | None    | Mark episode start                    |
| `finalize_episode(episode_stats)`         | None    | Compute outcomes, finalize            |
| `compute_causal_estimates()`              | dict    | Match-based ATE computation           |
| `save_summary()`                          | dict    | Generate and save summary JSON        |

**Step data schema:**

```python
{
    'episode': int,
    'step': int,
    'obs': dict,
    'raw_obs': dict,
    'action': int,
    'reward': float,
    'shaped_reward': float,
    'done': bool,
    'value': float,
    'llm_treatment': bool,
    'llm_hint_strength': float,
    'llm_treatment_id': int | None
}
```

---

## Causal Estimation

### DoublyRobustEstimator

Causal effect estimator.

```python
from causal_analysis import DoublyRobustEstimator

estimator = DoublyRobustEstimator(
    steps_file: str,
    interventions_file: str,
    summary_file: str = None
)
```

**Methods:**

| Method                             | Returns   | Description            |
| ---------------------------------- | --------- | ---------------------- |
| `load_data()`                      | DataFrame | Load JSONL files       |
| `prepare_features()`               | None      | Engineer features      |
| `estimate_propensity_scores()`     | None      | Fit propensity model   |
| `estimate_outcome_models()`        | None      | Fit outcome models     |
| `compute_doubly_robust_ate()`      | dict      | Main ATE computation   |
| `analyze_heterogeneous_effects()`  | None      | Effect by subgroups    |
| `analyze_strategy_effectiveness()` | None      | Effect by LLM strategy |
| `visualize_results(output_dir)`    | None      | Generate plots         |
| `run_full_analysis()`              | dict      | Complete pipeline      |

**ATE result schema:**

```python
{
    'ate': float,
    'se': float,
    'ci_lower': float,
    'ci_upper': float,
    'p_value': float,
    'significant': bool
}
```

---

## Utilities

### create_nethack_env

Create configured NetHack environment.

```python
from causal_filtered import create_nethack_env

env = create_nethack_env()
# Returns: gym.Env with NetHackScore-v0
```

### NetHackRewardShaper

Reward shaping for better learning.

```python
from causal_filtered import NetHackRewardShaper

shaper = NetHackRewardShaper()
shaped_reward = shaper.shape_reward(obs, raw_reward, done, info)
shaper.reset()  # Call at episode start
```

**Reward components:**

- Exploration: +0.01 per new position
- Health change: +0.0001 per HP gained
- Level up: +5.0
- Gold: +0.001 per gold
- Kill: +1.0
- Death: -5.0
