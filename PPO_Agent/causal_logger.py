import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import deque, defaultdict


class ImprovedCausalLogger:
    """
    Enhanced causal logger that tracks BOTH treated and control observations.
    
    Key improvements:
    1. Logs ALL steps (not just LLM interventions)
    2. Marks which steps received LLM treatment
    3. Enables proper causal inference (treatment vs control comparison)
    4. Tracks pre/post windows for each observation
    5. Computes counterfactual estimates
    """
    
    def __init__(self, log_dir="causal_logs_v2", window_size=50):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.steps_file = self.log_dir / f"steps_{timestamp}.jsonl"
        self.interventions_file = self.log_dir / f"interventions_{timestamp}.jsonl"
        self.summary_file = self.log_dir / f"summary_{timestamp}.json"
        
        # Core tracking
        self.current_episode = 0
        self.current_step = 0
        self.global_step = 0
        
        # Step-level tracking (ALL steps)
        self.all_steps = deque(maxlen=10000)  # Keep last 10k steps in memory
        
        # LLM intervention tracking
        self.llm_interventions = []  # All LLM calls with their IDs
        self.active_treatments = {}  # {step_num: treatment_info}
        
        # Performance tracking windows
        self.window_size = window_size
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_shaped_rewards = deque(maxlen=window_size)
        
        # Causal inference data structures
        self.treatment_windows = []  # List of (treatment_step, pre_window, post_window)
        self.control_windows = []    # List of (step, pre_window, post_window) - no treatment
        
        # Statistics
        self.total_steps_logged = 0
        self.total_interventions = 0
        self.episodes_completed = 0
        
        print(f"📊 Improved Causal Logger initialized")
        print(f"   Steps log: {self.steps_file}")
        print(f"   Interventions log: {self.interventions_file}")
        print(f"   Window size: {window_size}")
    
    # ========================================
    # STEP-LEVEL LOGGING (ALL STEPS)
    # ========================================
    
    def log_step(self, step_data):
        """
        Log EVERY step (both treated and control).
        
        Args:
            step_data: dict with:
                - episode: episode number
                - step: step within episode
                - obs: processed observation dict
                - raw_obs: raw observation
                - action: action taken
                - reward: raw reward
                - shaped_reward: shaped reward
                - done: episode done
                - value: critic's value estimate
                - llm_treatment: bool, whether LLM guidance was active this step
                - llm_hint_strength: float, strength of LLM guidance (0 if no treatment)
        """
        self.current_episode = step_data['episode']
        self.current_step = step_data['step']
        self.global_step += 1
        
        # Extract key features
        record = {
            # Identifiers
            'global_step': self.global_step,
            'episode': self.current_episode,
            'step': self.current_step,
            'timestamp': time.time(),
            
            # State features
            'health_ratio': self._extract_health_ratio(step_data['raw_obs']),
            'position': self._extract_position(step_data['raw_obs']),
            'level': self._extract_level(step_data['raw_obs']),
            'depth': self._extract_depth(step_data['raw_obs']),
            'gold': self._extract_gold(step_data['raw_obs']),
            
            # Action and outcome
            'action': step_data['action'],
            'reward': step_data['reward'],
            'shaped_reward': step_data['shaped_reward'],
            'value': step_data['value'],
            'done': step_data['done'],
            
            # CRITICAL: Treatment indicator
            'treated': step_data.get('llm_treatment', False),
            'treatment_strength': step_data.get('llm_hint_strength', 0.0),
            'treatment_id': step_data.get('llm_treatment_id', None),
            
            # Context features (for matching)
            'recent_reward_mean': float(np.mean(list(self.recent_rewards))) if self.recent_rewards else 0.0,
            'recent_reward_std': float(np.std(list(self.recent_rewards))) if len(self.recent_rewards) > 1 else 0.0,
        }
        
        # Update rolling windows
        self.recent_rewards.append(step_data['reward'])
        self.recent_shaped_rewards.append(step_data['shaped_reward'])
        
        # Store in memory
        self.all_steps.append(record)
        
        # Write to disk (JSONL - one line per step)
        self._write_step_record(record)
        
        self.total_steps_logged += 1
    
    # ========================================
    # LLM INTERVENTION LOGGING
    # ========================================
    
    def log_llm_intervention(self, intervention_data):
        """
        Log an LLM intervention (treatment event).
        
        Args:
            intervention_data: dict with:
                - episode: episode number
                - step: step number
                - semantic_description: text sent to LLM
                - llm_response: LLM's raw response
                - parsed_strategy: extracted strategy
                - action_hints: hint vector
                - boosted_actions: list of boosted action IDs
                - call_duration: time taken
                - performance_before: dict with avg_reward, avg_length
        """
        intervention_id = len(self.llm_interventions)
        
        # Extract pre-treatment context
        pre_window = self._extract_window_before(self.global_step, self.window_size)
        
        record = {
            # Identifiers
            'intervention_id': intervention_id,
            'global_step': self.global_step,
            'episode': intervention_data['episode'],
            'step': intervention_data['step'],
            'timestamp': time.time(),
            
            # Pre-treatment state
            'pre_treatment': {
                'health_ratio': self._extract_health_ratio(intervention_data['raw_obs']),
                'level': self._extract_level(intervention_data['raw_obs']),
                'depth': self._extract_depth(intervention_data['raw_obs']),
                'recent_rewards': [s['reward'] for s in pre_window],
                'recent_actions': [s['action'] for s in pre_window],
                'avg_reward_before': float(np.mean([s['reward'] for s in pre_window])) if pre_window else 0.0,
                'performance_metrics': intervention_data['performance_before'].copy(),
            },
            
            # LLM treatment details
            'treatment': {
                'semantic_description': intervention_data['semantic_description'],
                'llm_response': intervention_data['llm_response'],
                'parsed_strategy': intervention_data['parsed_strategy'],
                'action_hints': intervention_data['action_hints'].tolist() if hasattr(intervention_data['action_hints'], 'tolist') else list(intervention_data['action_hints']),
                'boosted_actions': intervention_data['boosted_actions'],
                'hint_strength': float(np.max(intervention_data['action_hints'])),
                'n_boosted_actions': len(intervention_data['boosted_actions']),
                'call_duration': intervention_data['call_duration'],
            },
            
            # Post-treatment outcomes (to be filled)
            'post_treatment': {
                'immediate': {},  # Next 10 steps
                'short_term': {},  # Next 50 steps
                'long_term': {},   # Next 100 steps
            },
            
            # Causal inference fields
            'causal_estimates': {
                'estimated_ate': None,  # Average Treatment Effect
                'estimated_counterfactual': None,  # What would have happened without LLM
                'confidence_interval': None,
            }
        }
        
        # Store intervention
        self.llm_interventions.append(record)
        self.total_interventions += 1
        
        # Mark this step as treated
        self.active_treatments[self.global_step] = intervention_id
        
        # Write to interventions log
        self._write_intervention_record(record)
        
        print(f"📝 Logged intervention #{intervention_id} at global step {self.global_step}")
        
        return intervention_id
    
    # ========================================
    # POST-TREATMENT OUTCOME TRACKING
    # ========================================
    
    def update_intervention_outcomes(self, intervention_id, outcome_type='immediate'):
        """
        Compute and update outcomes for an intervention.
        
        Args:
            intervention_id: ID of intervention to update
            outcome_type: 'immediate', 'short_term', or 'long_term'
        """
        if intervention_id >= len(self.llm_interventions):
            return
        
        intervention = self.llm_interventions[intervention_id]
        treatment_step = intervention['global_step']
        
        # Define window sizes
        window_sizes = {
            'immediate': 10,
            'short_term': 50,
            'long_term': 100,
        }
        
        window_size = window_sizes[outcome_type]
        
        # Extract post-treatment window
        post_window = self._extract_window_after(treatment_step, window_size)
        
        if not post_window:
            return
        
        # Compute outcomes
        outcomes = {
            'n_steps': len(post_window),
            'rewards': [s['reward'] for s in post_window],
            'shaped_rewards': [s['shaped_reward'] for s in post_window],
            'actions': [s['action'] for s in post_window],
            'avg_reward': float(np.mean([s['reward'] for s in post_window])),
            'total_reward': float(np.sum([s['reward'] for s in post_window])),
            'avg_shaped_reward': float(np.mean([s['shaped_reward'] for s in post_window])),
            'reward_std': float(np.std([s['reward'] for s in post_window])),
            'max_reward': float(np.max([s['reward'] for s in post_window])),
            'min_reward': float(np.min([s['reward'] for s in post_window])),
            'action_diversity': self._compute_action_diversity([s['action'] for s in post_window]),
            'survival_rate': sum(1 for s in post_window if not s['done']) / len(post_window),
        }
        
        # Compare to baseline
        pre_avg_reward = intervention['pre_treatment']['avg_reward_before']
        outcomes['reward_delta'] = outcomes['avg_reward'] - pre_avg_reward
        outcomes['relative_improvement'] = (outcomes['reward_delta'] / abs(pre_avg_reward)) if pre_avg_reward != 0 else 0.0
        
        # Update intervention record
        intervention['post_treatment'][outcome_type] = outcomes
        
        # Re-write updated intervention
        self._write_intervention_record(intervention)
    
    # ========================================
    # CAUSAL INFERENCE: MATCHING & ESTIMATION
    # ========================================
    
    def compute_causal_estimates(self):
        """
        Compute causal estimates using matching on observables.
        
        For each treated observation, find similar untreated observations
        and compute Average Treatment Effect (ATE).
        """
        print(f"\n🔬 Computing causal estimates...")
        
        # Get all steps
        all_steps_list = list(self.all_steps)
        
        # Separate treated and control
        treated_steps = [s for s in all_steps_list if s['treated']]
        control_steps = [s for s in all_steps_list if not s['treated']]
        
        print(f"   Treated steps: {len(treated_steps)}")
        print(f"   Control steps: {len(control_steps)}")
        
        if len(treated_steps) == 0 or len(control_steps) == 0:
            print("   ⚠️ Need both treated and control observations!")
            return None
        
        # For each treated observation, find matched controls
        treatment_effects = []
        
        for treated_obs in treated_steps:
            # Find similar control observations (propensity score matching)
            matched_controls = self._find_matched_controls(
                treated_obs, 
                control_steps,
                n_matches=5
            )
            
            if not matched_controls:
                continue
            
            # Compute treatment effect
            treated_outcome = treated_obs['shaped_reward']
            control_outcomes = [c['shaped_reward'] for c in matched_controls]
            avg_control_outcome = np.mean(control_outcomes)
            
            effect = treated_outcome - avg_control_outcome
            treatment_effects.append(effect)
        
        if not treatment_effects:
            print("   ⚠️ Could not compute treatment effects!")
            return None
        
        # Aggregate estimates
        ate = float(np.mean(treatment_effects))
        ate_std = float(np.std(treatment_effects))
        ate_se = ate_std / np.sqrt(len(treatment_effects))
        
        # 95% confidence interval
        ci_lower = ate - 1.96 * ate_se
        ci_upper = ate + 1.96 * ate_se
        
        causal_estimates = {
            'average_treatment_effect': ate,
            'ate_std': ate_std,
            'ate_se': ate_se,
            'confidence_interval_95': [ci_lower, ci_upper],
            'n_treated': len(treated_steps),
            'n_control': len(control_steps),
            'n_matched_pairs': len(treatment_effects),
            'statistically_significant': not (ci_lower <= 0 <= ci_upper),
        }
        
        print(f"\n📊 Causal Estimates:")
        print(f"   ATE: {ate:.4f} ± {ate_se:.4f}")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   Significant: {causal_estimates['statistically_significant']}")
        
        return causal_estimates
    
    def _find_matched_controls(self, treated_obs, control_pool, n_matches=5):
        """
        Find control observations similar to treated observation.
        
        Uses covariate matching on:
        - Health ratio
        - Level
        - Recent reward mean
        - Recent reward std
        """
        treated_features = np.array([
            treated_obs['health_ratio'],
            treated_obs['level'] / 30.0,  # Normalize
            treated_obs['recent_reward_mean'],
            treated_obs['recent_reward_std'],
        ])
        
        # Compute distances
        distances = []
        for control_obs in control_pool:
            control_features = np.array([
                control_obs['health_ratio'],
                control_obs['level'] / 30.0,
                control_obs['recent_reward_mean'],
                control_obs['recent_reward_std'],
            ])
            
            # Mahalanobis distance (or just Euclidean for simplicity)
            distance = np.linalg.norm(treated_features - control_features)
            distances.append((distance, control_obs))
        
        # Sort by distance and take closest matches
        distances.sort(key=lambda x: x[0])
        matched_controls = [obs for _, obs in distances[:n_matches]]
        
        return matched_controls
    
    # ========================================
    # EPISODE MANAGEMENT
    # ========================================
    
    def start_episode(self, episode_num):
        """Mark start of new episode"""
        self.current_episode = episode_num
        self.current_step = 0
    
    def finalize_episode(self, episode_stats):
        """
        Finalize episode and compute all pending outcomes.
        
        Args:
            episode_stats: dict with:
                - total_reward
                - total_steps
                - died
                - final_health
        """
        # Update all interventions in this episode
        episode_interventions = [
            i for i in self.llm_interventions 
            if i['episode'] == self.current_episode
        ]
        
        for intervention in episode_interventions:
            intervention_id = intervention['intervention_id']
            
            # Update immediate (if not already done)
            if not intervention['post_treatment']['immediate']:
                self.update_intervention_outcomes(intervention_id, 'immediate')
            
            # Update short-term
            if not intervention['post_treatment']['short_term']:
                self.update_intervention_outcomes(intervention_id, 'short_term')
            
            # Update long-term
            if not intervention['post_treatment']['long_term']:
                self.update_intervention_outcomes(intervention_id, 'long_term')
        
        self.episodes_completed += 1
    
    # ========================================
    # SUMMARY & EXPORT
    # ========================================
    
    def save_summary(self):
        """Generate and save comprehensive summary"""
        print(f"\n📊 Generating summary...")
        
        # Compute causal estimates
        causal_estimates = self.compute_causal_estimates()
        
        # Aggregate statistics
        all_steps_list = list(self.all_steps)
        treated_steps = [s for s in all_steps_list if s['treated']]
        control_steps = [s for s in all_steps_list if not s['treated']]
        
        summary = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_steps_logged': self.total_steps_logged,
                'total_interventions': self.total_interventions,
                'episodes_completed': self.episodes_completed,
            },
            
            'sample_composition': {
                'n_treated': len(treated_steps),
                'n_control': len(control_steps),
                'treatment_rate': len(treated_steps) / len(all_steps_list) if all_steps_list else 0,
            },
            
            'descriptive_statistics': {
                'treated': self._compute_descriptive_stats(treated_steps),
                'control': self._compute_descriptive_stats(control_steps),
            },
            
            'causal_estimates': causal_estimates,
            
            'intervention_summaries': {
                'by_strategy': self._summarize_by_strategy(),
                'best_performing': self._find_best_interventions(top_k=5),
                'worst_performing': self._find_worst_interventions(top_k=5),
            },
        }
        
        # Save to file
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Summary saved to {self.summary_file}")
        
        return summary
    
    def _compute_descriptive_stats(self, steps):
        """Compute descriptive statistics for a set of steps"""
        if not steps:
            return {}
        
        rewards = [s['reward'] for s in steps]
        shaped_rewards = [s['shaped_reward'] for s in steps]
        
        return {
            'n_observations': len(steps),
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'avg_shaped_reward': float(np.mean(shaped_rewards)),
            'avg_health': float(np.mean([s['health_ratio'] for s in steps])),
            'avg_level': float(np.mean([s['level'] for s in steps])),
        }
    
    def _summarize_by_strategy(self):
        """Summarize interventions by strategy type"""
        strategy_outcomes = defaultdict(list)
        
        for intervention in self.llm_interventions:
            strategy = intervention['treatment']['parsed_strategy']
            
            if intervention['post_treatment']['immediate']:
                outcome = intervention['post_treatment']['immediate']['avg_reward']
                strategy_outcomes[strategy].append(outcome)
        
        # Aggregate
        summary = {}
        for strategy, outcomes in strategy_outcomes.items():
            summary[strategy] = {
                'n_uses': len(outcomes),
                'avg_outcome': float(np.mean(outcomes)),
                'std_outcome': float(np.std(outcomes)),
            }
        
        return summary
    
    def _find_best_interventions(self, top_k=5):
        """Find top-performing interventions"""
        interventions_with_outcomes = [
            (i, i['post_treatment']['immediate']['avg_reward'])
            for i in self.llm_interventions
            if i['post_treatment']['immediate']
        ]
        
        interventions_with_outcomes.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {
                'intervention_id': i['intervention_id'],
                'strategy': i['treatment']['parsed_strategy'],
                'outcome': outcome,
            }
            for i, outcome in interventions_with_outcomes[:top_k]
        ]
    
    def _find_worst_interventions(self, top_k=5):
        """Find worst-performing interventions"""
        interventions_with_outcomes = [
            (i, i['post_treatment']['immediate']['avg_reward'])
            for i in self.llm_interventions
            if i['post_treatment']['immediate']
        ]
        
        interventions_with_outcomes.sort(key=lambda x: x[1])
        
        return [
            {
                'intervention_id': i['intervention_id'],
                'strategy': i['treatment']['parsed_strategy'],
                'outcome': outcome,
            }
            for i, outcome in interventions_with_outcomes[:top_k]
        ]
    
    # ========================================
    # HELPER METHODS
    # ========================================
    
    def _extract_window_before(self, current_step, window_size):
        """Extract steps before current step"""
        return [
            s for s in self.all_steps
            if s['global_step'] < current_step 
            and s['global_step'] >= current_step - window_size
        ]
    
    def _extract_window_after(self, current_step, window_size):
        """Extract steps after current step"""
        return [
            s for s in self.all_steps
            if s['global_step'] > current_step
            and s['global_step'] <= current_step + window_size
        ]
    
    def _extract_health_ratio(self, raw_obs):
        """Extract health ratio from observation"""
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        
        stats = raw_obs.get('blstats', np.zeros(26))
        if len(stats) > 11:
            hp = float(stats[10])
            max_hp = float(stats[11])
            return hp / max_hp if max_hp > 0 else 0.5
        return 0.5
    
    def _extract_position(self, raw_obs):
        """Extract player position"""
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        
        stats = raw_obs.get('blstats', np.zeros(26))
        if len(stats) > 1:
            return (int(stats[0]), int(stats[1]))
        return (0, 0)
    
    def _extract_level(self, raw_obs):
        """Extract player level"""
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        
        stats = raw_obs.get('blstats', np.zeros(26))
        return int(stats[18]) if len(stats) > 18 else 0
    
    def _extract_depth(self, raw_obs):
        """Extract dungeon depth"""
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        
        stats = raw_obs.get('blstats', np.zeros(26))
        return int(stats[12]) if len(stats) > 12 else 0
    
    def _extract_gold(self, raw_obs):
        """Extract gold amount"""
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        
        stats = raw_obs.get('blstats', np.zeros(26))
        return int(stats[13]) if len(stats) > 13 else 0
    
    def _compute_action_diversity(self, actions):
        """Compute diversity of actions"""
        if not actions:
            return 0.0
        
        unique_actions = len(set(actions))
        return unique_actions / len(actions)
    
    def _write_step_record(self, record):
        """Write step record to JSONL file"""
        with open(self.steps_file, 'a') as f:
            json.dump(record, f)
            f.write('\n')
    
    def _write_intervention_record(self, record):
        """Write intervention record to JSONL file"""
        with open(self.interventions_file, 'a') as f:
            json.dump(record, f)
            f.write('\n')


# ========================================
# USAGE EXAMPLE
# ========================================

def example_usage():
    """Example of how to use the improved causal logger"""
    
    logger = ImprovedCausalLogger(log_dir="causal_logs_v2", window_size=50)
    
    # Start episode
    logger.start_episode(episode_num=0)
    
    # Log regular steps (control observations)
    for step in range(100):
        step_data = {
            'episode': 0,
            'step': step,
            'obs': {},  # processed obs
            'raw_obs': {'blstats': np.random.randn(26)},
            'action': np.random.randint(0, 23),
            'reward': np.random.randn(),
            'shaped_reward': np.random.randn(),
            'done': False,
            'value': np.random.randn(),
            'llm_treatment': False,  # NO TREATMENT
            'llm_hint_strength': 0.0,
        }
        logger.log_step(step_data)
        
        # Occasionally add LLM intervention (treatment)
        if step == 50:
            intervention_data = {
                'episode': 0,
                'step': step,
                'semantic_description': "Example description",
                'llm_response': "explore",
                'parsed_strategy': "explore",
                'action_hints': np.random.rand(23),
                'boosted_actions': [0, 1, 2],
                'call_duration': 0.5,
                'performance_before': {'avg_reward': 0.0, 'avg_length': 50},
                'raw_obs': {'blstats': np.random.randn(26)},
            }
            intervention_id = logger.log_llm_intervention(intervention_data)
            
            # Mark next few steps as treated
            for treated_step in range(step + 1, step + 11):
                step_data = {
                    'episode': 0,
                    'step': treated_step,
                    'obs': {},
                    'raw_obs': {'blstats': np.random.randn(26)},
                    'action': np.random.randint(0, 23),
                    'reward': np.random.randn() + 0.5,  # Better performance
                    'shaped_reward': np.random.randn() + 0.5,
                    'done': False,
                    'value': np.random.randn(),
                    'llm_treatment': True,  # TREATED
                    'llm_hint_strength': 0.2,
                    'llm_treatment_id': intervention_id,
                }
                logger.log_step(step_data)
    
    # Finalize episode
    logger.finalize_episode({
        'total_reward': 10.0,
        'total_steps': 100,
        'died': False,
        'final_health': 0.8,
    })
    
    # Generate summary with causal estimates
    summary = logger.save_summary()
    
    print("\n✅ Example complete!")
    print(f"   Logged {logger.total_steps_logged} steps")
    print(f"   Logged {logger.total_interventions} interventions")


if __name__ == "__main__":
    example_usage()