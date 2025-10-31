import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import deque


class CausalLogger:
    """
    Logs comprehensive data for causal analysis of LLM advice effectiveness.
    
    Captures:
    - Pre-advice state (game state, agent performance, context)
    - LLM advice (what was suggested, how it was used)
    - Post-advice outcomes (immediate and delayed effects)
    - Counterfactual markers (what would have happened without LLM)
    """
    
    def __init__(self, log_dir="causal_logs", buffer_size=100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"causal_log_{timestamp}.jsonl"
        self.summary_file = self.log_dir / f"summary_{timestamp}.json"
        
        # Tracking structures
        self.llm_calls = []  # All LLM call records
        self.current_episode = 0
        self.current_step = 0
        
        # Pre-advice tracking (for causal inference)
        self.pre_advice_buffer = deque(maxlen=buffer_size)  # States before LLM
        self.post_advice_buffer = deque(maxlen=buffer_size)  # States after LLM
        
        # Performance tracking windows
        self.performance_window = deque(maxlen=50)  # Recent performance
        self.last_llm_call_step = -1000  # Track time since last LLM call
        
        print(f"📊 Causal Logger initialized")
        print(f"   Log file: {self.log_file}")
        print(f"   Summary file: {self.summary_file}")
    
    def start_episode(self, episode_num):
        """Mark start of new episode"""
        self.current_episode = episode_num
        self.current_step = 0
    
    def log_step(self, step_data):
        """
        Log every step for baseline tracking.
        
        Args:
            step_data: dict with keys:
                - 'episode': episode number
                - 'step': step number
                - 'obs': processed observation dict
                - 'raw_obs': raw observation
                - 'action': action taken
                - 'reward': reward received
                - 'shaped_reward': shaped reward
                - 'done': episode done
                - 'value': critic's value estimate
        """
        self.current_step = step_data['step']
        
        # Store in pre-advice buffer (rolling window)
        self.pre_advice_buffer.append({
            'episode': self.current_episode,
            'step': self.current_step,
            'action': step_data['action'],
            'reward': step_data['reward'],
            'shaped_reward': step_data['shaped_reward'],
            'value': step_data['value'],
            'health_ratio': self._extract_health_ratio(step_data['raw_obs']),
            'position': self._extract_position(step_data['raw_obs']),
            'timestamp': time.time()
        })
        
        self.performance_window.append(step_data['shaped_reward'])
    
    def log_llm_call(self, llm_call_data):
        """
        Log LLM advice call with full context.
        
        Args:
            llm_call_data: dict with keys:
                - 'episode': episode number
                - 'step': step number
                - 'semantic_description': full text sent to LLM
                - 'llm_response': raw LLM response
                - 'strategy': parsed strategy (explore/combat/retreat/etc)
                - 'action_hints': action hint vector (23-dim)
                - 'boosted_actions': list of action IDs that got boosted
                - 'performance_metrics': dict with avg_reward, avg_length
                - 'raw_obs': raw observation at time of call
                - 'processed_obs': processed observation
                - 'call_duration': time taken for LLM call (seconds)
        """
        
        # Extract pre-advice context (what was happening before LLM)
        pre_advice_context = self._extract_pre_advice_context()
        
        # Create comprehensive record
        record = {
            # Identifiers
            'call_id': len(self.llm_calls),
            'episode': llm_call_data['episode'],
            'step': llm_call_data['step'],
            'timestamp': time.time(),
            
            # PRE-ADVICE STATE (Causal Input Variables)
            'pre_advice': {
                'context': pre_advice_context,
                'game_state': self._extract_game_state(llm_call_data['raw_obs']),
                'performance': llm_call_data['performance_metrics'].copy(),
                'recent_performance_trend': self._compute_performance_trend(),
                'steps_since_last_llm': self.current_step - self.last_llm_call_step,
            },
            
            # LLM ADVICE (Treatment Variable)
            'llm_advice': {
                'semantic_description': llm_call_data['semantic_description'],
                'raw_response': llm_call_data['llm_response'],
                'parsed_strategy': llm_call_data['strategy'],
                'action_hints': llm_call_data['action_hints'].tolist(),
                'boosted_actions': llm_call_data['boosted_actions'],
                'call_duration': llm_call_data['call_duration'],
            },
            
            # ADVICE CHARACTERISTICS (Moderator Variables)
            'advice_features': {
                'n_boosted_actions': len(llm_call_data['boosted_actions']),
                'max_hint_value': float(np.max(llm_call_data['action_hints'])),
                'hint_entropy': self._compute_hint_entropy(llm_call_data['action_hints']),
                'strategy_category': llm_call_data['strategy'],
            },
            
            # POST-ADVICE TRACKING (Will be filled later)
            'post_advice': {
                'immediate': {},  # Next 10 steps
                'short_term': {},  # Next 50 steps
                'episode_end': {},  # Until episode ends
            },
            
            # COUNTERFACTUAL MARKERS
            'counterfactual': {
                'expected_action_without_llm': None,  # Will be filled
                'expected_reward_without_llm': float(np.mean(list(self.performance_window))) if self.performance_window else 0.0,
            }
        }
        
        self.llm_calls.append(record)
        self.last_llm_call_step = self.current_step
        
        # Write immediately (append mode)
        self._write_record(record)
        
        print(f"📝 Logged LLM call #{record['call_id']} at step {self.current_step}")
        
        return record['call_id']  # Return ID for later updating
    
    def log_post_advice_outcome(self, call_id, outcome_data, outcome_type='immediate'):
        """
        Update LLM call record with post-advice outcomes.
        
        Args:
            call_id: ID of the LLM call to update
            outcome_data: dict with outcome metrics
            outcome_type: 'immediate' (next 10 steps), 'short_term' (next 50), or 'episode_end'
        """
        if call_id >= len(self.llm_calls):
            return
        
        self.llm_calls[call_id]['post_advice'][outcome_type] = outcome_data
        
        # Re-write updated record
        self._write_record(self.llm_calls[call_id])
    
    def compute_post_advice_metrics(self, call_id, steps_ahead=10):
        """
        Compute outcomes for N steps after LLM advice.
        
        Returns dict with:
            - avg_reward: average reward over next N steps
            - reward_improvement: compared to baseline
            - actions_taken: list of actions
            - action_diversity: entropy of actions
            - survival_steps: how many steps survived
            - goal_progress: specific game progress metrics
        """
        if call_id >= len(self.llm_calls):
            return {}
        
        call_record = self.llm_calls[call_id]
        call_step = call_record['step']
        
        # Extract next N steps from post_advice_buffer
        future_steps = [s for s in self.post_advice_buffer 
                       if s['step'] > call_step and s['step'] <= call_step + steps_ahead]
        
        if not future_steps:
            return {}
        
        rewards = [s['reward'] for s in future_steps]
        shaped_rewards = [s['shaped_reward'] for s in future_steps]
        actions = [s['action'] for s in future_steps]
        
        # Compute metrics
        metrics = {
            'n_steps': len(future_steps),
            'avg_reward': float(np.mean(rewards)),
            'total_reward': float(np.sum(rewards)),
            'avg_shaped_reward': float(np.mean(shaped_rewards)),
            'reward_std': float(np.std(rewards)),
            'actions_taken': actions,
            'action_diversity': self._compute_action_diversity(actions),
            'survival_steps': len(future_steps),
            'died': future_steps[-1].get('done', False) if future_steps else False,
        }
        
        # Compare to baseline (expected reward without LLM)
        baseline_reward = call_record['counterfactual']['expected_reward_without_llm']
        metrics['reward_improvement'] = metrics['avg_reward'] - baseline_reward
        metrics['relative_improvement'] = (metrics['reward_improvement'] / abs(baseline_reward)) if baseline_reward != 0 else 0.0
        
        return metrics
    
    def finalize_episode(self, episode_stats):
        """
        Finalize episode and compute all pending post-advice metrics.
        
        Args:
            episode_stats: dict with:
                - total_reward
                - total_steps
                - final_health
                - died
        """
        # For each LLM call in this episode, compute final outcomes
        episode_llm_calls = [c for c in self.llm_calls if c['episode'] == self.current_episode]
        
        for call in episode_llm_calls:
            call_id = call['call_id']
            
            # Compute immediate outcomes (next 10 steps)
            if not call['post_advice']['immediate']:
                immediate_metrics = self.compute_post_advice_metrics(call_id, steps_ahead=10)
                self.log_post_advice_outcome(call_id, immediate_metrics, 'immediate')
            
            # Compute short-term outcomes (next 50 steps)
            if not call['post_advice']['short_term']:
                short_term_metrics = self.compute_post_advice_metrics(call_id, steps_ahead=50)
                self.log_post_advice_outcome(call_id, short_term_metrics, 'short_term')
            
            # Episode-end outcomes
            steps_to_end = episode_stats['total_steps'] - call['step']
            episode_end_metrics = {
                'steps_to_episode_end': steps_to_end,
                'episode_total_reward': episode_stats['total_reward'],
                'episode_died': episode_stats['died'],
                'episode_final_health': episode_stats.get('final_health', 0),
            }
            self.log_post_advice_outcome(call_id, episode_end_metrics, 'episode_end')
    
    def save_summary(self):
        """Generate and save summary statistics"""
        if not self.llm_calls:
            print("⚠️ No LLM calls to summarize")
            return
        
        summary = {
            'total_llm_calls': len(self.llm_calls),
            'episodes_with_llm': len(set(c['episode'] for c in self.llm_calls)),
            
            'strategy_distribution': self._compute_strategy_distribution(),
            'average_outcomes': self._compute_average_outcomes(),
            'best_strategies': self._rank_strategies_by_outcome(),
            
            'causal_indicators': {
                'avg_reward_improvement': self._compute_avg_improvement(),
                'success_rate': self._compute_success_rate(),
                'advice_followed_rate': self._compute_advice_followed_rate(),
            },
            
            'generated_at': datetime.now().isoformat(),
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary saved to {self.summary_file}")
        return summary
    
    # ========================================
    # HELPER METHODS
    # ========================================
    
    def _extract_pre_advice_context(self):
        """Extract what was happening before LLM call"""
        if len(self.pre_advice_buffer) < 5:
            return {}
        
        recent_steps = list(self.pre_advice_buffer)[-10:]
        
        return {
            'recent_actions': [s['action'] for s in recent_steps],
            'recent_rewards': [s['reward'] for s in recent_steps],
            'recent_health': [s['health_ratio'] for s in recent_steps],
            'action_repetition': self._compute_action_repetition(recent_steps),
            'reward_trend': self._compute_trend([s['reward'] for s in recent_steps]),
        }
    
    def _extract_game_state(self, raw_obs):
        """Extract key game state features"""
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        
        stats = raw_obs.get('blstats', np.zeros(26))
        
        return {
            'health_ratio': self._extract_health_ratio(raw_obs),
            'level': int(stats[18]) if len(stats) > 18 else 0,
            'experience': int(stats[19]) if len(stats) > 19 else 0,
            'depth': int(stats[12]) if len(stats) > 12 else 0,
            'gold': int(stats[13]) if len(stats) > 13 else 0,
        }
    
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
    
    def _compute_performance_trend(self):
        """Compute trend in recent performance"""
        if len(self.performance_window) < 10:
            return 0.0
        
        recent = list(self.performance_window)[-10:]
        return self._compute_trend(recent)
    
    def _compute_trend(self, values):
        """Compute linear trend of values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _compute_hint_entropy(self, hints):
        """Compute entropy of action hints (how diverse/uncertain)"""
        hints = np.array(hints)
        hints = hints[hints > 0]  # Only non-zero
        
        if len(hints) == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = hints / np.sum(hints)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return float(entropy)
    
    def _compute_action_diversity(self, actions):
        """Compute diversity of actions taken"""
        if not actions:
            return 0.0
        
        unique_actions = len(set(actions))
        return unique_actions / len(actions)
    
    def _compute_action_repetition(self, steps):
        """Check if actions are repeating (stuck)"""
        actions = [s['action'] for s in steps]
        if len(actions) < 3:
            return 0.0
        
        from collections import Counter
        most_common_count = Counter(actions).most_common(1)[0][1]
        
        return most_common_count / len(actions)
    
    def _compute_strategy_distribution(self):
        """Count how often each strategy was used"""
        from collections import Counter
        strategies = [c['llm_advice']['parsed_strategy'] for c in self.llm_calls]
        return dict(Counter(strategies))
    
    def _compute_average_outcomes(self):
        """Average outcomes across all LLM calls"""
        if not self.llm_calls:
            return {}
        
        immediate_rewards = []
        short_term_rewards = []
        
        for call in self.llm_calls:
            if call['post_advice']['immediate']:
                immediate_rewards.append(call['post_advice']['immediate'].get('avg_reward', 0))
            if call['post_advice']['short_term']:
                short_term_rewards.append(call['post_advice']['short_term'].get('avg_reward', 0))
        
        return {
            'avg_immediate_reward': float(np.mean(immediate_rewards)) if immediate_rewards else 0.0,
            'avg_short_term_reward': float(np.mean(short_term_rewards)) if short_term_rewards else 0.0,
        }
    
    def _rank_strategies_by_outcome(self):
        """Rank strategies by their outcomes"""
        strategy_outcomes = {}
        
        for call in self.llm_calls:
            strategy = call['llm_advice']['parsed_strategy']
            
            if strategy not in strategy_outcomes:
                strategy_outcomes[strategy] = []
            
            if call['post_advice']['immediate']:
                reward_improvement = call['post_advice']['immediate'].get('reward_improvement', 0)
                strategy_outcomes[strategy].append(reward_improvement)
        
        # Average improvement per strategy
        rankings = {}
        for strategy, improvements in strategy_outcomes.items():
            rankings[strategy] = float(np.mean(improvements)) if improvements else 0.0
        
        # Sort by improvement
        return dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))
    
    def _compute_avg_improvement(self):
        """Average reward improvement across all LLM calls"""
        improvements = []
        
        for call in self.llm_calls:
            if call['post_advice']['immediate']:
                improvements.append(call['post_advice']['immediate'].get('reward_improvement', 0))
        
        return float(np.mean(improvements)) if improvements else 0.0
    
    def _compute_success_rate(self):
        """What % of LLM advice led to improvement?"""
        successes = 0
        total = 0
        
        for call in self.llm_calls:
            if call['post_advice']['immediate']:
                total += 1
                if call['post_advice']['immediate'].get('reward_improvement', 0) > 0:
                    successes += 1
        
        return successes / total if total > 0 else 0.0
    
    def _compute_advice_followed_rate(self):
        """What % of advised actions were actually taken?"""
        # Would need to track this during action selection
        # Placeholder for now
        return 0.0
    
    def _write_record(self, record):
        """Write record to JSONL file (append mode)"""
        with open(self.log_file, 'a') as f:
            json.dump(record, f)
            f.write('\n')