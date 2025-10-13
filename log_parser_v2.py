import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class NetHackLogParser:
    """Parse NetHack LLM+RL agent logs for causal analysis with proper temporal alignment."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.data = []
        self.advice_episodes = []  # Track advice episodes
        
    def parse_logs(self) -> pd.DataFrame:
        """Main parsing function with temporal tracking."""
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        current_advice = None
        current_llm_data = None
        advice_given_at_step = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Parse LLM advice line
            if line.startswith("LLM Advice:"):
                current_advice = line.replace("LLM Advice:", "").strip()
                current_llm_data = None
                advice_given_at_step = None  # Will be set at next step
                
            # Parse step action line
            elif line.startswith("Step"):
                step_data = self._parse_step_line(line)
                if step_data:
                    # Mark when advice was given
                    if current_advice and advice_given_at_step is None:
                        advice_given_at_step = step_data['step']
                    
                    step_data['llm_advice'] = current_advice
                    step_data['llm_data'] = current_llm_data
                    step_data['advice_given_at_step'] = advice_given_at_step
                    step_data['steps_since_advice'] = step_data['step'] - advice_given_at_step if advice_given_at_step else None
                    
                    self.data.append(step_data)
            
            # Parse debug LLM response
            elif line.startswith("[DEBUG] Raw LLM response"):
                # Look ahead to get the JSON
                j = i + 1
                json_lines = []
                while j < len(lines) and not lines[j].strip().startswith("LLM Advice:") and not lines[j].strip().startswith("Step"):
                    json_lines.append(lines[j])
                    j += 1
                
                json_str = ''.join(json_lines).strip()
                try:
                    json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                    if json_match:
                        current_llm_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    current_llm_data = None
                    
            i += 1
        
        df = pd.DataFrame(self.data)
        df = self._enrich_dataframe(df)
        df = self._compute_temporal_alignment(df)
        return df
    
    def _parse_step_line(self, line: str) -> Optional[Dict]:
        """Parse a step line into structured data."""
        # Pattern: Step 1100 | Action: drink | R: -0.01 | SR: -0.010 | HP: 180.0% | Lvl: 14
        pattern = r'Step\s+(\d+)\s+\|\s+Action:\s+(\w+)\s+\|\s+R:\s+([-\d.]+)\s+\|\s+SR:\s+([-\d.]+)\s+\|\s+HP:\s+([\d.]+)%\s+\|\s+Lvl:\s+(\d+)'
        match = re.match(pattern, line)
        
        if match:
            return {
                'step': int(match.group(1)),
                'action': match.group(2),
                'reward': float(match.group(3)),
                'smoothed_reward': float(match.group(4)),
                'hp': float(match.group(5)),
                'level': int(match.group(6))
            }
        return None
    
    def _compute_temporal_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute proper temporal alignment between advice and actions.
        
        Key insight: An action "follows advice" if it matches suggestions 
        within a reasonable time window after advice is given.
        """
        if df.empty:
            return df
        
        # Extract LLM suggestions
        df['llm_suggestions'] = df['llm_data'].apply(
            lambda x: x.get('action_suggestions', []) if isinstance(x, dict) else []
        )
        
        # For each step, check if action matches ANY suggestion in the current advice window
        df['action_matches_any_suggestion'] = df.apply(
            lambda row: any(
                row['action'].startswith(sugg) or sugg in row['action'] 
                for sugg in row['llm_suggestions']
            ) if row['llm_suggestions'] else False,
            axis=1
        )
        
        # More sophisticated matching: exact match, partial match, semantic match
        df['action_exact_match'] = df.apply(
            lambda row: row['action'] in row['llm_suggestions'] if row['llm_suggestions'] else False,
            axis=1
        )
        
        df['action_partial_match'] = df.apply(
            lambda row: any(
                sugg in row['action'] or row['action'] in sugg
                for sugg in row['llm_suggestions']
            ) if row['llm_suggestions'] else False,
            axis=1
        )
        
        # Compute alignment score: how well actions align with advice over time window
        window_size = 5  # Consider 5 steps after advice
        
        df['alignment_score'] = 0.0
        for idx in df.index:
            if pd.isna(df.loc[idx, 'advice_given_at_step']):
                continue
            
            steps_since = df.loc[idx, 'steps_since_advice']
            if steps_since is not None and 0 <= steps_since <= window_size:
                # Weight decreases with time since advice
                time_weight = 1.0 - (steps_since / window_size)
                
                if df.loc[idx, 'action_exact_match']:
                    df.loc[idx, 'alignment_score'] = 1.0 * time_weight
                elif df.loc[idx, 'action_partial_match']:
                    df.loc[idx, 'alignment_score'] = 0.5 * time_weight
        
        # Create advice episodes: group steps by advice period
        advice_changes = (df['advice_given_at_step'] != df['advice_given_at_step'].shift(1)).cumsum()
        df['advice_episode'] = advice_changes
        
        # Compute episode-level statistics
        df['episode_alignment_rate'] = df.groupby('advice_episode')['action_matches_any_suggestion'].transform('mean')
        df['episode_avg_reward'] = df.groupby('advice_episode')['reward'].transform('mean')
        df['episode_reward_sum'] = df.groupby('advice_episode')['reward'].transform('sum')
        df['episode_length'] = df.groupby('advice_episode')['step'].transform('count')
        
        # Binary treatment: Did agent follow advice within reasonable time window?
        # Consider an episode as "treated" if alignment rate > threshold
        alignment_threshold = 0.3
        df['followed_advice'] = (df['episode_alignment_rate'] >= alignment_threshold).astype(int)
        
        # Alternative treatment: First action after advice
        df['is_first_action_after_advice'] = (df['steps_since_advice'] == 0).astype(int)
        df['first_action_matches'] = (df['is_first_action_after_advice'] & df['action_matches_any_suggestion']).astype(int)
        
        return df
    
    def _enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features and causal variables."""
        if df.empty:
            return df
        
        # Sort by step
        df = df.sort_values('step').reset_index(drop=True)
        
        # Extract LLM structured data
        df['llm_priority'] = df['llm_data'].apply(
            lambda x: x.get('immediate_priority', '') if isinstance(x, dict) else ''
        )
        df['llm_risk'] = df['llm_data'].apply(
            lambda x: x.get('risk_assessment', '') if isinstance(x, dict) else ''
        )
        df['llm_strategy'] = df['llm_data'].apply(
            lambda x: x.get('strategy', '') if isinstance(x, dict) else ''
        )
        df['llm_opportunities'] = df['llm_data'].apply(
            lambda x: x.get('opportunities', '') if isinstance(x, dict) else ''
        )
        
        # Change metrics (outcomes)
        df['hp_change'] = df['hp'].diff()
        df['reward_change'] = df['reward'].diff()
        df['sr_change'] = df['smoothed_reward'].diff()
        df['level_up'] = (df['level'].diff() > 0).astype(int)
        
        # Lag features (confounders - state before advice)
        df['hp_lag1'] = df['hp'].shift(1)
        df['hp_lag2'] = df['hp'].shift(2)
        df['reward_lag1'] = df['reward'].shift(1)
        df['sr_lag1'] = df['smoothed_reward'].shift(1)
        df['level_lag1'] = df['level'].shift(1)
        
        # Rolling statistics (context)
        df['reward_ma5'] = df['reward'].rolling(window=5, min_periods=1).mean()
        df['reward_ma10'] = df['reward'].rolling(window=10, min_periods=1).mean()
        df['hp_ma5'] = df['hp'].rolling(window=5, min_periods=1).mean()
        df['reward_std5'] = df['reward'].rolling(window=5, min_periods=1).std()
        df['hp_std5'] = df['hp'].rolling(window=5, min_periods=1).std()
        
        # Volatility measures
        df['reward_volatility'] = df['reward_std5'] / (abs(df['reward_ma5']) + 0.001)
        df['hp_volatility'] = df['hp_std5'] / (df['hp_ma5'] + 0.001)
        
        # Categorize advice
        df['advice_category'] = df['llm_advice'].apply(self._categorize_advice)
        df['priority_category'] = df['llm_priority'].apply(self._categorize_advice)
        df['action_category'] = df['action'].apply(self._categorize_action)
        
        # Critical states (confounders)
        df['critical_hp'] = (df['hp'] < 50).astype(int)
        df['low_hp'] = (df['hp'] < 100).astype(int)
        df['high_hp'] = (df['hp'] > 150).astype(int)
        df['very_high_hp'] = (df['hp'] > 200).astype(int)
        
        # Performance states
        df['positive_reward_trend'] = (df['reward_ma5'] > 0).astype(int)
        df['improving_performance'] = (df['smoothed_reward'].diff() > 0).astype(int)
        
        # Reward momentum
        df['reward_momentum'] = df['smoothed_reward'].diff().rolling(window=3, min_periods=1).mean()
        df['reward_acceleration'] = df['reward_momentum'].diff()
        
        # Episode stage (early, mid, late game)
        df['episode_stage'] = pd.cut(df['step'], bins=3, labels=['early', 'mid', 'late'])
        
        # Progress metrics
        df['steps_per_level'] = df['step'] / (df['level'] + 1)
        df['hp_per_level'] = df['hp'] / (df['level'] + 1)
        
        return df
    
    def _categorize_advice(self, advice: str) -> str:
        """Categorize LLM advice into types."""
        if not advice or pd.isna(advice):
            return 'none'
        
        advice_lower = advice.lower()
        
        # Priority-based categorization
        if any(word in advice_lower for word in ['eat', 'health', 'restore', 'heal', 'food', 'hunger']):
            return 'survival'
        elif any(word in advice_lower for word in ['attack', 'fight', 'combat', 'kill', 'enemy', 'monster']):
            return 'combat'
        elif any(word in advice_lower for word in ['flee', 'escape', 'run', 'avoid', 'danger', 'retreat']):
            return 'defensive'
        elif any(word in advice_lower for word in ['search', 'explore', 'look', 'find', 'area']):
            return 'exploration'
        elif any(word in advice_lower for word in ['equip', 'wear', 'wield', 'take off', 'armor', 'weapon']):
            return 'equipment'
        elif any(word in advice_lower for word in ['read', 'drink', 'use', 'apply', 'potion', 'scroll']):
            return 'item_use'
        elif any(word in advice_lower for word in ['move', 'go', 'navigate', 'direction']):
            return 'movement'
        elif any(word in advice_lower for word in ['drop', 'pick', 'inventory', 'take']):
            return 'inventory'
        else:
            return 'other'
    
    def _categorize_action(self, action: str) -> str:
        """Categorize actions into types."""
        if pd.isna(action):
            return 'none'
        
        action_lower = action.lower()
        
        action_map = {
            'movement': ['move_north', 'move_south', 'move_east', 'move_west', 
                        'move_northeast', 'move_northwest', 'move_southeast', 'move_southwest'],
            'combat': ['attack', 'fight', 'fire', 'throw', 'kick'],
            'item_use': ['eat', 'drink', 'read', 'zap', 'apply', 'quaff'],
            'equipment': ['wear', 'take_off', 'takeoff', 'wield', 'remove', 'put_on'],
            'interaction': ['open', 'close', 'open_door', 'close_door'],
            'exploration': ['search', 'look'],
            'inventory': ['take', 'drop', 'pickup', 'pick']
        }
        
        for category, actions in action_map.items():
            if any(action_lower.startswith(a) for a in actions):
                return category
        
        return 'other'
    
    def save_processed_data(self, output_path: str, df: pd.DataFrame):
        """Save processed data in multiple formats."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # CSV for general use
        df.to_csv(output_path / 'processed_data.csv', index=False)
        
        # Parquet for efficient storage
        df.to_parquet(output_path / 'processed_data.parquet', index=False)
        
        # Episode-level summary
        episode_summary = self._create_episode_summary(df)
        episode_summary.to_csv(output_path / 'episode_summary.csv', index=False)
        
        # Summary statistics
        summary = self._generate_summary(df)
        with open(output_path / 'summary_stats.txt', 'w') as f:
            f.write(summary)
        
        # Metadata about temporal structure
        metadata = self._generate_metadata(df)
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Data saved to {output_path}")
        print(f"Total steps: {len(df)}")
        print(f"Total advice episodes: {df['advice_episode'].nunique()}")
        print(f"Columns: {len(df.columns)}")
    
    def _create_episode_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create episode-level summary for episode-level causal analysis."""
        
        episode_stats = []
        
        for episode_id in df['advice_episode'].unique():
            if pd.isna(episode_id):
                continue
            
            episode_df = df[df['advice_episode'] == episode_id]
            
            stats = {
                'advice_episode': episode_id,
                'start_step': episode_df['step'].min(),
                'end_step': episode_df['step'].max(),
                'episode_length': len(episode_df),
                'llm_advice': episode_df['llm_advice'].iloc[0] if len(episode_df) > 0 else '',
                'advice_category': episode_df['advice_category'].iloc[0] if len(episode_df) > 0 else '',
                'llm_priority': episode_df['llm_priority'].iloc[0] if len(episode_df) > 0 else '',
                
                # Treatment
                'alignment_rate': episode_df['action_matches_any_suggestion'].mean(),
                'followed_advice': episode_df['followed_advice'].iloc[0] if len(episode_df) > 0 else 0,
                'first_action_matches': episode_df['first_action_matches'].sum() > 0,
                
                # Outcomes
                'total_reward': episode_df['reward'].sum(),
                'avg_reward': episode_df['reward'].mean(),
                'final_sr': episode_df['smoothed_reward'].iloc[-1] if len(episode_df) > 0 else None,
                'sr_change': episode_df['smoothed_reward'].iloc[-1] - episode_df['smoothed_reward'].iloc[0] if len(episode_df) > 0 else 0,
                'hp_start': episode_df['hp'].iloc[0] if len(episode_df) > 0 else None,
                'hp_end': episode_df['hp'].iloc[-1] if len(episode_df) > 0 else None,
                'hp_change': episode_df['hp_change'].sum(),
                'level_ups': episode_df['level_up'].sum(),
                
                # Confounders (pre-treatment state)
                'hp_at_advice': episode_df['hp_lag1'].iloc[0] if len(episode_df) > 0 else None,
                'reward_before': episode_df['reward_lag1'].iloc[0] if len(episode_df) > 0 else None,
                'sr_before': episode_df['sr_lag1'].iloc[0] if len(episode_df) > 0 else None,
                'critical_hp_at_start': episode_df['critical_hp'].iloc[0] if len(episode_df) > 0 else 0,
                'level_at_start': episode_df['level'].iloc[0] if len(episode_df) > 0 else None,
                
                # Action distribution
                'most_common_action': episode_df['action_category'].mode()[0] if len(episode_df) > 0 else 'none',
                'unique_actions': episode_df['action'].nunique(),
            }
            
            episode_stats.append(stats)
        
        return pd.DataFrame(episode_stats)
    
    def _generate_metadata(self, df: pd.DataFrame) -> dict:
        """Generate metadata about temporal structure."""
        return {
            'total_steps': len(df),
            'total_advice_episodes': int(df['advice_episode'].nunique()),
            'avg_episode_length': float(df['episode_length'].mean()),
            'median_episode_length': float(df['episode_length'].median()),
            'avg_steps_since_advice': float(df['steps_since_advice'].mean()),
            'overall_alignment_rate': float(df['action_matches_any_suggestion'].mean()),
            'episodes_following_advice': int(df['followed_advice'].sum()),
            'treatment_balance': {
                'followed_advice': int((df['followed_advice'] == 1).sum()),
                'not_followed': int((df['followed_advice'] == 0).sum())
            }
        }
    
    def _generate_summary(self, df: pd.DataFrame) -> str:
        """Generate summary statistics."""
        summary = []
        summary.append("=" * 70)
        summary.append("NetHack LLM+RL Agent Log Summary (Temporally Aligned)")
        summary.append("=" * 70)
        summary.append(f"\nTotal Steps: {len(df)}")
        summary.append(f"Step Range: {df['step'].min()} - {df['step'].max()}")
        summary.append(f"Total Advice Episodes: {df['advice_episode'].nunique()}")
        summary.append(f"Average Episode Length: {df['episode_length'].mean():.1f} steps")
        
        summary.append(f"\n\nTemporal Alignment Metrics:")
        summary.append(f"  Overall alignment rate: {df['action_matches_any_suggestion'].mean():.2%}")
        summary.append(f"  Episodes following advice (>{0.3:.0%} alignment): {df['followed_advice'].mean():.2%}")
        summary.append(f"  First action matches advice: {df['first_action_matches'].sum()} times")
        summary.append(f"  Average steps until action: {df['steps_since_advice'].mean():.1f}")
        
        summary.append(f"\n\nPerformance Metrics:")
        summary.append(f"  Level Range: {df['level'].min()} - {df['level'].max()}")
        summary.append(f"  HP Range: {df['hp'].min():.1f}% - {df['hp'].max():.1f}%")
        summary.append(f"  Total Reward: {df['reward'].sum():.3f}")
        summary.append(f"  Smoothed Reward Range: {df['smoothed_reward'].min():.3f} - {df['smoothed_reward'].max():.3f}")
        
        summary.append(f"\n\nAdvice Categories:")
        for cat, count in df.groupby('advice_episode')['advice_category'].first().value_counts().items():
            summary.append(f"  {cat}: {count}")
        
        summary.append(f"\n\nAction Categories:")
        for cat, count in df['action_category'].value_counts().head(10).items():
            summary.append(f"  {cat}: {count}")
        
        summary.append(f"\n\nCritical States:")
        summary.append(f"  Critical HP episodes: {df['critical_hp'].sum()}")
        summary.append(f"  Level ups: {df['level_up'].sum()}")
        
        # Compare outcomes by treatment
        followed = df[df['followed_advice'] == 1]
        not_followed = df[df['followed_advice'] == 0]
        
        if len(followed) > 0 and len(not_followed) > 0:
            summary.append(f"\n\nOutcome Comparison (Raw):")
            summary.append(f"  When advice followed:")
            summary.append(f"    - Avg reward: {followed['reward'].mean():.4f}")
            summary.append(f"    - Avg HP change: {followed['hp_change'].mean():.4f}")
            summary.append(f"  When advice not followed:")
            summary.append(f"    - Avg reward: {not_followed['reward'].mean():.4f}")
            summary.append(f"    - Avg HP change: {not_followed['hp_change'].mean():.4f}")
            summary.append(f"  Naive difference: {followed['reward'].mean() - not_followed['reward'].mean():.4f}")
            summary.append(f"  (Note: This is not causal - need to control for confounders)")
        
        return "\n".join(summary)


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse NetHack LLM+RL logs')
    parser.add_argument('--log_file', type=str, default='nethack_logs.txt',
                       help='Path to log file')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Parse logs
    log_parser = NetHackLogParser(args.log_file)
    
    print("Parsing logs with temporal alignment...")
    df = log_parser.parse_logs()
    
    print(f"\nParsed {len(df)} steps across {df['advice_episode'].nunique()} advice episodes")
    print(f"Columns: {len(df.columns)}")
    
    # Save processed data
    log_parser.save_processed_data(args.output_dir, df)
    
    # Display sample
    print("\n" + "="*70)
    print("Sample of processed data:")
    print("="*70)
    print(df[['step', 'action', 'reward', 'hp', 'steps_since_advice',
              'action_matches_any_suggestion', 'alignment_score', 
              'followed_advice']].head(15))
    
    print("\n" + "="*70)
    print("Causal Variables for Analysis:")
    print("="*70)
    print("\nStep-level Analysis:")
    print("  Treatment: action_matches_any_suggestion, alignment_score")
    print("  Outcome: reward, hp_change, sr_change")
    print("  Confounders: hp_lag1, reward_lag1, sr_lag1, critical_hp, level_lag1")
    print("\nEpisode-level Analysis (recommended):")
    print("  Treatment: followed_advice, alignment_rate, first_action_matches")
    print("  Outcome: total_reward, sr_change, hp_change")
    print("  Confounders: hp_at_advice, reward_before, sr_before, critical_hp_at_start")