import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class NetHackLogParser:
    """Parse NetHack LLM+RL agent logs with semantic action matching."""
    
    def __init__(self, log_file: str, alignment_window: int = 10):
        self.log_file = log_file
        self.data = []
        self.alignment_window = alignment_window
        
        # CRITICAL: Semantic mapping of advice to actual game actions
        self.semantic_mappings = {
            'search': ['move_north', 'move_south', 'move_east', 'move_west', 
                      'move_northeast', 'move_northwest', 'move_southeast', 'move_southwest',
                      'search', 'look'],
            'explore': ['move_north', 'move_south', 'move_east', 'move_west',
                       'move_northeast', 'move_northwest', 'move_southeast', 'move_southwest',
                       'search', 'look', 'open'],
            'move': ['move_north', 'move_south', 'move_east', 'move_west',
                    'move_northeast', 'move_northwest', 'move_southeast', 'move_southwest'],
            'eat': ['eat', 'apply', 'drop'],  # apply often used for eating in NetHack
            'food': ['eat', 'apply', 'drop'],
            'heal': ['eat', 'apply', 'quaff', 'drink'],
            'drink': ['quaff', 'apply', 'drink'],
            'rest': ['wait', 'search', 'move_north'],  # resting often = moving/waiting
            'take': ['take', 'pickup', 'pick'],
            'inventory': ['take', 'drop', 'pickup', 'pick', 'apply'],
            'attack': ['attack', 'fight', 'fire', 'throw', 'kick'],
            'fight': ['attack', 'fight', 'fire', 'throw', 'kick'],
            'flee': ['move_north', 'move_south', 'move_east', 'move_west',
                    'move_northeast', 'move_northwest', 'move_southeast', 'move_southwest'],
            'escape': ['move_north', 'move_south', 'move_east', 'move_west',
                      'move_northeast', 'move_northwest', 'move_southeast', 'move_southwest'],
            'equip': ['wear', 'wield', 'takeoff', 'take_off', 'put_on'],
            'wear': ['wear', 'put_on'],
            'wield': ['wield'],
            'read': ['read'],
            'open': ['open', 'open_door'],
            'close': ['close', 'close_door'],
        }
        
    def _semantically_matches(self, action: str, suggestion: str) -> bool:
        """
        Check if action semantically matches suggestion.
        
        Examples:
        - suggestion="search", action="move_northeast" -> True (moving is searching)
        - suggestion="eat", action="apply" -> True (applying food is eating)
        - suggestion="take", action="drop" -> False
        """
        action_lower = action.lower()
        suggestion_lower = suggestion.lower()
        
        # Direct match
        if suggestion_lower in action_lower or action_lower in suggestion_lower:
            return True
        
        # Semantic match via mapping
        if suggestion_lower in self.semantic_mappings:
            valid_actions = self.semantic_mappings[suggestion_lower]
            for valid_action in valid_actions:
                if action_lower.startswith(valid_action.lower()) or valid_action.lower() in action_lower:
                    return True
        
        # Check if action category matches suggestion category
        action_cat = self._categorize_action(action)
        suggestion_cat = self._categorize_advice_keyword(suggestion)
        
        if action_cat != 'other' and suggestion_cat != 'other' and action_cat == suggestion_cat:
            return True
        
        return False
    
    def _categorize_advice_keyword(self, keyword: str) -> str:
        """Categorize advice keywords."""
        if not keyword or pd.isna(keyword):
            return 'other'
        
        keyword_lower = keyword.lower()
        
        if keyword_lower in ['search', 'explore', 'look', 'find']:
            return 'exploration'
        elif keyword_lower in ['move', 'go', 'navigate', 'flee', 'escape', 'run']:
            return 'movement'
        elif keyword_lower in ['eat', 'food', 'heal', 'restore', 'drink', 'quaff']:
            return 'item_use'
        elif keyword_lower in ['attack', 'fight', 'kill', 'combat']:
            return 'combat'
        elif keyword_lower in ['take', 'drop', 'pickup', 'inventory']:
            return 'inventory'
        elif keyword_lower in ['equip', 'wear', 'wield']:
            return 'equipment'
        else:
            return 'other'
    
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
                advice_given_at_step = None
                
            # Parse step action line
            elif line.startswith("Step"):
                step_data = self._parse_step_line(line)
                if step_data:
                    if current_advice and advice_given_at_step is None:
                        advice_given_at_step = step_data['step']
                    
                    step_data['llm_advice'] = current_advice
                    step_data['llm_data'] = current_llm_data
                    step_data['advice_given_at_step'] = advice_given_at_step
                    step_data['steps_since_advice'] = step_data['step'] - advice_given_at_step if advice_given_at_step else None
                    
                    self.data.append(step_data)
            
            # Parse debug LLM response
            elif line.startswith("[DEBUG] Raw LLM response"):
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
        df = self._compute_semantic_alignment(df)
        return df
    
    def _parse_step_line(self, line: str) -> Optional[Dict]:
        """Parse a step line into structured data."""
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
    
    def _compute_semantic_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute alignment using SEMANTIC matching, not just string matching.
        
        KEY: When LLM says "search", agent doing "move_east" IS following advice!
        """
        if df.empty:
            return df
        
        # Extract suggestions and advice text
        df['llm_suggestions'] = df['llm_data'].apply(
            lambda x: x.get('action_suggestions', []) if isinstance(x, dict) else []
        )
        
        df['llm_priority'] = df['llm_data'].apply(
            lambda x: x.get('immediate_priority', '') if isinstance(x, dict) else ''
        )
        
        # Extract keywords from advice text and priority
        df['advice_keywords'] = df.apply(self._extract_advice_keywords, axis=1)
        
        # Check semantic matches at step level
        df['action_matches_any_suggestion'] = False
        df['action_matches_semantic'] = False
        df['matched_suggestion'] = None
        
        for idx in df.index:
            action = df.loc[idx, 'action']
            suggestions = df.loc[idx, 'llm_suggestions']
            keywords = df.loc[idx, 'advice_keywords']
            
            # Check against explicit suggestions
            for sugg in suggestions:
                if self._semantically_matches(action, sugg):
                    df.loc[idx, 'action_matches_any_suggestion'] = True
                    df.loc[idx, 'action_matches_semantic'] = True
                    df.loc[idx, 'matched_suggestion'] = sugg
                    break
            
            # Check against advice keywords (priority and advice text)
            if not df.loc[idx, 'action_matches_semantic']:
                for keyword in keywords:
                    if self._semantically_matches(action, keyword):
                        df.loc[idx, 'action_matches_semantic'] = True
                        df.loc[idx, 'matched_suggestion'] = keyword
                        break
        
        # Create advice episodes
        advice_changes = (df['advice_given_at_step'] != df['advice_given_at_step'].shift(1)).cumsum()
        df['advice_episode'] = advice_changes
        
        # For each episode, check if ANY action in window matches
        df['alignment_window_matched'] = False
        df['steps_until_match'] = None
        
        for episode_id in df['advice_episode'].unique():
            if pd.isna(episode_id):
                continue
            
            episode_mask = df['advice_episode'] == episode_id
            episode_indices = df[episode_mask].index
            
            if len(episode_indices) == 0:
                continue
            
            # Check semantic matches within window
            window_size = min(len(episode_indices), self.alignment_window)
            window_indices = episode_indices[:window_size]
            
            matches = df.loc[window_indices, 'action_matches_semantic'].values
            
            if matches.any():
                first_match_idx = np.where(matches)[0][0]
                df.loc[episode_indices, 'alignment_window_matched'] = True
                df.loc[episode_indices, 'steps_until_match'] = first_match_idx
        
        # Episode-level metrics
        df['episode_alignment_rate'] = df.groupby('advice_episode')['action_matches_semantic'].transform('mean')
        df['episode_matched_in_window'] = df.groupby('advice_episode')['alignment_window_matched'].transform('first')
        
        # Treatment definitions
        df['followed_advice_strict'] = (df['episode_alignment_rate'] >= 0.3).astype(int)
        df['followed_advice_lenient'] = df['episode_matched_in_window'].fillna(False).astype(int)
        df['alignment_strength'] = df['episode_alignment_rate']
        df['followed_advice'] = df['followed_advice_lenient']
        
        # Episode stats
        df['episode_avg_reward'] = df.groupby('advice_episode')['reward'].transform('mean')
        df['episode_reward_sum'] = df.groupby('advice_episode')['reward'].transform('sum')
        df['episode_length'] = df.groupby('advice_episode')['step'].transform('count')
        
        return df
    
    def _extract_advice_keywords(self, row) -> Set[str]:
        """Extract actionable keywords from advice and priority."""
        keywords = set()
        
        # From priority
        priority = row.get('llm_priority', '')
        if priority and not pd.isna(priority):
            # Common action keywords
            action_words = ['search', 'explore', 'eat', 'drink', 'attack', 'fight', 
                          'move', 'take', 'drop', 'wear', 'wield', 'read', 'flee',
                          'heal', 'rest', 'open', 'close', 'look', 'find']
            
            priority_lower = priority.lower()
            for word in action_words:
                if word in priority_lower:
                    keywords.add(word)
        
        # From suggestions
        suggestions = row.get('llm_suggestions', [])
        if suggestions:
            keywords.update(suggestions)
        
        # From advice text
        advice = row.get('llm_advice', '')
        if advice and not pd.isna(advice):
            advice_lower = advice.lower()
            for word in ['search', 'explore', 'eat', 'drink', 'attack', 'fight', 
                        'move', 'take', 'drop', 'wear', 'wield', 'read', 'flee',
                        'heal', 'rest', 'open', 'close', 'look', 'find']:
                if word in advice_lower:
                    keywords.add(word)
        
        return keywords
    
    def _enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features and causal variables."""
        if df.empty:
            return df
        
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
        
        # Change metrics
        df['hp_change'] = df['hp'].diff()
        df['reward_change'] = df['reward'].diff()
        df['sr_change'] = df['smoothed_reward'].diff()
        df['level_up'] = (df['level'].diff() > 0).astype(int)
        
        # Lag features (confounders)
        df['hp_lag1'] = df['hp'].shift(1)
        df['reward_lag1'] = df['reward'].shift(1)
        df['sr_lag1'] = df['smoothed_reward'].shift(1)
        df['level_lag1'] = df['level'].shift(1)
        
        # Rolling statistics
        df['reward_ma5'] = df['reward'].rolling(window=5, min_periods=1).mean()
        df['hp_ma5'] = df['hp'].rolling(window=5, min_periods=1).mean()
        df['reward_std5'] = df['reward'].rolling(window=5, min_periods=1).std()
        
        # Categorize
        df['advice_category'] = df['llm_advice'].apply(self._categorize_advice)
        df['action_category'] = df['action'].apply(self._categorize_action)
        
        # Critical states
        df['critical_hp'] = (df['hp'] < 50).astype(int)
        df['low_hp'] = (df['hp'] < 100).astype(int)
        df['high_hp'] = (df['hp'] > 150).astype(int)
        
        return df
    
    def _categorize_advice(self, advice: str) -> str:
        """Categorize LLM advice into types."""
        if not advice or pd.isna(advice):
            return 'none'
        
        advice_lower = advice.lower()
        
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
    
    def generate_alignment_report(self, df: pd.DataFrame) -> str:
        """Generate detailed alignment report with examples."""
        report = []
        report.append("=" * 80)
        report.append("SEMANTIC ALIGNMENT DETECTION REPORT")
        report.append("=" * 80)
        
        total_episodes = df['advice_episode'].nunique()
        matched_lenient = (df.groupby('advice_episode')['followed_advice_lenient'].first() == 1).sum()
        matched_strict = (df.groupby('advice_episode')['followed_advice_strict'].first() == 1).sum()
        
        report.append(f"\nTotal advice episodes: {total_episodes}")
        report.append(f"\nAlignment Detection (semantic matching, {self.alignment_window}-step window):")
        report.append(f"  Lenient (≥1 semantic match):  {matched_lenient} episodes ({matched_lenient/total_episodes*100:.1f}%)")
        report.append(f"  Strict (≥30% actions match):  {matched_strict} episodes ({matched_strict/total_episodes*100:.1f}%)")
        
        report.append(f"\nStep-level semantic alignment:")
        report.append(f"  Actions semantically matching advice: {df['action_matches_semantic'].sum()} / {len(df)} ({df['action_matches_semantic'].mean()*100:.1f}%)")
        
        # Show what gets matched
        matched_df = df[df['action_matches_semantic'] == True]
        if len(matched_df) > 0:
            report.append(f"\nTop semantic matches:")
            match_counts = matched_df.groupby(['matched_suggestion', 'action']).size().sort_values(ascending=False).head(10)
            for (sugg, action), count in match_counts.items():
                report.append(f"  '{sugg}' → {action}: {count} times")
        
        # Distribution of match timing
        matched_episodes = df[df['episode_matched_in_window'] == True].groupby('advice_episode')['steps_until_match'].first()
        if len(matched_episodes) > 0:
            report.append(f"\nWhen advice IS followed:")
            report.append(f"  Average steps until match: {matched_episodes.mean():.1f}")
            report.append(f"  Immediate (step 0): {(matched_episodes == 0).sum()} episodes")
            report.append(f"  Delayed (step 1-5): {((matched_episodes >= 1) & (matched_episodes <= 5)).sum()} episodes")
        
        # Sample matched episodes
        report.append(f"\n" + "="*80)
        report.append("SAMPLE MATCHED EPISODES:")
        report.append("="*80)
        
        matched_eps = df[df['episode_matched_in_window'] == True]['advice_episode'].unique()[:5]
        for ep_id in matched_eps:
            ep_df = df[df['advice_episode'] == ep_id].head(min(10, self.alignment_window))
            if len(ep_df) > 0:
                advice = ep_df['llm_advice'].iloc[0]
                keywords = ep_df['advice_keywords'].iloc[0]
                
                report.append(f"\nEpisode {ep_id}:")
                report.append(f"  Advice: {advice[:80]}...")
                report.append(f"  Keywords: {keywords}")
                report.append(f"  Actions:")
                for _, row in ep_df.iterrows():
                    match_mark = "✓" if row['action_matches_semantic'] else " "
                    matched_on = f" (matched '{row['matched_suggestion']}')" if row['matched_suggestion'] else ""
                    report.append(f"    [{match_mark}] {row['action']}{matched_on}")
        
        return "\n".join(report)
    
    def save_processed_data(self, output_path: str, df: pd.DataFrame):
        """Save processed data with alignment report."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path / 'processed_data.csv', index=False)
        df.to_parquet(output_path / 'processed_data.parquet', index=False)
        
        episode_summary = self._create_episode_summary(df)
        episode_summary.to_csv(output_path / 'episode_summary.csv', index=False)
        
        alignment_report = self.generate_alignment_report(df)
        with open(output_path / 'alignment_report.txt', 'w') as f:
            f.write(alignment_report)
        
        metadata = self._generate_metadata(df)
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Data saved to {output_path}")
        print(f"✓ Episodes following advice: {df['followed_advice'].sum()} / {df['advice_episode'].nunique()} ({df['followed_advice'].sum()/df['advice_episode'].nunique()*100:.1f}%)")
        print(f"\n📊 See alignment_report.txt for semantic matching details")
    
    def _create_episode_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create episode-level summary."""
        episode_stats = []
        
        for episode_id in df['advice_episode'].unique():
            if pd.isna(episode_id):
                continue
            
            ep = df[df['advice_episode'] == episode_id]
            
            stats = {
                'advice_episode': episode_id,
                'start_step': ep['step'].min(),
                'episode_length': len(ep),
                'llm_advice': ep['llm_advice'].iloc[0] if len(ep) > 0 else '',
                'advice_category': ep['advice_category'].iloc[0] if len(ep) > 0 else '',
                
                'followed_advice_lenient': ep['followed_advice_lenient'].iloc[0],
                'followed_advice_strict': ep['followed_advice_strict'].iloc[0],
                'alignment_rate': ep['episode_alignment_rate'].iloc[0],
                'matched_in_window': ep['episode_matched_in_window'].iloc[0] if len(ep) > 0 else False,
                
                'total_reward': ep['reward'].sum(),
                'avg_reward': ep['reward'].mean(),
                'sr_change': ep['smoothed_reward'].iloc[-1] - ep['smoothed_reward'].iloc[0] if len(ep) > 0 else 0,
                'hp_change': ep['hp'].iloc[-1] - ep['hp'].iloc[0] if len(ep) > 0 else 0,
                
                'hp_before': ep['hp'].iloc[0] if len(ep) > 0 else None,
                'reward_before': ep['reward_lag1'].iloc[0] if len(ep) > 0 and not pd.isna(ep['reward_lag1'].iloc[0]) else 0,
                'sr_before': ep['smoothed_reward'].iloc[0] if len(ep) > 0 else None,
                'level_before': ep['level'].iloc[0] if len(ep) > 0 else None,
                'critical_hp': int(ep['hp'].iloc[0] < 50) if len(ep) > 0 else 0,
            }
            
            episode_stats.append(stats)
        
        return pd.DataFrame(episode_stats)
    
    def _generate_metadata(self, df: pd.DataFrame) -> dict:
        """Generate metadata."""
        return {
            'total_steps': len(df),
            'total_advice_episodes': int(df['advice_episode'].nunique()),
            'alignment_window': self.alignment_window,
            'semantic_alignment_rate': float(df['action_matches_semantic'].mean()),
            'episodes_following_advice': int(df['followed_advice'].sum()),
            'treatment_balance': {
                'followed': int(df['followed_advice'].sum()),
                'not_followed': int((df['followed_advice'] == 0).sum())
            }
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse NetHack logs with semantic alignment')
    parser.add_argument('--log_file', type=str, default='nethack_logs.txt')
    parser.add_argument('--output_dir', type=str, default='processed_data')
    parser.add_argument('--window', type=int, default=10)
    
    args = parser.parse_args()
    
    log_parser = NetHackLogParser(args.log_file, alignment_window=args.window)
    
    print(f"Parsing with SEMANTIC matching (window={args.window})...")
    df = log_parser.parse_logs()
    
    print(f"\n✓ Parsed {len(df)} steps")
    log_parser.save_processed_data(args.output_dir, df)