import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class NetHackLogParser:
    """Parse NetHack LLM+RL agent logs for causal analysis."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.data = []
        self.raw_logs = []
        
    def parse_logs(self) -> pd.DataFrame:
        """Main parsing function."""
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        current_advice = None
        current_llm_data = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Parse LLM advice line
            if line.startswith("LLM Advice:"):
                current_advice = line.replace("LLM Advice:", "").strip()
                current_llm_data = None
                
            # Parse step action line
            elif line.startswith("Step"):
                step_data = self._parse_step_line(line)
                if step_data:
                    step_data['llm_advice'] = current_advice
                    step_data['llm_data'] = current_llm_data
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
                    # Extract JSON from the debug output
                    json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                    if json_match:
                        current_llm_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    current_llm_data = None
                    
            i += 1
        
        df = pd.DataFrame(self.data)
        df = self._enrich_dataframe(df)
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
        df['llm_suggestions'] = df['llm_data'].apply(
            lambda x: x.get('action_suggestions', []) if isinstance(x, dict) else []
        )
        
        # Action alignment with LLM suggestions
        df['action_in_suggestions'] = df.apply(
            lambda row: row['action'] in row['llm_suggestions'] if row['llm_suggestions'] else False,
            axis=1
        )
        
        # Change metrics (outcomes)
        df['hp_change'] = df['hp'].diff()
        df['reward_change'] = df['reward'].diff()
        df['sr_change'] = df['smoothed_reward'].diff()
        df['level_up'] = (df['level'].diff() > 0).astype(int)
        
        # Lag features (confounders)
        df['hp_lag1'] = df['hp'].shift(1)
        df['reward_lag1'] = df['reward'].shift(1)
        df['sr_lag1'] = df['smoothed_reward'].shift(1)
        
        # Rolling statistics (context)
        df['reward_ma5'] = df['reward'].rolling(window=5, min_periods=1).mean()
        df['hp_ma5'] = df['hp'].rolling(window=5, min_periods=1).mean()
        df['reward_std5'] = df['reward'].rolling(window=5, min_periods=1).std()
        
        # Categorize advice
        df['advice_category'] = df['llm_advice'].apply(self._categorize_advice)
        df['action_category'] = df['action'].apply(self._categorize_action)
        
        # Critical states
        df['critical_hp'] = (df['hp'] < 50).astype(int)
        df['low_hp'] = (df['hp'] < 100).astype(int)
        df['high_hp'] = (df['hp'] > 150).astype(int)
        
        # Time since last advice change
        advice_changes = (df['llm_advice'] != df['llm_advice'].shift(1)).cumsum()
        df['steps_since_advice_change'] = df.groupby(advice_changes).cumcount()
        
        # Reward momentum
        df['reward_momentum'] = df['smoothed_reward'].diff().rolling(window=3, min_periods=1).mean()
        
        # Episode stage (early, mid, late game)
        df['episode_stage'] = pd.cut(df['step'], bins=3, labels=['early', 'mid', 'late'])
        
        return df
    
    def _categorize_advice(self, advice: str) -> str:
        """Categorize LLM advice into types."""
        if not advice:
            return 'none'
        
        advice_lower = advice.lower()
        
        if any(word in advice_lower for word in ['eat', 'health', 'restore', 'heal']):
            return 'survival'
        elif any(word in advice_lower for word in ['search', 'explore', 'look', 'find']):
            return 'exploration'
        elif any(word in advice_lower for word in ['attack', 'fight', 'combat', 'kill']):
            return 'combat'
        elif any(word in advice_lower for word in ['equip', 'wear', 'wield', 'take off']):
            return 'equipment'
        elif any(word in advice_lower for word in ['read', 'drink', 'use']):
            return 'item_use'
        elif any(word in advice_lower for word in ['move', 'go', 'flee', 'escape']):
            return 'movement'
        else:
            return 'other'
    
    def _categorize_action(self, action: str) -> str:
        """Categorize actions into types."""
        action_map = {
            'move': ['move_north', 'move_south', 'move_east', 'move_west'],
            'combat': ['attack', 'fight'],
            'item_use': ['eat', 'drink', 'read', 'zap', 'apply'],
            'equipment': ['wear', 'take_off', 'wield', 'remove'],
            'interaction': ['open_door', 'close_door', 'kick'],
            'exploration': ['search', 'look'],
            'inventory': ['take', 'drop', 'pickup']
        }
        
        for category, actions in action_map.items():
            if any(action.startswith(a) for a in actions):
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
        
        # JSON for human readability
        df.to_json(output_path / 'processed_data.json', orient='records', indent=2)
        
        # Summary statistics
        summary = self._generate_summary(df)
        with open(output_path / 'summary_stats.txt', 'w') as f:
            f.write(summary)
        
        print(f"Data saved to {output_path}")
        print(f"Total steps: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        
    def _generate_summary(self, df: pd.DataFrame) -> str:
        """Generate summary statistics."""
        summary = []
        summary.append("=" * 60)
        summary.append("NetHack LLM+RL Agent Log Summary")
        summary.append("=" * 60)
        summary.append(f"\nTotal Steps: {len(df)}")
        summary.append(f"Step Range: {df['step'].min()} - {df['step'].max()}")
        summary.append(f"\nLevel Range: {df['level'].min()} - {df['level'].max()}")
        summary.append(f"HP Range: {df['hp'].min():.1f}% - {df['hp'].max():.1f}%")
        summary.append(f"\nTotal Reward: {df['reward'].sum():.3f}")
        summary.append(f"Smoothed Reward Range: {df['smoothed_reward'].min():.3f} - {df['smoothed_reward'].max():.3f}")
        
        summary.append(f"\n\nAdvice Categories:")
        for cat, count in df['advice_category'].value_counts().items():
            summary.append(f"  {cat}: {count}")
        
        summary.append(f"\n\nAction Categories:")
        for cat, count in df['action_category'].value_counts().items():
            summary.append(f"  {cat}: {count}")
        
        summary.append(f"\n\nAction Alignment:")
        summary.append(f"  Actions matching suggestions: {df['action_in_suggestions'].sum()} ({df['action_in_suggestions'].mean()*100:.1f}%)")
        
        summary.append(f"\n\nCritical HP Episodes: {df['critical_hp'].sum()}")
        summary.append(f"Level Ups: {df['level_up'].sum()}")
        
        return "\n".join(summary)


# Main execution
if __name__ == "__main__":
    # Example usage
    parser = NetHackLogParser('tests/example_run_v2.log')
    
    print("Parsing logs...")
    df = parser.parse_logs()
    
    print(f"\nParsed {len(df)} steps")
    print(f"Columns: {list(df.columns)}")
    
    # Save processed data
    parser.save_processed_data('processed_data', df)
    
    # Display sample
    print("\n" + "="*60)
    print("Sample of processed data:")
    print("="*60)
    print(df[['step', 'action', 'reward', 'hp', 'llm_advice', 
              'action_category', 'advice_category', 'action_in_suggestions']].head(10))
    
    print("\n" + "="*60)
    print("Causal Analysis Ready Variables:")
    print("="*60)
    print("\nTreatment: action_in_suggestions (binary)")
    print("Outcome: reward, hp_change, sr_change, level_up")
    print("Confounders: hp_lag1, reward_lag1, sr_lag1, critical_hp, episode_stage")
    print("Mediators: action_category, steps_since_advice_change")