import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class NetHackCausalLogParser:
    """
    Parse NetHack LLM+RL agent logs for causal effect estimation.
    
    Key assumption: Agent ALWAYS follows LLM advice by picking from suggested actions.
    Goal: Extract data to measure causal effect of LLM advice quality on outcomes.
    """
    
    def __init__(self, log_file: str, outcome_window: int = 10):
        self.log_file = log_file
        self.data = []
        self.outcome_window = outcome_window
        
    def parse_logs(self) -> pd.DataFrame:
        """Main parsing function extracting causal variables."""
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        current_llm_data = None
        current_advice_text = None
        advice_start_step = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Parse debug LLM response FIRST (comes before advice text)
            if line.startswith("[DEBUG] Raw LLM response"):
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
            
            # Parse LLM advice line (comes after debug response)
            elif line.startswith("LLM Advice:"):
                current_advice_text = line.replace("LLM Advice:", "").strip()
                advice_start_step = None  # Will be set at next step
                
            # Parse step action line
            elif line.startswith("Step"):
                step_data = self._parse_step_line(line)
                if step_data:
                    # Mark when advice starts (first step after new advice)
                    if current_advice_text and advice_start_step is None:
                        advice_start_step = step_data['step']
                    
                    step_data['llm_advice_text'] = current_advice_text
                    step_data['llm_data'] = current_llm_data
                    step_data['advice_start_step'] = advice_start_step
                    
                    if advice_start_step is not None:
                        step_data['steps_since_advice'] = step_data['step'] - advice_start_step
                    else:
                        step_data['steps_since_advice'] = None
                    
                    self.data.append(step_data)
                    
            i += 1
        
        df = pd.DataFrame(self.data)
        
        if df.empty:
            return df
        
        print(f"\n✓ Parsed {len(df)} steps")
        print(f"✓ Found {df['llm_advice_text'].notna().sum()} steps with advice")
        
        # Enrich with causal variables
        df = self._enrich_for_causal_analysis(df)
        
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
    
    def _enrich_for_causal_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add causal variables for effect estimation.
        
        Treatment: Quality/characteristics of LLM advice
        Outcome: Performance in next N steps
        Confounders: Game state when advice given
        """
        if df.empty:
            return df
        
        df = df.sort_values('step').reset_index(drop=True)
        
        print("\n" + "="*80)
        print("EXTRACTING CAUSAL VARIABLES")
        print("="*80)
        
        # =====================================================================
        # 1. CREATE ADVICE EPISODES
        # =====================================================================
        df['advice_text'] = df['llm_advice_text'].fillna('')
        advice_changes = (df['advice_text'] != df['advice_text'].shift(1))
        df['advice_episode'] = advice_changes.cumsum()
        
        print(f"\n✓ Created {df['advice_episode'].nunique()} advice episodes")
        
        # =====================================================================
        # 2. EXTRACT LLM ADVICE FEATURES (Treatment Variables)
        # =====================================================================
        print("\n📋 Extracting treatment variables (LLM advice features)...")
        
        # Basic structured fields
        df['llm_suggestions'] = df['llm_data'].apply(
            lambda x: x.get('action_suggestions', []) if isinstance(x, dict) else []
        )
        df['llm_priority'] = df['llm_data'].apply(
            lambda x: x.get('immediate_priority', '') if isinstance(x, dict) else ''
        )
        df['llm_risk'] = df['llm_data'].apply(
            lambda x: x.get('risk_assessment', '') if isinstance(x, dict) else ''
        )
        df['llm_opportunities'] = df['llm_data'].apply(
            lambda x: x.get('opportunities', '') if isinstance(x, dict) else ''
        )
        df['llm_strategy'] = df['llm_data'].apply(
            lambda x: x.get('strategy', '') if isinstance(x, dict) else ''
        )
        
        # Treatment features
        df['num_suggestions'] = df['llm_suggestions'].apply(len)
        df['has_json_advice'] = df['llm_data'].notna().astype(int)
        
        # Categorize advice type
        df['advice_category'] = df['llm_advice_text'].apply(self._categorize_advice)
        df['advice_specificity'] = df['llm_priority'].apply(self._measure_specificity)
        df['advice_urgency'] = df['llm_risk'].apply(self._measure_urgency)
        
        # Extract mentioned actions
        df['combat_mentioned'] = df['llm_advice_text'].apply(
            lambda x: 1 if any(w in str(x).lower() for w in ['attack', 'fight', 'combat', 'kill', 'kick']) else 0
        )
        df['survival_mentioned'] = df['llm_advice_text'].apply(
            lambda x: 1 if any(w in str(x).lower() for w in ['eat', 'heal', 'health', 'food', 'rest']) else 0
        )
        df['exploration_mentioned'] = df['llm_advice_text'].apply(
            lambda x: 1 if any(w in str(x).lower() for w in ['search', 'explore', 'move', 'find']) else 0
        )
        
        print(f"  - Advice categories: {df['advice_category'].value_counts().to_dict()}")
        print(f"  - Avg suggestions per advice: {df['num_suggestions'].mean():.1f}")
        print(f"  - JSON parsed: {df['has_json_advice'].sum()} / {len(df)}")
        
        # =====================================================================
        # 3. EXTRACT CONFOUNDERS (Pre-treatment game state)
        # =====================================================================
        print("\n🎮 Extracting confounders (game state at advice time)...")
        
        # Game state at step level
        df['hp_level'] = pd.cut(df['hp'], bins=[0, 50, 100, 150, 300], labels=['critical', 'low', 'medium', 'high'])
        df['critical_hp'] = (df['hp'] < 50).astype(int)
        df['low_hp'] = (df['hp'] < 100).astype(int)
        df['high_hp'] = (df['hp'] >= 150).astype(int)
        
        # Recent performance (confounders)
        df['reward_ma5'] = df['reward'].rolling(window=5, min_periods=1).mean()
        df['hp_ma5'] = df['hp'].rolling(window=5, min_periods=1).mean()
        df['reward_trend'] = df['reward'].rolling(window=5, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
        )
        
        # Changes
        df['hp_change'] = df['hp'].diff()
        df['reward_change'] = df['reward'].diff()
        df['level_up'] = (df['level'].diff() > 0).astype(int)
        
        # =====================================================================
        # 4. COMPUTE OUTCOMES (Post-treatment performance)
        # =====================================================================
        print(f"\n📊 Computing outcomes (performance in next {self.outcome_window} steps)...")
        
        # For each advice episode, compute outcomes
        episode_outcomes = []
        
        for episode_id in df['advice_episode'].unique():
            if pd.isna(episode_id):
                continue
            
            ep_df = df[df['advice_episode'] == episode_id].copy()
            
            if len(ep_df) == 0:
                continue
            
            # Take first step as baseline (when advice given)
            baseline = ep_df.iloc[0]
            
            # Take window for outcome measurement
            outcome_window_df = ep_df.head(min(len(ep_df), self.outcome_window))
            
            # Compute outcomes
            outcomes = {
                'advice_episode': episode_id,
                'start_step': baseline['step'],
                'episode_length': len(ep_df),
                
                # Treatment variables
                'llm_advice_text': baseline['llm_advice_text'],
                'advice_category': baseline['advice_category'],
                'advice_specificity': baseline['advice_specificity'],
                'advice_urgency': baseline['advice_urgency'],
                'num_suggestions': baseline['num_suggestions'],
                'has_json_advice': baseline['has_json_advice'],
                'combat_mentioned': baseline['combat_mentioned'],
                'survival_mentioned': baseline['survival_mentioned'],
                'exploration_mentioned': baseline['exploration_mentioned'],
                
                # Confounders (pre-treatment state)
                'hp_before': baseline['hp'],
                'level_before': baseline['level'],
                'reward_before': baseline['reward'],
                'sr_before': baseline['smoothed_reward'],
                'critical_hp_before': baseline['critical_hp'],
                'low_hp_before': baseline['low_hp'],
                'reward_ma5_before': baseline['reward_ma5'],
                'hp_ma5_before': baseline['hp_ma5'],
                'reward_trend_before': baseline['reward_trend'],
                
                # Outcomes (performance in window)
                'total_reward': outcome_window_df['reward'].sum(),
                'avg_reward': outcome_window_df['reward'].mean(),
                'max_reward': outcome_window_df['reward'].max(),
                'min_reward': outcome_window_df['reward'].min(),
                'reward_std': outcome_window_df['reward'].std(),
                
                'sr_change': outcome_window_df['smoothed_reward'].iloc[-1] - baseline['smoothed_reward'],
                'hp_change': outcome_window_df['hp'].iloc[-1] - baseline['hp'],
                'level_gained': outcome_window_df['level'].iloc[-1] - baseline['level'],
                
                'positive_reward_rate': (outcome_window_df['reward'] > 0).mean(),
                'negative_reward_rate': (outcome_window_df['reward'] < 0).mean(),
                'survived': int(outcome_window_df['hp'].iloc[-1] > 0),
                
                # Action diversity (did agent explore different actions?)
                'unique_actions': outcome_window_df['action'].nunique(),
                'action_entropy': self._compute_action_entropy(outcome_window_df['action']),
            }
            
            episode_outcomes.append(outcomes)
        
        episode_df = pd.DataFrame(episode_outcomes)
        
        print(f"  ✓ Created {len(episode_df)} episode-level observations")
        print(f"  ✓ Avg total reward: {episode_df['total_reward'].mean():.3f}")
        print(f"  ✓ Avg SR change: {episode_df['sr_change'].mean():.3f}")
        
        # =====================================================================
        # 5. PROPENSITY SCORE FEATURES
        # =====================================================================
        print("\n🎯 Creating propensity score features...")
        
        # Binary treatment: high-quality advice vs low-quality
        episode_df['high_quality_advice'] = (
            (episode_df['num_suggestions'] >= 3) & 
            (episode_df['has_json_advice'] == 1) &
            (episode_df['advice_specificity'] >= 2)
        ).astype(int)
        
        # Alternative: specific vs generic advice
        episode_df['specific_advice'] = (episode_df['advice_specificity'] >= 2).astype(int)
        
        # Alternative: combat-focused vs exploration-focused
        episode_df['combat_advice'] = (episode_df['combat_mentioned'] == 1).astype(int)
        
        print(f"  ✓ High quality advice: {episode_df['high_quality_advice'].sum()} / {len(episode_df)} ({episode_df['high_quality_advice'].mean()*100:.1f}%)")
        print(f"  ✓ Specific advice: {episode_df['specific_advice'].sum()} / {len(episode_df)} ({episode_df['specific_advice'].mean()*100:.1f}%)")
        print(f"  ✓ Combat advice: {episode_df['combat_advice'].sum()} / {len(episode_df)} ({episode_df['combat_advice'].mean()*100:.1f}%)")
        
        return episode_df
    
    def _categorize_advice(self, advice: str) -> str:
        """Categorize LLM advice into types."""
        if not advice or pd.isna(advice):
            return 'none'
        
        advice_lower = str(advice).lower()
        
        # Priority order matters
        if any(word in advice_lower for word in ['flee', 'escape', 'run', 'avoid', 'danger', 'retreat']):
            return 'defensive'
        elif any(word in advice_lower for word in ['eat', 'health', 'restore', 'heal', 'food', 'hunger']):
            return 'survival'
        elif any(word in advice_lower for word in ['attack', 'fight', 'combat', 'kill', 'enemy', 'monster', 'kick']):
            return 'combat'
        elif any(word in advice_lower for word in ['search', 'explore', 'look', 'find', 'area', 'move']):
            return 'exploration'
        elif any(word in advice_lower for word in ['equip', 'wear', 'wield', 'take off', 'armor', 'weapon']):
            return 'equipment'
        elif any(word in advice_lower for word in ['read', 'drink', 'use', 'apply', 'potion', 'scroll', 'pickup']):
            return 'item_use'
        else:
            return 'other'
    
    def _measure_specificity(self, priority_text: str) -> int:
        """
        Measure advice specificity (0=generic, 3=highly specific).
        
        Examples:
        - "clear single goal" -> 1 (vague)
        - "kill kobold north" -> 3 (specific action + target + direction)
        """
        if not priority_text or pd.isna(priority_text):
            return 0
        
        text = str(priority_text).lower()
        score = 0
        
        # Has specific action verb?
        action_verbs = ['kill', 'attack', 'pickup', 'eat', 'drink', 'move', 'search', 'open', 'kick']
        if any(verb in text for verb in action_verbs):
            score += 1
        
        # Has specific target?
        if any(word in text for word in ['kobold', 'gold', 'door', 'scroll', 'potion', 'monster', 'enemy']):
            score += 1
        
        # Has direction or location?
        if any(word in text for word in ['north', 'south', 'east', 'west', 'close', 'nearby', 'adjacent']):
            score += 1
        
        return score
    
    def _measure_urgency(self, risk_text: str) -> int:
        """Measure advice urgency (0=low, 2=high)."""
        if not risk_text or pd.isna(risk_text):
            return 0
        
        text = str(risk_text).lower()
        
        if any(word in text for word in ['high', 'critical', 'danger', 'urgent', 'immediate']):
            return 2
        elif any(word in text for word in ['medium', 'moderate', 'caution']):
            return 1
        else:
            return 0
    
    def _compute_action_entropy(self, actions: pd.Series) -> float:
        """Compute Shannon entropy of action distribution."""
        if len(actions) == 0:
            return 0.0
        
        counts = actions.value_counts()
        probs = counts / len(actions)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    def save_processed_data(self, output_path: str, episode_df: pd.DataFrame):
        """Save processed data for causal analysis."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save episode-level data (main dataset for causal analysis)
        episode_df.to_csv(output_path / 'causal_episodes.csv', index=False)
        episode_df.to_parquet(output_path / 'causal_episodes.parquet', index=False)
        
        # Generate causal analysis report
        report = self._generate_causal_report(episode_df)
        with open(output_path / 'causal_report.txt', 'w') as f:
            f.write(report)
        
        # Save metadata
        metadata = {
            'total_episodes': len(episode_df),
            'outcome_window': self.outcome_window,
            'treatment_variable_options': [
                'high_quality_advice',
                'specific_advice', 
                'combat_advice',
                'advice_category'
            ],
            'outcome_variables': [
                'total_reward',
                'sr_change',
                'hp_change',
                'positive_reward_rate'
            ],
            'confounder_variables': [
                'hp_before',
                'level_before',
                'reward_ma5_before',
                'critical_hp_before'
            ],
            'treatment_balance': {
                'high_quality': {
                    'treated': int(episode_df['high_quality_advice'].sum()),
                    'control': int((episode_df['high_quality_advice'] == 0).sum())
                },
                'specific': {
                    'treated': int(episode_df['specific_advice'].sum()),
                    'control': int((episode_df['specific_advice'] == 0).sum())
                },
                'combat': {
                    'treated': int(episode_df['combat_advice'].sum()),
                    'control': int((episode_df['combat_advice'] == 0).sum())
                }
            }
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*80}")
        print("DATA SAVED")
        print('='*80)
        print(f"📁 Output directory: {output_path}")
        print(f"📊 Episodes: {len(episode_df)}")
        print(f"📈 Ready for causal analysis!")
        print(f"\n💡 Suggested treatment variables:")
        print(f"   - high_quality_advice: {episode_df['high_quality_advice'].sum()} treated")
        print(f"   - specific_advice: {episode_df['specific_advice'].sum()} treated")
        print(f"   - combat_advice: {episode_df['combat_advice'].sum()} treated")
    
    def _generate_causal_report(self, df: pd.DataFrame) -> str:
        """Generate report for causal analysis setup."""
        report = []
        report.append("=" * 80)
        report.append("CAUSAL ANALYSIS DATASET REPORT")
        report.append("=" * 80)
        
        report.append(f"\n📊 Dataset Overview")
        report.append(f"   Total episodes: {len(df)}")
        report.append(f"   Outcome window: {self.outcome_window} steps")
        
        report.append(f"\n🎯 Treatment Variables (LLM Advice Features)")
        report.append(f"   Advice categories: {df['advice_category'].value_counts().to_dict()}")
        report.append(f"   Avg suggestions: {df['num_suggestions'].mean():.1f} (std: {df['num_suggestions'].std():.1f})")
        report.append(f"   JSON parsed: {df['has_json_advice'].sum()} ({df['has_json_advice'].mean()*100:.1f}%)")
        
        report.append(f"\n   Binary treatments:")
        for treatment in ['high_quality_advice', 'specific_advice', 'combat_advice']:
            n_treated = df[treatment].sum()
            pct = n_treated / len(df) * 100
            report.append(f"   - {treatment}: {n_treated} / {len(df)} ({pct:.1f}%)")
        
        report.append(f"\n📈 Outcome Variables (Performance)")
        outcome_vars = ['total_reward', 'sr_change', 'hp_change', 'positive_reward_rate']
        for var in outcome_vars:
            report.append(f"   {var}:")
            report.append(f"      Mean: {df[var].mean():.3f}, Std: {df[var].std():.3f}")
            report.append(f"      Range: [{df[var].min():.3f}, {df[var].max():.3f}]")
        
        report.append(f"\n🎮 Confounders (Pre-treatment State)")
        confounder_vars = ['hp_before', 'level_before', 'reward_ma5_before', 'critical_hp_before']
        for var in confounder_vars:
            if var in df.columns:
                report.append(f"   {var}:")
                report.append(f"      Mean: {df[var].mean():.3f}, Std: {df[var].std():.3f}")
        
        report.append(f"\n⚖️  Treatment Balance Check")
        for treatment in ['high_quality_advice', 'specific_advice', 'combat_advice']:
            treated = df[df[treatment] == 1]
            control = df[df[treatment] == 0]
            
            report.append(f"\n   {treatment}:")
            report.append(f"      N (treated): {len(treated)}, N (control): {len(control)}")
            
            if len(treated) > 0 and len(control) > 0:
                report.append(f"      HP before - Treated: {treated['hp_before'].mean():.1f}, Control: {control['hp_before'].mean():.1f}")
                report.append(f"      Avg reward - Treated: {treated['total_reward'].mean():.3f}, Control: {control['total_reward'].mean():.3f}")
        
        report.append(f"\n💡 Next Steps for Causal Analysis:")
        report.append(f"   1. Check covariate balance between treatment groups")
        report.append(f"   2. Estimate propensity scores using confounders")
        report.append(f"   3. Use matching, IPW, or doubly robust estimation")
        report.append(f"   4. Estimate ATE/ATT for each treatment variable")
        report.append(f"   5. Conduct sensitivity analysis for unmeasured confounding")
        
        return "\n".join(report)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse NetHack logs for causal analysis')
    parser.add_argument('--log_file', type=str, default='nethack_logs.txt')
    parser.add_argument('--output_dir', type=str, default='causal_data')
    parser.add_argument('--window', type=int, default=10, help='Outcome measurement window')
    
    args = parser.parse_args()
    
    parser = NetHackCausalLogParser(args.log_file, outcome_window=args.window)
    
    print(f"Parsing logs for causal analysis (outcome window={args.window})...")
    episode_df = parser.parse_logs()
    
    if len(episode_df) > 0:
        parser.save_processed_data(args.output_dir, episode_df)
        print(f"\n✅ Ready for causal effect estimation!")
        print(f"   Use 'causal_episodes.csv' with your preferred causal inference method")
    else:
        print("❌ ERROR: No data parsed. Check log file format.")