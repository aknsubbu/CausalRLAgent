import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class DoublyRobustEstimator:
    """
    Doubly Robust Estimator for Causal Effect of LLM Advice
    
    Combines:
    1. Propensity score weighting (inverse probability weighting)
    2. Outcome regression (predicted counterfactuals)
    
    More robust than either method alone - if either model is correct, 
    the estimate is consistent.
    """
    
    def __init__(self, steps_file, interventions_file, summary_file=None):
        self.steps_file = Path(steps_file)
        self.interventions_file = Path(interventions_file)
        self.summary_file = Path(summary_file) if summary_file else None
        
        # Models
        self.propensity_model = None
        self.outcome_model_treated = None
        self.outcome_model_control = None
        self.scaler = StandardScaler()
        
        print(f"📊 Doubly Robust Estimator initialized")
        print(f"   Steps file: {self.steps_file}")
        print(f"   Interventions file: {self.interventions_file}")
    
    def load_data(self):
        """Load and prepare data from JSONL files"""
        print("\n📥 Loading data...")
        
        # Load steps
        steps_data = []
        with open(self.steps_file, 'r') as f:
            for line in f:
                steps_data.append(json.loads(line))
        
        self.df = pd.DataFrame(steps_data)
        
        print(f"   Loaded {len(self.df)} steps")
        print(f"   Treated: {self.df['treated'].sum()}")
        print(f"   Control: {(~self.df['treated']).sum()}")
        
        # Load interventions for strategy analysis
        interventions_data = []
        with open(self.interventions_file, 'r') as f:
            for line in f:
                interventions_data.append(json.loads(line))
        
        self.interventions_df = pd.DataFrame(interventions_data)
        print(f"   Loaded {len(self.interventions_df)} interventions")
        
        return self.df
    
    def prepare_features(self):
        """Extract features for causal inference"""
        print("\n🔧 Preparing features...")
        
        # Covariates (confounders)
        self.df['health'] = self.df['health_ratio']
        self.df['level_norm'] = self.df['level'] / 30.0  # Normalize
        self.df['depth_norm'] = self.df['depth'] / 50.0
        self.df['gold_norm'] = np.log1p(self.df['gold'])
        
        # Lag features (what happened before)
        self.df['reward_lag1'] = self.df.groupby('episode')['reward'].shift(1).fillna(0)
        self.df['reward_lag2'] = self.df.groupby('episode')['reward'].shift(2).fillna(0)
        self.df['shaped_reward_lag1'] = self.df.groupby('episode')['shaped_reward'].shift(1).fillna(0)
        
        # Moving averages
        self.df['reward_ma5'] = self.df.groupby('episode')['reward'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        self.df['reward_ma10'] = self.df.groupby('episode')['reward'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
        
        # Episode progress
        episode_lengths = self.df.groupby('episode')['step'].transform('max')
        self.df['episode_progress'] = self.df['step'] / episode_lengths
        
        # Define feature set
        self.feature_cols = [
            'health', 'level_norm', 'depth_norm', 'gold_norm',
            'reward_lag1', 'reward_lag2', 'shaped_reward_lag1',
            'reward_ma5', 'reward_ma10', 'episode_progress',
            'recent_reward_mean', 'recent_reward_std'
        ]
        
        # Fill any remaining NaNs
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)
        
        print(f"   Features: {len(self.feature_cols)}")
        print(f"   Feature list: {self.feature_cols}")
    
    def estimate_propensity_scores(self):
        """
        Estimate propensity scores: P(Treatment=1 | X)
        
        This models the probability of receiving LLM advice given covariates.
        """
        print("\n🎯 Estimating propensity scores...")
        
        X = self.df[self.feature_cols].values
        T = self.df['treated'].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Use Random Forest for flexible propensity model
        self.propensity_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=50,
            random_state=42
        )
        
        self.propensity_model.fit(X_scaled, T)
        
        # Predict propensity scores
        self.df['propensity_score'] = self.propensity_model.predict_proba(X_scaled)[:, 1]
        
        # Clip extreme propensities to avoid instability
        self.df['propensity_score'] = np.clip(self.df['propensity_score'], 0.01, 0.99)
        
        print(f"   Mean propensity score: {self.df['propensity_score'].mean():.4f}")
        print(f"   Propensity score range: [{self.df['propensity_score'].min():.4f}, {self.df['propensity_score'].max():.4f}]")
        
        # Feature importance
        importance = self.propensity_model.feature_importances_
        feature_importance = sorted(zip(self.feature_cols, importance), key=lambda x: x[1], reverse=True)
        print("\n   Top 5 features predicting treatment:")
        for feat, imp in feature_importance[:5]:
            print(f"     {feat}: {imp:.4f}")
    
    def estimate_outcome_models(self):
        """
        Estimate outcome models: E[Y | X, T=1] and E[Y | X, T=0]
        
        These predict what the outcome (shaped_reward) would be under 
        treatment and control.
        """
        print("\n📈 Estimating outcome models...")
        
        X = self.scaler.transform(self.df[self.feature_cols].values)
        Y = self.df['shaped_reward'].values
        T = self.df['treated'].values
        
        # Separate treated and control
        X_treated = X[T == True]
        Y_treated = Y[T == True]
        X_control = X[T == False]
        Y_control = Y[T == False]
        
        print(f"   Training on {len(X_treated)} treated, {len(X_control)} control observations")
        
        # Model for treated group: E[Y | X, T=1]
        self.outcome_model_treated = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            random_state=42
        )
        self.outcome_model_treated.fit(X_treated, Y_treated)
        
        # Model for control group: E[Y | X, T=0]
        if len(X_control) > 50:  # Need enough control observations
            self.outcome_model_control = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=20,
                random_state=42
            )
            self.outcome_model_control.fit(X_control, Y_control)
        else:
            print("   ⚠️ Too few control observations for outcome model, using treated model")
            self.outcome_model_control = self.outcome_model_treated
        
        # Predict potential outcomes for everyone
        self.df['Y1_hat'] = self.outcome_model_treated.predict(X)  # Predicted outcome if treated
        self.df['Y0_hat'] = self.outcome_model_control.predict(X)  # Predicted outcome if not treated
        
        print(f"   Mean predicted Y1 (treated): {self.df['Y1_hat'].mean():.4f}")
        print(f"   Mean predicted Y0 (control): {self.df['Y0_hat'].mean():.4f}")
    
    def compute_doubly_robust_ate(self):
        """
        Compute Doubly Robust Average Treatment Effect (ATE)
        
        Formula:
        ATE = E[ T*(Y - Y1_hat)/e(X) + Y1_hat ] - E[ (1-T)*(Y - Y0_hat)/(1-e(X)) + Y0_hat ]
        
        Where:
        - T = treatment indicator
        - Y = observed outcome
        - Y1_hat, Y0_hat = predicted outcomes
        - e(X) = propensity score
        """
        print("\n🔬 Computing Doubly Robust ATE...")
        
        T = self.df['treated'].values
        Y = self.df['shaped_reward'].values
        Y1_hat = self.df['Y1_hat'].values
        Y0_hat = self.df['Y0_hat'].values
        e = self.df['propensity_score'].values
        
        # Doubly robust estimator for treated potential outcome
        mu1_dr = (T * (Y - Y1_hat) / e) + Y1_hat
        
        # Doubly robust estimator for control potential outcome
        mu0_dr = ((1 - T) * (Y - Y0_hat) / (1 - e)) + Y0_hat
        
        # ATE is the difference
        self.ate_dr = np.mean(mu1_dr - mu0_dr)
        
        # Standard error (conservative)
        tau = mu1_dr - mu0_dr
        self.ate_se = np.std(tau) / np.sqrt(len(tau))
        
        # 95% confidence interval
        self.ci_lower = self.ate_dr - 1.96 * self.ate_se
        self.ci_upper = self.ate_dr + 1.96 * self.ate_se
        
        # Statistical significance
        self.p_value = 2 * (1 - stats.norm.cdf(abs(self.ate_dr / self.ate_se)))
        self.is_significant = self.p_value < 0.05
        
        print(f"\n📊 DOUBLY ROBUST RESULTS:")
        print(f"   ATE: {self.ate_dr:.6f}")
        print(f"   Standard Error: {self.ate_se:.6f}")
        print(f"   95% CI: [{self.ci_lower:.6f}, {self.ci_upper:.6f}]")
        print(f"   P-value: {self.p_value:.6f}")
        print(f"   Significant: {'YES ✓' if self.is_significant else 'NO ✗'}")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        if self.is_significant:
            if self.ate_dr > 0:
                print(f"   ✅ LLM advice IMPROVES performance by {self.ate_dr:.6f} shaped reward per step")
            else:
                print(f"   ❌ LLM advice HURTS performance by {abs(self.ate_dr):.6f} shaped reward per step")
        else:
            print(f"   ⚠️ No statistically significant effect detected")
        
        return {
            'ate': self.ate_dr,
            'se': self.ate_se,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'p_value': self.p_value,
            'significant': self.is_significant
        }
    
    def analyze_heterogeneous_effects(self):
        """
        Analyze treatment effect heterogeneity across MULTIPLE dimensions
        - Does LLM advice help more in certain situations?
        """
        print("\n🔍 Analyzing heterogeneous treatment effects...")
        
        T = self.df['treated'].values
        Y = self.df['shaped_reward'].values
        
        # Compute individual treatment effects (ITE)
        Y1_hat = self.df['Y1_hat'].values
        Y0_hat = self.df['Y0_hat'].values
        self.df['ite'] = Y1_hat - Y0_hat
        
        # 1. BY EPISODE PROGRESS (most important according to propensity model!)
        print("\n📊 Treatment Effect by Episode Progress:")
        progress_bins = pd.cut(self.df['episode_progress'], 
                               bins=[0, 0.25, 0.5, 0.75, 1.0], 
                               labels=['Early', 'Mid-Early', 'Mid-Late', 'Late'],
                               include_lowest=True)
        self.df['progress_bin'] = progress_bins
        
        for progress_cat in ['Early', 'Mid-Early', 'Mid-Late', 'Late']:
            mask = self.df['progress_bin'] == progress_cat
            if mask.sum() > 100:  # Only show if enough data
                ate = self.df[mask]['ite'].mean()
                se = self.df[mask]['ite'].std() / np.sqrt(mask.sum())
                n = mask.sum()
                sig = "✓" if abs(ate/se) > 1.96 else "✗"
                print(f"   {progress_cat:12s}: ATE={ate:+.6f} ±{se:.6f} (n={n:6d}) {sig}")
        
        # 2. BY RECENT PERFORMANCE
        print("\n📊 Treatment Effect by Recent Performance:")
        perf_bins = pd.qcut(self.df['reward_ma10'], q=4, 
                            labels=['Struggling', 'Below-Avg', 'Above-Avg', 'Doing-Well'],
                            duplicates='drop')
        self.df['perf_bin'] = perf_bins
        
        for perf_cat in perf_bins.cat.categories:
            mask = self.df['perf_bin'] == perf_cat
            if mask.sum() > 100:
                ate = self.df[mask]['ite'].mean()
                se = self.df[mask]['ite'].std() / np.sqrt(mask.sum())
                n = mask.sum()
                sig = "✓" if abs(ate/se) > 1.96 else "✗"
                print(f"   {perf_cat:12s}: ATE={ate:+.6f} ±{se:.6f} (n={n:6d}) {sig}")
        
        # 3. BY DUNGEON DEPTH
        print("\n📊 Treatment Effect by Dungeon Depth:")
        try:
            depth_bins = pd.qcut(self.df['depth'], q=3, 
                                labels=['Shallow', 'Mid', 'Deep'],
                                duplicates='drop')
            self.df['depth_bin'] = depth_bins
            
            for depth_cat in depth_bins.cat.categories:
                mask = self.df['depth_bin'] == depth_cat
                if mask.sum() > 100:
                    ate = self.df[mask]['ite'].mean()
                    se = self.df[mask]['ite'].std() / np.sqrt(mask.sum())
                    n = mask.sum()
                    sig = "✓" if abs(ate/se) > 1.96 else "✗"
                    print(f"   {depth_cat:12s}: ATE={ate:+.6f} ±{se:.6f} (n={n:6d}) {sig}")
        except:
            print("   ⚠️ Depth data too uniform for binning")
        
        # 4. BY REWARD VOLATILITY (risk/uncertainty)
        print("\n📊 Treatment Effect by Reward Volatility:")
        vol_bins = pd.qcut(self.df['recent_reward_std'], q=3,
                           labels=['Stable', 'Moderate', 'Volatile'],
                           duplicates='drop')
        self.df['vol_bin'] = vol_bins
        
        for vol_cat in vol_bins.cat.categories:
            mask = self.df['vol_bin'] == vol_cat
            if mask.sum() > 100:
                ate = self.df[mask]['ite'].mean()
                se = self.df[mask]['ite'].std() / np.sqrt(mask.sum())
                n = mask.sum()
                sig = "✓" if abs(ate/se) > 1.96 else "✗"
                print(f"   {vol_cat:12s}: ATE={ate:+.6f} ±{se:.6f} (n={n:6d}) {sig}")
        
        # 5. BY CHARACTER LEVEL
        print("\n📊 Treatment Effect by Character Level:")
        try:
            level_bins = pd.cut(self.df['level'], 
                               bins=[0, 1, 2, 3, 30],
                               labels=['Level-1', 'Level-2', 'Level-3', 'Level-4+'],
                               include_lowest=True)
            self.df['level_bin'] = level_bins
            
            for level_cat in ['Level-1', 'Level-2', 'Level-3', 'Level-4+']:
                mask = self.df['level_bin'] == level_cat
                if mask.sum() > 100:
                    ate = self.df[mask]['ite'].mean()
                    se = self.df[mask]['ite'].std() / np.sqrt(mask.sum())
                    n = mask.sum()
                    sig = "✓" if abs(ate/se) > 1.96 else "✗"
                    print(f"   {level_cat:12s}: ATE={ate:+.6f} ±{se:.6f} (n={n:6d}) {sig}")
        except:
            print("   ⚠️ Level data too uniform for binning")
        
        # 6. BY GOLD COLLECTED (proxy for success)
        print("\n📊 Treatment Effect by Gold Collected:")
        gold_bins = pd.cut(self.df['gold'], 
                          bins=[-1, 0, 10, 50, 10000],
                          labels=['None', 'Little', 'Some', 'Lots'],
                          include_lowest=True)
        self.df['gold_bin'] = gold_bins
        
        for gold_cat in ['None', 'Little', 'Some', 'Lots']:
            mask = self.df['gold_bin'] == gold_cat
            if mask.sum() > 100:
                ate = self.df[mask]['ite'].mean()
                se = self.df[mask]['ite'].std() / np.sqrt(mask.sum())
                n = mask.sum()
                sig = "✓" if abs(ate/se) > 1.96 else "✗"
                print(f"   {gold_cat:12s}: ATE={ate:+.6f} ±{se:.6f} (n={n:6d}) {sig}")
        
        # 7. INTERACTION: Progress × Performance
        print("\n📊 Treatment Effect by Progress × Performance Interaction:")
        for progress_cat in ['Early', 'Late']:
            for perf_cat in ['Struggling', 'Doing-Well']:
                mask = (self.df['progress_bin'] == progress_cat) & \
                       (self.df['perf_bin'] == perf_cat)
                if mask.sum() > 50:
                    ate = self.df[mask]['ite'].mean()
                    se = self.df[mask]['ite'].std() / np.sqrt(mask.sum())
                    n = mask.sum()
                    sig = "✓" if abs(ate/se) > 1.96 else "✗"
                    print(f"   {progress_cat} + {perf_cat:12s}: ATE={ate:+.6f} ±{se:.6f} (n={n:5d}) {sig}")
    
    def analyze_strategy_effectiveness(self):
        """Analyze which LLM strategies are most effective"""
        print("\n🎮 Analyzing Strategy Effectiveness...")
        
        # Extract strategy from interventions
        strategy_map = {}
        for _, intervention in self.interventions_df.iterrows():
            strategy_map[intervention['global_step']] = intervention['treatment']['parsed_strategy']
        
        # Map strategies to steps
        self.df['strategy'] = self.df.apply(
            lambda row: strategy_map.get(row['global_step'], 'none') if row['treated'] else 'none',
            axis=1
        )
        
        # Compute average outcomes by strategy
        strategy_outcomes = self.df[self.df['treated']].groupby('strategy').agg({
            'shaped_reward': ['mean', 'std', 'count'],
            'reward': ['mean', 'std']
        })
        
        print("\n📊 Average Outcomes by Strategy:")
        print(strategy_outcomes)
        
        # Statistical tests between strategies
        strategies = self.df[self.df['treated']]['strategy'].unique()
        strategies = [s for s in strategies if s != 'none']
        
        if len(strategies) > 1:
            print("\n🔬 Strategy Comparisons (t-tests):")
            from itertools import combinations
            for s1, s2 in combinations(strategies[:5], 2):  # Top 5 strategies
                outcomes_s1 = self.df[(self.df['strategy'] == s1)]['shaped_reward'].values
                outcomes_s2 = self.df[(self.df['strategy'] == s2)]['shaped_reward'].values
                
                if len(outcomes_s1) > 30 and len(outcomes_s2) > 30:
                    t_stat, p_val = stats.ttest_ind(outcomes_s1, outcomes_s2)
                    sig = "✓" if p_val < 0.05 else "✗"
                    print(f"   {s1:12s} vs {s2:12s}: t={t_stat:+.3f}, p={p_val:.4f} {sig}")
    
    def visualize_results(self, output_dir='causal_analysis'):
        """Create comprehensive visualizations"""
        print(f"\n📊 Generating visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # FIGURE 1: Core doubly robust analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Propensity scores
        ax = axes[0, 0]
        treated_scores = self.df[self.df['treated']]['propensity_score']
        control_scores = self.df[~self.df['treated']]['propensity_score']
        
        ax.hist(treated_scores, bins=50, alpha=0.6, label='Treated', density=True, color='blue')
        ax.hist(control_scores, bins=50, alpha=0.6, label='Control', density=True, color='orange')
        ax.set_xlabel('Propensity Score')
        ax.set_ylabel('Density')
        ax.set_title('Propensity Score Distribution\n(Overlap Check)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Individual Treatment Effects
        ax = axes[0, 1]
        ax.hist(self.df['ite'], bins=100, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(self.ate_dr, color='red', linestyle='--', linewidth=2, label=f'ATE={self.ate_dr:.4f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Individual Treatment Effect')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Individual Treatment Effects')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Treatment Effect by Progress
        ax = axes[1, 0]
        if 'progress_bin' in self.df.columns:
            progress_effects = self.df.groupby('progress_bin')['ite'].agg(['mean', 'std', 'count'])
            progress_effects['se'] = progress_effects['std'] / np.sqrt(progress_effects['count'])
            
            ax.bar(range(len(progress_effects)), progress_effects['mean'],
                   yerr=progress_effects['se'], capsize=5,
                   color='steelblue', edgecolor='black', alpha=0.7)
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_xticks(range(len(progress_effects)))
            ax.set_xticklabels(progress_effects.index, rotation=45)
            ax.set_ylabel('Average Treatment Effect')
            ax.set_title('Treatment Effect by Episode Progress')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Treatment Effect by Performance
        ax = axes[1, 1]
        if 'perf_bin' in self.df.columns:
            perf_effects = self.df.groupby('perf_bin')['ite'].agg(['mean', 'std', 'count'])
            perf_effects['se'] = perf_effects['std'] / np.sqrt(perf_effects['count'])
            
            ax.bar(range(len(perf_effects)), perf_effects['mean'],
                   yerr=perf_effects['se'], capsize=5,
                   color='coral', edgecolor='black', alpha=0.7)
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_xticks(range(len(perf_effects)))
            ax.set_xticklabels(perf_effects.index, rotation=45)
            ax.set_ylabel('Average Treatment Effect')
            ax.set_title('Treatment Effect by Recent Performance')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'doubly_robust_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_dir / 'doubly_robust_analysis.png'}")
        plt.close()
        
        # FIGURE 2: Multi-dimensional heterogeneity
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        dimensions = [
            ('progress_bin', 'Episode Progress', axes[0, 0]),
            ('perf_bin', 'Recent Performance', axes[0, 1]),
            ('vol_bin', 'Reward Volatility', axes[0, 2]),
            ('depth_bin', 'Dungeon Depth', axes[1, 0]),
            ('gold_bin', 'Gold Collected', axes[1, 1]),
            ('level_bin', 'Character Level', axes[1, 2])
        ]
        
        for col_name, title, ax in dimensions:
            if col_name in self.df.columns:
                effects = self.df.groupby(col_name)['ite'].agg(['mean', 'std', 'count'])
                effects = effects[effects['count'] > 100]  # Only show if enough data
                
                if len(effects) > 0:
                    effects['se'] = effects['std'] / np.sqrt(effects['count'])
                    
                    bars = ax.bar(range(len(effects)), effects['mean'],
                                 yerr=effects['se'], capsize=5,
                                 color='mediumseagreen', edgecolor='black', alpha=0.7)
                    
                    # Color code bars (red for negative, green for positive)
                    for i, bar in enumerate(bars):
                        if effects.iloc[i]['mean'] < 0:
                            bar.set_color('indianred')
                    
                    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
                    ax.set_xticks(range(len(effects)))
                    ax.set_xticklabels(effects.index, rotation=45, ha='right')
                    ax.set_ylabel('ATE')
                    ax.set_title(f'Effect by {title}')
                    ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, f'{title}\nData not available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Effect by {title}')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'heterogeneous_effects.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_dir / 'heterogeneous_effects.png'}")
        plt.close()
        
        # FIGURE 3: Strategy-specific analysis (if available)
        if 'strategy' in self.df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Strategy effectiveness
            ax = axes[0]
            strategy_data = self.df[self.df['treated']].groupby('strategy')['shaped_reward'].agg(['mean', 'std', 'count'])
            strategy_data = strategy_data[strategy_data['count'] > 50]
            strategy_data = strategy_data.sort_values('mean', ascending=False)
            strategy_data['se'] = strategy_data['std'] / np.sqrt(strategy_data['count'])
            
            ax.bar(range(len(strategy_data)), strategy_data['mean'], 
                   yerr=strategy_data['se'], capsize=5,
                   color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xticks(range(len(strategy_data)))
            ax.set_xticklabels(strategy_data.index, rotation=45, ha='right')
            ax.set_ylabel('Average Shaped Reward')
            ax.set_title('Performance by LLM Strategy')
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Strategy distribution
            ax = axes[1]
            strategy_counts = self.df[self.df['treated']]['strategy'].value_counts()
            strategy_counts = strategy_counts[strategy_counts > 50]
            
            ax.bar(range(len(strategy_counts)), strategy_counts.values,
                   color='coral', edgecolor='black', alpha=0.7)
            ax.set_xticks(range(len(strategy_counts)))
            ax.set_xticklabels(strategy_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Frequency')
            ax.set_title('Strategy Usage Distribution')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'strategy_effectiveness.png', dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: {output_dir / 'strategy_effectiveness.png'}")
            plt.close()
    
    def run_full_analysis(self):
        """Run complete doubly robust analysis pipeline"""
        print("="*80)
        print("🔬 DOUBLY ROBUST CAUSAL ANALYSIS")
        print("="*80)
        
        # Load and prepare data
        self.load_data()
        self.prepare_features()
        
        # Estimate models
        self.estimate_propensity_scores()
        self.estimate_outcome_models()
        
        # Compute treatment effects
        results = self.compute_doubly_robust_ate()
        
        # Additional analyses
        self.analyze_heterogeneous_effects()
        self.analyze_strategy_effectiveness()
        
        # Visualize
        self.visualize_results()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)
        
        return results


# ========================================
# USAGE
# ========================================

if __name__ == "__main__":
    # Update with your actual file paths
    estimator = DoublyRobustEstimator(
        steps_file="causal_logs_v2/steps_20251101_222630.jsonl",
        interventions_file="causal_logs_v2/interventions_20251101_222630.jsonl",
        summary_file="causal_logs_v2/summary_20251101_222630.json"
    )
    
    results = estimator.run_full_analysis()
    
    print(f"\n📋 FINAL VERDICT:")
    print(f"   Treatment Effect: {results['ate']:.6f}")
    print(f"   95% CI: [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
    print(f"   Significant: {results['significant']}")