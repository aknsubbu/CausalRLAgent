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
        Analyze treatment effect heterogeneity
        - Does LLM advice help more in certain situations?
        """
        print("\n🔍 Analyzing heterogeneous treatment effects...")
        
        T = self.df['treated'].values
        Y = self.df['shaped_reward'].values
        
        # Compute individual treatment effects (ITE)
        Y1_hat = self.df['Y1_hat'].values
        Y0_hat = self.df['Y0_hat'].values
        self.df['ite'] = Y1_hat - Y0_hat
        
        # Analyze by health status
        health_bins = pd.qcut(self.df['health'], q=4, labels=['Critical', 'Low', 'Moderate', 'Good'])
        self.df['health_bin'] = health_bins
        
        print("\n📊 Treatment Effect by Health Status:")
        for health_cat in ['Critical', 'Low', 'Moderate', 'Good']:
            mask = self.df['health_bin'] == health_cat
            ate_health = self.df[mask]['ite'].mean()
            n = mask.sum()
            print(f"   {health_cat:12s}: ATE={ate_health:+.6f} (n={n:6d})")
        
        # Analyze by episode progress
        progress_bins = pd.cut(self.df['episode_progress'], bins=[0, 0.25, 0.5, 0.75, 1.0], 
                               labels=['Early', 'Mid-Early', 'Mid-Late', 'Late'])
        self.df['progress_bin'] = progress_bins
        
        print("\n📊 Treatment Effect by Episode Progress:")
        for progress_cat in ['Early', 'Mid-Early', 'Mid-Late', 'Late']:
            mask = self.df['progress_bin'] == progress_cat
            if mask.sum() > 0:
                ate_progress = self.df[mask]['ite'].mean()
                n = mask.sum()
                print(f"   {progress_cat:12s}: ATE={ate_progress:+.6f} (n={n:6d})")
    
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
        
        # 1. Propensity Score Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Propensity scores
        ax = axes[0, 0]
        treated_scores = self.df[self.df['treated']]['propensity_score']
        control_scores = self.df[~self.df['treated']]['propensity_score']
        
        ax.hist(treated_scores, bins=50, alpha=0.6, label='Treated', density=True)
        ax.hist(control_scores, bins=50, alpha=0.6, label='Control', density=True)
        ax.set_xlabel('Propensity Score')
        ax.set_ylabel('Density')
        ax.set_title('Propensity Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Individual Treatment Effects
        ax = axes[0, 1]
        ax.hist(self.df['ite'], bins=100, alpha=0.7, edgecolor='black')
        ax.axvline(self.ate_dr, color='red', linestyle='--', linewidth=2, label=f'ATE={self.ate_dr:.4f}')
        ax.set_xlabel('Individual Treatment Effect')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Individual Treatment Effects')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Treatment Effect by Health
        ax = axes[1, 0]
        health_effects = self.df.groupby('health_bin')['ite'].mean()
        health_effects.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Health Status')
        ax.set_ylabel('Average Treatment Effect')
        ax.set_title('Treatment Effect by Health Status')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Outcomes comparison
        ax = axes[1, 1]
        outcomes_data = {
            'Treated (Observed)': self.df[self.df['treated']]['shaped_reward'].mean(),
            'Control (Observed)': self.df[~self.df['treated']]['shaped_reward'].mean(),
            'Treated (Predicted)': self.df['Y1_hat'].mean(),
            'Control (Predicted)': self.df['Y0_hat'].mean(),
        }
        
        bars = ax.bar(range(len(outcomes_data)), list(outcomes_data.values()), 
                      color=['blue', 'orange', 'lightblue', 'lightsalmon'],
                      edgecolor='black')
        ax.set_xticks(range(len(outcomes_data)))
        ax.set_xticklabels(list(outcomes_data.keys()), rotation=45, ha='right')
        ax.set_ylabel('Average Shaped Reward')
        ax.set_title('Observed vs Predicted Outcomes')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'doubly_robust_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_dir / 'doubly_robust_analysis.png'}")
        plt.close()
        
        # 2. Strategy-specific analysis
        if 'strategy' in self.df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            strategy_data = self.df[self.df['treated']].groupby('strategy')['shaped_reward'].agg(['mean', 'std', 'count'])
            strategy_data = strategy_data[strategy_data['count'] > 50]  # Only strategies with enough data
            strategy_data = strategy_data.sort_values('mean', ascending=False)
            
            ax.bar(range(len(strategy_data)), strategy_data['mean'], 
                   yerr=strategy_data['std'] / np.sqrt(strategy_data['count']),
                   capsize=5, color='steelblue', edgecolor='black')
            ax.set_xticks(range(len(strategy_data)))
            ax.set_xticklabels(strategy_data.index, rotation=45, ha='right')
            ax.set_ylabel('Average Shaped Reward')
            ax.set_title('Performance by LLM Strategy (with SE)')
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
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
        steps_file="causal_logs_v2/steps_20251101_222048.jsonl",
        interventions_file="causal_logs_v2/interventions_20251101_222048.jsonl",
        summary_file="causal_logs_v2/summary_20251101_222048.json"
    )
    
    results = estimator.run_full_analysis()
    
    print(f"\n📋 FINAL VERDICT:")
    print(f"   Treatment Effect: {results['ate']:.6f}")
    print(f"   95% CI: [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
    print(f"   Significant: {results['significant']}")