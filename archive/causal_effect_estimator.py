import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


class CausalEffectEstimator:
    """
    Estimate causal effects of LLM advice quality on NetHack performance.
    
    Implements multiple causal inference methods:
    1. Naive comparison (biased baseline)
    2. Regression adjustment
    3. Propensity score matching
    4. Inverse probability weighting (IPW)
    5. Doubly robust estimation
    """
    
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.results = {}
        
        # Define variable sets
        self.treatments = ['high_quality_advice', 'specific_advice', 'combat_advice']
        
        self.outcomes = [
            'total_reward',
            'sr_change', 
            'hp_change',
            'positive_reward_rate'
        ]
        
        self.confounders = [
            'hp_before',
            'level_before', 
            'reward_ma5_before',
            'hp_ma5_before',
            'reward_trend_before',
            'critical_hp_before',
            'low_hp_before'
        ]
        
        # Filter to only confounders that exist
        self.confounders = [c for c in self.confounders if c in self.df.columns]
        
        print(f"Loaded {len(self.df)} episodes")
        print(f"Treatments: {self.treatments}")
        print(f"Outcomes: {self.outcomes}")
        print(f"Confounders: {self.confounders}")
    
    def estimate_all_effects(self, treatment: str, outcome: str) -> Dict:
        """
        Estimate causal effect using all methods.
        
        Returns dict with effect estimates, standard errors, and p-values.
        """
        print(f"\n{'='*80}")
        print(f"ESTIMATING CAUSAL EFFECT: {treatment} → {outcome}")
        print(f"{'='*80}")
        
        results = {}
        
        # Check sample sizes
        T = self.df[treatment].values
        Y = self.df[outcome].values
        n_treated = T.sum()
        n_control = len(T) - n_treated
        
        print(f"\nSample: N={len(T)}, Treated={n_treated}, Control={n_control}")
        
        if n_treated < 30 or n_control < 30:
            print("⚠️  WARNING: Small sample sizes may lead to unreliable estimates")
        
        # 1. Naive comparison (biased)
        print("\n1️⃣  Naive Comparison (No Adjustment)")
        results['naive'] = self._naive_comparison(T, Y)
        
        # 2. Regression adjustment
        print("\n2️⃣  Regression Adjustment")
        results['regression'] = self._regression_adjustment(treatment, outcome)
        
        # 3. Propensity score methods
        print("\n3️⃣  Propensity Score Methods")
        ps_model = self._estimate_propensity_scores(treatment)
        results['ps_model'] = ps_model
        
        # 3a. PS Matching
        print("\n   3a. Propensity Score Matching")
        results['ps_matching'] = self._ps_matching(treatment, outcome, ps_model['ps'])
        
        # 3b. Inverse Probability Weighting
        print("\n   3b. Inverse Probability Weighting (IPW)")
        results['ipw'] = self._ipw(treatment, outcome, ps_model['ps'])
        
        # 4. Doubly Robust
        print("\n4️⃣  Doubly Robust Estimation")
        results['doubly_robust'] = self._doubly_robust(treatment, outcome, ps_model['ps'])
        
        # 5. Summary
        print("\n" + "="*80)
        print("SUMMARY OF ESTIMATES")
        print("="*80)
        self._print_summary(results)
        
        return results
    
    def _naive_comparison(self, T: np.ndarray, Y: np.ndarray) -> Dict:
        """Naive difference in means (biased if confounding exists)."""
        Y_treated = Y[T == 1]
        Y_control = Y[T == 0]
        
        ate = Y_treated.mean() - Y_control.mean()
        
        # T-test
        t_stat, p_value = stats.ttest_ind(Y_treated, Y_control)
        
        # Standard error
        se = np.sqrt(Y_treated.var()/len(Y_treated) + Y_control.var()/len(Y_control))
        
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        print(f"   ATE: {ate:.4f} (SE: {se:.4f})")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   p-value: {p_value:.4f}")
        
        return {
            'ate': ate,
            'se': se,
            'ci': (ci_lower, ci_upper),
            'p_value': p_value,
            't_stat': t_stat,
            'n_treated': len(Y_treated),
            'n_control': len(Y_control)
        }
    
    def _regression_adjustment(self, treatment: str, outcome: str) -> Dict:
        """
        Regression adjustment: E[Y|T,X] model.
        
        Estimates ATE by predicting Y(1) and Y(0) for all units.
        """
        # Prepare data
        X_conf = self.df[self.confounders].values
        T = self.df[treatment].values.reshape(-1, 1)
        X = np.hstack([T, X_conf])
        Y = self.df[outcome].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model: Y ~ T + X
        model = LinearRegression()
        model.fit(X_scaled, Y)
        
        # Predict Y(1) and Y(0) for everyone
        X_treated = X.copy()
        X_treated[:, 0] = 1
        X_treated_scaled = scaler.transform(X_treated)
        Y1_pred = model.predict(X_treated_scaled)
        
        X_control = X.copy()
        X_control[:, 0] = 0
        X_control_scaled = scaler.transform(X_control)
        Y0_pred = model.predict(X_control_scaled)
        
        # ATE = average(Y1 - Y0)
        ate = (Y1_pred - Y0_pred).mean()
        
        # Bootstrap for SE
        n_bootstrap = 500
        ate_bootstrap = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(Y), len(Y), replace=True)
            X_boot = X_scaled[idx]
            Y_boot = Y[idx]
            
            model_boot = LinearRegression()
            model_boot.fit(X_boot, Y_boot)
            
            Y1_boot = model_boot.predict(X_treated_scaled[idx])
            Y0_boot = model_boot.predict(X_control_scaled[idx])
            ate_bootstrap.append((Y1_boot - Y0_boot).mean())
        
        se = np.std(ate_bootstrap)
        ci_lower = np.percentile(ate_bootstrap, 2.5)
        ci_upper = np.percentile(ate_bootstrap, 97.5)
        p_value = 2 * min(np.mean(np.array(ate_bootstrap) >= 0), 
                          np.mean(np.array(ate_bootstrap) <= 0))
        
        print(f"   ATE: {ate:.4f} (SE: {se:.4f})")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   p-value: {p_value:.4f}")
        
        return {
            'ate': ate,
            'se': se,
            'ci': (ci_lower, ci_upper),
            'p_value': p_value,
            'model': model,
            'treatment_coef': model.coef_[0]
        }
    
    def _estimate_propensity_scores(self, treatment: str) -> Dict:
        """
        Estimate propensity scores: P(T=1|X).
        
        Returns model and diagnostics.
        """
        X = self.df[self.confounders].values
        T = self.df[treatment].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit logistic regression
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X_scaled, T)
        
        # Get propensity scores
        ps = ps_model.predict_proba(X_scaled)[:, 1]
        
        # Diagnostics
        auc = roc_auc_score(T, ps)
        
        # Check overlap
        ps_treated = ps[T == 1]
        ps_control = ps[T == 0]
        
        print(f"   PS Model AUC: {auc:.3f}")
        print(f"   PS range (treated): [{ps_treated.min():.3f}, {ps_treated.max():.3f}]")
        print(f"   PS range (control): [{ps_control.min():.3f}, {ps_control.max():.3f}]")
        
        # Check for poor overlap
        if ps_treated.min() < 0.1 or ps_control.max() > 0.9:
            print("   ⚠️  WARNING: Poor overlap detected. Estimates may be unreliable.")
        
        return {
            'model': ps_model,
            'ps': ps,
            'auc': auc,
            'ps_treated': ps_treated,
            'ps_control': ps_control,
            'scaler': scaler
        }
    
    def _ps_matching(self, treatment: str, outcome: str, ps: np.ndarray) -> Dict:
        """
        Propensity score matching (1:1 nearest neighbor).
        """
        T = self.df[treatment].values
        Y = self.df[outcome].values
        
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]
        
        # Match each treated to nearest control
        matched_pairs = []
        used_controls = set()
        
        for i in treated_idx:
            ps_i = ps[i]
            
            # Find nearest control (not yet used)
            distances = np.abs(ps[control_idx] - ps_i)
            available = [j for j in range(len(control_idx)) if control_idx[j] not in used_controls]
            
            if len(available) == 0:
                continue
            
            best_match_idx = available[np.argmin(distances[available])]
            best_match = control_idx[best_match_idx]
            
            matched_pairs.append((i, best_match))
            used_controls.add(best_match)
        
        # Compute ATE on matched sample
        effects = []
        for treated_i, control_i in matched_pairs:
            effects.append(Y[treated_i] - Y[control_i])
        
        ate = np.mean(effects)
        se = np.std(effects) / np.sqrt(len(effects))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        # T-test
        t_stat = ate / se
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), len(effects) - 1))
        
        print(f"   Matched pairs: {len(matched_pairs)}")
        print(f"   ATE: {ate:.4f} (SE: {se:.4f})")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   p-value: {p_value:.4f}")
        
        return {
            'ate': ate,
            'se': se,
            'ci': (ci_lower, ci_upper),
            'p_value': p_value,
            'n_matched': len(matched_pairs),
            'matched_pairs': matched_pairs
        }
    
    def _ipw(self, treatment: str, outcome: str, ps: np.ndarray) -> Dict:
        """
        Inverse Probability Weighting (IPW).
        
        Weight observations by inverse of propensity score.
        """
        T = self.df[treatment].values
        Y = self.df[outcome].values
        
        # Trim extreme weights (PS between 0.1 and 0.9)
        ps_trimmed = np.clip(ps, 0.1, 0.9)
        
        # IPW weights
        weights = T / ps_trimmed + (1 - T) / (1 - ps_trimmed)
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        # Weighted means
        Y1_weighted = np.sum(weights * T * Y) / np.sum(weights * T)
        Y0_weighted = np.sum(weights * (1 - T) * Y) / np.sum(weights * (1 - T))
        
        ate = Y1_weighted - Y0_weighted
        
        # Bootstrap for SE
        n_bootstrap = 500
        ate_bootstrap = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(Y), len(Y), replace=True)
            T_boot = T[idx]
            Y_boot = Y[idx]
            ps_boot = ps_trimmed[idx]
            
            weights_boot = T_boot / ps_boot + (1 - T_boot) / (1 - ps_boot)
            weights_boot = weights_boot / weights_boot.sum() * len(weights_boot)
            
            Y1_boot = np.sum(weights_boot * T_boot * Y_boot) / np.sum(weights_boot * T_boot)
            Y0_boot = np.sum(weights_boot * (1 - T_boot) * Y_boot) / np.sum(weights_boot * (1 - T_boot))
            
            ate_bootstrap.append(Y1_boot - Y0_boot)
        
        se = np.std(ate_bootstrap)
        ci_lower = np.percentile(ate_bootstrap, 2.5)
        ci_upper = np.percentile(ate_bootstrap, 97.5)
        p_value = 2 * min(np.mean(np.array(ate_bootstrap) >= 0),
                          np.mean(np.array(ate_bootstrap) <= 0))
        
        print(f"   ATE: {ate:.4f} (SE: {se:.4f})")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   p-value: {p_value:.4f}")
        print(f"   Effective sample size: {1 / np.sum((weights / weights.sum())**2):.0f}")
        
        return {
            'ate': ate,
            'se': se,
            'ci': (ci_lower, ci_upper),
            'p_value': p_value,
            'weights': weights
        }
    
    def _doubly_robust(self, treatment: str, outcome: str, ps: np.ndarray) -> Dict:
        """
        Doubly Robust Estimation (AIPW).
        
        Combines regression + IPW. Consistent if either model correct.
        """
        X = self.df[self.confounders].values
        T = self.df[treatment].values
        Y = self.df[outcome].values
        
        # Fit outcome models for treated and control
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Model for E[Y|T=1,X]
        treated_idx = T == 1
        model_treated = LinearRegression()
        model_treated.fit(X_scaled[treated_idx], Y[treated_idx])
        mu1 = model_treated.predict(X_scaled)
        
        # Model for E[Y|T=0,X]
        control_idx = T == 0
        model_control = LinearRegression()
        model_control.fit(X_scaled[control_idx], Y[control_idx])
        mu0 = model_control.predict(X_scaled)
        
        # Trim PS
        ps_trimmed = np.clip(ps, 0.1, 0.9)
        
        # Doubly robust estimator
        tau_dr = (
            T * (Y - mu1) / ps_trimmed + mu1 -
            (1 - T) * (Y - mu0) / (1 - ps_trimmed) - mu0
        )
        
        ate = tau_dr.mean()
        se = tau_dr.std() / np.sqrt(len(tau_dr))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        # T-test
        t_stat = ate / se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
        
        print(f"   ATE: {ate:.4f} (SE: {se:.4f})")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   p-value: {p_value:.4f}")
        
        return {
            'ate': ate,
            'se': se,
            'ci': (ci_lower, ci_upper),
            'p_value': p_value,
            't_stat': t_stat
        }
    
    def _print_summary(self, results: Dict):
        """Print summary table of all estimates."""
        methods = ['naive', 'regression', 'ps_matching', 'ipw', 'doubly_robust']
        method_names = ['Naive', 'Regression', 'PS Matching', 'IPW', 'Doubly Robust']
        
        print(f"\n{'Method':<20} {'ATE':>10} {'SE':>10} {'95% CI':>25} {'p-value':>10}")
        print("-" * 80)
        
        for method, name in zip(methods, method_names):
            if method in results:
                r = results[method]
                ci_str = f"[{r['ci'][0]:>6.4f}, {r['ci'][1]:>6.4f}]"
                sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
                print(f"{name:<20} {r['ate']:>10.4f} {r['se']:>10.4f} {ci_str:>25} {r['p_value']:>10.4f} {sig}")
        
        print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")
    
    def run_full_analysis(self, output_dir: str = 'causal_results'):
        """
        Run complete causal analysis for all treatment-outcome pairs.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for treatment in self.treatments:
            for outcome in self.outcomes:
                key = f"{treatment}__{outcome}"
                all_results[key] = self.estimate_all_effects(treatment, outcome)
        
        # Save results
        self._save_results(all_results, output_path)
        
        # Generate report
        self._generate_report(all_results, output_path)
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_path}")
    
    def _save_results(self, results: Dict, output_path: Path):
        """Save results to CSV."""
        records = []
        
        for key, result in results.items():
            treatment, outcome = key.split('__')
            
            for method in ['naive', 'regression', 'ps_matching', 'ipw', 'doubly_robust']:
                if method in result:
                    r = result[method]
                    records.append({
                        'treatment': treatment,
                        'outcome': outcome,
                        'method': method,
                        'ate': r['ate'],
                        'se': r['se'],
                        'ci_lower': r['ci'][0],
                        'ci_upper': r['ci'][1],
                        'p_value': r['p_value'],
                        'significant': r['p_value'] < 0.05
                    })
        
        df_results = pd.DataFrame(records)
        df_results.to_csv(output_path / 'causal_estimates.csv', index=False)
    
    def _generate_report(self, results: Dict, output_path: Path):
        """Generate comprehensive report."""
        report = []
        report.append("="*80)
        report.append("CAUSAL EFFECT ESTIMATION REPORT")
        report.append("="*80)
        
        report.append("\n🎯 MAIN FINDINGS\n")
        
        # For each treatment, summarize across outcomes
        for treatment in self.treatments:
            report.append(f"\n{treatment.upper().replace('_', ' ')}:")
            
            for outcome in self.outcomes:
                key = f"{treatment}__{outcome}"
                if key not in results:
                    continue
                
                result = results[key]
                
                # Use doubly robust as primary estimate
                if 'doubly_robust' in result:
                    dr = result['doubly_robust']
                    sig = "***" if dr['p_value'] < 0.001 else "**" if dr['p_value'] < 0.01 else "*" if dr['p_value'] < 0.05 else "NS"
                    
                    report.append(f"  {outcome}:")
                    report.append(f"    ATE = {dr['ate']:.4f} ({sig})")
                    report.append(f"    95% CI: [{dr['ci'][0]:.4f}, {dr['ci'][1]:.4f}]")
                    
                    # Interpretation
                    if dr['p_value'] < 0.05:
                        direction = "increases" if dr['ate'] > 0 else "decreases"
                        report.append(f"    → Treatment {direction} outcome (p={dr['p_value']:.4f})")
                    else:
                        report.append(f"    → No significant effect (p={dr['p_value']:.4f})")
        
        report.append("\n\n" + "="*80)
        report.append("INTERPRETATION GUIDE")
        report.append("="*80)
        report.append("""
- ATE (Average Treatment Effect): Expected change in outcome from treatment
- Positive ATE: Treatment improves outcome
- Negative ATE: Treatment worsens outcome
- p-value < 0.05: Statistically significant effect
- Doubly robust estimate is most reliable (consistent if either PS or outcome model correct)
        """)
        
        with open(output_path / 'causal_report.txt', 'w') as f:
            f.write('\n'.join(report))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Estimate causal effects of LLM advice')
    parser.add_argument('--data', type=str, default='causal_data/causal_episodes.csv')
    parser.add_argument('--output', type=str, default='causal_results')
    parser.add_argument('--treatment', type=str, default=None, help='Specific treatment to analyze')
    parser.add_argument('--outcome', type=str, default=None, help='Specific outcome to analyze')
    
    args = parser.parse_args()
    
    estimator = CausalEffectEstimator(args.data)
    
    if args.treatment and args.outcome:
        # Single analysis
        results = estimator.estimate_all_effects(args.treatment, args.outcome)
    else:
        # Full analysis
        estimator.run_full_analysis(args.output)