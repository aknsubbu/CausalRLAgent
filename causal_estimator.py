import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings

# DoWhy for causal inference
from dowhy import CausalModel
from dowhy.causal_estimators.propensity_score_weighting_estimator import PropensityScoreWeightingEstimator

# Sklearn for ML models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Stats
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class CausalEffectEstimator:
    """
    Estimate causal effect of following LLM advice on agent performance.
    
    Uses DoWhy framework with multiple estimation strategies:
    - Inverse Propensity Weighting (IPW)
    - Doubly Robust (DR) 
    - Propensity Score Matching
    - Stratification
    """
    
    def __init__(
        self, 
        data_path: str,
        treatment_col: str = 'followed_advice_lenient',
        outcome_cols: List[str] = None,
        confounder_cols: List[str] = None
    ):
        """
        Initialize causal estimator.
        
        Args:
            data_path: Path to episode_summary.csv from parser
            treatment_col: Column indicating advice following (0/1)
            outcome_cols: List of outcome variables to analyze
            confounder_cols: List of pre-treatment confounders
        """
        self.data_path = Path(data_path)
        self.treatment_col = treatment_col
        
        # Default outcomes
        self.outcome_cols = outcome_cols or [
            'total_reward',
            'sr_change', 
            'hp_change',
            'avg_reward'
        ]
        
        # Default confounders (pre-treatment variables)
        self.confounder_cols = confounder_cols or [
            'hp_before',
            'level_before',
            'sr_before',
            'reward_before',
            'critical_hp'
        ]
        
        self.df = None
        self.results = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load episode data and prepare for causal analysis."""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        df = pd.read_csv(self.data_path)
        print(f"\n✓ Loaded {len(df)} episodes")
        
        # Ensure treatment is binary
        df[self.treatment_col] = df[self.treatment_col].astype(int)
        
        # Remove episodes with missing confounders
        missing_before = len(df)
        df = df.dropna(subset=self.confounder_cols + self.outcome_cols)
        missing_after = len(df)
        
        if missing_before - missing_after > 0:
            print(f"⚠ Dropped {missing_before - missing_after} episodes with missing values")
        
        # Check treatment balance
        n_treated = df[self.treatment_col].sum()
        n_control = len(df) - n_treated
        
        print(f"\nTreatment Balance:")
        print(f"  Followed advice (T=1): {n_treated} ({n_treated/len(df)*100:.1f}%)")
        print(f"  Did not follow (T=0):  {n_control} ({n_control/len(df)*100:.1f}%)")
        
        if n_treated < 30 or n_control < 30:
            warnings.warn("Small sample sizes may lead to unstable estimates!")
        
        self.df = df
        return df
    
    def run_dowhy_analysis(self, outcome: str) -> Dict:
        """
        Run complete DoWhy causal analysis for one outcome.
        
        Returns:
            Dictionary with causal estimates and diagnostics
        """
        print(f"\n{'='*80}")
        print(f"CAUSAL ANALYSIS: {outcome}")
        print(f"{'='*80}")
        
        # Step 1: Model the causal graph
        print("\n[1/4] Creating causal model...")
        
        model = CausalModel(
            data=self.df,
            treatment=self.treatment_col,
            outcome=outcome,
            common_causes=self.confounder_cols,
            effect_modifiers=[]
        )
        
        print("✓ Causal graph created")
        print(f"  Treatment: {self.treatment_col}")
        print(f"  Outcome: {outcome}")
        print(f"  Confounders: {', '.join(self.confounder_cols)}")
        
        # Step 2: Identify causal effect (backdoor criterion)
        print("\n[2/4] Identifying causal effect...")
        
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True
        )
        
        print("✓ Effect identified using backdoor adjustment")
        
        # Step 3: Estimate causal effect using multiple methods
        print("\n[3/4] Estimating causal effects...")
        
        estimates = {}
        
        # Method 1: Propensity Score Weighting
        try:
            print("\n  [a] Inverse Propensity Weighting...")
            ipw_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_weighting",
                method_params={
                    "weighting_scheme": "ips_weight",  # Inverse propensity score
                    "propensity_score_model": LogisticRegression(max_iter=1000)
                }
            )
            estimates['ipw'] = {
                'ate': ipw_estimate.value,
                'estimator': 'Inverse Propensity Weighting'
            }
            print(f"    ATE = {ipw_estimate.value:.4f}")
        except Exception as e:
            print(f"    ✗ IPW failed: {e}")
            estimates['ipw'] = None
        
        # Method 2: Linear Regression (outcome regression)
        try:
            print("\n  [b] Linear Regression (Backdoor)...")
            lr_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                control_value=0,
                treatment_value=1
            )
            estimates['linear_regression'] = {
                'ate': lr_estimate.value,
                'estimator': 'Linear Regression'
            }
            print(f"    ATE = {lr_estimate.value:.4f}")
        except Exception as e:
            print(f"    ✗ Linear regression failed: {e}")
            estimates['linear_regression'] = None
        
        # Method 3: Propensity Score Stratification
        try:
            print("\n  [c] Propensity Score Stratification...")
            strat_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_stratification",
                method_params={
                    "num_strata": 5,
                    "propensity_score_model": LogisticRegression(max_iter=1000)
                }
            )
            estimates['stratification'] = {
                'ate': strat_estimate.value,
                'estimator': 'Propensity Score Stratification'
            }
            print(f"    ATE = {strat_estimate.value:.4f}")
        except Exception as e:
            print(f"    ✗ Stratification failed: {e}")
            estimates['stratification'] = None
        
        # Method 4: Propensity Score Matching
        try:
            print("\n  [d] Propensity Score Matching...")
            match_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching",
                method_params={
                    "propensity_score_model": LogisticRegression(max_iter=1000)
                }
            )
            estimates['matching'] = {
                'ate': match_estimate.value,
                'estimator': 'Propensity Score Matching'
            }
            print(f"    ATE = {match_estimate.value:.4f}")
        except Exception as e:
            print(f"    ✗ Matching failed: {e}")
            estimates['matching'] = None
        
        # Step 4: Refutation tests
        print("\n[4/4] Running sensitivity analysis...")
        
        refutations = {}
        
        # Use first successful estimate for refutation
        primary_estimate = None
        for est_name, est in estimates.items():
            if est is not None:
                primary_estimate = (est_name, est)
                break
        
        if primary_estimate:
            est_name, _ = primary_estimate
            print(f"\n  Using {est_name} for refutation tests...")
            
            # Re-estimate for refutation (need to use model.estimate_effect again)
            if est_name == 'ipw':
                test_estimate = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.propensity_score_weighting"
                )
            elif est_name == 'linear_regression':
                test_estimate = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.linear_regression"
                )
            else:
                test_estimate = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.propensity_score_stratification"
                )
            
            # Refutation 1: Random common cause
            try:
                print("\n    [a] Adding random confounder...")
                refute_random = model.refute_estimate(
                    identified_estimand,
                    test_estimate,
                    method_name="random_common_cause"
                )
                refutations['random_cause'] = {
                    'new_effect': refute_random.new_effect,
                    'p_value': getattr(refute_random, 'refutation_result', {}).get('p_value', None)
                }
                print(f"      New ATE = {refute_random.new_effect:.4f}")
            except Exception as e:
                print(f"      ✗ Failed: {e}")
                refutations['random_cause'] = None
            
            # Refutation 2: Placebo treatment
            try:
                print("\n    [b] Placebo treatment test...")
                refute_placebo = model.refute_estimate(
                    identified_estimand,
                    test_estimate,
                    method_name="placebo_treatment_refuter"
                )
                refutations['placebo'] = {
                    'new_effect': refute_placebo.new_effect,
                    'p_value': getattr(refute_placebo, 'refutation_result', {}).get('p_value', None)
                }
                print(f"      Placebo ATE = {refute_placebo.new_effect:.4f}")
            except Exception as e:
                print(f"      ✗ Failed: {e}")
                refutations['placebo'] = None
        
        return {
            'outcome': outcome,
            'model': model,
            'identified_estimand': identified_estimand,
            'estimates': estimates,
            'refutations': refutations
        }
    
    def estimate_with_custom_dr(self, outcome: str) -> Dict:
        """
        Doubly Robust estimator (custom implementation).
        
        DR is robust to misspecification of either propensity or outcome model.
        """
        print(f"\n{'='*80}")
        print(f"DOUBLY ROBUST ESTIMATION: {outcome}")
        print(f"{'='*80}")
        
        X = self.df[self.confounder_cols].values
        T = self.df[self.treatment_col].values
        Y = self.df[outcome].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Step 1: Estimate propensity scores e(X) = P(T=1|X)
        print("\n[1/3] Estimating propensity scores...")
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X_scaled, T)
        ps = ps_model.predict_proba(X_scaled)[:, 1]
        
        # Clip propensity scores to avoid extreme weights
        ps = np.clip(ps, 0.01, 0.99)
        
        print(f"  Propensity score range: [{ps.min():.3f}, {ps.max():.3f}]")
        print(f"  Mean PS for T=1: {ps[T==1].mean():.3f}")
        print(f"  Mean PS for T=0: {ps[T==0].mean():.3f}")
        
        # Step 2: Estimate outcome regressions μ₀(X), μ₁(X)
        print("\n[2/3] Estimating outcome regressions...")
        
        # μ₀(X) = E[Y|X, T=0]
        mu0_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        mu0_model.fit(X_scaled[T==0], Y[T==0])
        mu0 = mu0_model.predict(X_scaled)
        
        # μ₁(X) = E[Y|X, T=1]
        mu1_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        mu1_model.fit(X_scaled[T==1], Y[T==1])
        mu1 = mu1_model.predict(X_scaled)
        
        print(f"  Outcome model R² (T=0): {mu0_model.score(X_scaled[T==0], Y[T==0]):.3f}")
        print(f"  Outcome model R² (T=1): {mu1_model.score(X_scaled[T==1], Y[T==1]):.3f}")
        
        # Step 3: Compute DR estimate
        print("\n[3/3] Computing doubly robust ATE...")
        
        # DR formula:
        # ATE = E[(T(Y - μ₁(X)))/e(X) - ((1-T)(Y - μ₀(X)))/(1-e(X)) + μ₁(X) - μ₀(X)]
        
        term1 = T * (Y - mu1) / ps
        term2 = (1 - T) * (Y - mu0) / (1 - ps)
        term3 = mu1 - mu0
        
        ate_dr = np.mean(term1 - term2 + term3)
        
        # Standard error via bootstrap
        n_boot = 1000
        boot_ates = []
        
        for _ in range(n_boot):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            boot_ate = np.mean(
                term1[idx] - term2[idx] + term3[idx]
            )
            boot_ates.append(boot_ate)
        
        se = np.std(boot_ates)
        ci_lower = np.percentile(boot_ates, 2.5)
        ci_upper = np.percentile(boot_ates, 97.5)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(ate_dr / se)))
        
        print(f"\n✓ Doubly Robust ATE = {ate_dr:.4f}")
        print(f"  Standard Error = {se:.4f}")
        print(f"  95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  p-value = {p_value:.4f}")
        
        return {
            'ate': ate_dr,
            'se': se,
            'ci': (ci_lower, ci_upper),
            'p_value': p_value,
            'propensity_scores': ps,
            'outcome_models': (mu0_model, mu1_model)
        }
    
    def run_all_analyses(self) -> Dict[str, Dict]:
        """Run complete causal analysis for all outcomes."""
        self.load_and_prepare_data()
        
        print(f"\n{'='*80}")
        print("RUNNING CAUSAL ANALYSES")
        print(f"{'='*80}")
        
        results = {}
        
        for outcome in self.outcome_cols:
            if outcome not in self.df.columns:
                print(f"\n⚠ Skipping {outcome} (not in data)")
                continue
            
            # DoWhy analysis
            dowhy_results = self.run_dowhy_analysis(outcome)
            
            # Custom DR estimator
            dr_results = self.estimate_with_custom_dr(outcome)
            
            results[outcome] = {
                'dowhy': dowhy_results,
                'doubly_robust': dr_results
            }
        
        self.results = results
        return results
    
    def generate_report(self, output_dir: str = 'causal_analysis'):
        """Generate comprehensive causal analysis report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print("GENERATING REPORT")
        print(f"{'='*80}")
        
        # Summary table
        summary_rows = []
        
        for outcome, results in self.results.items():
            row = {'outcome': outcome}
            
            # DoWhy estimates
            if results['dowhy']['estimates']:
                for method, est in results['dowhy']['estimates'].items():
                    if est:
                        row[f'ate_{method}'] = est['ate']
            
            # DR estimate
            if results['doubly_robust']:
                row['ate_dr'] = results['doubly_robust']['ate']
                row['ate_dr_se'] = results['doubly_robust']['se']
                row['ate_dr_p'] = results['doubly_robust']['p_value']
            
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / 'causal_estimates_summary.csv', index=False)
        
        print(f"\n✓ Summary saved to {output_dir / 'causal_estimates_summary.csv'}")
        print("\nCausal Effect Estimates:")
        print(summary_df.to_string())
        
        # Detailed report
        report = []
        report.append("="*80)
        report.append("CAUSAL EFFECT ESTIMATION REPORT")
        report.append("="*80)
        report.append(f"\nTreatment: {self.treatment_col}")
        report.append(f"Confounders: {', '.join(self.confounder_cols)}")
        report.append(f"Sample size: {len(self.df)} episodes")
        
        for outcome, results in self.results.items():
            report.append(f"\n{'='*80}")
            report.append(f"OUTCOME: {outcome}")
            report.append(f"{'='*80}")
            
            # DoWhy estimates
            report.append("\n--- DoWhy Estimates ---")
            for method, est in results['dowhy']['estimates'].items():
                if est:
                    report.append(f"{est['estimator']:40s} ATE = {est['ate']:8.4f}")
            
            # DR estimate
            if results['doubly_robust']:
                dr = results['doubly_robust']
                report.append("\n--- Doubly Robust Estimate ---")
                report.append(f"ATE = {dr['ate']:.4f} (SE = {dr['se']:.4f})")
                report.append(f"95% CI = [{dr['ci'][0]:.4f}, {dr['ci'][1]:.4f}]")
                report.append(f"p-value = {dr['p_value']:.4f}")
                
                if dr['p_value'] < 0.05:
                    report.append("✓ Statistically significant at α=0.05")
                else:
                    report.append("✗ Not statistically significant at α=0.05")
            
            # Refutations
            if results['dowhy']['refutations']:
                report.append("\n--- Sensitivity Tests ---")
                for test_name, refute in results['dowhy']['refutations'].items():
                    if refute:
                        report.append(f"{test_name:20s}: New ATE = {refute['new_effect']:.4f}")
        
        report_text = "\n".join(report)
        
        with open(output_dir / 'causal_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Full report saved to {output_dir / 'causal_report.txt'}")
        
        return summary_df


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Causal effect estimation')
    parser.add_argument('--data', type=str, default='processed_data/episode_summary.csv')
    parser.add_argument('--output', type=str, default='causal_analysis')
    parser.add_argument('--treatment', type=str, default='followed_advice_lenient',
                       choices=['followed_advice_lenient', 'followed_advice_strict'])
    
    args = parser.parse_args()
    
    # Initialize estimator
    estimator = CausalEffectEstimator(
        data_path=args.data,
        treatment_col=args.treatment
    )
    
    # Run analyses
    results = estimator.run_all_analyses()
    
    # Generate report
    summary = estimator.generate_report(args.output)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {args.output}/")
    print("  - causal_estimates_summary.csv")
    print("  - causal_report.txt")