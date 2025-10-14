#!/usr/bin/env python3
"""
Diagnostic script to analyze what went wrong with your NetHack PPO training.
This will help identify the specific problems and provide actionable solutions.
"""

import json
import os
from datetime import datetime

def analyze_evaluation_results(results_path):
    """Analyze evaluation results to diagnose problems"""
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("🔍 DIAGNOSTIC ANALYSIS")
    print("=" * 50)
    
    # Basic stats
    stats = results['summary_statistics']
    action_freq = results['action_frequencies']
    episodes = results['episode_summaries']
    
    print(f"📊 Basic Performance:")
    print(f"  Episodes: {stats['num_episodes']}")
    print(f"  Mean Reward: {stats['mean_reward']:.2f}")
    print(f"  Episode Length: {stats['mean_length']:.0f}")
    print(f"  Survival Rate: {stats['survival_rate']:.1%}")
    
    # Action Analysis
    print(f"\n🎮 Action Analysis:")
    print(f"  Total Actions Used: {len(action_freq)}/23")
    print(f"  Action Frequencies:")
    
    total_actions = sum(action_freq.values())
    for action_id, count in action_freq.items():
        percentage = (count / total_actions) * 100
        print(f"    Action {action_id}: {count:,} times ({percentage:.1f}%)")
    
    # Diagnose specific problems
    print(f"\n🚨 PROBLEM DIAGNOSIS:")
    
    # Problem 1: Stuck behavior
    if len(action_freq) <= 3:
        print(f"  ❌ CRITICAL: Severe action limitation!")
        print(f"     - Only using {len(action_freq)} out of 23 possible actions")
        print(f"     - Agent is stuck in repetitive behavior")
        print(f"     - Root cause: Policy collapsed during training")
    
    # Problem 2: No exploration
    avg_unique_pos = sum(ep['unique_positions'] for ep in episodes) / len(episodes)
    if avg_unique_pos < 10:
        print(f"  ❌ CRITICAL: No exploration!")
        print(f"     - Average unique positions: {avg_unique_pos:.1f}")
        print(f"     - Agent is not moving around the environment")
        print(f"     - Likely stuck against walls or in corners")
    
    # Problem 3: Poor rewards
    if stats['mean_reward'] < -20:
        print(f"  ❌ CRITICAL: Extremely poor rewards!")
        print(f"     - Mean reward: {stats['mean_reward']:.2f}")
        print(f"     - Agent is getting maximum negative rewards")
        print(f"     - Not learning basic survival skills")
    
    # Problem 4: No survival
    if stats['survival_rate'] == 0:
        print(f"  ❌ CRITICAL: Zero survival rate!")
        print(f"     - Agent dies or times out in every episode")
        print(f"     - Not learning to stay alive")
    
    # Problem 5: All episodes same length
    lengths = [ep['length'] for ep in episodes]
    if len(set(lengths)) == 1 and lengths[0] >= 5000:
        print(f"  ❌ CRITICAL: All episodes timeout!")
        print(f"     - Every episode reaches maximum steps ({lengths[0]})")
        print(f"     - Agent never dies naturally or wins")
        print(f"     - Completely stuck behavior")
    
    print(f"\n💡 ROOT CAUSE ANALYSIS:")
    
    # Determine primary root cause
    if len(action_freq) == 1:
        main_action = list(action_freq.keys())[0]
        print(f"  🎯 PRIMARY ISSUE: Complete policy collapse")
        print(f"     - Agent learned to ONLY use action {main_action}")
        print(f"     - This action is likely:")
        print(f"       • Moving into a wall (gets stuck)")
        print(f"       • No-op action (does nothing)")
        print(f"       • Invalid move (causes problems)")
    
    print(f"\n🛠️ RECOMMENDED FIXES:")
    print(f"  1. 🎯 Retrain with conservative reward shaping")
    print(f"     - Current rewards were too inflated")
    print(f"     - Need smaller, more realistic bonuses")
    print(f"  ")
    print(f"  2. 🚶 Increase exploration incentives")
    print(f"     - Higher entropy coefficient (0.05-0.1)")
    print(f"     - Better exploration rewards")
    print(f"  ")
    print(f"  3. ⏱️ Train for longer")
    print(f"     - Your training stopped too early")
    print(f"     - Need 200-500+ episodes minimum")
    print(f"  ")
    print(f"  4. 🎛️ Lower learning rate") 
    print(f"     - Use 1e-5 to 5e-5 instead of 1e-4")
    print(f"     - Prevents policy collapse")
    print(f"  ")
    print(f"  5. 🎯 Realistic target rewards")
    print(f"     - Set target to 10-20, not 50+")
    print(f"     - Prevents premature stopping")
    
    print(f"\n📋 ACTION PLAN:")
    print(f"  1. Run: python fixed_training.py")
    print(f"  2. Wait for 500+ episodes to complete")  
    print(f"  3. Evaluate with: python simple_evaluate.py --model_path FIXED_*.pth")
    print(f"  4. Look for:")
    print(f"     - Multiple actions used (>10 different actions)")
    print(f"     - Higher unique positions (>20 per episode)")
    print(f"     - Rewards closer to 0 or positive")
    print(f"     - Some episodes ending before timeout")

def main():
    """Main diagnostic function"""
    
    # Look for recent evaluation results
    results_dir = "evaluation_results"
    if os.path.exists(results_dir):
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if json_files:
            # Use most recent
            latest_file = max(json_files)
            results_path = os.path.join(results_dir, latest_file)
            analyze_evaluation_results(results_path)
        else:
            print("❌ No evaluation JSON files found in evaluation_results/")
    else:
        print("❌ No evaluation_results directory found")
        print("Run evaluation first: python simple_evaluate.py --model_path your_model.pth")

if __name__ == "__main__":
    main()