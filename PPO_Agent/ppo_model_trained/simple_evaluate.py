#!/usr/bin/env python3
"""
Simplified NetHack PPO evaluation script with basic visualizations.
Use this if the full evaluate_model.py has dependency issues.

Usage: python simple_evaluate.py --model_path your_model.pth
"""

import argparse
import os
import time
import json
from datetime import datetime
from collections import defaultdict

try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.distributions import Categorical
    
    # Try to import from your training script
    from improved_reward_shaping import (
        EnhancedNetHackPPOAgent,
        create_nethack_env
    )
    
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing dependencies with:")
    print("pip install torch numpy matplotlib gymnasium nle")
    DEPENDENCIES_OK = False


def simple_evaluate_model(model_path, num_episodes=10, max_steps=3000):
    """Simple model evaluation with basic metrics"""
    
    if not DEPENDENCIES_OK:
        return None
    
    print(f"🤖 Loading model from: {model_path}")
    
    # Create environment and agent
    env = create_nethack_env()
    agent = EnhancedNetHackPPOAgent(action_dim=env.action_space.n, use_wandb=False)
    
    # Load the trained model
    try:
        agent.load_model(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    episode_details = []
    action_counts = defaultdict(int)
    
    print(f"\n🎯 Running {num_episodes} evaluation episodes...")
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}", end=" - ")
        
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        # Reset agent state
        agent.actor.reset_hidden_states()
        agent.critic.reset_hidden_states()
        agent.last_action = None
        
        reset_hidden = True
        
        while steps < max_steps:
            # Get action from policy
            processed_obs = agent.process_observation(obs)
            
            with torch.no_grad():
                action_logits = agent.actor(processed_obs, reset_hidden)
                # Use deterministic action for evaluation
                action = torch.argmax(action_logits, dim=-1).item()
                reset_hidden = False
            
            agent.last_action = action
            action_counts[action] += 1
            
            # Take step
            step_result = env.step(action)
            
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            total_reward += reward
            steps += 1
            obs = next_obs
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Check if died
        died = False
        try:
            if isinstance(obs, dict) and 'blstats' in obs:
                final_stats = obs['blstats']
                died = len(final_stats) > 0 and final_stats[0] <= 0
            elif hasattr(obs, '__getitem__'):
                obs_dict = obs[0] if isinstance(obs, tuple) else obs
                if isinstance(obs_dict, dict) and 'blstats' in obs_dict:
                    final_stats = obs_dict['blstats']
                    died = len(final_stats) > 0 and final_stats[0] <= 0
        except:
            pass
        
        episode_details.append({
            'reward': total_reward,
            'length': steps,
            'died': died
        })
        
        print(f"Reward: {total_reward:.1f}, Steps: {steps}, Died: {died}")
    
    env.close()
    
    # Calculate statistics
    results = {
        'num_episodes': num_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_details': episode_details,
        'action_counts': dict(action_counts),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'survival_rate': sum(1 for ep in episode_details if not ep['died']) / num_episodes
    }
    
    return results


def create_simple_plots(results, model_name):
    """Create basic visualization plots"""
    
    if not results:
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Model Evaluation Results: {model_name}', fontsize=16)
    
    # 1. Episode Rewards
    ax1 = axes[0, 0]
    episodes = range(1, len(results['episode_rewards']) + 1)
    ax1.bar(episodes, results['episode_rewards'], alpha=0.7, color='skyblue')
    ax1.axhline(results['mean_reward'], color='red', linestyle='--', 
               label=f"Mean: {results['mean_reward']:.1f}")
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Lengths
    ax2 = axes[0, 1]
    ax2.bar(episodes, results['episode_lengths'], alpha=0.7, color='lightgreen')
    ax2.axhline(results['mean_length'], color='red', linestyle='--',
               label=f"Mean: {results['mean_length']:.0f}")
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reward Distribution
    ax3 = axes[0, 2]
    ax3.hist(results['episode_rewards'], bins=min(8, len(results['episode_rewards'])), 
            alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(results['mean_reward'], color='red', linestyle='--', linewidth=2)
    ax3.set_title('Reward Distribution')
    ax3.set_xlabel('Total Reward')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. Action Usage
    ax4 = axes[1, 0]
    actions = list(results['action_counts'].keys())
    counts = list(results['action_counts'].values())
    if actions:
        ax4.bar(actions, counts, alpha=0.7, color='orange')
        ax4.set_title('Action Usage')
        ax4.set_xlabel('Action ID')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
    
    # 5. Survival Analysis
    ax5 = axes[1, 1]
    survived = sum(1 for ep in results['episode_details'] if not ep['died'])
    died = results['num_episodes'] - survived
    
    labels = ['Survived', 'Died']
    sizes = [survived, died]
    colors = ['lightgreen', 'lightcoral']
    
    ax5.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax5.set_title('Survival Rate')
    
    # 6. Performance Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
    EVALUATION SUMMARY
    
    Episodes: {results['num_episodes']}
    
    Rewards:
    • Mean: {results['mean_reward']:.2f}
    • Std: {results['std_reward']:.2f}
    • Min: {results['min_reward']:.1f}
    • Max: {results['max_reward']:.1f}
    
    Performance:
    • Mean Length: {results['mean_length']:.0f}
    • Survival Rate: {results['survival_rate']:.1%}
    • Actions Used: {len(results['action_counts'])}/23
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'simple_evaluation_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Plot saved as: {plot_filename}")
    
    return plot_filename


def main():
    parser = argparse.ArgumentParser(description='Simple NetHack PPO model evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=3000,
                       help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    if not DEPENDENCIES_OK:
        print("❌ Dependencies not available. Please install required packages.")
        return
    
    print("🚀 Simple NetHack PPO Evaluation")
    print(f"📁 Model: {args.model_path}")
    print(f"🎯 Episodes: {args.episodes}")
    
    # Run evaluation
    start_time = time.time()
    results = simple_evaluate_model(args.model_path, args.episodes, args.max_steps)
    eval_time = time.time() - start_time
    
    if results is None:
        print("❌ Evaluation failed!")
        return
    
    print(f"\n✅ Evaluation completed in {eval_time:.1f} seconds!")
    
    # Print summary
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} (±{results['std_reward']:.2f})")
    print(f"  Best Episode: {results['max_reward']:.1f}")
    print(f"  Mean Length: {results['mean_length']:.0f} steps")
    print(f"  Survival Rate: {results['survival_rate']:.1%}")
    print(f"  Actions Used: {len(results['action_counts'])}/23")
    
    # Generate performance assessment
    print(f"\n💡 PERFORMANCE ASSESSMENT:")
    if results['mean_reward'] > 50:
        print("  ✅ Excellent performance!")
    elif results['mean_reward'] > 20:
        print("  👍 Good performance!")
    elif results['mean_reward'] > 0:
        print("  ⚠️ Moderate performance.")
    else:
        print("  ❌ Needs improvement.")
    
    if results['survival_rate'] > 0.5:
        print("  ✅ Good survival rate!")
    else:
        print("  ⚠️ Low survival rate - agent dies frequently.")
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    model_name = os.path.basename(args.model_path)
    plot_file = create_simple_plots(results, model_name)
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'simple_evaluation_results_{timestamp}.json'
    
    # Prepare JSON-friendly results
    json_results = {
        'timestamp': timestamp,
        'model_path': args.model_path,
        'evaluation_params': {
            'num_episodes': args.episodes,
            'max_steps': args.max_steps
        },
        'summary_stats': {
            'mean_reward': float(results['mean_reward']),
            'std_reward': float(results['std_reward']),
            'min_reward': float(results['min_reward']),
            'max_reward': float(results['max_reward']),
            'mean_length': float(results['mean_length']),
            'survival_rate': float(results['survival_rate'])
        },
        'episode_details': results['episode_details'],
        'action_counts': results['action_counts']
    }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"📝 Results saved to: {results_file}")
    print(f"\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()