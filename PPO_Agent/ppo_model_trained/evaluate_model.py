#!/usr/bin/env python3
"""
NetHack PPO Model Evaluation Script

This script loads a trained PPO model and evaluates its performance in the NetHack environment.
It provides comprehensive visualizations and performance metrics.

Usage:
    python evaluate_model.py --model_path <path_to_model.pth> --episodes 20 --render
    
Example:
    python evaluate_model.py --model_path enhanced_nethack_ppo_20251013_133042.pth --episodes 10
"""

import argparse
import gymnasium as gym
import nle.env
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
from datetime import datetime
import os
import json
import time
from typing import Dict, List, Tuple

# Import the necessary classes from your training script
from improved_reward_shaping import (
    EnhancedNetHackPPOAgent,
    NetHackObservationProcessor,
    RecurrentPPOActor,
    RecurrentPPOCritic,
    create_nethack_env
)

class ModelEvaluator:
    """Comprehensive model evaluation with detailed analysis"""
    
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the trained model
        self.agent = self._load_model()
        
        # Evaluation metrics
        self.episode_data = []
        self.step_data = []
        self.action_frequencies = defaultdict(int)
        self.position_heatmap = defaultdict(int)
        
    def _load_model(self) -> EnhancedNetHackPPOAgent:
        """Load the trained model"""
        print(f"Loading model from {self.model_path}")
        
        # Create agent (you may need to adjust hyperparameters to match training)
        agent = EnhancedNetHackPPOAgent(
            action_dim=23,  # NetHack action space
            learning_rate=1e-4,
            use_wandb=False
        )
        
        # Load the model weights
        try:
            agent.load_model(self.model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
            
        return agent
        
    def evaluate_episodes(self, env, num_episodes: int = 10, max_steps: int = 5000, 
                         render: bool = False, save_trajectories: bool = True) -> Dict:
        """Evaluate the model over multiple episodes"""
        
        print(f"🎯 Starting evaluation for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        episode_details = []
        
        for episode in range(num_episodes):
            print(f"\n📊 Episode {episode + 1}/{num_episodes}")
            
            episode_data = self._run_single_episode(
                env, episode, max_steps, render, save_trajectories
            )
            
            episode_rewards.append(episode_data['total_reward'])
            episode_lengths.append(episode_data['length'])
            episode_details.append(episode_data)
            
            # Print episode summary
            print(f"  Reward: {episode_data['total_reward']:.2f}")
            print(f"  Length: {episode_data['length']} steps")
            print(f"  Survival: {'Yes' if not episode_data['died'] else 'No'}")
            print(f"  Unique Positions: {episode_data['unique_positions']}")
            
        # Calculate summary statistics
        summary = {
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'survival_rate': sum(1 for ep in episode_details if not ep['died']) / num_episodes,
            'episode_details': episode_details
        }
        
        return summary
        
    def _run_single_episode(self, env, episode_num: int, max_steps: int, 
                           render: bool, save_trajectory: bool) -> Dict:
        """Run a single episode and collect detailed data"""
        
        obs = env.reset()
        total_reward = 0
        steps = 0
        died = False
        
        # Episode tracking
        positions_visited = set()
        actions_taken = []
        rewards_per_step = []
        observations = [] if save_trajectory else None
        
        # Reset agent state
        self.agent.actor.reset_hidden_states()
        self.agent.critic.reset_hidden_states()
        self.agent.last_action = None
        
        reset_hidden = True
        
        while steps < max_steps:
            if render:
                env.render()
                
            # Process observation and select action
            processed_obs = self.agent.process_observation(obs)
            
            # Get action from trained policy (deterministic for evaluation)
            with torch.no_grad():
                action_logits = self.agent.actor(processed_obs, reset_hidden)
                action_dist = Categorical(logits=action_logits)
                
                # Use deterministic action (argmax) for evaluation
                action = torch.argmax(action_logits, dim=-1).item()
                
                reset_hidden = False
            
            # Store action for next observation processing
            self.agent.last_action = action
            
            # Take step in environment
            step_result = env.step(action)
            
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
                
            # Track data
            total_reward += reward
            actions_taken.append(action)
            rewards_per_step.append(reward)
            self.action_frequencies[action] += 1
            
            # Track position if available
            if isinstance(next_obs, dict) and 'blstats' in next_obs:
                stats = next_obs['blstats']
                if len(stats) >= 2:
                    pos = (int(stats[0]), int(stats[1]))
                    positions_visited.add(pos)
                    self.position_heatmap[pos] += 1
            elif hasattr(next_obs, '__getitem__') and len(next_obs) > 0:
                # Handle tuple observation
                obs_dict = next_obs[0] if isinstance(next_obs, tuple) else next_obs
                if isinstance(obs_dict, dict) and 'blstats' in obs_dict:
                    stats = obs_dict['blstats']
                    if len(stats) >= 2:
                        pos = (int(stats[0]), int(stats[1]))
                        positions_visited.add(pos)
                        self.position_heatmap[pos] += 1
            
            if save_trajectory and observations is not None:
                observations.append({
                    'step': steps,
                    'action': action,
                    'reward': reward,
                    'done': done
                })
                
            obs = next_obs
            steps += 1
            
            if done:
                # Check if player died
                if isinstance(obs, dict) and 'blstats' in obs:
                    final_stats = obs['blstats']
                    died = len(final_stats) > 0 and final_stats[0] <= 0
                elif hasattr(obs, '__getitem__'):
                    obs_dict = obs[0] if isinstance(obs, tuple) else obs
                    if isinstance(obs_dict, dict) and 'blstats' in obs_dict:
                        final_stats = obs_dict['blstats']
                        died = len(final_stats) > 0 and final_stats[0] <= 0
                break
                
        # Compile episode data
        episode_data = {
            'episode': episode_num,
            'total_reward': total_reward,
            'length': steps,
            'died': died,
            'unique_positions': len(positions_visited),
            'actions_taken': actions_taken,
            'rewards_per_step': rewards_per_step,
            'mean_reward_per_step': total_reward / max(steps, 1),
            'trajectory': observations
        }
        
        return episode_data
        
    def create_visualizations(self, eval_results: Dict, save_dir: str = "evaluation_results"):
        """Create comprehensive visualizations of evaluation results"""
        
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create main figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Episode Rewards
        ax1 = plt.subplot(3, 4, 1)
        episode_rewards = [ep['total_reward'] for ep in eval_results['episode_details']]
        episodes = range(1, len(episode_rewards) + 1)
        
        plt.bar(episodes, episode_rewards, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.axhline(y=eval_results['mean_reward'], color='red', linestyle='--', 
                   label=f'Mean: {eval_results["mean_reward"]:.2f}')
        plt.title('Episode Rewards', fontsize=12, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Episode Lengths
        ax2 = plt.subplot(3, 4, 2)
        episode_lengths = [ep['length'] for ep in eval_results['episode_details']]
        
        plt.bar(episodes, episode_lengths, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        plt.axhline(y=eval_results['mean_length'], color='red', linestyle='--',
                   label=f'Mean: {eval_results["mean_length"]:.1f}')
        plt.title('Episode Lengths', fontsize=12, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Reward vs Length Scatter
        ax3 = plt.subplot(3, 4, 3)
        plt.scatter(episode_lengths, episode_rewards, alpha=0.7, s=100, c='purple')
        plt.xlabel('Episode Length')
        plt.ylabel('Total Reward')
        plt.title('Reward vs Length', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(episode_lengths, episode_rewards)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 4. Action Distribution
        ax4 = plt.subplot(3, 4, 4)
        actions = list(self.action_frequencies.keys())
        frequencies = list(self.action_frequencies.values())
        
        plt.bar(actions, frequencies, alpha=0.7, color='orange')
        plt.title('Action Frequency Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Action ID')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 5. Survival Analysis
        ax5 = plt.subplot(3, 4, 5)
        survival_data = ['Survived' if not ep['died'] else 'Died' 
                        for ep in eval_results['episode_details']]
        survival_counts = {status: survival_data.count(status) for status in set(survival_data)}
        
        colors = ['lightgreen' if status == 'Survived' else 'lightcoral' 
                 for status in survival_counts.keys()]
        plt.pie(survival_counts.values(), labels=survival_counts.keys(), 
               autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Survival Rate', fontsize=12, fontweight='bold')
        
        # 6. Reward Distribution
        ax6 = plt.subplot(3, 4, 6)
        plt.hist(episode_rewards, bins=min(10, len(episode_rewards)), 
                alpha=0.7, color='mediumpurple', edgecolor='black')
        plt.axvline(eval_results['mean_reward'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {eval_results["mean_reward"]:.2f}')
        plt.title('Reward Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Exploration Efficiency
        ax7 = plt.subplot(3, 4, 7)
        unique_positions = [ep['unique_positions'] for ep in eval_results['episode_details']]
        exploration_efficiency = [pos/length for pos, length in zip(unique_positions, episode_lengths)]
        
        plt.bar(episodes, exploration_efficiency, alpha=0.7, color='gold', edgecolor='darkorange')
        plt.title('Exploration Efficiency', fontsize=12, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Unique Pos / Steps')
        plt.grid(True, alpha=0.3)
        
        # 8. Performance Trends
        ax8 = plt.subplot(3, 4, 8)
        # Moving average for trends (if enough episodes)
        if len(episode_rewards) >= 5:
            window = min(5, len(episode_rewards) // 2)
            ma_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ma_episodes = range(window, len(episode_rewards) + 1)
            plt.plot(ma_episodes, ma_rewards, 'r-', linewidth=2, label=f'MA-{window}')
            
        plt.plot(episodes, episode_rewards, 'b-', alpha=0.5, label='Raw')
        plt.title('Performance Trend', fontsize=12, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Step-wise Reward Analysis (for first few episodes)
        ax9 = plt.subplot(3, 4, 9)
        if len(eval_results['episode_details']) > 0:
            # Plot reward progression for first 3 episodes
            for i, ep_data in enumerate(eval_results['episode_details'][:3]):
                rewards = ep_data['rewards_per_step']
                cumulative_rewards = np.cumsum(rewards)
                plt.plot(cumulative_rewards, alpha=0.7, label=f'Episode {i+1}')
                
        plt.title('Cumulative Reward Progression', fontsize=12, fontweight='bold')
        plt.xlabel('Step')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 10. Action Sequence Analysis (most common actions)
        ax10 = plt.subplot(3, 4, 10)
        top_actions = sorted(self.action_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_actions:
            actions, counts = zip(*top_actions)
            plt.barh(range(len(actions)), counts, alpha=0.7, color='teal')
            plt.yticks(range(len(actions)), [f'Action {a}' for a in actions])
            plt.title('Top 10 Actions Used', fontsize=12, fontweight='bold')
            plt.xlabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        # 11. Performance Statistics Box
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        stats_text = f"""
        EVALUATION STATISTICS
        
        Episodes: {eval_results['num_episodes']}
        
        Rewards:
        • Mean: {eval_results['mean_reward']:.2f}
        • Std: {eval_results['std_reward']:.2f}
        • Min: {eval_results['min_reward']:.2f}
        • Max: {eval_results['max_reward']:.2f}
        
        Episode Length:
        • Mean: {eval_results['mean_length']:.1f}
        • Std: {eval_results['std_length']:.1f}
        
        Performance:
        • Survival Rate: {eval_results['survival_rate']:.1%}
        • Avg Exploration: {np.mean(unique_positions):.1f}
        • Total Actions: {sum(self.action_frequencies.values())}
        """
        
        ax11.text(0.05, 0.95, stats_text, transform=ax11.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # 12. Position Heatmap (if we have position data)
        ax12 = plt.subplot(3, 4, 12)
        if self.position_heatmap:
            # Create a simplified heatmap
            positions = list(self.position_heatmap.keys())
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                weights = [self.position_heatmap[pos] for pos in positions]
                
                # Create 2D histogram
                plt.hist2d(x_coords, y_coords, weights=weights, bins=20, cmap='YlOrRd')
                plt.colorbar(label='Visit Frequency')
                plt.title('Position Heatmap', fontsize=12, fontweight='bold')
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
        
        plt.tight_layout(pad=2.0)
        
        # Save the comprehensive plot
        plot_path = os.path.join(save_dir, f'comprehensive_evaluation_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"📊 Comprehensive evaluation plots saved to: {plot_path}")
        
        # Save detailed results to JSON
        results_path = os.path.join(save_dir, f'evaluation_results_{timestamp}.json')
        
        # Prepare JSON-serializable data
        json_results = {
            'timestamp': timestamp,
            'model_path': self.model_path,
            'summary_statistics': {
                'num_episodes': eval_results['num_episodes'],
                'mean_reward': float(eval_results['mean_reward']),
                'std_reward': float(eval_results['std_reward']),
                'min_reward': float(eval_results['min_reward']),
                'max_reward': float(eval_results['max_reward']),
                'mean_length': float(eval_results['mean_length']),
                'std_length': float(eval_results['std_length']),
                'survival_rate': float(eval_results['survival_rate'])
            },
            'action_frequencies': dict(self.action_frequencies),
            'episode_summaries': [
                {
                    'episode': ep['episode'],
                    'total_reward': float(ep['total_reward']),
                    'length': int(ep['length']),
                    'died': bool(ep['died']),
                    'unique_positions': int(ep['unique_positions']),
                    'mean_reward_per_step': float(ep['mean_reward_per_step'])
                }
                for ep in eval_results['episode_details']
            ]
        }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        print(f"📝 Detailed results saved to: {results_path}")
        
        return plot_path, results_path


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained NetHack PPO model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model (.pth file)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate (default: 10)')
    parser.add_argument('--max_steps', type=int, default=5000,
                       help='Maximum steps per episode (default: 5000)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during evaluation')
    parser.add_argument('--save_trajectories', action='store_true',
                       help='Save detailed episode trajectories')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results (default: evaluation_results)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        return
        
    print("🚀 Starting NetHack PPO Model Evaluation...")
    print(f"📁 Model: {args.model_path}")
    print(f"🎯 Episodes: {args.episodes}")
    print(f"👁️ Render: {args.render}")
    
    # Create environment
    print("\n🌍 Creating NetHack environment...")
    env = create_nethack_env()
    
    # Create evaluator and load model
    print("\n🤖 Loading trained model...")
    evaluator = ModelEvaluator(args.model_path)
    
    # Run evaluation
    print(f"\n🎮 Running evaluation...")
    start_time = time.time()
    
    results = evaluator.evaluate_episodes(
        env=env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_trajectories=args.save_trajectories
    )
    
    eval_time = time.time() - start_time
    
    # Print summary
    print(f"\n✅ Evaluation completed in {eval_time:.2f} seconds!")
    print("\n📊 EVALUATION SUMMARY:")
    print(f"  Episodes: {results['num_episodes']}")
    print(f"  Mean Reward: {results['mean_reward']:.2f} (±{results['std_reward']:.2f})")
    print(f"  Best Reward: {results['max_reward']:.2f}")
    print(f"  Worst Reward: {results['min_reward']:.2f}")
    print(f"  Mean Length: {results['mean_length']:.1f} steps")
    print(f"  Survival Rate: {results['survival_rate']:.1%}")
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    plot_path, results_path = evaluator.create_visualizations(results, args.output_dir)
    
    # Provide recommendations
    print("\n💡 PERFORMANCE ANALYSIS:")
    
    if results['mean_reward'] > 50:
        print("  ✅ Excellent performance! The model is performing very well.")
    elif results['mean_reward'] > 20:
        print("  👍 Good performance. The model has learned effective strategies.")
    elif results['mean_reward'] > 0:
        print("  ⚠️ Moderate performance. There's room for improvement.")
    else:
        print("  ❌ Poor performance. The model may need more training.")
        
    if results['survival_rate'] > 0.7:
        print("  ✅ High survival rate - agent avoids early death well.")
    elif results['survival_rate'] > 0.3:
        print("  ⚠️ Moderate survival rate - agent sometimes dies early.")
    else:
        print("  ❌ Low survival rate - agent frequently dies early.")
        
    # Action diversity analysis
    total_actions = sum(evaluator.action_frequencies.values())
    unique_actions = len(evaluator.action_frequencies)
    action_diversity = unique_actions / 23  # Total possible actions
    
    print(f"  📊 Action Diversity: {action_diversity:.1%} ({unique_actions}/23 actions used)")
    
    if action_diversity > 0.7:
        print("  ✅ High action diversity - agent uses varied strategies.")
    elif action_diversity > 0.4:
        print("  ⚠️ Moderate action diversity - agent has some preferred actions.")
    else:
        print("  ❌ Low action diversity - agent may be stuck in limited behaviors.")
    
    env.close()
    print(f"\n🎉 Evaluation complete! Check '{args.output_dir}' for detailed results.")


if __name__ == "__main__":
    main()