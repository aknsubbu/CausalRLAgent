#!/usr/bin/env python3
"""
ML Presentation Graphs Generator
Creates professional visualizations for technical ML presentations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional presentations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path("ml_presentation_graphs")
output_dir.mkdir(exist_ok=True)

# Load the training data
try:
    episodes_df = pd.read_csv('training_logs/nethack_ppo_20251014_142607_episodes.csv')
    training_df = pd.read_csv('training_logs/nethack_ppo_20251014_142607_training.csv')
    extensive_df = pd.read_csv('training_logs/nethack_ppo_20251014_142607_extensive.csv')
    print("✅ Data loaded successfully")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Configure matplotlib for high-quality output
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_reward_learning_curve():
    """Create the main learning curve with moving averages"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Raw rewards with moving average
    episodes = episodes_df['episode'].values
    raw_rewards = episodes_df['raw_reward'].values
    shaped_rewards = episodes_df['shaped_reward'].values
    
    # Calculate moving averages
    window = 10
    raw_ma = pd.Series(raw_rewards).rolling(window=window, center=True).mean()
    shaped_ma = pd.Series(shaped_rewards).rolling(window=window, center=True).mean()
    
    # Raw rewards
    ax1.plot(episodes, raw_rewards, alpha=0.3, color='steelblue', linewidth=1, label='Raw Rewards')
    ax1.plot(episodes, raw_ma, color='steelblue', linewidth=3, label=f'{window}-Episode MA')
    
    # Highlight peak performance
    peak_idx = np.argmax(raw_rewards)
    ax1.scatter(episodes[peak_idx], raw_rewards[peak_idx], color='red', s=100, zorder=5, 
               label=f'Peak: {raw_rewards[peak_idx]:.1f} (Episode {episodes[peak_idx]})')
    
    ax1.set_title('NetHack PPO Learning Curve - Raw Rewards', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Raw Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Shaped rewards
    ax2.plot(episodes, shaped_rewards, alpha=0.3, color='darkorange', linewidth=1, label='Shaped Rewards')
    ax2.plot(episodes, shaped_ma, color='darkorange', linewidth=3, label=f'{window}-Episode MA')
    
    ax2.set_title('Reward Shaping Effect on Training Signal', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Shaped Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curve.png')
    plt.close()
    print("✅ Learning curve created")

def create_training_dynamics():
    """Create training dynamics visualization (losses, entropy)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert step to thousands for readability
    steps = training_df['step'].values / 1000
    
    # Actor Loss
    actor_loss = training_df['actor_loss'].values
    ax1.plot(steps, actor_loss, color='purple', linewidth=2, marker='o', markersize=3)
    ax1.set_title('Actor Network Loss', fontweight='bold')
    ax1.set_xlabel('Training Steps (K)')
    ax1.set_ylabel('Actor Loss')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(steps, actor_loss, 1)
    p = np.poly1d(z)
    ax1.plot(steps, p(steps), "--", alpha=0.8, color='red', 
             label=f'Trend: {z[0]:.5f}x + {z[1]:.3f}')
    ax1.legend()
    
    # Critic Loss
    critic_loss = training_df['critic_loss'].values
    ax2.plot(steps, critic_loss, color='green', linewidth=2, marker='s', markersize=3)
    ax2.set_title('Critic Network Loss', fontweight='bold')
    ax2.set_xlabel('Training Steps (K)')
    ax2.set_ylabel('Critic Loss')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(steps, critic_loss, 1)
    p = np.poly1d(z)
    ax2.plot(steps, p(steps), "--", alpha=0.8, color='red',
             label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')
    ax2.legend()
    
    # Entropy (exploration)
    entropy = training_df['entropy'].values
    ax3.plot(steps, entropy, color='orange', linewidth=2, marker='^', markersize=3)
    ax3.set_title('Policy Entropy (Exploration)', fontweight='bold')
    ax3.set_xlabel('Training Steps (K)')
    ax3.set_ylabel('Entropy')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=np.mean(entropy), color='red', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(entropy):.3f}')
    ax3.legend()
    
    # Policy vs Value Loss
    policy_loss = training_df['policy_loss'].values
    value_loss = training_df['value_loss'].values
    ax4.plot(steps, policy_loss, color='blue', linewidth=2, label='Policy Loss', marker='o', markersize=3)
    ax4.plot(steps, value_loss, color='red', linewidth=2, label='Value Loss', marker='s', markersize=3)
    ax4.set_title('Policy vs Value Learning', fontweight='bold')
    ax4.set_xlabel('Training Steps (K)')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_dynamics.png')
    plt.close()
    print("✅ Training dynamics created")

def create_performance_phases():
    """Create performance phases analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define phases
    phases = [
        (0, 25, "Initial Learning"),
        (25, 50, "Optimization"), 
        (50, 75, "Peak Performance"),
        (75, 100, "Consolidation")
    ]
    
    phase_stats = []
    for start, end, name in phases:
        phase_data = episodes_df[(episodes_df['episode'] >= start) & (episodes_df['episode'] < end)]
        stats = {
            'phase': name,
            'mean_reward': phase_data['raw_reward'].mean(),
            'std_reward': phase_data['raw_reward'].std(),
            'mean_length': phase_data['episode_length'].mean(),
            'exploration_eff': phase_data['unique_positions'].mean() / phase_data['episode_length'].mean()
        }
        phase_stats.append(stats)
    
    phase_df = pd.DataFrame(phase_stats)
    
    # Performance by phase
    x_pos = np.arange(len(phase_df))
    bars = ax1.bar(x_pos, phase_df['mean_reward'], yerr=phase_df['std_reward'], 
                   capsize=5, alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax1.set_title('Performance by Training Phase', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Training Phase')
    ax1.set_ylabel('Mean Raw Reward ± Std')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(phase_df['phase'], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, phase_df['mean_reward'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Exploration efficiency by phase
    bars2 = ax2.bar(x_pos, phase_df['exploration_eff'], alpha=0.8, 
                    color=['steelblue', 'crimson', 'forestgreen', 'orange'])
    ax2.set_title('Exploration Efficiency by Phase', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Training Phase')
    ax2.set_ylabel('Unique Positions / Episode Length')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(phase_df['phase'], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, phase_df['exploration_eff'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_phases.png')
    plt.close()
    print("✅ Performance phases created")

def create_extensive_metrics_heatmap():
    """Create heatmap of extensive metrics over time"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Select key metrics for heatmap
    metrics_cols = ['avg_raw_reward', 'avg_shaped_reward', 'success_rate', 'survival_rate', 
                   'exploration_efficiency', 'avg_actor_loss', 'avg_critic_loss', 'avg_entropy']
    
    # Normalize the data for better visualization
    heatmap_data = extensive_df[metrics_cols].copy()
    for col in metrics_cols:
        heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())
    
    # Create heatmap
    sns.heatmap(heatmap_data.T, 
                xticklabels=[f"Episodes {i*10+1}-{(i+1)*10}" for i in range(len(extensive_df))],
                yticklabels=['Avg Raw Reward', 'Avg Shaped Reward', 'Success Rate', 'Survival Rate',
                           'Exploration Eff.', 'Actor Loss', 'Critic Loss', 'Entropy'],
                cmap='RdYlBu_r', annot=True, fmt='.2f', cbar_kws={'label': 'Normalized Value'},
                ax=ax)
    
    ax.set_title('Training Metrics Evolution (Normalized)', fontweight='bold', fontsize=16)
    ax.set_xlabel('Training Batches')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png')
    plt.close()
    print("✅ Metrics heatmap created")

def create_reward_distribution():
    """Create reward distribution analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    raw_rewards = episodes_df['raw_reward'].values
    shaped_rewards = episodes_df['shaped_reward'].values
    
    # Reward histogram
    ax1.hist(raw_rewards, bins=20, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    ax1.axvline(np.mean(raw_rewards), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(raw_rewards):.1f}')
    ax1.axvline(np.median(raw_rewards), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(raw_rewards):.1f}')
    ax1.set_title('Raw Reward Distribution', fontweight='bold')
    ax1.set_xlabel('Raw Reward')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax2.boxplot([raw_rewards, shaped_rewards], labels=['Raw Rewards', 'Shaped Rewards'])
    ax2.set_title('Reward Distribution Comparison', fontweight='bold')
    ax2.set_ylabel('Reward Value')
    ax2.grid(True, alpha=0.3)
    
    # Cumulative rewards
    cumulative_raw = np.cumsum(raw_rewards)
    cumulative_shaped = np.cumsum(shaped_rewards)
    ax3.plot(episodes_df['episode'], cumulative_raw, color='blue', linewidth=2, label='Raw Rewards')
    ax3.plot(episodes_df['episode'], cumulative_shaped, color='orange', linewidth=2, label='Shaped Rewards')
    ax3.set_title('Cumulative Reward Progression', fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cumulative Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Q-Q plot for normality
    from scipy import stats
    stats.probplot(raw_rewards, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot: Raw Rewards vs Normal Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_analysis.png')
    plt.close()
    print("✅ Reward distribution analysis created")

def create_model_checkpoints_analysis():
    """Analyze performance of saved model checkpoints"""
    # Define checkpoint episodes and their rewards
    checkpoints = [
        (0, 81.84, "Initial Model"),
        (14, 86.36, "Early Improvement"), 
        (76, 112.48, "Peak Performance"),
        (99, episodes_df.iloc[-1]['raw_reward'], "Final Model")
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Checkpoint performance
    episodes_cp, rewards_cp, labels = zip(*checkpoints)
    colors = ['lightblue', 'yellow', 'gold', 'lightgreen']
    
    bars = ax1.bar(range(len(checkpoints)), rewards_cp, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Model Checkpoint Performance', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Model Checkpoint')
    ax1.set_ylabel('Reward')
    ax1.set_xticks(range(len(checkpoints)))
    ax1.set_xticklabels([f"{label}\n(Ep {ep})" for ep, _, label in checkpoints])
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, reward in zip(bars, rewards_cp):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{reward:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Performance improvement over time
    ax2.plot(episodes_cp, rewards_cp, 'o-', linewidth=3, markersize=10, color='darkgreen')
    ax2.set_title('Checkpoint Performance Trajectory', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.grid(True, alpha=0.3)
    
    # Annotate points
    for ep, reward, label in checkpoints:
        ax2.annotate(f'{label}\n{reward:.1f}', (ep, reward), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'checkpoint_analysis.png')
    plt.close()
    print("✅ Checkpoint analysis created")

def create_comprehensive_dashboard():
    """Create a comprehensive dashboard view"""
    fig = plt.figure(figsize=(20, 16))
    
    # Create subplot grid
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main learning curve (top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    episodes = episodes_df['episode'].values
    raw_rewards = episodes_df['raw_reward'].values
    window = 10
    raw_ma = pd.Series(raw_rewards).rolling(window=window, center=True).mean()
    
    ax1.plot(episodes, raw_rewards, alpha=0.4, color='steelblue', linewidth=1)
    ax1.plot(episodes, raw_ma, color='steelblue', linewidth=3, label='10-Episode MA')
    peak_idx = np.argmax(raw_rewards)
    ax1.scatter(episodes[peak_idx], raw_rewards[peak_idx], color='red', s=100, zorder=5)
    ax1.set_title('Learning Progress', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Raw Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Key metrics summary (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    metrics_text = f"""
    KEY METRICS
    
    Peak Reward: {np.max(raw_rewards):.1f}
    Mean Reward: {np.mean(raw_rewards):.1f}
    Std Reward: {np.std(raw_rewards):.1f}
    
    Episodes: {len(episodes)}
    Training Steps: {training_df['step'].iloc[-1]:,}
    
    Best Model: Episode {episodes[peak_idx]}
    Improvement: +{((np.max(raw_rewards) - raw_rewards[0])/raw_rewards[0]*100):.1f}%
    """
    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. Training losses (second row, left)
    ax3 = fig.add_subplot(gs[1, 0])
    steps = training_df['step'].values / 1000
    ax3.plot(steps, training_df['actor_loss'], color='purple', linewidth=2, label='Actor')
    ax3.plot(steps, training_df['critic_loss'], color='green', linewidth=2, label='Critic')
    ax3.set_title('Training Losses', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Entropy (second row, middle)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(steps, training_df['entropy'], color='orange', linewidth=2)
    ax4.set_title('Policy Entropy', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Entropy')
    ax4.grid(True, alpha=0.3)
    
    # 5. Episode lengths (second row, right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(episodes, episodes_df['episode_length'], color='brown', alpha=0.7)
    ax5.set_title('Episode Lengths', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Steps')
    ax5.grid(True, alpha=0.3)
    
    # 6. Reward distribution (third row, left)
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(raw_rewards, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax6.axvline(np.mean(raw_rewards), color='red', linestyle='--', linewidth=2)
    ax6.set_title('Reward Distribution', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Reward')
    
    # 7. Exploration metrics (third row, middle)
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(episodes, episodes_df['unique_positions'], color='green', alpha=0.7)
    ax7.set_title('Exploration (Unique Positions)', fontweight='bold', fontsize=12)
    ax7.set_ylabel('Positions')
    ax7.grid(True, alpha=0.3)
    
    # 8. Success metrics (third row, right)
    ax8 = fig.add_subplot(gs[2, 2])
    success_rates = extensive_df['success_rate'].values
    exploration_effs = extensive_df['exploration_efficiency'].values
    batch_labels = [f"{i*10+1}-{(i+1)*10}" for i in range(len(extensive_df))]
    
    x_pos = np.arange(len(success_rates))
    ax8.bar(x_pos, success_rates, alpha=0.7, color='lightcoral')
    ax8.set_title('Success Rate by Batch', fontweight='bold', fontsize=12)
    ax8.set_ylabel('Success Rate')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels([f"{i*10+1}-{(i+1)*10}" for i in range(len(extensive_df))], rotation=45)
    
    # 9. Model checkpoints (bottom, spans all columns)
    ax9 = fig.add_subplot(gs[3, :])
    checkpoints_ep = [0, 14, 76, 99]
    checkpoints_reward = [81.84, 86.36, 112.48, episodes_df.iloc[-1]['raw_reward']]
    
    ax9.plot(episodes, raw_rewards, alpha=0.3, color='gray', label='All Episodes')
    ax9.scatter(checkpoints_ep, checkpoints_reward, color='red', s=150, zorder=5, 
               label='Saved Models', edgecolor='black', linewidth=2)
    
    for ep, reward in zip(checkpoints_ep, checkpoints_reward):
        ax9.annotate(f'Model\n{reward:.1f}', (ep, reward), 
                    xytext=(0, 20), textcoords='offset points',
                    ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax9.set_title('Model Checkpoints Performance', fontweight='bold', fontsize=14)
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Reward')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('NetHack PPO Training - Comprehensive Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_dashboard.png')
    plt.close()
    print("✅ Comprehensive dashboard created")

def create_summary_statistics():
    """Create a summary statistics visualization"""
    print("\n📊 TRAINING SUMMARY STATISTICS")
    print("=" * 50)
    
    # Episode statistics
    raw_rewards = episodes_df['raw_reward'].values
    shaped_rewards = episodes_df['shaped_reward'].values
    episode_lengths = episodes_df['episode_length'].values
    
    print(f"📈 REWARD STATISTICS:")
    print(f"   Raw Rewards - Mean: {np.mean(raw_rewards):.2f} ± {np.std(raw_rewards):.2f}")
    print(f"   Raw Rewards - Range: [{np.min(raw_rewards):.2f}, {np.max(raw_rewards):.2f}]")
    print(f"   Peak Episode: {np.argmax(raw_rewards)} (Reward: {np.max(raw_rewards):.2f})")
    print(f"   Improvement: +{((np.max(raw_rewards) - raw_rewards[0])/raw_rewards[0]*100):.1f}% over baseline")
    
    print(f"\n🎮 EPISODE STATISTICS:")
    print(f"   Total Episodes: {len(episodes_df)}")
    print(f"   Mean Length: {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f} steps")
    print(f"   Total Steps: {np.sum(episode_lengths):,}")
    
    print(f"\n🧠 TRAINING STATISTICS:")
    print(f"   Training Updates: {len(training_df)}")
    print(f"   Final Actor Loss: {training_df['actor_loss'].iloc[-1]:.4f}")
    print(f"   Final Critic Loss: {training_df['critic_loss'].iloc[-1]:.4f}")
    print(f"   Mean Entropy: {np.mean(training_df['entropy']):.4f}")
    
    print(f"\n🎯 MODEL CHECKPOINTS:")
    checkpoints = [(0, 81.84), (14, 86.36), (76, 112.48)]
    for ep, reward in checkpoints:
        print(f"   Episode {ep:2d}: {reward:.2f} reward")

def main():
    """Generate all visualizations"""
    print("🎨 Generating ML Presentation Graphs...")
    print("=" * 50)
    
    # Create all visualizations
    create_reward_learning_curve()
    create_training_dynamics() 
    create_performance_phases()
    create_extensive_metrics_heatmap()
    create_reward_distribution()
    create_model_checkpoints_analysis()
    create_comprehensive_dashboard()
    create_summary_statistics()
    
    print("\n" + "=" * 50)
    print("🎉 ALL GRAPHS GENERATED SUCCESSFULLY!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print("\n📊 Generated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"   ✅ {file.name}")
    
    print("\n💡 Usage for ML Presentations:")
    print("   • learning_curve.png - Main results slide")
    print("   • training_dynamics.png - Technical deep dive")
    print("   • comprehensive_dashboard.png - Overview slide")
    print("   • checkpoint_analysis.png - Model progression")
    print("   • performance_phases.png - Training analysis")
    print("   • metrics_heatmap.png - Advanced analytics")

if __name__ == "__main__":
    main()