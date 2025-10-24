#!/usr/bin/env python3
"""
Create a simple visual diagram of the neural network architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_architecture_diagram():
    """Create a visual diagram of the neural network architecture"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',
        'processing': '#B8E6B8', 
        'fusion': '#FFE5B4',
        'output': '#FFB6C1'
    }
    
    # Title
    ax.text(5, 9.5, 'NetHack PPO Neural Network Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Input layer boxes
    inputs = [
        ('Glyphs\n21×79\n(Visual)', 0.5, 7.5),
        ('Stats\n26D\n(Numbers)', 2.5, 7.5), 
        ('Messages\n256D\n(Text)', 4.5, 7.5),
        ('Inventory\n55D\n(Items)', 6.5, 7.5),
        ('Actions\n50D\n(History)', 8.5, 7.5)
    ]
    
    for text, x, y in inputs:
        box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                           boxstyle="round,pad=0.1", 
                           facecolor=colors['input'],
                           edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Processing layer
    processors = [
        ('CNN+LSTM\n→ 256D', 0.5, 6),
        ('LSTM\n→ 64D', 2.5, 6),
        ('Dense\n→ 128D', 4.5, 6), 
        ('Dense\n→ 64D', 6.5, 6),
        ('Dense\n→ 32D', 8.5, 6)
    ]
    
    for text, x, y in processors:
        box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['processing'], 
                           edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9)
        
        # Draw arrows from inputs to processors
        ax.arrow(x, 7.2, 0, -0.9, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Fusion layer
    fusion_box = FancyBboxPatch((2, 4.2), 6, 0.6,
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['fusion'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(5, 4.5, 'Feature Fusion: Concatenate All → 544D → 512D → 256D\n(Shared Representation)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw arrows from processors to fusion
    for _, x, y in processors:
        ax.arrow(x, 5.7, (5-x)*0.3, -0.8, head_width=0.08, head_length=0.08, fc='blue', ec='blue')
    
    # Output heads
    actor_box = FancyBboxPatch((1.5, 2.2), 3, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['output'],
                             edgecolor='black', linewidth=2)
    ax.add_patch(actor_box)
    ax.text(3, 2.6, 'ACTOR NETWORK\n(Policy)\n256D → 23D\n"What action to take?"', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    critic_box = FancyBboxPatch((5.5, 2.2), 3, 0.8, 
                              boxstyle="round,pad=0.1",
                              facecolor=colors['output'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(critic_box)
    ax.text(7, 2.6, 'CRITIC NETWORK\n(Value Function)\n256D → 1D\n"How good is this state?"',
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw arrows from fusion to outputs
    ax.arrow(3.5, 4.2, -0.3, -1.1, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax.arrow(6.5, 4.2, 0.3, -1.1, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
    
    # Final outputs
    ax.text(3, 1.2, 'Action Probabilities\n[0.3, 0.2, 0.1, ...]', 
            ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    ax.text(7, 1.2, 'State Value\nScore: +15.7', 
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Parameter counts
    ax.text(9.5, 8.5, 'Parameter Counts:', fontsize=12, fontweight='bold')
    ax.text(9.5, 8.1, 'Total: 4.2M', fontsize=11, color='red', fontweight='bold')
    ax.text(9.5, 7.8, 'Actor: 2.1M', fontsize=10)
    ax.text(9.5, 7.5, 'Critic: 2.1M', fontsize=10)
    
    # Memory components
    ax.text(0.5, 0.8, 'Memory Components:', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.5, '• LSTM hidden states', fontsize=9)
    ax.text(0.5, 0.3, '• Position history (100 steps)', fontsize=9)
    ax.text(0.5, 0.1, '• Action sequences (50 steps)', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/Users/kishore/CausalRLAgent/october_14_training/ml_presentation_graphs/neural_architecture_diagram.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Neural architecture diagram created!")

def create_data_flow_diagram():
    """Create a simplified data flow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(5, 7.5, 'Data Flow: From Game to Decision', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Step boxes
    steps = [
        ('1. NetHack\nGame State', 1, 6, '#FFE5E5'),
        ('2. Multi-Modal\nProcessing', 3, 6, '#E5F2FF'),
        ('3. Feature\nFusion', 5, 6, '#E5FFE5'), 
        ('4. Actor/Critic\nHeads', 7, 6, '#FFF5E5'),
        ('5. Action +\nValue', 9, 6, '#F5E5FF')
    ]
    
    for i, (text, x, y, color) in enumerate(steps):
        box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor=color,
                           edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(steps) - 1:
            ax.arrow(x+0.6, y, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Details for each step
    details = [
        ('Visual: 21×79 grid\nStats: 26 numbers\nText: messages\nItems: inventory\nMemory: actions', 1, 4.5),
        ('CNN for vision\nLSTM for memory\nFC for text/items\nDimension reduction', 3, 4.5),
        ('Concatenate features\n544D → 256D\nShared representation', 5, 4.5),
        ('Policy network\nValue network\nBoth use same features', 7, 4.5),
        ('23 action probs\n1 value score\nPPO training', 9, 4.5)
    ]
    
    for text, x, y in details:
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Key insights
    ax.text(5, 2.5, 'Key Architectural Insights', fontsize=14, fontweight='bold', ha='center')
    
    insights = [
        '🧠 Multi-modal processing handles different information types',
        '🔄 LSTM provides memory for temporal dependencies', 
        '🔗 Shared features ensure Actor and Critic understand world consistently',
        '⚖️ Actor-Critic design balances exploration and exploitation',
        '📈 4.2M parameters provide sufficient capacity for complex game'
    ]
    
    for i, insight in enumerate(insights):
        ax.text(1, 2 - i*0.3, insight, fontsize=11, ha='left')
    
    plt.tight_layout()
    plt.savefig('/Users/kishore/CausalRLAgent/october_14_training/ml_presentation_graphs/data_flow_diagram.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Data flow diagram created!")

if __name__ == "__main__":
    print("🎨 Creating neural network architecture diagrams...")
    create_architecture_diagram()
    create_data_flow_diagram()
    print("📁 Diagrams saved to ml_presentation_graphs/")
    print("   • neural_architecture_diagram.png")
    print("   • data_flow_diagram.png")