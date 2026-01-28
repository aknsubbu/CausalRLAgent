#!/usr/bin/env python3
"""
Quick test to verify the EnhancedNetHackPPOAgent can be instantiated properly.
"""

try:
    from improved_reward_shaping import EnhancedNetHackPPOAgent
    
    print("🧪 Testing EnhancedNetHackPPOAgent instantiation...")
    
    # Try creating the agent with the parameters that were causing the error
    agent = EnhancedNetHackPPOAgent(
        action_dim=23,
        learning_rate=1e-4,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.02,
        value_coef=0.5,
        max_grad_norm=0.5,
        use_wandb=False
    )
    
    print("✅ SUCCESS! Agent created successfully!")
    print(f"  Device: {agent.device}")
    print(f"  Entropy Coefficient: {agent.entropy_coef}")
    print(f"  Value Coefficient: {agent.value_coef}")
    print(f"  Max Grad Norm: {agent.max_grad_norm}")
    print(f"  Use Wandb: {agent.use_wandb}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()