import gymnasium as gym
import nle
import numpy as np

def explore_nethack_environment():
    """Understand what NetHack gives us"""
    print("=== NetHack Learning Environment Demo ===\n")
    
    # Create environment
    env = gym.make("NetHackScore-v0")
    obs, info = env.reset()
    
    print("✅ Environment created successfully!")
    print(f"Action space size: {env.action_space.n}")
    print(f"Available observations: {list(obs.keys())}")
    
    # Explore observation structure
    print(f"\nObservation details:")
    print(f"- Screen (glyphs): {obs['glyphs'].shape}")
    print(f"- Stats array (blstats): {obs['blstats'].shape}")  # Fixed: use 'blstats'
    print(f"- Sample stats: {obs['blstats'][:15]}")  # First 15 stats
    print(f"- Characters: {obs['chars'].shape}")
    print(f"- Colors: {obs['colors'].shape}")
    
    # Take a few random actions to see what happens
    print(f"\n=== Taking 5 random actions ===")
    total_reward = 0
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Extract key info using correct field name
        blstats = obs['blstats']
        hp = blstats[10] if len(blstats) > 10 else 0      # HP
        max_hp = blstats[11] if len(blstats) > 11 else 0  # Max HP
        depth = blstats[12] if len(blstats) > 12 else 1   # Dungeon level
        gold = blstats[13] if len(blstats) > 13 else 0    # Gold
        
        print(f"Step {step+1}: Action={action}, Reward={reward}")
        print(f"  -> HP={hp}/{max_hp}, Depth={depth}, Gold={gold}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step+1}")
            break
    
    print(f"\nTotal reward: {total_reward}")
    print(f"Final stats sample: {obs['blstats'][:20]}")  # Show more stats
    
    env.close()
    print("✅ Environment exploration complete!")

if __name__ == "__main__":
    explore_nethack_environment()