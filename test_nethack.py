# test_nethack.py
import gymnasium as gym
import nle

# List available NetHack environments
print("Available NetHack environments:")
for env_id in gym.envs.registry.keys():
    if 'NetHack' in env_id:
        print(f"  - {env_id}")

# Test basic environment creation - use NetHackScore-v0 which is more stable
try:
    env = gym.make("NetHackScore-v0")
    obs, info = env.reset()
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")
    
    # Print observation structure
    if isinstance(obs, dict):
        print(f"\nObservation keys: {list(obs.keys())}")
        for key, value in obs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Run a few random steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        print(f"Step {i}: Action={action}, Reward={reward}, Done={done}")
        
        if done:
            print(f"Episode ended. Total reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0
            break
    
    env.close()
    print("\nNetHack environment test successful!")

except Exception as e:
    print(f"Error creating NetHack environment: {e}")
    print("\nTrying to install NLE properly...")
    print("Run: pip uninstall nle && pip install nle")