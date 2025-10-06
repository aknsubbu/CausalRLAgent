# inspect_nethack_observations.py
import gymnasium as gym
import nle
import numpy as np

def inspect_observations():
    """Detailed inspection of NetHack observations to understand the structure"""
    
    print("=== NetHack Observation Analysis ===")
    
    env = gym.make("NetHackScore-v0")
    obs, info = env.reset()
    
    print(f"\nObservation space type: {type(env.observation_space)}")
    print(f"Observation space: {env.observation_space}")
    
    if hasattr(env.observation_space, 'spaces'):
        print(f"\nDict observation with {len(env.observation_space.spaces)} components:")
        for key, space in env.observation_space.spaces.items():
            print(f"  {key}: {space}")
    
    print(f"\nActual observation type: {type(obs)}")
    
    if isinstance(obs, dict):
        print(f"\nObservation components ({len(obs)} keys):")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}:")
                print(f"    Shape: {value.shape}")
                print(f"    Dtype: {value.dtype}")
                print(f"    Range: {value.min()} to {value.max()}")
                if value.size < 50:  # Only print small arrays
                    print(f"    Sample values: {value.flatten()[:10]}")
            else:
                print(f"  {key}: {type(value)} = {value}")
    
    print(f"\nAction space: {env.action_space}")
    print(f"Action space size: {env.action_space.n}")
    
    # Test a few actions to see observation changes
    print(f"\nTesting a few random actions:")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Action {i}: {action}, Reward: {reward}, Done: {terminated or truncated}")
        
        if terminated or truncated:
            print("  Episode ended, resetting...")
            obs, info = env.reset()
            break
    
    env.close()
    print("\n=== Analysis Complete ===")

def test_policy_compatibility():
    """Test which policy types work with NetHack"""
    
    print("\n=== Policy Compatibility Test ===")
    
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3 import PPO
    
    env = make_vec_env("NetHackScore-v0", n_envs=1)
    
    policies_to_test = [
        ("MultiInputPolicy", "For dict observation spaces"),
        ("CnnPolicy", "For image-like observations"),
        ("MlpPolicy", "For vector observations")
    ]
    
    for policy_name, description in policies_to_test:
        try:
            print(f"\nTesting {policy_name} ({description})...")
            model = PPO(policy_name, env, verbose=0)
            print(f"  ✅ {policy_name} works!")
            
            # Quick test - single step
            obs = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            print(f"  ✅ Prediction works, action: {action}")
            
            # Clean up
            del model
            
        except Exception as e:
            print(f"  ❌ {policy_name} failed: {str(e)[:100]}...")
    
    env.close()
    print("\n=== Policy Test Complete ===")

if __name__ == "__main__":
    inspect_observations()
    test_policy_compatibility()