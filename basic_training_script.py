# train_basic_agent.py
import gymnasium as gym
import nle
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

def main():
    # Create environment - start with simpler NetHackScore
    env_id = "NetHackScore-v0"  
    
    # Create vectorized environment (for parallel training)
    env = make_vec_env(env_id, n_envs=2)  # Reduced from 4 to 2 for stability
    
    # Create evaluation environment
    eval_env = gym.make(env_id)
    
    # Set up logging
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    # Create PPO agent
    model = PPO(
        "CnnPolicy",  # CNN policy for visual observations
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.00025,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        device="auto"  # Use GPU if available
    )
    
    print("Starting training...")
    # Train the agent
    model.learn(
        total_timesteps=50000,  # Reduced for initial testing
        callback=eval_callback,
        tb_log_name="ppo_nethack"
    )
    
    # Save the final model
    model.save("ppo_nethack_basic")
    print("Training completed!")

if __name__ == "__main__":
    main()