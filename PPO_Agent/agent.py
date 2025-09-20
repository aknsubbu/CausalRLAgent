import gymnasium as gym
import nle.env  # This registers the NetHack environments
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os

class NetHackObservationProcessor:
    """Processes NetHack observations into a format suitable for neural networks"""
    
    def __init__(self):
        # NetHack observation dimensions
        self.glyph_shape = (21, 79)  # Standard NetHack view size
        self.stats_dim = 26  # bl_stats dimension
        self.message_dim = 256  # Max message length
        self.inventory_dim = 55  # Inventory slots
        
    def process_observation(self, obs):
        """
        Convert raw NetHack observation to processed tensors
        Returns: dict with processed components
        """
        processed = {}
        
        # Handle tuple format from gymnasium/NLE - extract the main observation dict
        if isinstance(obs, tuple):
            # The first element is usually the main observation
            obs = obs[0]
        
        # Now obs should be a dictionary
        if not isinstance(obs, dict):
            raise ValueError(f"Expected dict observation, got {type(obs)}")
        
        # Process glyphs (main game view) - normalize to 0-1
        if 'glyphs' in obs:
            glyphs = np.array(obs['glyphs']).astype(np.float32) / 5976.0  # Max glyph value
            processed['glyphs'] = glyphs
        else:
            # Fallback: create dummy glyphs if not available
            processed['glyphs'] = np.zeros(self.glyph_shape, dtype=np.float32)
        
        # Process stats (health, level, etc.)
        if 'blstats' in obs:
            stats = np.array(obs['blstats']).astype(np.float32)
            # Normalize some key stats safely
            stats_normalized = stats.copy()
            if len(stats) > 1 and stats[1] > 0:
                stats_normalized[0] = stats[0] / stats[1]  # HP ratio
            if len(stats) > 7:
                stats_normalized[7] = min(stats[7] / 30.0, 1.0)  # Level (cap at 30)
            
            # Pad or truncate to expected size
            if len(stats_normalized) < self.stats_dim:
                padded_stats = np.zeros(self.stats_dim, dtype=np.float32)
                padded_stats[:len(stats_normalized)] = stats_normalized
                processed['stats'] = padded_stats
            else:
                processed['stats'] = stats_normalized[:self.stats_dim]
        else:
            processed['stats'] = np.zeros(self.stats_dim, dtype=np.float32)
        
        # Process message (recent game messages)
        if 'message' in obs:
            message = np.array(obs['message']).astype(np.float32)
            # Ensure message is the right size
            if len(message) < self.message_dim:
                padded_message = np.zeros(self.message_dim, dtype=np.float32)
                padded_message[:len(message)] = message / 255.0
                processed['message'] = padded_message
            else:
                processed['message'] = message[:self.message_dim] / 255.0
        else:
            processed['message'] = np.zeros(self.message_dim, dtype=np.float32)
        
        # Process inventory
        if 'inv_strs' in obs:
            inventory = obs['inv_strs']  # String representation
            # Convert to simple numerical representation (simplified for now)
            inv_features = np.zeros(self.inventory_dim, dtype=np.float32)
            for i, item in enumerate(inventory):
                if i < len(inv_features):
                    # Check if item exists and has content
                    item_str = str(item) if item is not None else ""
                    if len(item_str.strip()) > 0 and item_str.strip() != "b''":
                        inv_features[i] = 1.0  # Item present
            processed['inventory'] = inv_features
        else:
            processed['inventory'] = np.zeros(self.inventory_dim, dtype=np.float32)
        
        return processed

class NetHackCNN(nn.Module):
    """CNN for processing NetHack glyphs"""
    
    def __init__(self, input_shape=(21, 79), output_dim=512):
        super(NetHackCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        conv_out_size = self._get_conv_out_size(input_shape)
        self.fc = nn.Linear(conv_out_size, output_dim)
        
    def _get_conv_out_size(self, shape):
        """Calculate output size after convolutions"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *shape)
            dummy_output = self._forward_conv(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        return x
    
    def forward(self, x):
        # Add channel dimension if not present
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class PPOActor(nn.Module):
    """PPO Actor network for NetHack"""
    
    def __init__(self, action_dim=23):  # Updated to match your environment's action space
        super(PPOActor, self).__init__()
        
        # Feature extractors
        self.glyph_cnn = NetHackCNN(output_dim=512)
        self.stats_fc = nn.Linear(26, 128)
        self.message_fc = nn.Linear(256, 128)
        self.inventory_fc = nn.Linear(55, 64)
        
        # Combined feature processing
        combined_dim = 512 + 128 + 128 + 64  # 832
        self.combined_fc1 = nn.Linear(combined_dim, 512)
        self.combined_fc2 = nn.Linear(512, 256)
        
        # Action head
        self.action_head = nn.Linear(256, action_dim)
        
    def forward(self, obs):
        # Process each observation component
        glyph_features = self.glyph_cnn(obs['glyphs'])
        stats_features = F.relu(self.stats_fc(obs['stats']))
        message_features = F.relu(self.message_fc(obs['message']))
        inventory_features = F.relu(self.inventory_fc(obs['inventory']))
        
        # Combine all features
        combined = torch.cat([
            glyph_features, stats_features, 
            message_features, inventory_features
        ], dim=1)
        
        # Process combined features
        x = F.relu(self.combined_fc1(combined))
        x = F.relu(self.combined_fc2(x))
        
        # Output action logits
        action_logits = self.action_head(x)
        return action_logits

class PPOCritic(nn.Module):
    """PPO Critic network for NetHack"""
    
    def __init__(self):
        super(PPOCritic, self).__init__()
        
        # Feature extractors (same as actor)
        self.glyph_cnn = NetHackCNN(output_dim=512)
        self.stats_fc = nn.Linear(26, 128)
        self.message_fc = nn.Linear(256, 128)
        self.inventory_fc = nn.Linear(55, 64)
        
        # Combined feature processing
        combined_dim = 512 + 128 + 128 + 64
        self.combined_fc1 = nn.Linear(combined_dim, 512)
        self.combined_fc2 = nn.Linear(512, 256)
        
        # Value head
        self.value_head = nn.Linear(256, 1)
        
    def forward(self, obs):
        # Same feature extraction as actor
        glyph_features = self.glyph_cnn(obs['glyphs'])
        stats_features = F.relu(self.stats_fc(obs['stats']))
        message_features = F.relu(self.message_fc(obs['message']))
        inventory_features = F.relu(self.inventory_fc(obs['inventory']))
        
        combined = torch.cat([
            glyph_features, stats_features, 
            message_features, inventory_features
        ], dim=1)
        
        x = F.relu(self.combined_fc1(combined))
        x = F.relu(self.combined_fc2(x))
        
        value = self.value_head(x)
        return value

class PPOBuffer:
    """Buffer for storing PPO training data"""
    
    def __init__(self, max_size=2048):
        self.max_size = max_size
        self.clear()
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages(self, gamma=0.99, lam=0.95):
        """Compute GAE advantages"""
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]
            
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + gamma * lam * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batch(self, batch_size):
        """Get random batch for training"""
        indices = np.random.choice(len(self.observations), batch_size, replace=False)
        
        batch_obs = {}
        for key in self.observations[0].keys():
            batch_obs[key] = torch.stack([self.observations[i][key] for i in indices])
        
        batch_actions = torch.tensor([self.actions[i] for i in indices])
        batch_log_probs = torch.tensor([self.log_probs[i] for i in indices])
        batch_returns = torch.tensor([self.returns[i] for i in indices])
        batch_advantages = torch.tensor([self.advantages[i] for i in indices])
        
        return batch_obs, batch_actions, batch_log_probs, batch_returns, batch_advantages
    
    def __len__(self):
        return len(self.observations)

class NetHackPPOAgent:
    """PPO Agent for NetHack"""
    
    def __init__(self, action_dim=23, learning_rate=3e-4, gamma=0.99, clip_ratio=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.actor = PPOActor(action_dim=action_dim).to(self.device)
        self.critic = PPOCritic().to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.buffer = PPOBuffer()
        
        # Observation processor
        self.obs_processor = NetHackObservationProcessor()
        
        # Training stats
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def process_observation(self, obs):
        """Process and convert observation to tensors"""
        processed = self.obs_processor.process_observation(obs)
        
        # Convert to tensors
        tensor_obs = {}
        for key, value in processed.items():
            tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
        
        return tensor_obs
    
    def select_action(self, obs):
        """Select action using current policy"""
        with torch.no_grad():
            action_logits = self.actor(obs)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.critic(obs)
            
            return action.item(), log_prob.item(), value.item()
    
    def update(self, epochs=4, batch_size=64):
        """Update actor and critic networks"""
        if len(self.buffer) < batch_size:
            return
        
        # Compute advantages
        self.buffer.compute_advantages(self.gamma)
        
        # Normalize advantages
        advantages = torch.tensor(self.buffer.advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.tolist()
        
        for _ in range(epochs):
            batch_obs, batch_actions, old_log_probs, batch_returns, batch_advantages = \
                self.buffer.get_batch(min(batch_size, len(self.buffer)))
            
            # Actor update
            action_logits = self.actor(batch_obs)
            action_dist = Categorical(logits=action_logits)
            new_log_probs = action_dist.log_prob(batch_actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Critic update
            values = self.critic(batch_obs).squeeze()
            critic_loss = F.mse_loss(values, batch_returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
    
    def train(self, env, num_episodes=1000, update_freq=2048):
        """Train the PPO agent"""
        step_count = 0
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                processed_obs = self.process_observation(obs)
                action, log_prob, value = self.select_action(processed_obs)
                
                # Convert tensors back to single values for buffer
                processed_obs_for_buffer = {}
                for key, tensor_val in processed_obs.items():
                    processed_obs_for_buffer[key] = tensor_val.squeeze(0).cpu()
                
                step_result = env.step(action)
                
                # Handle both old and new gymnasium formats
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                    terminated = truncated = done
                else:  # len == 5
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                self.buffer.add(processed_obs_for_buffer, action, reward, value, log_prob, done)
                
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                step_count += 1
                
                if done:
                    break
                
                # Update networks periodically
                if step_count % update_freq == 0:
                    print(f"Updating networks at step {step_count}")
                    self.update()
                    self.buffer.clear()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if episode % 10 == 0:
                avg_reward = np.mean(list(self.episode_rewards))
                avg_length = np.mean(list(self.episode_lengths))
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
        
        return list(self.episode_rewards)
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {path}")

def create_nethack_env():
    """Create and configure NetHack environment"""
    # Import nle.env to register environments
    import nle.env
    
    # You can experiment with different NetHack environments:
    # - NetHackChallenge-v0: Standard challenge environment
    # - NetHackScore-v0: Optimize for game score
    # - NetHackStaircase-v0: Focus on reaching stairs
    # - NetHack-v0: Basic NetHack environment
    
    try:
        env = gym.make("NetHackScore-v0")
    except:
        # Fallback to basic NetHack environment if challenge doesn't exist
        env = gym.make("NetHack-v0")
    
    return env

def debug_observation(obs):
    """Debug function to understand observation structure"""
    print(f"Observation type: {type(obs)}")
    
    if isinstance(obs, tuple):
        print(f"Tuple length: {len(obs)}")
        for i, item in enumerate(obs):
            print(f"  [{i}]: {type(item)} - shape: {getattr(item, 'shape', 'N/A')}")
            if isinstance(item, dict):
                print(f"    Dictionary keys: {list(item.keys())}")
                for key, value in list(item.items())[:3]:  # Show first 3 keys
                    print(f"      {key}: {type(value)} - shape: {getattr(value, 'shape', 'N/A')}")
    elif isinstance(obs, dict):
        print("Dictionary keys:")
        for key, value in obs.items():
            print(f"  {key}: {type(value)} - shape: {getattr(value, 'shape', 'N/A')}")
    else:
        print(f"Unexpected observation format: {type(obs)}")

def main():
    """Main training function"""
    print("Setting up NetHack PPO Training...")
    
    # Create environment
    env = create_nethack_env()
    print(f"Environment action space: {env.action_space.n}")
    print(f"Environment observation space keys: {list(env.observation_space.spaces.keys())}")
    
    # Debug: Let's see what the actual observation looks like
    print("\nDebugging first observation...")
    obs = env.reset()
    debug_observation(obs)
    
    # Create agent with correct action dimension
    agent = NetHackPPOAgent(action_dim=env.action_space.n)
    
    # Test observation processing
    print("\nTesting observation processing...")
    try:
        processed_obs = agent.process_observation(obs)
        print("Observation processing successful!")
        for key, value in processed_obs.items():
            print(f"  {key}: {value.shape}")
    except Exception as e:
        print(f"Error in observation processing: {e}")
        return
    
    # Train agent
    print("\nStarting training...")
    rewards = agent.train(env, num_episodes=100)  # Reduced for initial testing
    
    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"nethack_ppo_model_{timestamp}.pth"
    agent.save_model(model_path)
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('NetHack PPO Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.savefig(f'nethack_training_progress_{timestamp}.png')
    plt.show()
    
    env.close()
    print("Training completed!")

if __name__ == "__main__":
    main()