import gymnasium as gym
import nle
import time
from state_parser import SimpleStateParser
from llm_advisor import GroqLLMAdvisor, GeminiLLMAdvisor
from action_translator import SimpleActionTranslator

class LLMRLAgent:
    """Complete LLM-RL Agent Demo"""
    
    def __init__(self):
        self.state_parser = SimpleStateParser()
        self.action_translator = SimpleActionTranslator()
        
        # Try different LLM providers
        try:
            self.llm_advisor = GroqLLMAdvisor()
            self.llm_name = "Groq"
        except:
            try:
                self.llm_advisor = GeminiLLMAdvisor()
                self.llm_name = "Gemini"
            except:
                self.llm_advisor = None
                self.llm_name = "None (using fallback)"
        
        self.step_count = 0
        self.advice_history = []
        self.action_history = []
    
    def get_action(self, obs):
        """Main decision loop: obs -> advice -> action"""
        self.step_count += 1
        
        # Get state description
        state_desc = self.state_parser.describe_state(obs)
        
        # Get LLM advice (every few steps to save API calls)
        if self.step_count % 3 == 1:  # Every 3rd step
            if self.llm_advisor:
                advice = self.llm_advisor.get_advice(state_desc)
            else:
                advice = "Explore carefully and maintain health"
            
            self.advice_history.append(advice)
        else:
            advice = self.advice_history[-1] if self.advice_history else "Explore"
        
        # Translate advice to action
        stats = self.state_parser.parse_stats(obs['blstats'])
        action = self.action_translator.translate_to_action(advice, stats)
        
        # Store for logging
        self.action_history.append(action)
        
        return action, advice, state_desc
    
    def demo_run(self, max_steps=20):
        """Run a demo episode"""
        print(f"=== LLM-RL Agent Demo (using {self.llm_name}) ===\n")
        
        env = gym.make("NetHackScore-v0")
        obs, info = env.reset()
        
        total_reward = 0
        
        for step in range(max_steps):
            print(f"--- Step {step + 1} ---")
            
            # Get action from our agent
            action, advice, state_desc = self.get_action(obs)
            
            # Show decision process
            print(f"State: {state_desc.split(chr(10))[1]}")  # Just the health line
            print(f"LLM Advice: {advice[:80]}...")
            print(f"Action: {self.action_translator.explain_action(action)}")
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Reward: {reward} (Total: {total_reward})")
            
            if terminated or truncated:
                print(f"Episode ended after {step + 1} steps!")
                break
            
            print()
            time.sleep(0.5)  # Pause for readability
        
        env.close()
        
        print(f"=== Demo Complete ===")
        print(f"Total steps: {self.step_count}")
        print(f"Total reward: {total_reward}")
        print(f"LLM advice given {len(self.advice_history)} times")
        
        return total_reward

def run_comparison_demo():
    """Compare LLM agent vs random agent"""
    print("=== Comparison: LLM-RL vs Random Agent ===\n")
    
    # Test LLM agent
    llm_agent = LLMRLAgent()
    llm_score = llm_agent.demo_run(max_steps=15)
    
    print("\n" + "="*50 + "\n")
    
    # Test random agent
    print("=== Random Baseline Agent ===\n")
    env = gym.make("NetHackScore-v0")
    obs, info = env.reset()
    
    random_score = 0
    for step in range(15):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        random_score += reward
        
        if step % 5 == 0:
            hp = obs['blstats'][9] if len(obs['blstats']) > 9 else 0
            print(f"Step {step + 1}: Random action, HP: {hp}, Reward: {reward}")
        
        if terminated or truncated:
            print(f"Random agent episode ended at step {step + 1}")
            break
    
    env.close()
    
    print(f"\n=== Results ===")
    print(f"LLM-RL Agent Score: {llm_score}")
    print(f"Random Agent Score: {random_score}")
    print(f"Difference: {llm_score - random_score}")
    
    if llm_score > random_score:
        print("✅ LLM guidance shows improvement!")
    else:
        print("⚠️ LLM guidance needs refinement")

if __name__ == "__main__":
    run_comparison_demo()