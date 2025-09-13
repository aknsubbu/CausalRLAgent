# nethack_demo.py - NetHack Environment Demonstration for Presentation
import gymnasium as gym
import nle
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import defaultdict
import json

class NetHackDemo:
    """
    Comprehensive NetHack environment demonstration for research presentation.
    Shows the complexity, observability, and challenges of the environment.
    """
    
    def __init__(self, env_id="NetHackScore-v0"):
        self.env_id = env_id
        self.env = None
        self.observation_history = []
        self.action_history = []
        
        # NetHack symbol mappings for visualization
        self.symbol_map = {
            32: ' ',   # space
            46: '.',   # floor
            35: '#',   # wall
            43: '+',   # door (closed)
            47: '/',   # door (open)
            60: '<',   # upstairs
            62: '>',   # downstairs
            64: '@',   # player
            37: '%',   # food
            41: ')',   # weapon
            91: '[',   # armor
            61: '=',   # ring
            63: '?',   # scroll
            33: '!',   # potion
            43: '+',   # spellbook
            34: '"',   # amulet
            40: '(',   # tool
            36: '$',   # gold
            42: '*',   # gem
            47: '/',   # wand
            # Common monsters
            100: 'd',  # dog
            102: 'f',  # cat
            114: 'r',  # rat
            107: 'k',  # kobold
            111: 'o',  # orc
        }
        
        # Action mappings
        self.action_names = {
            0: "North", 1: "South", 2: "West", 3: "East",
            4: "NorthWest", 5: "NorthEast", 6: "SouthWest", 7: "SouthEast",
            8: "Wait/Search", 9: "Open", 10: "Kick", 11: "Eat",
            12: "Apply", 13: "Read", 14: "Zap", 15: "Drop",
            16: "Inventory", 17: "Look", 18: "Pick Up", 19: "Put On"
        }
    
    def setup_environment(self):
        """Initialize the NetHack environment and show basic info"""
        print("=" * 60)
        print("🎮 NETHACK LEARNING ENVIRONMENT DEMO")
        print("=" * 60)
        
        # Create environment
        self.env = gym.make(self.env_id)
        
        # Show environment details
        print(f"\n📋 Environment: {self.env_id}")
        print(f"Action Space: {self.env.action_space}")
        print(f"Action Space Size: {self.env.action_space.n}")
        print("\nObservation Space Components:")
        
        for key, space in self.env.observation_space.spaces.items():
            if hasattr(space, 'shape'):
                print(f"  • {key}: {space.shape} ({space.dtype})")
            else:
                print(f"  • {key}: {space}")
        
        return True
    
    def demonstrate_observation_complexity(self):
        """Show the rich observation space of NetHack"""
        print("\n" + "=" * 50)
        print("🔍 OBSERVATION SPACE COMPLEXITY")
        print("=" * 50)
        
        obs, info = self.env.reset()
        
        print("\n1. VISUAL OBSERVATION (tty_chars):")
        tty_chars = obs['tty_chars']
        print(f"   Shape: {tty_chars.shape}")
        print(f"   Type: ASCII character codes")
        print(f"   Range: {tty_chars.min()} - {tty_chars.max()}")
        
        # Show a sample of the screen
        print("\n   Sample of game screen (converted to characters):")
        self._display_screen_sample(tty_chars)
        
        print("\n2. PLAYER STATISTICS (blstats):")
        blstats = obs['blstats']
        stat_names = [
            "strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma",
            "score", "hitpoints", "max_hitpoints", "depth", "gold", "energy", "max_energy",
            "armor_class", "monster_level", "experience_level", "experience_points",
            "time", "hunger_state", "carrying_capacity", "dungeon_number", "level_number"
        ]
        
        print(f"   Total stats: {len(blstats)} values")
        for i, (name, value) in enumerate(zip(stat_names[:10], blstats[:10])):
            print(f"   • {name}: {int(value)}")
        print("   ... (and more)")
        
        print("\n3. GAME MESSAGES:")
        message = obs.get('message', b'').decode('utf-8', errors='ignore')
        print(f"   Current message: '{message}'")
        
        print("\n4. INVENTORY STRINGS:")
        inv_strs = obs.get('inv_strs', [])
        print(f"   Inventory items: {len(inv_strs)}")
        if inv_strs:
            for i, item in enumerate(inv_strs[:3]):
                if isinstance(item, bytes):
                    item = item.decode('utf-8', errors='ignore')
                print(f"   • {item}")
        
        return obs
    
    def _display_screen_sample(self, tty_chars, sample_size=(10, 20)):
        """Display a sample portion of the NetHack screen"""
        height, width = tty_chars.shape
        start_row = height // 2 - sample_size[0] // 2
        start_col = width // 2 - sample_size[1] // 2
        
        print("   ┌" + "─" * sample_size[1] + "┐")
        for i in range(sample_size[0]):
            row_idx = start_row + i
            if 0 <= row_idx < height:
                line = "   │"
                for j in range(sample_size[1]):
                    col_idx = start_col + j
                    if 0 <= col_idx < width:
                        char_code = tty_chars[row_idx, col_idx]
                        char = self.symbol_map.get(char_code, chr(char_code) if 32 <= char_code <= 126 else '?')
                        line += char
                    else:
                        line += " "
                line += "│"
                print(line)
        print("   └" + "─" * sample_size[1] + "┘")
    
    def demonstrate_partial_observability(self):
        """Show how NetHack has partial observability"""
        print("\n" + "=" * 50)
        print("👁️ PARTIAL OBSERVABILITY CHALLENGES")
        print("=" * 50)
        
        print("\n🌫️ Information Limitations:")
        print("   • Player can only see nearby tiles (limited vision)")
        print("   • Unexplored areas are unknown")
        print("   • Monster behavior is not fully predictable")
        print("   • Item properties may be unknown until identification")
        print("   • Dungeon layout is procedurally generated")
        
        print("\n🎲 Stochastic Elements:")
        print("   • Random number generation affects combat")
        print("   • Procedural dungeon generation")
        print("   • Random monster spawning")
        print("   • Item generation and placement")
        
        print("\n🧩 Hidden State Information:")
        print("   • Monster AI state and intentions")
        print("   • Exact damage calculations")
        print("   • Future dungeon layout")
        print("   • Unidentified item properties")
    
    def demonstrate_action_space(self):
        """Show the complexity of the action space"""
        print("\n" + "=" * 50)
        print("🎯 ACTION SPACE COMPLEXITY")
        print("=" * 50)
        
        print(f"\nTotal Actions Available: {self.env.action_space.n}")
        print("\n📋 Action Categories:")
        
        # Group actions by category
        action_categories = {
            "Movement": list(range(8)),
            "Interaction": [8, 9, 10, 18],  # wait, open, kick, pick up
            "Inventory": [11, 15, 16, 19],  # eat, drop, inventory, put on
            "Magic/Items": [12, 13, 14],    # apply, read, zap
            "Information": [17],            # look
        }
        
        for category, actions in action_categories.items():
            print(f"\n   {category}:")
            for action_id in actions:
                if action_id < len(self.action_names):
                    print(f"     {action_id}: {self.action_names.get(action_id, 'Unknown')}")
        
        print(f"\n   ... and {self.env.action_space.n - 20} more specialized actions")
    
    def demonstrate_interactive_gameplay(self, steps=10):
        """Show interactive gameplay with analysis"""
        print("\n" + "=" * 50)
        print("🎮 INTERACTIVE GAMEPLAY DEMONSTRATION")
        print("=" * 50)
        
        obs, info = self.env.reset()
        total_reward = 0
        
        print(f"\nRunning {steps} random actions to show gameplay dynamics...")
        print("\nStep-by-step Analysis:")
        
        for step in range(steps):
            # Choose a reasonable action (movement or wait)
            action = np.random.choice([0, 1, 2, 3, 8])  # movements + wait
            
            print(f"\n--- Step {step + 1} ---")
            print(f"Action: {action} ({self.action_names.get(action, 'Unknown')})")
            
            # Show current state briefly
            current_hp = obs['blstats'][10] if len(obs['blstats']) > 10 else 0
            max_hp = obs['blstats'][11] if len(obs['blstats']) > 11 else 0
            score = obs['blstats'][9] if len(obs['blstats']) > 9 else 0
            
            print(f"Before: HP={int(current_hp)}/{int(max_hp)}, Score={int(score)}")
            
            # Take action
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            
            # Show results
            new_hp = obs['blstats'][10] if len(obs['blstats']) > 10 else 0
            new_score = obs['blstats'][9] if len(obs['blstats']) > 9 else 0
            message = obs.get('message', b'').decode('utf-8', errors='ignore')
            
            print(f"After:  HP={int(new_hp)}/{int(max_hp)}, Score={int(new_score)}")
            print(f"Reward: {reward:.2f}")
            if message:
                print(f"Message: '{message}'")
            
            if done or truncated:
                print("🏁 Episode ended!")
                break
            
            # Small delay for readability
            time.sleep(0.5)
        
        print(f"\n📊 Episode Summary:")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Steps Taken: {step + 1}")
        print(f"   Episode Status: {'Completed' if done else 'Ongoing'}")
    
    def analyze_state_space_complexity(self):
        """Analyze and demonstrate the complexity of the state space"""
        print("\n" + "=" * 50)
        print("📈 STATE SPACE COMPLEXITY ANALYSIS")
        print("=" * 50)
        
        obs, _ = self.env.reset()
        
        # Calculate state space dimensions
        tty_shape = obs['tty_chars'].shape
        blstats_shape = obs['blstats'].shape
        
        print(f"\n🔢 Dimensionality:")
        print(f"   Visual State: {tty_shape} = {np.prod(tty_shape):,} values")
        print(f"   Character Range: {2**8} possible values per cell")
        print(f"   Theoretical Visual States: {2**8}^{np.prod(tty_shape):,}")
        print(f"   Player Stats: {blstats_shape[0]} continuous values")
        
        # Analyze visual complexity
        unique_chars = len(np.unique(obs['tty_chars']))
        non_empty_cells = np.sum(obs['tty_chars'] != 32)  # non-space characters
        
        print(f"\n🎨 Current Screen Analysis:")
        print(f"   Unique Characters: {unique_chars}")
        print(f"   Non-empty Cells: {non_empty_cells}/{np.prod(tty_shape)}")
        print(f"   Screen Density: {non_empty_cells/np.prod(tty_shape)*100:.1f}%")
        
        print(f"\n🌍 Environment Characteristics:")
        print(f"   • Procedurally generated dungeons")
        print(f"   • ~400+ item types with combinations")
        print(f"   • ~200+ monster types")
        print(f"   • Multiple dungeon branches")
        print(f"   • Complex interaction rules")
        
        print(f"\n🧠 Learning Challenges:")
        print(f"   • Sparse rewards over long horizons")
        print(f"   • Complex strategy requirements")
        print(f"   • Partial observability")
        print(f"   • Stochastic environment dynamics")
        print(f"   • Need for exploration vs exploitation balance")
    
    def demonstrate_long_horizon_dependencies(self):
        """Show examples of long-term strategic dependencies"""
        print("\n" + "=" * 50)
        print("⏳ LONG-HORIZON DECISION DEPENDENCIES")
        print("=" * 50)
        
        print("\n🎯 Strategic Decision Examples:")
        
        strategies = [
            {
                "name": "Resource Management",
                "description": "Food consumption affects long-term survival",
                "horizon": "100-500 steps",
                "example": "Eating when not hungry wastes food, starving leads to death"
            },
            {
                "name": "Equipment Optimization", 
                "description": "Early equipment choices affect late-game combat ability",
                "horizon": "1000+ steps",
                "example": "Sacrificing early protection for better weapons"
            },
            {
                "name": "Exploration Strategy",
                "description": "Dungeon exploration order affects resource availability",
                "horizon": "500-2000 steps", 
                "example": "Which areas to explore first for optimal item collection"
            },
            {
                "name": "Character Development",
                "description": "Skill and attribute development impacts entire game",
                "horizon": "Entire episode",
                "example": "Experience allocation affects combat effectiveness"
            }
        ]
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n{i}. {strategy['name']}:")
            print(f"   Description: {strategy['description']}")
            print(f"   Time Horizon: {strategy['horizon']}")
            print(f"   Example: {strategy['example']}")
        
        print(f"\n💡 Why This Matters for RL:")
        print(f"   • Requires sophisticated credit assignment")
        print(f"   • Traditional RL struggles with sparse rewards")
        print(f"   • LLM guidance can provide strategic insights")
        print(f"   • Safety becomes critical over long episodes")
    
    def save_demonstration_data(self, filename="nethack_demo_data.json"):
        """Save demonstration data for later analysis"""
        demo_data = {
            "environment_id": self.env_id,
            "observation_space": str(self.env.observation_space),
            "action_space_size": self.env.action_space.n,
            "demo_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "observation_history": self.observation_history,
            "action_history": self.action_history
        }
        
        with open(filename, 'w') as f:
            json.dump(demo_data, f, indent=2, default=str)
        
        print(f"\n💾 Demo data saved to: {filename}")
    
    def run_complete_demo(self):
        """Run the complete demonstration sequence"""
        try:
            # Setup
            self.setup_environment()
            
            # Core demonstrations
            self.demonstrate_observation_complexity()
            self.demonstrate_partial_observability()
            self.demonstrate_action_space()
            self.analyze_state_space_complexity()
            self.demonstrate_long_horizon_dependencies()
            self.demonstrate_interactive_gameplay(steps=15)
            
            # Wrap up
            print("\n" + "=" * 60)
            print("✅ DEMO COMPLETE - KEY TAKEAWAYS")
            print("=" * 60)
            
            takeaways = [
                "NetHack provides a rich, complex environment for RL research",
                "Partial observability creates realistic uncertainty",
                "Large action space requires intelligent action selection",
                "Long-horizon dependencies demand strategic thinking",
                "State space complexity challenges traditional RL approaches",
                "Perfect testbed for LLM-guided RL safety research"
            ]
            
            for i, takeaway in enumerate(takeaways, 1):
                print(f"{i}. {takeaway}")
            
            print(f"\n🚀 This environment will allow us to test:")
            print(f"   • LLM strategic guidance capabilities")
            print(f"   • Safety validation under uncertainty")
            print(f"   • Robustness to adversarial inputs")
            print(f"   • Trust calibration mechanisms")
            
            # Save data
            self.save_demonstration_data()
            
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            print("Make sure you have installed: pip install nle gymnasium")
            
        finally:
            if self.env:
                self.env.close()
                print(f"\n🔒 Environment closed successfully")

def create_visualization_plots():
    """Create some basic visualizations for the presentation"""
    print("\n" + "=" * 50)
    print("📊 CREATING VISUALIZATION PLOTS")
    print("=" * 50)
    
    # Create sample complexity visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. State space growth
    levels = np.arange(1, 11)
    complexity = 2 ** levels
    ax1.semilogy(levels, complexity, 'b-o')
    ax1.set_title('State Space Complexity Growth')
    ax1.set_xlabel('Dungeon Level')
    ax1.set_ylabel('Approximate State Count (log scale)')
    ax1.grid(True)
    
    # 2. Action distribution simulation
    actions = ['Move', 'Wait', 'Combat', 'Use Item', 'Inventory', 'Other']
    action_counts = [35, 15, 20, 12, 8, 10]
    ax2.pie(action_counts, labels=actions, autopct='%1.1f%%')
    ax2.set_title('Typical Action Distribution')
    
    # 3. Reward sparsity simulation
    steps = np.arange(0, 1000, 10)
    rewards = np.zeros_like(steps, dtype=float)
    # Add sparse positive rewards
    reward_steps = [100, 250, 400, 650, 800, 950]
    for step in reward_steps:
        idx = step // 10
        if idx < len(rewards):
            rewards[idx] = np.random.exponential(2)
    
    ax3.plot(steps, rewards, 'g-')
    ax3.set_title('Sparse Reward Structure')
    ax3.set_xlabel('Game Steps')
    ax3.set_ylabel('Reward')
    ax3.grid(True)
    
    # 4. Survival curve simulation
    episodes = np.arange(1, 101)
    survival_rate = 100 * np.exp(-episodes / 30)  # Exponential decay
    ax4.plot(episodes, survival_rate, 'r-')
    ax4.set_title('Agent Survival Rate (Simulated)')
    ax4.set_xlabel('Episode Number')
    ax4.set_ylabel('Survival Rate (%)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('nethack_complexity_analysis.png', dpi=300, bbox_inches='tight')
    print("📈 Saved complexity analysis plots to: nethack_complexity_analysis.png")
    
    plt.show()

if __name__ == "__main__":
    print("🎮 NetHack Environment Demo for Research Presentation")
    print("=" * 60)
    
    # Check if NLE is installed
    try:
        import nle
        print("✅ NetHack Learning Environment is installed")
    except ImportError:
        print("❌ Please install NLE first: pip install nle")
        exit(1)
    
    # Run the main demo
    demo = NetHackDemo()
    demo.run_complete_demo()
    
    # Create visualizations
    try:
        create_visualization_plots()
    except Exception as e:
        print(f"⚠️  Visualization creation failed: {e}")
        print("   (This is optional - demo data is still available)")
    
    print(f"\n🎯 Demo complete! Use this information in your presentation to show:")
    print(f"   • Environment complexity and richness")
    print(f"   • Challenges that make it perfect for your research")
    print(f"   • Why LLM guidance + safety validation is needed")