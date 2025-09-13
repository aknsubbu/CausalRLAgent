import numpy as np

class SimpleStateParser:
    """Convert NetHack observations to readable text"""
    
    def __init__(self):
        # NetHack blstats indices (correct indices for NLE)
        self.stat_indices = {
            'x': 0,          # X position
            'y': 1,          # Y position  
            'str': 2,        # Strength
            'dex': 3,        # Dexterity
            'con': 4,        # Constitution
            'int': 5,        # Intelligence
            'wis': 6,        # Wisdom
            'cha': 7,        # Charisma
            'score': 8,      # Score
            'hp': 10,        # Current HP
            'max_hp': 11,    # Max HP
            'depth': 12,     # Dungeon level
            'gold': 13,      # Gold
            'energy': 14,    # Energy
            'max_energy': 15, # Max energy
            'armor_class': 16, # Armor class
            'monster_level': 17, # Monster level
            'experience': 18, # Experience
            'time': 19       # Time
        }
    
    def parse_stats(self, blstats):
        """Extract key stats from blstats array"""
        parsed = {}
        for name, idx in self.stat_indices.items():
            if idx < len(blstats):
                parsed[name] = int(blstats[idx])
            else:
                parsed[name] = 0
        return parsed
    
    def get_surrounding_info(self, glyphs):
        """Get basic info about surroundings"""
        h, w = glyphs.shape
        
        # Player position from stats would be more accurate, but this is simpler
        center_y, center_x = h // 2, w // 2
        
        # Check 3x3 around center
        wall_count = 0
        floor_count = 0
        unknown_count = 0
        
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue  # Skip player position
                y, x = center_y + dy, center_x + dx
                if 0 <= y < h and 0 <= x < w:
                    glyph_val = glyphs[y, x]
                    # Basic glyph classification (this is simplified)
                    if glyph_val >= 2359 and glyph_val <= 2395:  # Wall range (approximate)
                        wall_count += 1
                    elif glyph_val >= 2396 and glyph_val <= 2400:  # Floor range (approximate)
                        floor_count += 1
                    else:
                        unknown_count += 1
        
        if wall_count > floor_count:
            return "enclosed area with walls"
        elif floor_count > 0:
            return "open area with floors"
        else:
            return "mixed terrain"
    
    def describe_state(self, obs):
        """Main method: observation -> text description"""
        stats = self.parse_stats(obs['blstats'])  # Use 'blstats' instead of 'stats'
        surroundings = self.get_surrounding_info(obs['glyphs'])
        
        # Get message if available
        message = ""
        if 'message' in obs:
            try:
                if isinstance(obs['message'], bytes):
                    message = obs['message'].decode('utf-8', errors='ignore')
                else:
                    message = str(obs['message'])
                # Clean up message
                message = message.strip().replace('\x00', '')[:100]
            except:
                message = "No readable message"
        
        description = f"""Current NetHack State:
- Health: {stats['hp']}/{stats['max_hp']} HP
- Depth: Level {stats['depth']} 
- Gold: {stats['gold']} pieces
- Energy: {stats['energy']}/{stats['max_energy']}
- Armor Class: {stats['armor_class']}
- Experience: {stats['experience']} points
- Position: ({stats['x']}, {stats['y']})
- Surroundings: {surroundings}
- Recent message: {message if message else 'None'}"""
        
        return description

# Test the parser
if __name__ == "__main__":
    import gymnasium as gym
    import nle
    
    print("Testing State Parser...")
    
    env = gym.make("NetHackScore-v0")
    obs, _ = env.reset()
    
    parser = SimpleStateParser()
    description = parser.describe_state(obs)
    print("\n" + "="*50)
    print(description)
    print("="*50)
    
    # Show raw blstats for debugging
    print(f"\nRaw blstats (first 20): {obs['blstats'][:20]}")
    print(f"Glyphs shape: {obs['glyphs'].shape}")
    print(f"Message: {obs.get('message', 'No message')}")
    
    env.close()
    print("\n✅ State parser working!")