import random
import numpy as np

class SimpleActionTranslator:
    """Convert LLM advice to NetHack actions"""
    
    def __init__(self):
        # NetHack action mappings (simplified)
        self.action_map = {
            # Movement (most common)
            'north': 0, 'northeast': 1, 'east': 2, 'southeast': 3,
            'south': 4, 'southwest': 5, 'west': 6, 'northwest': 7,
            'wait': 8,
            
            # Common actions
            'search': 15,  # 's' key
            'rest': 16,    # '.' key  
            'inventory': 17, # 'i' key
            'down': 18,    # '>' key (stairs down)
            'up': 19,      # '<' key (stairs up)
        }
        
        # Strategy mappings
        self.strategy_actions = {
            'explore': [0, 1, 2, 3, 4, 5, 6, 7, 15],  # Move + search
            'rest': [16, 8],  # Rest and wait
            'retreat': [4, 5, 6, 7],  # Move away (south/west directions)
            'advance': [0, 1, 2, 3],  # Move forward (north/east directions)  
            'search': [15, 8],  # Search and wait
            'stairs': [18, 19],  # Use stairs
        }
    
    def extract_keywords(self, advice):
        """Extract action keywords from LLM advice"""
        advice_lower = advice.lower()
        keywords = []
        
        # Check for explicit actions
        if any(word in advice_lower for word in ['explore', 'search', 'look']):
            keywords.append('explore')
        if any(word in advice_lower for word in ['rest', 'wait', 'heal']):
            keywords.append('rest')
        if any(word in advice_lower for word in ['retreat', 'back', 'away', 'escape']):
            keywords.append('retreat')
        if any(word in advice_lower for word in ['advance', 'forward', 'attack', 'go']):
            keywords.append('advance')
        if any(word in advice_lower for word in ['stairs', 'down', 'up', 'level']):
            keywords.append('stairs')
        
        # Direction keywords
        directions = ['north', 'south', 'east', 'west', 'up', 'down']
        for direction in directions:
            if direction in advice_lower:
                keywords.append(direction)
        
        return keywords
    
    def translate_to_action(self, advice, current_stats=None):
        """Main translation: advice -> action number"""
        keywords = self.extract_keywords(advice)
        
        # Priority system
        if not keywords:
            # No clear direction, default exploration
            return random.choice(self.strategy_actions['explore'])
        
        # Handle specific directions first
        for keyword in keywords:
            if keyword in self.action_map:
                return self.action_map[keyword]
        
        # Handle strategies
        for keyword in keywords:
            if keyword in self.strategy_actions:
                return random.choice(self.strategy_actions[keyword])
        
        # Default: explore
        return random.choice(self.strategy_actions['explore'])
    
    def explain_action(self, action_num):
        """Convert action number back to description"""
        action_names = {
            0: "Move North", 1: "Move Northeast", 2: "Move East", 3: "Move Southeast",
            4: "Move South", 5: "Move Southwest", 6: "Move West", 7: "Move Northwest",
            8: "Wait", 15: "Search", 16: "Rest", 17: "Check Inventory",
            18: "Go Down Stairs", 19: "Go Up Stairs"
        }
        return action_names.get(action_num, f"Action {action_num}")

# Test translator
if __name__ == "__main__":
    translator = SimpleActionTranslator()
    
    test_advice = [
        "Explore the area carefully to find items",
        "Rest and recover your health",  
        "Go north to find the stairs",
        "Retreat from danger immediately"
    ]
    
    for advice in test_advice:
        action = translator.translate_to_action(advice)
        explanation = translator.explain_action(action)
        print(f"Advice: '{advice}'")
        print(f"-> Action {action}: {explanation}\n")
    
    print("✅ Action translator working!")