# Neural Network Architecture Explained - NetHack PPO Agent

## 🧠 Architecture Overview: The Big Picture

Think of your neural network as a **smart brain** that needs to:
1. **See** the game world (like human vision)
2. **Remember** what happened before (like human memory)
3. **Understand** different types of information (multi-modal processing)
4. **Decide** what action to take (policy)
5. **Evaluate** how good the current situation is (value function)

---

## 🏗️ The Complete Architecture Breakdown

### **Two Main Networks Working Together**

```
Your RL Agent = Actor Network + Critic Network

Actor Network (The Decision Maker):
"What action should I take next?"
Input: Game state → Output: Action probabilities

Critic Network (The Evaluator):  
"How good is my current situation?"
Input: Game state → Output: Value score
```

---

## 👁️ Step 1: Multi-Modal Input Processing

### **Why Multi-Modal?**
NetHack gives you different types of information, like how humans use multiple senses:
- **Visual**: What you see on screen (like your eyes)
- **Statistical**: Your health, level, etc. (like feeling your pulse)  
- **Textual**: Game messages (like hearing sounds)
- **Memory**: What you did recently (like remembering your path)

### **The 5 Input Streams**

```
1. GLYPHS (Visual Information) [21 × 79 grid]
   ┌─────────────────────────────────┐
   │ @ = Player    # = Wall          │
   │ . = Floor     d = Dog           │  
   │ ) = Weapon    % = Food          │
   └─────────────────────────────────┘
   This is like a "screenshot" of the game world

2. STATS (Game Statistics) [26 numbers]
   ┌─────────────────────────────────┐
   │ Health: 15/15   Level: 3        │
   │ Strength: 18    Experience: 145 │
   │ Position: (5,10) etc...         │
   └─────────────────────────────────┘
   
3. MESSAGES (Text Information) [256 characters]
   ┌─────────────────────────────────┐
   │ "You kill the goblin!"          │
   │ "You feel hungry."              │
   └─────────────────────────────────┘

4. INVENTORY (Items You Have) [55 slots]
   ┌─────────────────────────────────┐
   │ Sword: Yes    Potion: No        │
   │ Shield: Yes   Food: Yes         │
   └─────────────────────────────────┘

5. ACTION HISTORY (Recent Actions) [50 actions]
   ┌─────────────────────────────────┐
   │ Last 50 moves: North, North,    │
   │ Attack, South, Pickup, etc.     │
   └─────────────────────────────────┘
```

---

## 🔄 Step 2: Processing Each Input Stream

### **Stream 1: Visual Processing (Glyphs)**
```
Raw Glyphs [21×79] 
    ↓ (Convolutional Neural Network - like image recognition)
Conv Layer 1: [21×79×1] → [21×79×32]  (Find basic patterns)
    ↓ (Pool down to half size)
Conv Layer 2: [11×40×32] → [11×40×64]  (Find complex patterns)  
    ↓ (Pool down again)
Conv Layer 3: [6×20×64] → [6×20×128]   (Find high-level features)
    ↓ (Flatten to 1D)
Dense Layer: [15360] → [512]           (Compress to manageable size)
    ↓ (Add memory with LSTM)
LSTM: [512] → [256]                    (Remember visual patterns over time)

Final Output: 256 visual features that capture "what I see and remember seeing"
```

**What this does**: Like how your brain processes vision - first detecting edges, then shapes, then objects, then remembering what you saw before.

### **Stream 2: Statistics Processing (Game Stats)**
```
Raw Stats [26 numbers]
    ↓ (LSTM for temporal patterns)
LSTM: [26] → [64]

Final Output: 64 features that capture "my character's condition over time"
```

**What this does**: Tracks how your health, level, experience change over time - like remembering "I'm getting stronger" or "I'm low on health".

### **Stream 3: Message Processing (Text)**
```
Raw Messages [256 characters]
    ↓ (Fully Connected layer)
Dense: [256] → [128]

Final Output: 128 features that capture "what the game is telling me"
```

### **Stream 4: Inventory Processing (Items)**
```
Raw Inventory [55 item slots]
    ↓ (Fully Connected layer)  
Dense: [55] → [64]

Final Output: 64 features that capture "what items I have"
```

### **Stream 5: Action History Processing (Memory)**
```
Raw Action History [50 recent actions]
    ↓ (Fully Connected layer)
Dense: [50] → [32]  

Final Output: 32 features that capture "what I've been doing lately"
```

---

## 🔗 Step 3: Combining Everything Together

### **Feature Fusion**
```
Visual Features:    256D  ┐
Stats Features:     64D   ├─► Concatenate → [544D combined vector]
Message Features:   128D  │
Inventory Features: 64D   │
Action Features:    32D   ┘

Combined Processing:
[544D] → Dense Layer → [512D] → ReLU activation
[512D] → Dense Layer → [256D] → ReLU activation

Final Representation: 256D vector that captures EVERYTHING
```

**What this does**: Like how your brain combines sight, sound, touch, and memory to understand a situation completely.

---

## 🎭 Step 4: The Two Heads - Actor & Critic

### **Actor Network (The Decision Maker)**
```
Combined Features [256D]
    ↓
Action Head: [256D] → [23D action probabilities]

Output: "I think I should go North (30%), Attack (25%), Pick up item (20%)..."
```

### **Critic Network (The Evaluator)**
```
Combined Features [256D] (same processing as Actor)
    ↓  
Value Head: [256D] → [1D value score]

Output: "This situation looks good/bad, score: +15.3"
```

---

## 🧮 Parameter Count Breakdown

### **Where do the 4.2M parameters come from?**

```
ACTOR NETWORK:
├── Glyph CNN: ~800K parameters
├── Glyph LSTM: ~500K parameters  
├── Stats LSTM: ~50K parameters
├── Message FC: ~33K parameters
├── Inventory FC: ~4K parameters
├── Action History FC: ~2K parameters
├── Combined FC1: ~280K parameters
├── Combined FC2: ~130K parameters
└── Action Head: ~6K parameters
TOTAL ACTOR: ~2.1M parameters

CRITIC NETWORK:  
├── Same feature extraction as Actor: ~1.8M parameters
└── Value Head: ~0.3K parameters  
TOTAL CRITIC: ~2.1M parameters

GRAND TOTAL: ~4.2M parameters
```

---

## 🔄 Memory & Recurrence Explained

### **Why LSTM (Long Short-Term Memory)?**

**Problem**: NetHack needs memory. If you see a monster, then it goes behind a wall, you should still remember it's there.

**Solution**: LSTM cells that can remember important information and forget unimportant stuff.

```
LSTM Cell at each timestep:
┌─────────────────────────────┐
│ Previous Memory + New Info  │
│         ↓                   │
│ Decide what to:             │  
│ • Remember (important info) │
│ • Forget (old/useless info) │
│ • Output (current decision) │
└─────────────────────────────┘

Example:
Step 1: See monster → Remember "monster at (5,7)"
Step 2: Monster moves behind wall → Still remember "monster nearby" 
Step 3: Explore → Use memory to avoid that area or prepare for fight
```

### **Hidden State Management**
```
Episode Start: Reset all memories (fresh start)
During Episode: Keep updating memories
Training: Process in batches, reset between episodes
```

---

## 🎯 How Training Works (PPO Algorithm)

### **The Learning Loop**
```
1. COLLECT EXPERIENCE:
   Actor decides actions → Environment gives rewards → Store in buffer

2. COMPUTE ADVANTAGES:  
   Critic evaluates "How much better/worse was each action than expected?"

3. UPDATE ACTOR:
   "Make good actions more likely, bad actions less likely"
   But don't change too much at once (that's the "clipping" part)

4. UPDATE CRITIC:
   "Get better at predicting how good each situation is"

5. REPEAT:
   Do this thousands of times until the agent gets really good
```

### **Why PPO Specifically?**
- **Stable**: Won't "forget" what it learned (no catastrophic forgetting)
- **Sample Efficient**: Learns from each experience multiple times
- **Robust**: Works well even if hyperparameters aren't perfect

---

## 🔍 Visual Summary: Information Flow

```
NetHack Game State
        ↓
┌───────────────────────────────────────┐
│     MULTI-MODAL PROCESSING           │
├─────────┬─────────┬─────────┬────────┤
│ Visual  │ Stats   │ Text    │ Memory │
│ CNN+    │ LSTM    │ Dense   │ Dense  │
│ LSTM    │ 26→64   │ 256→128 │ 50→32  │
│ 21×79→  │         │         │        │
│ 256     │         │         │        │
└─────────┴─────────┴─────────┴────────┘
        ↓ (Concatenate All)
┌───────────────────────────────────────┐
│       COMBINED REPRESENTATION        │
│     544D → 512D → 256D               │
│    (Shared by Actor & Critic)        │
└───────────────────────────────────────┘
        ↓                    ↓
┌─────────────────┐  ┌─────────────────┐
│  ACTOR HEAD     │  │  CRITIC HEAD    │
│  (Policy)       │  │  (Value)        │
│  256D → 23D     │  │  256D → 1D      │
│  "What to do?"  │  │  "How good?"    │
└─────────────────┘  └─────────────────┘
```

---

## 💡 Key Insights

### **Why This Architecture Works**
1. **Multi-Modal**: Handles different types of game information like human perception
2. **Recurrent**: Remembers important events over time  
3. **Hierarchical**: Processes low-level features → high-level understanding
4. **Shared Features**: Actor and Critic understand the world the same way
5. **Stable Training**: PPO ensures steady improvement without forgetting

### **Compared to Simpler Approaches**
- **Basic DQN**: Only handles one type of input, no memory
- **Simple Policy Gradient**: Less stable training, might forget good strategies  
- **Feed-forward only**: Can't remember what happened before

Your architecture is like having a smart agent with good vision, memory, and decision-making - that's why it achieved 37.4% improvement!

---

## 🤔 Common Questions

**Q: Why so many parameters (4.2M)?**
A: NetHack is complex! Need lots of "brain capacity" to handle visual patterns, text understanding, memory, and decision-making.

**Q: Why two networks (Actor + Critic)?** 
A: Actor focuses on "what to do", Critic focuses on "how good is this". Like having a decision-maker and an advisor.

**Q: Why LSTM instead of simpler RNN?**
A: LSTM can remember long-term patterns and forget irrelevant info. Regular RNNs forget too much or remember too much.

**Q: Could this work on other games?**
A: Yes! The multi-modal architecture could adapt to any game with visual, textual, and statistical information.

---

*This explanation breaks down the complex 4.2M parameter architecture into understandable components*  
*Each part serves a specific purpose in creating intelligent game-playing behavior*