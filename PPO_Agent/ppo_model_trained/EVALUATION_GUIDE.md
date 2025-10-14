# NetHack PPO Model Evaluation Guide

This guide will help you evaluate your trained NetHack PPO model and generate comprehensive visualizations of its performance.

## 📋 Prerequisites

1. **Python Environment**: Make sure you have Python 3.8+ installed
2. **Dependencies**: Install required packages
3. **Trained Model**: Have your `.pth` model file ready

## 🚀 Quick Setup

### Step 1: Install Dependencies

```bash
# Navigate to your PPO_Agent directory
cd /Users/kishore/CausalRLAgent/PPO_Agent

# Install evaluation requirements
pip install -r evaluation_requirements.txt
```

### Step 2: Prepare Your Model

Make sure you have your trained model file (`.pth`) in the PPO_Agent directory. For example:
- `enhanced_nethack_ppo_20251013_133042.pth`
- `best_model_episode_0_reward_68.69.pth`

## 🎯 Running Evaluation

### Basic Evaluation (10 episodes)
```bash
python evaluate_model.py --model_path enhanced_nethack_ppo_20251013_133042.pth
```

### Extended Evaluation (30 episodes with detailed analysis)
```bash
python evaluate_model.py --model_path enhanced_nethack_ppo_20251013_133042.pth --episodes 30 --save_trajectories
```

### Interactive Evaluation (with rendering - slower but visual)
```bash
python evaluate_model.py --model_path enhanced_nethack_ppo_20251013_133042.pth --episodes 5 --render
```

### Custom Evaluation
```bash
python evaluate_model.py \
  --model_path your_model.pth \
  --episodes 20 \
  --max_steps 10000 \
  --output_dir my_evaluation_results \
  --save_trajectories
```

## 📊 What You'll Get

### 1. Comprehensive Visualizations
The script generates a large dashboard with 12 different plots:

- **Episode Rewards**: Bar chart of rewards per episode
- **Episode Lengths**: How long each episode lasted
- **Reward vs Length**: Correlation between episode length and reward
- **Action Distribution**: Which actions the agent prefers
- **Survival Analysis**: Pie chart showing survival vs death rate
- **Reward Distribution**: Histogram of reward distribution
- **Exploration Efficiency**: How well the agent explores new areas
- **Performance Trends**: Moving average of performance over time
- **Cumulative Rewards**: Step-by-step reward accumulation
- **Top Actions**: Most frequently used actions
- **Statistics Summary**: Key performance metrics
- **Position Heatmap**: Where the agent spends most time

### 2. Detailed Results Files
- **comprehensive_evaluation_YYYYMMDD_HHMMSS.png**: Main visualization dashboard
- **evaluation_results_YYYYMMDD_HHMMSS.json**: Raw data in JSON format

### 3. Console Output
Real-time statistics and performance analysis with recommendations.

## 🎨 Sample Visualizations

After running the evaluation, you'll see:

1. **Performance Dashboard**: A comprehensive 20x16 inch plot with all metrics
2. **Interactive Console Output**: Real-time episode statistics
3. **Performance Recommendations**: AI-generated insights about your model

### Example Console Output:
```
🚀 Starting NetHack PPO Model Evaluation...
📁 Model: enhanced_nethack_ppo_20251013_133042.pth
🎯 Episodes: 10
👁️ Render: False

🌍 Creating NetHack environment...
🤖 Loading trained model...
✅ Model loaded successfully!

🎮 Running evaluation...

📊 Episode 1/10
  Reward: 68.25
  Length: 1850 steps
  Survival: No
  Unique Positions: 75

📊 Episode 2/10
  Reward: 82.15
  Length: 2100 steps
  Survival: No
  Unique Positions: 89

... (more episodes)

✅ Evaluation completed in 45.32 seconds!

📊 EVALUATION SUMMARY:
  Episodes: 10
  Mean Reward: 71.45 (±12.30)
  Best Reward: 95.20
  Worst Reward: 45.80
  Mean Length: 1920.5 steps
  Survival Rate: 20.0%

💡 PERFORMANCE ANALYSIS:
  ✅ Excellent performance! The model is performing very well.
  ⚠️ Moderate survival rate - agent sometimes dies early.
  ✅ High action diversity - agent uses varied strategies.

🎉 Evaluation complete! Check 'evaluation_results' for detailed results.
```

## 🔧 Command Line Options

```bash
python evaluate_model.py [OPTIONS]

Required:
  --model_path PATH          Path to your trained .pth model file

Optional:
  --episodes N               Number of episodes to run (default: 10)
  --max_steps N              Maximum steps per episode (default: 5000)
  --render                   Show the game visually (slower)
  --save_trajectories        Save detailed step-by-step data
  --output_dir DIR           Where to save results (default: evaluation_results)
```

## 📈 Understanding the Results

### Performance Metrics

- **Mean Reward**: Average score across all episodes
  - >50: Excellent performance
  - 20-50: Good performance  
  - 0-20: Moderate performance
  - <0: Needs improvement

- **Survival Rate**: Percentage of episodes where agent didn't die
  - >70%: Excellent survival skills
  - 30-70%: Moderate survival
  - <30%: Poor survival

- **Exploration Efficiency**: Unique positions per step
  - Higher values indicate better exploration

- **Action Diversity**: Percentage of available actions used
  - >70%: Great strategy variety
  - 40-70%: Moderate variety
  - <40%: Limited strategies

### Key Visualizations Explained

1. **Episode Rewards Chart**: Shows consistency and improvement trends
2. **Reward vs Length Scatter**: Negative correlation often indicates efficiency
3. **Action Distribution**: Reveals if agent has learned specific strategies
4. **Position Heatmap**: Shows exploration patterns and favorite areas
5. **Cumulative Rewards**: Reveals learning within episodes

## 🛠 Troubleshooting

### Common Issues

1. **Import Errors**: 
   ```bash
   pip install -r evaluation_requirements.txt
   ```

2. **Model Loading Errors**: 
   - Ensure model path is correct
   - Check model was saved properly during training

3. **Environment Issues**:
   ```bash
   pip install nle gymnasium
   ```

4. **Memory Issues**: 
   - Reduce `--episodes` number
   - Remove `--save_trajectories` flag

### Performance Tips

- Start with 5-10 episodes for quick analysis
- Use `--render` only for debugging (it's slow)
- Use `--save_trajectories` for detailed analysis but expect larger files
- Run 20+ episodes for statistical significance

## 📋 Checklist for Evaluation

- [ ] Install dependencies with `pip install -r evaluation_requirements.txt`
- [ ] Have your `.pth` model file ready
- [ ] Choose appropriate number of episodes (10-30 recommended)
- [ ] Run evaluation script
- [ ] Review the comprehensive dashboard
- [ ] Analyze the performance recommendations
- [ ] Compare with training metrics if available

## 🎯 Next Steps After Evaluation

Based on your results:

1. **If performance is good**: Try more challenging environments or longer episodes
2. **If performance is poor**: 
   - Check if more training is needed
   - Analyze action patterns for repetitive behaviors
   - Review exploration efficiency
3. **For production use**: Run evaluation on 100+ episodes for robust statistics

## 📞 Support

If you encounter issues:
1. Check the console output for specific error messages
2. Ensure all dependencies are installed correctly
3. Verify your model file is not corrupted
4. Try with fewer episodes first

Happy evaluating! 🎉