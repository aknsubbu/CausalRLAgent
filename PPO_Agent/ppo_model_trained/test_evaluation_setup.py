#!/usr/bin/env python3
"""
Quick test script to verify the evaluation setup works correctly.
This script tests the model loading and basic functionality.
"""

import os
import sys
import glob

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('gymnasium', 'Gymnasium'),
        ('nle', 'NetHack Learning Environment'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name} - OK")
        except ImportError:
            print(f"  ❌ {name} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r evaluation_requirements.txt")
        return False
    
    print("✅ All dependencies are available!")
    return True

def find_model_files():
    """Find available model files"""
    print("\n🔍 Looking for model files...")
    
    model_patterns = [
        "*.pth",
        "best_model*.pth",
        "enhanced_nethack*.pth"
    ]
    
    found_models = []
    for pattern in model_patterns:
        models = glob.glob(pattern)
        found_models.extend(models)
    
    # Remove duplicates
    found_models = list(set(found_models))
    
    if found_models:
        print(f"📁 Found {len(found_models)} model file(s):")
        for model in found_models:
            size = os.path.getsize(model) / (1024 * 1024)  # MB
            print(f"  • {model} ({size:.1f} MB)")
        return found_models
    else:
        print("❌ No model files found!")
        print("Expected files with extensions: .pth")
        print("Make sure your trained model is in this directory.")
        return []

def test_environment():
    """Test if NetHack environment can be created"""
    print("\n🔍 Testing NetHack environment...")
    
    try:
        from improved_reward_shaping import create_nethack_env
        env = create_nethack_env()
        print(f"  ✅ Environment created successfully!")
        print(f"  • Action space: {env.action_space.n}")
        
        # Test a quick reset
        obs = env.reset()
        print(f"  • Reset successful, observation type: {type(obs)}")
        env.close()
        
        return True
    except Exception as e:
        print(f"  ❌ Environment test failed: {e}")
        return False

def test_model_structure():
    """Test if the model classes can be imported"""
    print("\n🔍 Testing model structure...")
    
    try:
        from improved_reward_shaping import (
            EnhancedNetHackPPOAgent,
            RecurrentPPOActor,
            RecurrentPPOCritic
        )
        print("  ✅ Model classes imported successfully!")
        
        # Try creating an agent (without loading weights)
        agent = EnhancedNetHackPPOAgent(action_dim=23, use_wandb=False)
        print("  ✅ Agent created successfully!")
        
        return True
    except Exception as e:
        print(f"  ❌ Model structure test failed: {e}")
        return False

def generate_sample_command(model_files):
    """Generate sample commands for evaluation"""
    if not model_files:
        return
    
    print("\n🚀 Sample evaluation commands:")
    
    # Use the first model file found
    model_file = model_files[0]
    
    commands = [
        f"# Quick test (5 episodes)",
        f"python evaluate_model.py --model_path {model_file} --episodes 5",
        f"",
        f"# Standard evaluation (10 episodes)", 
        f"python evaluate_model.py --model_path {model_file} --episodes 10",
        f"",
        f"# Extended evaluation (20 episodes with trajectories)",
        f"python evaluate_model.py --model_path {model_file} --episodes 20 --save_trajectories",
        f"",
        f"# Visual evaluation (slower, but you can see the game)",
        f"python evaluate_model.py --model_path {model_file} --episodes 3 --render"
    ]
    
    for cmd in commands:
        print(f"  {cmd}")

def main():
    """Run all tests"""
    print("🧪 NetHack PPO Evaluation Setup Test")
    print("=" * 50)
    
    success = True
    
    # Test dependencies
    if not check_dependencies():
        success = False
    
    # Find model files
    model_files = find_model_files()
    if not model_files:
        success = False
    
    # Test environment
    if not test_environment():
        success = False
    
    # Test model structure
    if not test_model_structure():
        success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 All tests passed! You're ready to evaluate your model.")
        generate_sample_command(model_files)
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("  • pip install -r evaluation_requirements.txt")
        print("  • Make sure your .pth model file is in this directory")
        print("  • Check that improved_reward_shaping.py is available")
    
    print(f"\n📖 For detailed instructions, see: EVALUATION_GUIDE.md")

if __name__ == "__main__":
    main()