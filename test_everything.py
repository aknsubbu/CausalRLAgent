"""Quick test to verify everything works"""

def test_environment():
    try:
        import gymnasium as gym
        import nle
        env = gym.make("NetHackScore-v0")
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(0)
        env.close()
        print("✅ Environment: Working")
        return True
    except Exception as e:
        print(f"❌ Environment: {e}")
        return False

def test_state_parser():
    try:
        from state_parser import SimpleStateParser
        parser = SimpleStateParser()
        
        import gymnasium as gym
        import nle
        env = gym.make("NetHackScore-v0")
        obs, _ = env.reset()
        desc = parser.describe_state(obs)
        env.close()
        
        print("✅ State Parser: Working")
        return True
    except Exception as e:
        print(f"❌ State Parser: {e}")
        return False

def test_llm():
    try:
        from llm_advisor import GroqLLMAdvisor
        llm = GroqLLMAdvisor()
        advice = llm.get_advice("Test state")
        print("✅ LLM: Working")
        return True
    except Exception as e:
        print(f"❌ LLM: {e}")
        try:
            from llm_advisor import GeminiLLMAdvisor  
            llm = GeminiLLMAdvisor()
            advice = llm.get_advice("Test state")
            print("✅ LLM (Gemini): Working")
            return True
        except Exception as e2:
            print(f"❌ Both LLMs failed: {e2}")
            return False

def test_action_translator():
    try:
        from action_translator import SimpleActionTranslator
        translator = SimpleActionTranslator()
        action = translator.translate_to_action("explore north")
        print("✅ Action Translator: Working") 
        return True
    except Exception as e:
        print(f"❌ Action Translator: {e}")
        return False

def test_integration():
    try:
        from demo_complete_integration import LLMRLAgent
        agent = LLMRLAgent()
        print("✅ Integration: All components loaded")
        return True
    except Exception as e:
        print(f"❌ Integration: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing All Components ===\n")
    
    tests = [
        test_environment,
        test_state_parser, 
        test_llm,
        test_action_translator,
        test_integration
    ]
    
    results = [test() for test in tests]
    
    print(f"\n=== Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 Everything working! Ready for demo.")
        print("\nRun: python demo_complete_integration.py")
    else:
        print("⚠️ Some components need fixing")