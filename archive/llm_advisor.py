import os
from dotenv import load_dotenv
from groq import Groq
import requests
import json

load_dotenv()

class GroqLLMAdvisor:
    """LLM Advisor using official Groq client"""
    
    def __init__(self):
        # Initialize Groq client - it will use GROQ_API_KEY from environment
        self.client = Groq()
        # Use a good model that's available on Groq
        self.model = "llama-3.1-8b-instant"  # High quality model
    
    def get_advice(self, state_description):
        """Get strategic advice from Groq LLM"""
        
        prompt = f"""You are an expert NetHack player giving advice to survive and progress in this roguelike dungeon crawler.

Current Game State:
{state_description}

Provide exactly ONE specific, actionable recommendation (1-2 sentences). Focus on the most critical priority:

1. IMMEDIATE SURVIVAL: If HP is low or there's danger, prioritize safety
2. RESOURCE MANAGEMENT: If low on food/energy, find sustenance  
3. STRATEGIC EXPLORATION: If safe, explore for items/stairs
4. COMBAT TACTICS: If enemies present, advise on engagement

Be concise and specific about what action to take next."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a master NetHack player. Give concise, survival-focused advice."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_completion_tokens=100,  # Keep advice concise
                top_p=0.9,
                stop=None
            )
            
            advice = completion.choices[0].message.content
            return advice.strip()
            
        except Exception as e:
            # Fallback advice if LLM fails
            return f"LLM error ({str(e)[:30]}). Default advice: Explore carefully, maintain health above 50%, and search for food if energy is low."

class GroqLLMAdvisorStreaming:
    """Alternative version with streaming (like your example)"""
    
    def __init__(self):
        self.client = Groq()
        self.model = "llama-3.1-8b-instant"
    
    def get_advice(self, state_description):
        """Get advice with streaming response"""
        
        prompt = f"""NetHack Expert Advisor:

{state_description}

Give ONE specific action recommendation (max 2 sentences):"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_completion_tokens=100,
                top_p=0.9,
                stream=True,  # Enable streaming like your example
                stop=None
            )
            
            # Collect streaming response
            full_response = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                full_response += content
            
            return full_response.strip()
            
        except Exception as e:
            return f"Streaming LLM failed: {str(e)[:50]}. Explore safely and check HP frequently."

class GeminiLLMAdvisor:
    """Backup: Free Gemini API (keeping for fallback)"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"
    
    def get_advice(self, state_description):
        prompt = f"""NetHack expert advice needed:

{state_description}

Give ONE specific action recommendation (1-2 sentences) focusing on survival and progress."""
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 100, "temperature": 0.7}
        }
        
        try:
            response = requests.post(self.base_url, json=data, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            advice = result['candidates'][0]['content']['parts'][0]['text']
            return advice.strip()
            
        except Exception as e:
            return f"Gemini failed: {str(e)[:30]}. Default: Explore safely, maintain health."

# Test the advisor
if __name__ == "__main__":
    print("Testing LLM Advisors...")
    
    # Test state
    test_state = """Current NetHack State:
- Health: 12/15 HP (Low!)
- Depth: Level 2
- Gold: 45 pieces  
- Energy: 8/14
- Surroundings: open area with floors
- Recent message: You hear a door open."""
    
    # Test Groq (standard)
    try:
        advisor = GroqLLMAdvisor()
        advice = advisor.get_advice(test_state)
        print(f"✅ Groq Standard: {advice}")
    except Exception as e:
        print(f"❌ Groq Standard failed: {e}")
    
    # Test Groq (streaming)
    try:
        advisor_stream = GroqLLMAdvisorStreaming()
        advice = advisor_stream.get_advice(test_state)
        print(f"✅ Groq Streaming: {advice}")
    except Exception as e:
        print(f"❌ Groq Streaming failed: {e}")
    
    # Test Gemini (backup)
    try:
        advisor_gemini = GeminiLLMAdvisor()
        advice = advisor_gemini.get_advice(test_state)
        print(f"✅ Gemini Backup: {advice}")
    except Exception as e:
        print(f"❌ Gemini failed: {e}")
    
    print("\nLLM testing complete!")