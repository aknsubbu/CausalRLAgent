from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

def test_groq_connection():
    """Test basic Groq connection"""
    
    # Check if API key is set
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ No GROQ_API_KEY found in environment variables")
        print("Make sure you have a .env file with GROQ_API_KEY=your_key")
        return False
    
    print(f"✅ API Key found (starts with: {api_key[:10]}...)")
    
    # Test connection
    try:
        client = Groq()
        
        # Simple test with basic model
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Most reliable free model
            messages=[
                {
                    "role": "user", 
                    "content": "Say 'Hello' in exactly one word."
                }
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content
        print(f"✅ Groq connection successful!")
        print(f"Test response: '{result}'")
        return True
        
    except Exception as e:
        print(f"❌ Groq connection failed: {e}")
        return False

def list_available_models():
    """List available models"""
    try:
        client = Groq()
        models = client.models.list()
        print(f"\n📋 Available models:")
        for model in models.data:
            print(f"  - {model.id}")
    except Exception as e:
        print(f"❌ Could not list models: {e}")

if __name__ == "__main__":
    print("=== Testing Groq API Connection ===\n")
    
    success = test_groq_connection()
    
    if success:
        print("\n🎉 Groq is working! You can proceed with the LLM integration.")
        list_available_models()
    else:
        print("\n🔧 Troubleshooting steps:")
        print("1. Get API key from: https://console.groq.com/")
        print("2. Add to .env file: GROQ_API_KEY=your_actual_key")
        print("3. Make sure .env is in your project directory")
        print("4. Try running: pip install python-dotenv groq")