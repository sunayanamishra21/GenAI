#!/usr/bin/env python3
"""
Test script to verify LLM configuration and provide setup guidance
"""

import config
from rag_service import RAGService

def test_llm_configuration():
    """Test LLM configuration and provide setup guidance"""
    
    print("=== LLM Configuration Test ===")
    print()
    
    # Initialize RAG service
    rag_service = RAGService()
    
    # Check configuration
    print("Current Configuration:")
    print(f"  Provider: {config.LLM_PROVIDER}")
    print(f"  Model: {config.LLM_MODEL}")
    print(f"  Max Tokens: {config.MAX_TOKENS}")
    print(f"  Temperature: {config.TEMPERATURE}")
    print()
    
    # Check API key status
    if config.LLM_PROVIDER == "openai":
        if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "sk-your-openai-api-key-here":
            print("[SUCCESS] OpenAI API key is configured")
        else:
            print("[ERROR] OpenAI API key is not configured")
            print("        Current value:", config.OPENAI_API_KEY[:20] + "..." if len(config.OPENAI_API_KEY) > 20 else config.OPENAI_API_KEY)
    
    elif config.LLM_PROVIDER == "anthropic":
        if config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY != "":
            print("[SUCCESS] Anthropic API key is configured")
        else:
            print("[ERROR] Anthropic API key is not configured")
    
    print()
    
    # Test LLM service
    print("LLM Service Status:")
    if rag_service.is_llm_configured():
        print("[SUCCESS] LLM service is properly configured and ready")
        available_models = rag_service.get_available_models()
        print(f"         Available models: {', '.join(available_models)}")
    else:
        print("[ERROR] LLM service is not configured")
        print("        Will use fallback responses only")
    
    print()
    
    # Provide setup instructions
    if not rag_service.is_llm_configured():
        print("=== SETUP INSTRUCTIONS ===")
        print()
        
        if config.LLM_PROVIDER == "openai":
            print("To configure OpenAI:")
            print("1. Visit: https://platform.openai.com")
            print("2. Sign up or log in")
            print("3. Go to API Keys section")
            print("4. Create a new secret key")
            print("5. Copy the API key (starts with 'sk-')")
            print("6. Update config.py:")
            print("   OPENAI_API_KEY = 'sk-your-actual-key-here'")
            print()
            print("Example:")
            print("   OPENAI_API_KEY = 'sk-1234567890abcdef...'")
            
        elif config.LLM_PROVIDER == "anthropic":
            print("To configure Anthropic:")
            print("1. Visit: https://console.anthropic.com")
            print("2. Sign up or log in")
            print("3. Go to API Keys section")
            print("4. Create a new API key")
            print("5. Copy the API key (starts with 'sk-ant-')")
            print("6. Update config.py:")
            print("   ANTHROPIC_API_KEY = 'sk-ant-your-actual-key-here'")
            print()
            print("Example:")
            print("   ANTHROPIC_API_KEY = 'sk-ant-1234567890abcdef...'")
        
        print()
        print("After updating config.py, restart the Streamlit app:")
        print("   python -m streamlit run streamlit_app.py")
        
    else:
        print("=== READY TO USE ===")
        print()
        print("Your LLM is configured and ready!")
        print("You can now:")
        print("1. Launch the Streamlit app")
        print("2. Ask questions about your PDF content")
        print("3. Get AI-powered responses")
        print()
        print("Sample queries to try:")
        print("- What are the data subject rights under GDPR?")
        print("- How should data breaches be reported?")
        print("- What are the penalties for non-compliance?")

if __name__ == "__main__":
    test_llm_configuration()
