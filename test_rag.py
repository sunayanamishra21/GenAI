#!/usr/bin/env python3
"""
Test script for RAG integration
"""

from rag_service import RAGService
import config

def test_rag_service():
    """Test the RAG service"""
    
    print("=== RAG Service Test ===")
    
    # Initialize RAG service
    rag_service = RAGService()
    
    # Test configuration
    print(f"LLM Provider: {config.LLM_PROVIDER}")
    print(f"LLM Model: {config.LLM_MODEL}")
    print(f"LLM Configured: {rag_service.is_llm_configured()}")
    
    if rag_service.is_llm_configured():
        print("[SUCCESS] LLM is configured and ready")
        available_models = rag_service.get_available_models()
        print(f"[MODELS] Available models: {available_models}")
    else:
        print("[WARNING] LLM not configured - will use fallback responses")
    
    # Mock search results for testing
    mock_results = [
        {
            "score": 0.85,
            "payload": {
                "text": "Article 5 - Principles relating to processing of personal data. Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject.",
                "chunk_id": 1,
                "source": "CELEX_32016R0679_EN_TXT.pdf"
            }
        },
        {
            "score": 0.78,
            "payload": {
                "text": "Article 6 - Lawfulness of processing. Processing shall be lawful only if and to the extent that at least one of the following applies: the data subject has given consent to the processing of their personal data for one or more specific purposes.",
                "chunk_id": 2,
                "source": "CELEX_32016R0679_EN_TXT.pdf"
            }
        }
    ]
    
    # Test query
    test_query = "What are the principles for processing personal data?"
    
    print(f"\nTest Query: {test_query}")
    print("=" * 50)
    
    # Generate response
    response = rag_service.generate_response(test_query, mock_results)
    
    print("RAG Response:")
    print("-" * 30)
    print(response)
    print("-" * 30)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_rag_service()
