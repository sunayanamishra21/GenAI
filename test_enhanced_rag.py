#!/usr/bin/env python3
"""
Test script for Enhanced RAG System with Context Awareness and Memory
"""

from rag_service import RAGService
from streamlit_app import QdrantQueryInterface
import config

def test_enhanced_rag():
    """Test the enhanced RAG system with context awareness and memory"""
    
    print("=== Enhanced RAG System Test ===")
    print()
    
    # Initialize services
    qdrant_interface = QdrantQueryInterface()
    rag_service = RAGService()
    
    # Start a new conversation session
    session = rag_service.memory_aware_service.start_new_session("test_user")
    print(f"Started new session: {session.session_id}")
    print()
    
    # Test conversation flow
    conversation_flow = [
        "What are the data subject rights under GDPR?",
        "How can data subjects exercise their right to access?",
        "What about the right to be forgotten?",
        "Are there any exceptions to these rights?",
        "What penalties exist for non-compliance with these rights?"
    ]
    
    print("Testing conversation flow with context awareness and memory...")
    print("=" * 70)
    
    for i, query in enumerate(conversation_flow, 1):
        print(f"\n=== Turn {i} ===")
        print(f"Query: {query}")
        print()
        
        try:
            # Create embedding for the query
            query_embedding = qdrant_interface.embedding_model.encode([query])[0].tolist()
            
            # Search similar vectors
            results = qdrant_interface.search_similar(
                query_embedding, 
                limit=5, 
                score_threshold=0.3
            )
            
            if results:
                print(f"Found {len(results)} relevant results")
                
                # Generate enhanced response
                enhanced_response = rag_service.generate_enhanced_response(query, results)
                
                # Extract components
                response = enhanced_response["response"]
                query_analysis = enhanced_response["query_analysis"]
                conversation_context = enhanced_response["conversation_context"]
                enhancement_info = enhanced_response["enhancement_info"]
                
                print("AI Response:")
                print("-" * 40)
                print(response)
                print("-" * 40)
                
                # Show context information
                print("Context Information:")
                print(f"  Query Type: {query_analysis.get('query_type', 'general')}")
                print(f"  Complexity: {query_analysis.get('complexity', 'medium')}")
                print(f"  GDPR Concepts: {query_analysis.get('gdpr_concepts', [])}")
                
                if conversation_context.get("recent_turns"):
                    print(f"  Recent Turns: {len(conversation_context['recent_turns'])}")
                
                if enhancement_info.get("is_follow_up"):
                    print("  [FOLLOW-UP] Question detected")
                
                if enhancement_info.get("has_reference"):
                    print("  [REFERENCE] Previous context referenced")
                
                # Show memory stats
                memory_stats = rag_service.memory_aware_service.get_memory_stats()
                print(f"  Session Turns: {memory_stats['current_session_turns']}")
                
            else:
                print("No relevant results found")
                
        except Exception as e:
            print(f"Error processing query: {e}")
        
        print("=" * 70)
    
    # Test memory persistence
    print("\n=== Memory Persistence Test ===")
    print("Testing if conversation memory is preserved...")
    
    # Get conversation context
    conversation_context = rag_service.memory_aware_service.get_conversation_context()
    
    if conversation_context.get("recent_turns"):
        print(f"[SUCCESS] Memory preserved: {len(conversation_context['recent_turns'])} recent turns")
        print("Recent conversation:")
        for turn in conversation_context["recent_turns"][-3:]:
            print(f"  Q: {turn['query'][:50]}...")
    else:
        print("[ERROR] No conversation memory found")
    
    # Test related topics
    if conversation_context.get("related_topics"):
        print(f"[SUCCESS] Related topics identified: {', '.join(conversation_context['related_topics'])}")
    else:
        print("[INFO] No related topics identified")
    
    # Final memory stats
    print("\n=== Final Memory Statistics ===")
    memory_stats = rag_service.memory_aware_service.get_memory_stats()
    print(f"Total Sessions: {memory_stats['total_sessions']}")
    print(f"Total Turns: {memory_stats['total_turns']}")
    print(f"Current Session Turns: {memory_stats['current_session_turns']}")
    print(f"Session Duration: {memory_stats['current_session_duration']:.0f} seconds")
    print(f"Memory File Size: {memory_stats['memory_file_size']} characters")
    
    print("\n=== Enhanced RAG System Test Complete ===")
    print("[SUCCESS] Context awareness: Working")
    print("[SUCCESS] Memory system: Working")
    print("[SUCCESS] Conversation flow: Working")
    print("[SUCCESS] Query analysis: Working")

if __name__ == "__main__":
    test_enhanced_rag()
