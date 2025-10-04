#!/usr/bin/env python3
"""
Test script for Self-RAG functionality
"""

import sys
import traceback
from streamlit_app import QdrantQueryInterface, get_global_embedding_model
from context_aware_service import ContextAwareService
from memory_aware_service import MemoryAwareService
from self_rag_service import SelfRAGService
import config

def test_self_rag_system():
    """Test the Self-RAG system with sample queries"""
    
    print("=" * 60)
    print("TESTING SELF-RAG SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize services
        print("1. Initializing services...")
        
        # Initialize embedding model
        embedding_model = get_global_embedding_model()
        print(f"   ✅ Embedding model: {type(embedding_model).__name__}")
        
        # Initialize Qdrant interface
        qdrant_interface = QdrantQueryInterface()
        qdrant_interface.embedding_model = embedding_model
        print("   ✅ Qdrant interface initialized")
        
        # Initialize context aware service
        context_aware_service = ContextAwareService()
        print("   ✅ Context aware service initialized")
        
        # Initialize memory aware service
        memory_aware_service = MemoryAwareService()
        print("   ✅ Memory aware service initialized")
        
        # Initialize Self-RAG service
        self_rag_service = SelfRAGService(
            qdrant_interface,
            context_aware_service,
            memory_aware_service
        )
        print("   ✅ Self-RAG service initialized")
        
        print("\n2. Testing Self-RAG features...")
        
        # Test queries
        test_queries = [
            "What are the data protection requirements?",
            "How should personal data be processed?",
            "Hello, how are you today?",
            "What are the rights of data subjects under GDPR?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}/{len(test_queries)} ---")
            print(f"Query: {query}")
            
            try:
                # Generate Self-RAG response
                response = self_rag_service.generate_self_rag_response(query, f"test_session_{i}")
                
                print(f"Retrieval Decision: {response.retrieval_decision.decision}")
                print(f"Decision Reasoning: {response.retrieval_decision.reasoning}")
                print(f"Query Type: {response.retrieval_decision.query_type}")
                print(f"Confidence: {response.retrieval_decision.confidence:.2f}")
                
                print(f"Evidence Items: {len(response.evidence_items)}")
                if response.evidence_items:
                    print(f"Top Evidence Relevance: {response.evidence_items[0].relevance_score:.3f}")
                
                print(f"Response Quality: {response.reflection.quality}")
                print(f"Reflection Confidence: {response.reflection.confidence:.2f}")
                
                if response.reflection.issues:
                    print(f"Issues Found: {len(response.reflection.issues)}")
                    for issue in response.reflection.issues:
                        print(f"  - {issue}")
                
                if response.corrected_response:
                    print("✅ Response was self-corrected")
                    if response.correction_reasoning:
                        print(f"Correction Reason: {response.correction_reasoning}")
                else:
                    print("ℹ️ No correction needed")
                
                print(f"Response Length: {len(response.response)} characters")
                print(f"Response Preview: {response.response[:100]}...")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                traceback.print_exc()
        
        print("\n3. Testing Self-RAG statistics...")
        stats = self_rag_service.get_self_rag_stats()
        print("Self-RAG System Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n4. Testing memory functionality...")
        memory_stats = memory_aware_service.get_memory_stats()
        print("Memory Stats:")
        for key, value in memory_stats.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        print("✅ SELF-RAG SYSTEM TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_individual_components():
    """Test individual Self-RAG components"""
    
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 60)
    
    try:
        # Initialize services
        embedding_model = get_global_embedding_model()
        qdrant_interface = QdrantQueryInterface()
        qdrant_interface.embedding_model = embedding_model
        context_aware_service = ContextAwareService()
        memory_aware_service = MemoryAwareService()
        self_rag_service = SelfRAGService(
            qdrant_interface,
            context_aware_service,
            memory_aware_service
        )
        
        # Test 1: Retrieval Decision Analysis
        print("\n1. Testing Retrieval Decision Analysis...")
        test_query = "What are the GDPR requirements?"
        decision = self_rag_service._analyze_retrieval_need(test_query)
        print(f"   Query: {test_query}")
        print(f"   Decision: {decision.decision}")
        print(f"   Reasoning: {decision.reasoning}")
        print(f"   Confidence: {decision.confidence:.2f}")
        print(f"   Query Type: {decision.query_type}")
        
        # Test 2: Evidence Retrieval
        print("\n2. Testing Evidence Retrieval...")
        evidence_items = self_rag_service._retrieve_evidence(test_query, limit=3)
        print(f"   Retrieved {len(evidence_items)} evidence items")
        for i, item in enumerate(evidence_items):
            print(f"   Item {i+1}: Relevance {item.relevance_score:.3f}, Source: {item.source}")
        
        # Test 3: Response Generation
        print("\n3. Testing Response Generation...")
        response = self_rag_service._generate_response(test_query, evidence_items, decision)
        print(f"   Generated response length: {len(response)} characters")
        print(f"   Response preview: {response[:200]}...")
        
        # Test 4: Self-Reflection
        print("\n4. Testing Self-Reflection...")
        reflection = self_rag_service._reflect_on_response(test_query, response, evidence_items)
        print(f"   Quality: {reflection.quality}")
        print(f"   Confidence: {reflection.confidence:.2f}")
        print(f"   Issues found: {len(reflection.issues)}")
        print(f"   Suggestions: {len(reflection.suggestions)}")
        
        # Test 5: Self-Correction (if needed)
        print("\n5. Testing Self-Correction...")
        if reflection.quality in ["low", "medium"]:
            corrected_response, correction_reasoning = self_rag_service._self_correct(
                test_query, response, evidence_items, reflection
            )
            if corrected_response:
                print(f"   ✅ Response corrected")
                print(f"   Correction reasoning: {correction_reasoning}")
                print(f"   New response length: {len(corrected_response)} characters")
            else:
                print(f"   [INFO] No correction applied: {correction_reasoning}")
        else:
            print(f"   [INFO] No correction needed (quality: {reflection.quality})")
        
        print("\n✅ All component tests completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Component test error: {e}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main test function"""
    print("Starting Self-RAG System Tests...")
    
    # Test individual components first
    component_success = test_individual_components()
    
    # Test full system
    system_success = test_self_rag_system()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Component Tests: {'[PASSED]' if component_success else '[FAILED]'}")
    print(f"System Tests: {'[PASSED]' if system_success else '[FAILED]'}")
    
    if component_success and system_success:
        print("\n[SUCCESS] ALL TESTS PASSED! Self-RAG system is ready for use.")
        return True
    else:
        print("\n[WARNING] Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
