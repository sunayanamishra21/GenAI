"""
Self-RAG Streamlit Application
Advanced Retrieval-Augmented Generation with Self-Reflection
"""

import streamlit as st
import config
from streamlit_app import QdrantQueryInterface, get_global_embedding_model
from context_aware_service import ContextAwareService
from memory_aware_service import MemoryAwareService
from self_rag_service import SelfRAGService
import json
from datetime import datetime

def main():
    st.set_page_config(
        page_title="Self-RAG Vector Database",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧠 Self-RAG Vector Database Interface")
    st.markdown("**Self-Reflective Retrieval-Augmented Generation System**")
    
    # Initialize session state
    if 'self_rag_session_id' not in st.session_state:
        st.session_state.self_rag_session_id = f"session_{int(datetime.now().timestamp())}"
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Self-RAG Configuration")
        
        # Service status
        st.subheader("Service Status")
        
        # Initialize services
        if 'self_rag_services_initialized' not in st.session_state:
            with st.spinner("Initializing Self-RAG services..."):
                try:
                    # Initialize embedding model
                    if 'global_embedding_model' not in st.session_state:
                        st.session_state.global_embedding_model = get_global_embedding_model()
                    
                    # Initialize Qdrant interface
                    st.session_state.qdrant_interface = QdrantQueryInterface()
                    st.session_state.qdrant_interface.embedding_model = st.session_state.global_embedding_model
                    
                    # Initialize context aware service
                    st.session_state.context_aware_service = ContextAwareService()
                    
                    # Initialize memory aware service
                    st.session_state.memory_aware_service = MemoryAwareService()
                    
                    # Initialize Self-RAG service
                    st.session_state.self_rag_service = SelfRAGService(
                        st.session_state.qdrant_interface,
                        st.session_state.context_aware_service,
                        st.session_state.memory_aware_service
                    )
                    
                    st.session_state.self_rag_services_initialized = True
                    st.success("✅ Self-RAG services initialized!")
                    
                except Exception as e:
                    st.error(f"❌ Failed to initialize services: {e}")
                    st.session_state.self_rag_services_initialized = False
        
        # Display service status
        if st.session_state.self_rag_services_initialized:
            st.success("✅ All services ready")
            
            # Self-RAG stats
            stats = st.session_state.self_rag_service.get_self_rag_stats()
            with st.expander("Self-RAG System Info"):
                st.json(stats)
        else:
            st.error("❌ Services not initialized")
            if st.button("🔄 Retry Initialization"):
                del st.session_state['self_rag_services_initialized']
                st.rerun()
        
        # Session management
        st.subheader("Session Management")
        st.info(f"Session ID: {st.session_state.self_rag_session_id[:12]}...")
        
        if st.button("🆕 New Session"):
            st.session_state.self_rag_session_id = f"session_{int(datetime.now().timestamp())}"
            st.rerun()
        
        # Memory stats
        if st.session_state.self_rag_services_initialized:
            memory_stats = st.session_state.memory_aware_service.get_memory_stats()
            with st.expander("Memory Statistics"):
                st.json(memory_stats)
    
    # Main interface
    if not st.session_state.self_rag_services_initialized:
        st.warning("⚠️ Please initialize services in the sidebar first.")
        return
    
    # Query input
    st.header("🔍 Self-RAG Query Interface")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your query:",
            placeholder="Ask a question about the document...",
            key="self_rag_query"
        )
    
    with col2:
        if st.button("🚀 Generate Self-RAG Response", type="primary"):
            if query:
                with st.spinner("Generating Self-RAG response..."):
                    try:
                        # Generate Self-RAG response
                        self_rag_response = st.session_state.self_rag_service.generate_self_rag_response(
                            query, 
                            st.session_state.self_rag_session_id
                        )
                        
                        # Store in session state for display
                        st.session_state.last_self_rag_response = self_rag_response
                        
                    except Exception as e:
                        st.error(f"Error generating Self-RAG response: {e}")
                        st.session_state.last_self_rag_response = None
    
    # Display results
    if 'last_self_rag_response' in st.session_state and st.session_state.last_self_rag_response:
        response = st.session_state.last_self_rag_response
        
        # Main response
        st.header("📝 Self-RAG Response")
        
        # Response tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤖 Response", 
            "🔍 Retrieval Analysis", 
            "📊 Evidence", 
            "🪞 Self-Reflection", 
            "🔧 Correction"
        ])
        
        with tab1:
            st.subheader("AI Response")
            st.write(response.response)
            
            if response.corrected_response and response.corrected_response != response.response:
                st.info("🔄 This response was self-corrected based on quality analysis.")
        
        with tab2:
            st.subheader("Retrieval Decision Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Decision",
                    response.retrieval_decision.decision.title(),
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Confidence",
                    f"{response.retrieval_decision.confidence:.2f}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Query Type",
                    response.retrieval_decision.query_type.title(),
                    delta=None
                )
            
            st.write("**Reasoning:**")
            st.info(response.retrieval_decision.reasoning)
        
        with tab3:
            st.subheader("Retrieved Evidence")
            
            if response.evidence_items:
                st.write(f"Found {len(response.evidence_items)} evidence items:")
                
                for i, evidence in enumerate(response.evidence_items):
                    with st.expander(f"Evidence {i+1} (Relevance: {evidence.relevance_score:.3f})"):
                        st.write(f"**Source:** {evidence.source}")
                        st.write(f"**Chunk ID:** {evidence.chunk_id}")
                        st.write(f"**Point ID:** {evidence.point_id[:8]}...")
                        st.write("**Content:**")
                        st.text_area(
                            "",
                            value=evidence.text,
                            height=150,
                            key=f"evidence_{i}",
                            disabled=True
                        )
            else:
                st.info("No evidence was retrieved for this query.")
        
        with tab4:
            st.subheader("Self-Reflection Analysis")
            
            # Quality metrics
            col1, col2 = st.columns(2)
            
            with col1:
                quality_color = {
                    "high": "🟢",
                    "medium": "🟡", 
                    "low": "🔴"
                }
                st.metric(
                    "Response Quality",
                    f"{quality_color.get(response.reflection.quality, '⚪')} {response.reflection.quality.title()}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Assessment Confidence",
                    f"{response.reflection.confidence:.2f}",
                    delta=None
                )
            
            # Reasoning
            st.write("**Quality Assessment:**")
            st.info(response.reflection.reasoning)
            
            # Issues
            if response.reflection.issues:
                st.write("**Identified Issues:**")
                for issue in response.reflection.issues:
                    st.error(f"• {issue}")
            
            # Suggestions
            if response.reflection.suggestions:
                st.write("**Improvement Suggestions:**")
                for suggestion in response.reflection.suggestions:
                    st.success(f"• {suggestion}")
        
        with tab5:
            st.subheader("Self-Correction")
            
            if response.corrected_response:
                st.write("**Original Response:**")
                st.text_area(
                    "",
                    value=response.response,
                    height=150,
                    key="original_response",
                    disabled=True
                )
                
                st.write("**Corrected Response:**")
                st.text_area(
                    "",
                    value=response.corrected_response,
                    height=150,
                    key="corrected_response",
                    disabled=True
                )
                
                if response.correction_reasoning:
                    st.write("**Correction Reasoning:**")
                    st.info(response.correction_reasoning)
            else:
                st.info("No correction was needed for this response.")
        
        # Raw data
        with st.expander("🔍 Raw Self-RAG Data"):
            raw_data = {
                "retrieval_decision": {
                    "decision": response.retrieval_decision.decision,
                    "reasoning": response.retrieval_decision.reasoning,
                    "confidence": response.retrieval_decision.confidence,
                    "query_type": response.retrieval_decision.query_type
                },
                "evidence_count": len(response.evidence_items),
                "evidence_items": [
                    {
                        "text": item.text[:200] + "...",
                        "relevance_score": item.relevance_score,
                        "source": item.source,
                        "chunk_id": item.chunk_id
                    }
                    for item in response.evidence_items
                ],
                "reflection": {
                    "quality": response.reflection.quality,
                    "reasoning": response.reflection.reasoning,
                    "issues": response.reflection.issues,
                    "suggestions": response.reflection.suggestions,
                    "confidence": response.reflection.confidence
                },
                "correction_applied": response.corrected_response is not None
            }
            st.json(raw_data)
    
    # Sample queries
    st.header("💡 Sample Self-RAG Queries")
    
    sample_queries = [
        "What are the data protection requirements under GDPR?",
        "How should personal data be processed legally?",
        "What are the rights of data subjects?",
        "When is consent required for data processing?",
        "What are the penalties for GDPR violations?",
        "Explain the data breach notification requirements",
        "What is the role of data protection officers?",
        "How should data processing records be maintained?"
    ]
    
    cols = st.columns(2)
    for i, sample_query in enumerate(sample_queries):
        with cols[i % 2]:
            if st.button(f"🔍 {sample_query}", key=f"sample_{i}"):
                st.session_state.self_rag_query = sample_query
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🧠 **Self-RAG System** - Advanced RAG with self-reflection and self-correction capabilities. "
        "This system analyzes queries, retrieves relevant evidence, reflects on response quality, "
        "and self-corrects when needed for improved accuracy and reliability."
    )

if __name__ == "__main__":
    main()
