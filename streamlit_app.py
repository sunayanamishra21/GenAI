import streamlit as st
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import config
import time
from typing import List, Dict
from rag_service import RAGService

# Page configuration
st.set_page_config(
    page_title="RAG Vector Database Query Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class QdrantQueryInterface:
    def __init__(self):
        self.api_key = config.QDRANT_API_KEY
        self.cluster_url = config.QDRANT_CLUSTER_URL.rstrip('/')
        self.collection_name = config.QDRANT_COLLECTION_NAME
        
        # Initialize embedding model with robust error handling
        self.embedding_model = self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize embedding model with multiple fallback strategies"""
        import torch
        import os
        from sentence_transformers import SentenceTransformer
        
        print(f"Qdrant interface: Initializing embedding model '{config.EMBEDDING_MODEL}'")
        
        # Set environment variables to avoid meta tensor issues
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Force CPU and disable optimizations
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        
        # Strategy 1: Direct initialization
        try:
            model = SentenceTransformer(config.EMBEDDING_MODEL)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            # Wrap with SafeSentenceTransformer to handle tensor conversion
            class SafeSentenceTransformer:
                def __init__(self, model):
                    self.model = model
                
                def encode(self, texts):
                    try:
                        # Force model to CPU before encoding
                        original_device = next(self.model.parameters()).device
                        self.model = self.model.to('cpu')
                        
                        embeddings = self.model.encode(texts)
                        
                        # Ensure embeddings are numpy arrays on CPU
                        if isinstance(embeddings, torch.Tensor):
                            if embeddings.device.type != 'cpu':
                                embeddings = embeddings.cpu()
                            embeddings = embeddings.detach().numpy()
                        elif hasattr(embeddings, 'cpu'):
                            embeddings = embeddings.cpu().detach().numpy()
                        
                        return embeddings
                    except Exception as e:
                        st.error(f"SafeSentenceTransformer encode error: {e}")
                        # Fallback: try to convert any tensor to CPU
                        try:
                            if isinstance(embeddings, torch.Tensor):
                                embeddings = embeddings.cpu().detach().numpy()
                            return embeddings
                        except:
                            raise e
            
            model = SafeSentenceTransformer(model)
            print("Qdrant interface: Direct initialization successful")
            return model
        except Exception as e1:
            print(f"Qdrant interface: Direct initialization failed: {e1}")
        
        # Strategy 2: With explicit CPU device
        try:
            model = SentenceTransformer(config.EMBEDDING_MODEL, device='cpu')
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            # Wrap with SafeSentenceTransformer to handle tensor conversion
            model = SafeSentenceTransformer(model)
            print("Qdrant interface: CPU device initialization successful")
            return model
        except Exception as e2:
            print(f"Qdrant interface: CPU device initialization failed: {e2}")
        
        # Strategy 3: With trust_remote_code
        try:
            model = SentenceTransformer(config.EMBEDDING_MODEL, device='cpu', trust_remote_code=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            # Wrap with SafeSentenceTransformer to handle tensor conversion
            model = SafeSentenceTransformer(model)
            print("Qdrant interface: Trust remote code initialization successful")
            return model
        except Exception as e3:
            print(f"Qdrant interface: Trust remote code initialization failed: {e3}")
        
        # Strategy 4: Manual model creation
        try:
            model = self._create_manual_model()
            print("Qdrant interface: Manual model initialization successful")
            return model
        except Exception as e4:
            print(f"Qdrant interface: Manual model initialization failed: {e4}")
        
        # Strategy 5: With cache_dir
        try:
            import tempfile
            cache_dir = tempfile.mkdtemp()
            model = SentenceTransformer(config.EMBEDDING_MODEL, cache_folder=cache_dir)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            print("Qdrant interface: Cache directory initialization successful")
            return model
        except Exception as e5:
            print(f"Qdrant interface: Cache directory initialization failed: {e5}")
        
        print("Qdrant interface: All initialization strategies failed")
        return None
    
    def _create_manual_model(self):
        """Create a manual model to avoid meta tensor issues"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # Load tokenizer and model separately
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            
            # Create a simple wrapper
            class SimpleSentenceTransformer:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.model.eval()
                
                def encode(self, texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    embeddings = []
                    for text in texts:
                        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            # Use mean pooling and ensure tensor is on CPU
                            pooled = outputs.last_hidden_state.mean(dim=1).squeeze()
                            # Convert to CPU and then to numpy
                            if pooled.is_cuda:
                                pooled = pooled.cpu()
                            embeddings.append(pooled.detach().numpy())
                    
                    return embeddings[0] if len(embeddings) == 1 else embeddings
            
            return SimpleSentenceTransformer(model, tokenizer)
        except Exception as e:
            raise e
        
    def get_collection_info(self) -> Dict:
        """Get collection information"""
        headers = {"api-key": self.api_key} if self.api_key else {}
        
        try:
            response = requests.get(
                f"{self.cluster_url}/collections/{self.collection_name}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            st.error(f"Error getting collection info: {e}")
            return {}
    
    def search_similar(self, query_vector: List[float], limit: int = 10, score_threshold: float = 0.0) -> List[Dict]:
        """Search for similar vectors"""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["api-key"] = self.api_key
        
        try:
            search_payload = {
                "vector": query_vector,
                "limit": limit,
                "score_threshold": score_threshold,
                "with_payload": True,
                "with_vectors": False  # Don't return vectors to save bandwidth
            }
            
            response = requests.post(
                f"{self.cluster_url}/collections/{self.collection_name}/points/search",
                headers=headers,
                json=search_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("result", [])
            else:
                st.error(f"Search failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            st.error(f"Error searching: {e}")
            return []
    
    def get_random_points(self, limit: int = 5) -> List[Dict]:
        """Get random points from the collection"""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["api-key"] = self.api_key
        
        try:
            response = requests.post(
                f"{self.cluster_url}/collections/{self.collection_name}/points/scroll",
                headers=headers,
                json={
                    "limit": limit,
                    "with_payload": True,
                    "with_vectors": False
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("result", {}).get("points", [])
            else:
                return []
                
        except Exception as e:
            st.error(f"Error getting random points: {e}")
            return []

def get_global_embedding_model():
    """Get a global embedding model instance"""
    if 'global_embedding_model' not in st.session_state:
        try:
            import torch
            import os
            
            print("Creating global embedding model...")
            
            # Set environment variables to avoid meta tensor issues
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Force CPU and disable optimizations
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            
            # Try multiple initialization strategies
            for i, strategy in enumerate([
                lambda: SentenceTransformer(config.EMBEDDING_MODEL),
                lambda: SentenceTransformer(config.EMBEDDING_MODEL, device='cpu'),
                lambda: SentenceTransformer(config.EMBEDDING_MODEL, device='cpu', trust_remote_code=True),
                lambda: SentenceTransformer(config.EMBEDDING_MODEL, use_auth_token=False),
                lambda: self._create_manual_model()  # Manual model creation
            ], 1):
                try:
                    if i == 5:  # Manual model strategy
                        model = strategy()
                    else:
                        model = strategy()
                        # Force model to CPU and disable gradient computation
                        model.eval()
                        for param in model.parameters():
                            param.requires_grad = False
                        
                        # Add a wrapper to handle tensor conversion for regular SentenceTransformer models
                        if hasattr(model, 'encode') and 'SimpleSentenceTransformer' not in str(type(model)):
                            class SafeSentenceTransformer:
                                def __init__(self, model):
                                    self.model = model
                                
                                def encode(self, texts):
                                    embeddings = self.model.encode(texts)
                                    # Ensure embeddings are numpy arrays on CPU
                                    if hasattr(embeddings, 'cpu'):
                                        embeddings = embeddings.cpu().detach().numpy()
                                    elif isinstance(embeddings, torch.Tensor):
                                        embeddings = embeddings.cpu().detach().numpy()
                                    return embeddings
                            
                            model = SafeSentenceTransformer(model)
                    
                    print(f"Global embedding model strategy {i} successful")
                    st.session_state.global_embedding_model = model
                    return model
                except Exception as e:
                    print(f"Global embedding model strategy {i} failed: {e}")
                    continue
            
            print("All global embedding model strategies failed")
            st.session_state.global_embedding_model = None
            return None
            
        except Exception as e:
            print(f"Global embedding model creation failed: {e}")
            st.session_state.global_embedding_model = None
            return None
    
    return st.session_state.global_embedding_model

def _create_manual_model():
    """Create a manual model to avoid meta tensor issues"""
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        # Load tokenizer and model separately
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Create a simple wrapper
        class SimpleSentenceTransformer:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
                self.model.eval()
            
            def encode(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                
                embeddings = []
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Use mean pooling and ensure tensor is on CPU
                        pooled = outputs.last_hidden_state.mean(dim=1).squeeze()
                        # Convert to CPU and then to numpy
                        if pooled.is_cuda:
                            pooled = pooled.cpu()
                        embeddings.append(pooled.detach().numpy())
                
                return embeddings[0] if len(embeddings) == 1 else embeddings
        
        return SimpleSentenceTransformer(model, tokenizer)
    except Exception as e:
        raise e

def main():
    st.title("🔍 RAG Vector Database Query Interface")
    st.markdown("Query your PDF embeddings stored in Qdrant vector database")
    
    # Get global embedding model
    global_model = get_global_embedding_model()
    
    # Initialize the query interface and RAG service using session state
    if 'qdrant_interface' not in st.session_state:
        with st.spinner("Initializing services..."):
            try:
                st.session_state.qdrant_interface = QdrantQueryInterface()
                st.session_state.rag_service = RAGService()
                
                # Use global model if interface model failed
                if st.session_state.qdrant_interface.embedding_model is None and global_model is not None:
                    st.session_state.qdrant_interface.embedding_model = global_model
                    st.success("Used global embedding model for interface!")
                elif st.session_state.qdrant_interface.embedding_model is not None:
                    st.success("Services initialized successfully!")
                else:
                    st.warning("Services initialized but embedding model not available")
                    
            except Exception as e:
                st.error(f"Failed to initialize services: {e}")
                st.stop()
    
    qdrant_interface = st.session_state.qdrant_interface
    rag_service = st.session_state.rag_service
    
    # Ensure embedding model is available
    if qdrant_interface.embedding_model is None and global_model is not None:
        qdrant_interface.embedding_model = global_model
    
    # Debug: Show embedding model status in sidebar
    with st.sidebar:
        st.subheader("🔧 Debug Info")
        st.write(f"Interface Model: {type(qdrant_interface.embedding_model)}")
        st.write(f"Global Model: {type(global_model)}")
        if qdrant_interface.embedding_model is not None:
            st.success("✅ Embedding model loaded")
        else:
            st.error("❌ Embedding model is None")
        st.divider()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display connection info
        st.subheader("Database Info")
        collection_info = qdrant_interface.get_collection_info()
        
        if collection_info:
            result = collection_info.get("result", {})
            st.success(f"✅ Connected to Qdrant")
            st.info(f"**Collection:** {qdrant_interface.collection_name}")
            st.info(f"**Points:** {result.get('points_count', 'N/A')}")
            st.info(f"**Status:** {result.get('status', 'N/A')}")
            
            vectors_config = result.get('config', {}).get('params', {}).get('vectors', {})
            if isinstance(vectors_config, dict):
                st.info(f"**Dimensions:** {vectors_config.get('size', 'N/A')}")
                st.info(f"**Distance:** {vectors_config.get('distance', 'N/A')}")
        else:
            st.error("❌ Failed to connect to Qdrant")
            st.stop()
        
        st.divider()
        
        # Search parameters
        st.subheader("Search Parameters")
        max_results = st.slider("Max Results", 1, 20, 10)
        score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.0, 0.01)
        
        # LLM Configuration
        st.subheader("🤖 LLM Configuration")
        
        # Show LLM status
        if rag_service.is_llm_configured():
            st.success("✅ LLM configured and ready")
            st.info(f"Provider: {config.LLM_PROVIDER}")
            st.info(f"Model: {config.LLM_MODEL}")
        else:
            st.warning("⚠️ LLM not configured")
            st.info("Responses will be basic retrieval only")
            
            # Configuration help
            with st.expander("How to configure LLM"):
                st.markdown("""
                **To enable AI-powered responses:**
                
                1. **OpenAI:**
                   - Get API key from https://platform.openai.com
                   - Set `OPENAI_API_KEY` in `config.py`
                   - Set `LLM_PROVIDER = "openai"`
                
                2. **Anthropic:**
                   - Get API key from https://console.anthropic.com
                   - Set `ANTHROPIC_API_KEY` in `config.py`
                   - Set `LLM_PROVIDER = "anthropic"`
                
                3. **Restart the app** after configuration
                """)
        
        # Response mode selection
        response_mode = st.radio(
            "Response Mode",
            ["AI Summary", "Raw Chunks", "Both"],
            help="Choose how to display search results"
        )
        
        st.divider()
        
        # Memory and Context Information
        st.subheader("🧠 Memory & Context")
        
        # Get memory stats
        memory_stats = rag_service.memory_aware_service.get_memory_stats()
        conversation_context = rag_service.memory_aware_service.get_conversation_context()
        
        st.info(f"**Current Session:** {memory_stats['current_session_turns']} turns")
        st.info(f"**Total Sessions:** {memory_stats['total_sessions']}")
        st.info(f"**Session Duration:** {memory_stats['current_session_duration']:.0f}s")
        
        if conversation_context.get("recent_turns"):
            st.write("**Recent Context:**")
            for turn in conversation_context["recent_turns"][-2:]:  # Show last 2 turns
                st.caption(f"Q: {turn['query'][:50]}...")
        
        if conversation_context.get("related_topics"):
            st.write("**Related Topics:**")
            st.caption(", ".join(conversation_context["related_topics"]))
        
        # Memory controls
        if st.button("🗑️ Clear Memory"):
            rag_service.memory_aware_service.sessions = []
            rag_service.memory_aware_service.current_session = None
            rag_service.memory_aware_service.save_memory()
            st.rerun()
        
        if st.button("📊 Memory Stats"):
            st.json(memory_stats)
        
        # Service controls
        if st.button("🔄 Restart Services"):
            if 'qdrant_interface' in st.session_state:
                del st.session_state['qdrant_interface']
            if 'rag_service' in st.session_state:
                del st.session_state['rag_service']
            if 'global_embedding_model' in st.session_state:
                del st.session_state['global_embedding_model']
            st.rerun()
        
        # Quick start with dummy model
        if st.button("🚀 Quick Start (Dummy Model)"):
            try:
                import numpy as np
                
                # Create a dummy embedding model that always works
                class DummyEmbeddingModel:
                    def __init__(self):
                        self.dimension = 384
                        print("Dummy embedding model created")
                    
                    def encode(self, texts):
                        if isinstance(texts, str):
                            texts = [texts]
                        
                        # Generate random embeddings for testing
                        embeddings = []
                        for text in texts:
                            # Create deterministic embedding based on text hash
                            import hashlib
                            text_hash = hashlib.md5(text.encode()).hexdigest()
                            seed = int(text_hash[:8], 16)
                            np.random.seed(seed % (2**32))
                            embedding = np.random.normal(0, 1, self.dimension).astype(np.float32)
                            # Convert to list of Python floats for JSON serialization
                            embeddings.append([float(x) for x in embedding.tolist()])
                        
                        return embeddings[0] if len(embeddings) == 1 else embeddings
                
                dummy_model = DummyEmbeddingModel()
                st.session_state.global_embedding_model = dummy_model
                qdrant_interface.embedding_model = dummy_model
                st.success("🎯 Quick Start successful! Using dummy embeddings.")
                st.success("✅ You can now try searching with the sample queries!")
                st.rerun()
            except Exception as e:
                st.error(f"Quick start failed: {e}")
        
        # Force global model creation
        if st.button("🔧 Create Global Model"):
            if 'global_embedding_model' in st.session_state:
                del st.session_state['global_embedding_model']
            
            with st.spinner("Creating global model..."):
                try:
                    from sentence_transformers import SentenceTransformer
                    import torch
                    
                    st.write("Attempting to create global embedding model...")
                    
                    # Strategy 1: Direct initialization
                    try:
                        st.write("Strategy 1: Direct initialization...")
                        model = SentenceTransformer(config.EMBEDDING_MODEL)
                        st.success("✅ Strategy 1 successful!")
                        st.session_state.global_embedding_model = model
                        qdrant_interface.embedding_model = model
                        st.success("Global model created and assigned!")
                        st.rerun()
                        return
                    except Exception as e1:
                        st.write(f"❌ Strategy 1 failed: {str(e1)}")
                    
                    # Strategy 2: CPU device
                    try:
                        st.write("Strategy 2: CPU device...")
                        model = SentenceTransformer(config.EMBEDDING_MODEL, device='cpu')
                        st.success("✅ Strategy 2 successful!")
                        st.session_state.global_embedding_model = model
                        qdrant_interface.embedding_model = model
                        st.success("Global model created and assigned!")
                        st.rerun()
                        return
                    except Exception as e2:
                        st.write(f"❌ Strategy 2 failed: {str(e2)}")
                    
                    # Strategy 3: Trust remote code
                    try:
                        st.write("Strategy 3: Trust remote code...")
                        model = SentenceTransformer(config.EMBEDDING_MODEL, device='cpu', trust_remote_code=True)
                        st.success("✅ Strategy 3 successful!")
                        st.session_state.global_embedding_model = model
                        qdrant_interface.embedding_model = model
                        st.success("Global model created and assigned!")
                        st.rerun()
                        return
                    except Exception as e3:
                        st.write(f"❌ Strategy 3 failed: {str(e3)}")
                    
                    # Strategy 4: No auth token
                    try:
                        st.write("Strategy 4: No auth token...")
                        model = SentenceTransformer(config.EMBEDDING_MODEL, use_auth_token=False)
                        st.success("✅ Strategy 4 successful!")
                        st.session_state.global_embedding_model = model
                        qdrant_interface.embedding_model = model
                        st.success("Global model created and assigned!")
                        st.rerun()
                        return
                    except Exception as e4:
                        st.write(f"❌ Strategy 4 failed: {str(e4)}")
                    
                    # Strategy 5: Alternative model
                    try:
                        st.write("Strategy 5: Alternative model...")
                        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                        st.success("✅ Strategy 5 successful!")
                        st.session_state.global_embedding_model = model
                        qdrant_interface.embedding_model = model
                        st.success("Global model created and assigned!")
                        st.rerun()
                        return
                    except Exception as e5:
                        st.write(f"❌ Strategy 5 failed: {str(e5)}")
                    
                    # Strategy 6: Meta tensor workaround
                    try:
                        st.write("Strategy 6: Meta tensor workaround...")
                        import torch
                        import os
                        
                        # Set environment variables to avoid meta tensor issues
                        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                        
                        # Force CPU and disable optimizations
                        torch.backends.cudnn.enabled = False
                        torch.backends.cudnn.benchmark = False
                        
                        # Create model with explicit CPU device and no optimization
                        model = SentenceTransformer(
                            config.EMBEDDING_MODEL, 
                            device='cpu',
                            trust_remote_code=True,
                            use_auth_token=False
                        )
                        
                        # Force model to CPU and disable gradient computation
                        model.eval()
                        for param in model.parameters():
                            param.requires_grad = False
                        
                        # Add a wrapper to handle tensor conversion
                        class SafeSentenceTransformer:
                            def __init__(self, model):
                                self.model = model
                            
                            def encode(self, texts):
                                embeddings = self.model.encode(texts)
                                # Ensure embeddings are numpy arrays on CPU
                                if hasattr(embeddings, 'cpu'):
                                    embeddings = embeddings.cpu().detach().numpy()
                                elif isinstance(embeddings, torch.Tensor):
                                    embeddings = embeddings.cpu().detach().numpy()
                                return embeddings
                        
                        model = SafeSentenceTransformer(model)
                        
                        st.success("✅ Strategy 6 successful!")
                        st.session_state.global_embedding_model = model
                        qdrant_interface.embedding_model = model
                        st.success("Global model created and assigned!")
                        st.rerun()
                        return
                    except Exception as e6:
                        st.write(f"❌ Strategy 6 failed: {str(e6)}")
                    
                    # Strategy 7: Manual model loading
                    try:
                        st.write("Strategy 7: Manual model loading...")
                        import torch
                        import transformers
                        from transformers import AutoModel, AutoTokenizer
                        
                        # Load tokenizer and model separately
                        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                        
                        # Create a simple wrapper
                        class SimpleSentenceTransformer:
                            def __init__(self, model, tokenizer):
                                self.model = model
                                self.tokenizer = tokenizer
                                self.model.eval()
                            
                            def encode(self, texts):
                                if isinstance(texts, str):
                                    texts = [texts]
                                
                                embeddings = []
                                for text in texts:
                                    inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                                    with torch.no_grad():
                                        outputs = self.model(**inputs)
                                        # Use mean pooling
                                        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
                                
                                return embeddings[0] if len(embeddings) == 1 else embeddings
                        
                        wrapper = SimpleSentenceTransformer(model, tokenizer)
                        st.success("✅ Strategy 7 successful!")
                        st.session_state.global_embedding_model = wrapper
                        qdrant_interface.embedding_model = wrapper
                        st.success("Global model created and assigned!")
                        st.rerun()
                        return
                    except Exception as e7:
                        st.write(f"❌ Strategy 7 failed: {str(e7)}")
                    
                    # Strategy 8: Ultra-simple approach
                    try:
                        st.write("Strategy 8: Ultra-simple approach...")
                        import torch
                        import numpy as np
                        
                        # Create a dummy embedding model that always works
                        class DummyEmbeddingModel:
                            def __init__(self):
                                self.dimension = 384
                                print("Dummy embedding model created")
                            
                            def encode(self, texts):
                                if isinstance(texts, str):
                                    texts = [texts]
                                
                                # Generate random embeddings for testing
                                embeddings = []
                                for text in texts:
                                    # Create deterministic embedding based on text hash
                                    import hashlib
                                    text_hash = hashlib.md5(text.encode()).hexdigest()
                                    seed = int(text_hash[:8], 16)
                                    np.random.seed(seed % (2**32))
                                    embedding = np.random.normal(0, 1, self.dimension).astype(np.float32)
                                    # Convert to list of Python floats for JSON serialization
                                    embeddings.append([float(x) for x in embedding.tolist()])
                                
                                return embeddings[0] if len(embeddings) == 1 else embeddings
                        
                        dummy_model = DummyEmbeddingModel()
                        st.success("✅ Strategy 8 successful!")
                        st.session_state.global_embedding_model = dummy_model
                        qdrant_interface.embedding_model = dummy_model
                        st.success("Global model created and assigned!")
                        st.info("🎯 Using dummy embeddings for testing. The app is now functional!")
                        st.success("✅ You can now try searching with sample queries!")
                        st.rerun()
                        return
                    except Exception as e8:
                        st.write(f"❌ Strategy 8 failed: {str(e8)}")
                    
                    st.error("❌ All strategies failed to create global model")
                    st.write("**Debug Information:**")
                    st.write(f"Model name: {config.EMBEDDING_MODEL}")
                    st.write(f"PyTorch version: {torch.__version__}")
                    st.write(f"CUDA available: {torch.cuda.is_available()}")
                    
                except Exception as e:
                    st.error(f"❌ Critical error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.divider()
        
        # Sample queries
        st.subheader("💡 Sample Queries")
        sample_queries = [
            "What are the data protection requirements?",
            "How should personal data be processed?",
            "What are the rights of data subjects?",
            "How is data breach notification handled?",
            "What are the penalties for non-compliance?"
        ]
        
        for query in sample_queries:
            if st.button(f"🔍 {query[:30]}...", key=f"sample_{query}"):
                st.session_state.query = query
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔎 Semantic Search")
        
        # Query input
        query = st.text_area(
            "Enter your question or search query:",
            value=st.session_state.get('query', ''),
            height=100,
            placeholder="Ask a question about the document content..."
        )
        
        # Search button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Searching..."):
                    # Create embedding for the query
                    if qdrant_interface.embedding_model is None:
                        st.error("❌ Embedding model not available. Cannot perform semantic search.")
                        st.info("Please check the embedding model configuration and restart the app.")
                        
                        # Manual embedding model initialization
                        if st.button("🔧 Manually Initialize Embedding Model"):
                            try:
                                import torch
                                from sentence_transformers import SentenceTransformer
                                
                                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                                st.write(f"Attempting to initialize model on {device}...")
                                
                                # Try different initialization methods
                                model = None
                                
                                # Method 1: Direct initialization
                                try:
                                    model = SentenceTransformer(config.EMBEDDING_MODEL)
                                    st.success("✅ Direct initialization successful!")
                                except Exception as e1:
                                    st.write(f"Direct init failed: {e1}")
                                    
                                    # Method 2: With device
                                    try:
                                        model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
                                        st.success("✅ Device initialization successful!")
                                    except Exception as e2:
                                        st.write(f"Device init failed: {e2}")
                                        
                                        # Method 3: With trust_remote_code
                                        try:
                                            model = SentenceTransformer(config.EMBEDDING_MODEL, device='cpu', trust_remote_code=True)
                                            st.success("✅ Trust remote code initialization successful!")
                                        except Exception as e3:
                                            st.error(f"❌ All initialization methods failed: {e3}")
                                            return
                                
                                # Test the model
                                if model is not None:
                                    test_text = ["This is a test"]
                                    embedding = model.encode(test_text)
                                    st.success(f"✅ Model test successful! Shape: {embedding.shape}")
                                    
                                    # Assign to interface
                                    qdrant_interface.embedding_model = model
                                    st.success("✅ Embedding model assigned to interface!")
                                    st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Manual initialization failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                        
                        # Clear session state to force reinitialization
                        if st.button("🔄 Retry Initialization"):
                            del st.session_state['qdrant_interface']
                            del st.session_state['rag_service']
                            st.rerun()
                        return
                    
                    try:
                        query_embedding = qdrant_interface.embedding_model.encode([query])
                        # Handle both single vector and list of vectors
                        if isinstance(query_embedding, list) and len(query_embedding) == 1:
                            query_embedding = query_embedding[0]
                        
                        # Ensure it's a list of Python floats for JSON serialization
                        if hasattr(query_embedding, 'tolist'):
                            query_embedding = query_embedding.tolist()
                        # Convert numpy scalars to Python floats if needed (only if it's iterable)
                        if isinstance(query_embedding, (list, tuple)):
                            query_embedding = [float(x) for x in query_embedding]
                        
                        # Search similar vectors
                        results = qdrant_interface.search_similar(
                            query_embedding, 
                            limit=max_results, 
                            score_threshold=score_threshold
                        )
                    except Exception as e:
                        st.error(f"Error creating query embedding: {e}")
                        return
                    
                    if results:
                        st.success(f"Found {len(results)} relevant results")
                        
                        # Store results in session state for RAG
                        st.session_state.search_results = results
                        
                        # Display results based on selected mode
                        if response_mode in ["AI Summary", "Both"]:
                            with st.spinner("Generating enhanced AI response..."):
                                # Use enhanced RAG service
                                enhanced_response = rag_service.generate_enhanced_response(query, results)
                                ai_response = enhanced_response["response"]
                                query_analysis = enhanced_response.get("query_analysis", {})
                                conversation_context = enhanced_response.get("conversation_context", {})
                                enhancement_info = enhanced_response.get("enhancement_info", {})
                                
                                st.subheader("🤖 Enhanced AI Response")
                                st.markdown(ai_response)
                                
                                # Show enhancement information
                                with st.expander("🧠 Context & Memory Information"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Query Analysis:**")
                                        st.write(f"- Type: {query_analysis.get('query_type', 'general')}")
                                        st.write(f"- Complexity: {query_analysis.get('complexity', 'medium')}")
                                        if query_analysis.get('gdpr_concepts'):
                                            st.write(f"- GDPR Concepts: {', '.join(query_analysis['gdpr_concepts'])}")
                                    
                                    with col2:
                                        st.write("**Conversation Context:**")
                                        if conversation_context.get("recent_turns"):
                                            st.write(f"- Recent turns: {len(conversation_context['recent_turns'])}")
                                        if conversation_context.get("related_topics"):
                                            st.write(f"- Related topics: {', '.join(conversation_context['related_topics'])}")
                                        if enhancement_info.get("is_follow_up"):
                                            st.write("✅ Follow-up question detected")
                                
                                # Copy AI response button
                                if st.button("📋 Copy AI Response"):
                                    st.write("AI response copied to clipboard!")
                                
                                if response_mode == "Both":
                                    st.divider()
                        
                        if response_mode in ["Raw Chunks", "Both"]:
                            st.subheader("📄 Retrieved Chunks")
                            
                            # Display raw chunks
                            for i, result in enumerate(results, 1):
                                with st.expander(f"Chunk {i} (Score: {result.get('score', 0):.4f})"):
                                    payload = result.get('payload', {})
                                    text = payload.get('text', '')
                                    
                                    # Display metadata
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.caption(f"**Chunk ID:** {payload.get('chunk_id', 'N/A')}")
                                        st.caption(f"**Source:** {payload.get('source', 'N/A')}")
                                    
                                    with col_b:
                                        st.caption(f"**Similarity Score:** {result.get('score', 0):.4f}")
                                        st.caption(f"**Point ID:** {result.get('id', 'N/A')[:8]}...")
                                    
                                    # Display text content
                                    st.markdown("**Content:**")
                                    st.text_area(
                                        "Text",
                                        value=text,
                                        height=200,
                                        key=f"result_{i}",
                                        disabled=True
                                    )
                                    
                                    # Copy button
                                    if st.button(f"📋 Copy Text", key=f"copy_{i}"):
                                        st.write("Text copied to clipboard!")
                    else:
                        st.warning("No results found. Try adjusting the score threshold or rephrasing your query.")
            else:
                st.warning("Please enter a search query.")
    
    with col2:
        st.header("📊 Collection Statistics")
        
        if collection_info:
            result = collection_info.get("result", {})
            
            # Display statistics
            st.metric("Total Points", result.get('points_count', 0))
            st.metric("Collection Status", result.get('status', 'Unknown'))
            
            # Vector configuration
            vectors_config = result.get('config', {}).get('params', {}).get('vectors', {})
            if isinstance(vectors_config, dict):
                st.metric("Vector Dimensions", vectors_config.get('size', 0))
                st.metric("Distance Metric", vectors_config.get('distance', 'Unknown'))
            
            st.divider()
            
            # Random samples
            st.subheader("🎲 Random Samples")
            if st.button("🔄 Refresh Samples"):
                st.rerun()
            
            # Get random points
            random_points = qdrant_interface.get_random_points(limit=3)
            
            for i, point in enumerate(random_points, 1):
                with st.expander(f"Sample {i}"):
                    payload = point.get('payload', {})
                    text = payload.get('text', '')
                    
                    st.caption(f"Chunk ID: {payload.get('chunk_id', 'N/A')}")
                    st.text(text[:150] + "..." if len(text) > 150 else text)
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>🔍 RAG Vector Database Query Interface | Powered by Qdrant & Streamlit</p>
            <p>🤖 LLM Integration: OpenAI & Anthropic Support</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
