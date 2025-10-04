# Configuration file for RAG Vector Database
import os

# Qdrant Vector Database Configuration
QDRANT_API_KEY = "your-qdrant-api-key-here"
QDRANT_CLUSTER_URL = "https://your-cluster-url.qdrant.io"
QDRANT_COLLECTION_NAME = "documents"

# PDF Processing Configuration
PDF_FILE_PATH = "CELEX_32016R0679_EN_TXT.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector Database Settings
VECTOR_DIMENSIONS = 384
DISTANCE_METRIC = "Cosine"
BATCH_SIZE = 100

# File Paths
EMBEDDINGS_DATA_FILE = "embeddings_data.json"
EMBEDDINGS_VECTORS_FILE = "embeddings_vectors.npy"
TEXT_CHUNKS_FILE = "text_chunks.txt"

# LLM Configuration
LLM_PROVIDER = "openai"  # Options: "openai", "anthropic", "local"
OPENAI_API_KEY = "your-openai-api-key-here"
ANTHROPIC_API_KEY = ""  # Add your Anthropic API key here
LLM_MODEL = "gpt-3.5-turbo"  # Options: "gpt-3.5-turbo", "gpt-4", "claude-3-haiku", "claude-3-sonnet"
MAX_TOKENS = 1000
TEMPERATURE = 0.1
