# PDF to Vector Database Ingestion

This project provides tools to extract text from PDFs, create embeddings, and store them in vector databases for RAG (Retrieval-Augmented Generation) applications.

## Files Created

### Core Scripts
- `pdf_ingestion_flexible.py` - Main ingestion script with support for multiple vector databases
- `upload_to_qdrant.py` - Script to upload embeddings to Qdrant vector database
- `streamlit_app.py` - Web interface for querying the vector database with RAG
- `rag_service.py` - RAG service integrating LLM with vector search
- `view_qdrant_data.py` - Command-line tool to view and analyze vector data
- `test_rag.py` - Test script for RAG functionality
- `requirements.txt` - Python dependencies

### Launcher Scripts
- `run_streamlit.py` - Python launcher for the Streamlit app
- `start_app.bat` - Windows batch file to start the app
- `launch_app.bat` - Simple Windows launcher
- `config.py` - Configuration file (replaces .env)
- `LLM_SETUP_GUIDE.md` - Guide for configuring LLM providers

### Generated Data Files
- `embeddings_data.json` - Complete embedding data with metadata
- `embeddings_vectors.npy` - Numpy array of vectors only
- `text_chunks.txt` - Extracted text chunks for reference

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run PDF Ingestion
```bash
python pdf_ingestion_flexible.py
```

This script will:
- Extract text from `CELEX_32016R0679_EN_TXT.pdf`
- Split text into 1000-character chunks with 200-character overlap
- Create embeddings using the `all-MiniLM-L6-v2` model
- Save embeddings locally (since no vector database is configured)

### 3. Upload to Vector Database
```bash
python upload_to_qdrant.py
```

This will upload your embeddings to your Qdrant cluster.

### 4. Configure LLM (Optional but Recommended)
To enable AI-powered responses, configure an LLM provider in `config.py`:

```python
# For OpenAI
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "your-openai-api-key"

# For Anthropic
LLM_PROVIDER = "anthropic" 
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

See `LLM_SETUP_GUIDE.md` for detailed instructions.

### 5. Launch Streamlit Web Interface
```bash
streamlit run streamlit_app.py
```

Or use the launcher scripts:
```bash
# Python launcher
python run_streamlit.py

# Windows batch file
start_app.bat
```

The web interface will open at `http://localhost:8501` and allows you to:
- Query your PDF content using natural language
- Get AI-powered summaries using OpenAI or Anthropic LLMs
- View similarity scores and relevant text chunks
- Choose between AI responses, raw chunks, or both
- Browse collection statistics
- See random samples from your data

## Vector Database Support

The scripts support multiple vector database providers:
- **Pinecone** - `https://api.pinecone.io`
- **Weaviate** - `https://api.weaviate.io`
- **ChromaDB** - Local or hosted
- **Generic Vector DB** - Any REST API compatible database

## Configuration

### API Key
Your API key is already configured in the scripts:
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.xFcDgqa1ADs22JIIwFNvrN45FWcgRQQgqZ1AetfKoSU
```

### Custom Vector Database Endpoint
If you know your vector database endpoint, you can modify the scripts or use the upload script with option 2 to specify the correct URL.

## Results

The PDF processing extracted:
- **366,545 characters** of text
- **459 text chunks** (1000 characters each with 200-character overlap)
- **459 embeddings** (384-dimensional vectors)

## Next Steps

1. **Configure Vector Database**: Set up your vector database and update the endpoint in the scripts
2. **Query Interface**: Create a query interface to search the embeddings
3. **RAG Implementation**: Build a RAG system using the stored embeddings

## Troubleshooting

### Common Issues
1. **Vector Database Connection**: The script will save embeddings locally if the vector database is not accessible
2. **API Key**: Ensure your API key is valid for your vector database provider
3. **Dependencies**: Make sure all Python packages are installed correctly

### Local Backup
If vector database upload fails, embeddings are saved locally in multiple formats:
- JSON format with full metadata
- Numpy format for direct vector operations
- Text format for human reading

## File Structure
```
Assignment5_RAG/
├── CELEX_32016R0679_EN_TXT.pdf     # Source PDF
├── pdf_ingestion_flexible.py       # Main ingestion script
├── upload_to_qdrant.py             # Qdrant upload script
├── streamlit_app.py                # Web interface with RAG
├── rag_service.py                  # RAG service
├── view_qdrant_data.py             # CLI viewer
├── test_rag.py                     # RAG test script
├── run_streamlit.py                # Python launcher
├── start_app.bat                   # Windows launcher
├── launch_app.bat                  # Simple launcher
├── requirements.txt                 # Dependencies
├── config.py                       # Configuration
├── LLM_SETUP_GUIDE.md              # LLM setup guide
├── embeddings_data.json            # Complete embedding data
├── embeddings_vectors.npy          # Vector arrays
├── text_chunks.txt                 # Text chunks
└── README.md                       # This file
```
