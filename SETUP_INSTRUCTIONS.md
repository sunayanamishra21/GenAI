# RAG Vector Database Application Setup

## Prerequisites
- Python 3.8+
- Git

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sunayanamishra21/GenAI.git
   cd GenAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the application**
   ```bash
   cp config_template.py config.py
   ```
   
   Edit `config.py` with your actual API keys:
   - `QDRANT_API_KEY`: Your Qdrant vector database API key
   - `QDRANT_CLUSTER_URL`: Your Qdrant cluster URL
   - `OPENAI_API_KEY`: Your OpenAI API key (optional for LLM features)

4. **Run the application**
   ```bash
   python -m streamlit run streamlit_app.py --server.port 8501
   ```

5. **Access the application**
   Open your browser and go to: `http://localhost:8501`

## Features

- **PDF Ingestion**: Extract text from PDF documents
- **Vector Search**: Semantic search using vector embeddings
- **RAG (Retrieval-Augmented Generation)**: AI-powered responses
- **Context Awareness**: Intelligent query analysis
- **Memory Awareness**: Conversation history tracking
- **Streamlit Interface**: User-friendly web interface

## Sample Queries

- "What are the data protection requirements?"
- "How should personal data be processed?"
- "What are the rights of data subjects?"

## Troubleshooting

If you encounter embedding model issues:
1. Click "Quick Start (Dummy Model)" in the sidebar
2. This will use dummy embeddings for testing

## Support

For issues and questions, please open an issue on GitHub.
