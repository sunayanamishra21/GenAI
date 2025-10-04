# Self-RAG (Self-Reflective Retrieval-Augmented Generation) System

## Overview

This is an advanced RAG system that implements **Self-RAG** capabilities, including:

- **Intelligent Retrieval Decision**: Analyzes queries to determine if retrieval is needed
- **Evidence Relevance Scoring**: Uses LLM to score retrieved evidence relevance
- **Response Quality Reflection**: Self-assesses response quality and identifies issues
- **Self-Correction**: Automatically corrects responses when quality is low
- **Memory-Aware Conversations**: Maintains conversation context and history

## Key Features

### 1. Retrieval Decision Analysis
- Analyzes query intent and type
- Decides whether to retrieve from knowledge base
- Provides reasoning for retrieval decisions
- Confidence scoring for decisions

### 2. Evidence Retrieval and Scoring
- Retrieves relevant documents from vector database
- Uses LLM to score evidence relevance (0.0-1.0)
- Ranks evidence by relevance score
- Filters low-relevance evidence

### 3. Response Generation
- Generates responses using retrieved evidence
- Adapts response style based on query type
- Integrates evidence with proper citations
- Handles cases with no relevant evidence

### 4. Self-Reflection
- Evaluates response quality (high/medium/low)
- Identifies specific issues and problems
- Provides improvement suggestions
- Confidence scoring for assessments

### 5. Self-Correction
- Automatically corrects responses when quality is low
- Uses reflection insights to guide corrections
- Maintains correction reasoning
- Falls back gracefully when correction fails

## Architecture

```
Query Input
    ↓
Retrieval Decision Analysis
    ↓
Evidence Retrieval (if needed)
    ↓
Response Generation
    ↓
Self-Reflection
    ↓
Self-Correction (if needed)
    ↓
Final Response + Metadata
```

## Components

### Core Services

1. **SelfRAGService**: Main orchestrator
2. **QdrantQueryInterface**: Vector database interface
3. **ContextAwareService**: Query analysis and context
4. **MemoryAwareService**: Conversation memory management

### Data Structures

- **RetrievalDecision**: Decision about whether to retrieve
- **EvidenceItem**: Retrieved evidence with relevance score
- **ResponseReflection**: Self-assessment of response quality
- **SelfRAGResponse**: Complete response with all metadata

## Usage

### 1. Launch the Self-RAG Application

```bash
# Using the batch file (Windows)
start_self_rag.bat

# Or directly with Python
python -m streamlit run self_rag_app.py --server.port 8502
```

### 2. Access the Interface

Open your browser and go to: `http://localhost:8502`

### 3. Features Available

- **Query Interface**: Enter questions about your documents
- **Retrieval Analysis**: View decision-making process
- **Evidence Display**: See retrieved evidence with relevance scores
- **Self-Reflection**: View quality assessment and issues
- **Self-Correction**: See if and how responses were corrected
- **Memory Management**: Track conversation history

## Configuration

### API Keys Required

1. **Qdrant API Key**: For vector database access
2. **OpenAI API Key**: For LLM operations (recommended)
3. **Anthropic API Key**: Alternative LLM provider

### Configuration File

Copy `config_template.py` to `config.py` and update:

```python
# Qdrant Configuration
QDRANT_API_KEY = "your-qdrant-api-key"
QDRANT_CLUSTER_URL = "https://your-cluster.qdrant.io"

# LLM Configuration
LLM_PROVIDER = "openai"  # or "anthropic"
OPENAI_API_KEY = "your-openai-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

## Testing

### Run Self-RAG Tests

```bash
python test_self_rag.py
```

This will test:
- Individual component functionality
- End-to-end Self-RAG workflow
- Error handling and fallbacks
- Memory management

### Test Individual Components

```python
from self_rag_service import SelfRAGService

# Test retrieval decision
decision = self_rag_service._analyze_retrieval_need("What is GDPR?")

# Test evidence retrieval
evidence = self_rag_service._retrieve_evidence("What is GDPR?")

# Test self-reflection
reflection = self_rag_service._reflect_on_response(query, response, evidence)
```

## Sample Queries

### Factual Queries (Will Retrieve)
- "What are the data protection requirements under GDPR?"
- "How should personal data be processed legally?"
- "What are the rights of data subjects?"

### General Queries (May Not Retrieve)
- "Hello, how are you?"
- "What's the weather like?"
- "Tell me a joke"

### Analytical Queries (Complex Retrieval)
- "Compare GDPR requirements with other privacy laws"
- "Analyze the impact of data breach notifications"

## Response Quality Levels

### High Quality
- Accurate information
- Well-supported by evidence
- Complete answer to query
- Clear and well-structured

### Medium Quality
- Generally accurate
- Some evidence support
- Partially complete
- Reasonably clear

### Low Quality
- Inaccurate or unsupported
- Missing key information
- Poor structure
- Unclear or confusing

## Self-Correction Triggers

Responses are automatically corrected when:

1. **Quality is Low**: Response has significant issues
2. **Quality is Medium**: Response has minor issues that can be improved
3. **Issues Identified**: Specific problems found during reflection
4. **Suggestions Available**: Improvement suggestions can be applied

## Memory Management

The system maintains:

- **Session History**: All conversations in a session
- **Turn Metadata**: Self-RAG specific information for each turn
- **Quality Tracking**: Response quality over time
- **Correction History**: When and why responses were corrected

## Error Handling

### Graceful Fallbacks

1. **LLM Unavailable**: Falls back to heuristic decisions
2. **Vector DB Error**: Continues without retrieval
3. **Reflection Failure**: Uses simple quality assessment
4. **Correction Failure**: Returns original response

### Error Recovery

- Automatic retry for transient errors
- Fallback to simpler methods
- User-friendly error messages
- System continues operating

## Performance Considerations

### Optimization Strategies

1. **Caching**: Embedding model and responses
2. **Batching**: Multiple evidence scoring
3. **Limiting**: Evidence items and response length
4. **Streaming**: Real-time response generation

### Resource Usage

- **Memory**: Conversation history and embeddings
- **Compute**: LLM API calls for reflection/correction
- **Storage**: Vector database and conversation memory
- **Network**: API calls to external services

## Comparison with Basic RAG

| Feature | Basic RAG | Self-RAG |
|---------|-----------|----------|
| Retrieval | Always retrieves | Intelligent decision |
| Evidence Use | All retrieved items | Relevance-scored |
| Quality Control | None | Self-reflection |
| Error Correction | None | Self-correction |
| Adaptability | Static | Dynamic |
| Transparency | Limited | Full process visibility |

## Future Enhancements

### Planned Features

1. **Multi-Modal**: Image and document analysis
2. **Real-Time**: Streaming responses
3. **Collaborative**: Multi-user conversations
4. **Advanced**: Fine-tuned reflection models
5. **Integration**: External knowledge sources

### Research Directions

1. **Better Reflection**: Improved quality assessment
2. **Efficient Correction**: Faster self-correction
3. **Context Awareness**: Better query understanding
4. **Memory Optimization**: Smarter conversation management

## Troubleshooting

### Common Issues

1. **Service Initialization Failed**
   - Check API keys in config.py
   - Verify network connectivity
   - Ensure dependencies are installed

2. **No Evidence Retrieved**
   - Check vector database connection
   - Verify collection exists
   - Check embedding model compatibility

3. **LLM Errors**
   - Verify API keys
   - Check rate limits
   - Monitor token usage

4. **Memory Issues**
   - Clear conversation history
   - Restart services
   - Check disk space

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy config template: `cp config_template.py config.py`
4. Run tests: `python test_self_rag.py`

### Code Structure

- `self_rag_service.py`: Core Self-RAG implementation
- `self_rag_app.py`: Streamlit interface
- `test_self_rag.py`: Test suite
- `memory_aware_service.py`: Enhanced memory management

## License

This project is part of the GenAI Training Assignment 5.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite
3. Review the error logs
4. Create an issue with details

---

**Self-RAG System**: Advanced RAG with self-reflection and self-correction capabilities for improved accuracy and reliability.
