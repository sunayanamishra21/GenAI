# Parameter Tuning Guide for RAG System

This guide explains how to adjust various parameters in your RAG system to optimize performance, cost, and accuracy.

## Configuration Parameters

### 1. LLM Parameters (`config.py`)

#### `MAX_TOKENS`
- **Purpose**: Maximum tokens in LLM response
- **Range**: 100-4000
- **Default**: 1000
- **Impact**: 
  - Higher = Longer responses, higher cost
  - Lower = Shorter responses, lower cost
- **Recommendations**:
  - Simple queries: 500-800
  - Complex analysis: 1500-2000
  - Cost optimization: 300-500

#### `TEMPERATURE`
- **Purpose**: Controls randomness in responses
- **Range**: 0.0-1.0
- **Default**: 0.1
- **Impact**:
  - 0.0 = Most deterministic, factual
  - 1.0 = Most creative, random
- **Recommendations**:
  - Legal documents: 0.0-0.2
  - General Q&A: 0.1-0.3
  - Creative tasks: 0.5-0.8

#### `LLM_MODEL`
- **Purpose**: Which LLM model to use
- **Options**:
  - OpenAI: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo-preview`
  - Anthropic: `claude-3-haiku-20240307`, `claude-3-sonnet-20240229`
- **Impact**:
  - Faster models = Lower cost, less capable
  - Slower models = Higher cost, more capable
- **Recommendations**:
  - Cost-sensitive: `gpt-3.5-turbo` or `claude-3-haiku`
  - Quality-focused: `gpt-4` or `claude-3-sonnet`

### 2. Vector Search Parameters

#### `max_results` (Streamlit UI)
- **Purpose**: Number of chunks to retrieve
- **Range**: 1-20
- **Default**: 10
- **Impact**:
  - Higher = More context, higher cost
  - Lower = Less context, lower cost
- **Recommendations**:
  - Simple queries: 3-5
  - Complex queries: 8-12
  - Comprehensive analysis: 15-20

#### `score_threshold` (Streamlit UI)
- **Purpose**: Minimum similarity score for results
- **Range**: 0.0-1.0
- **Default**: 0.0
- **Impact**:
  - Higher = More relevant, fewer results
  - Lower = More results, may include irrelevant
- **Recommendations**:
  - High precision: 0.7-0.8
  - Balanced: 0.5-0.6
  - High recall: 0.2-0.4

### 3. PDF Processing Parameters

#### `CHUNK_SIZE`
- **Purpose**: Size of text chunks for embedding
- **Range**: 500-2000
- **Default**: 1000
- **Impact**:
  - Larger = More context per chunk, fewer chunks
  - Smaller = Less context, more chunks
- **Recommendations**:
  - Legal documents: 1000-1500
  - Technical docs: 800-1200
  - General text: 500-1000

#### `CHUNK_OVERLAP`
- **Purpose**: Overlap between consecutive chunks
- **Range**: 50-300
- **Default**: 200
- **Impact**:
  - Higher = Better context continuity
  - Lower = More unique chunks
- **Recommendations**:
  - Legal documents: 200-300
  - Technical docs: 100-200
  - General text: 50-150

## Performance Optimization

### 1. Cost Optimization

#### Reduce Token Usage
```python
# In config.py
MAX_TOKENS = 500  # Reduce from 1000
TEMPERATURE = 0.0  # Most deterministic
LLM_MODEL = "gpt-3.5-turbo"  # Cheapest option
```

#### Optimize Search Parameters
```python
# In Streamlit UI
max_results = 5  # Reduce from 10
score_threshold = 0.6  # Higher threshold
```

#### Use Fallback Mode
```python
# In config.py
LLM_PROVIDER = "local"  # No API calls
```

### 2. Quality Optimization

#### Increase Context
```python
# In config.py
MAX_TOKENS = 1500  # More detailed responses
CHUNK_SIZE = 1200  # Larger chunks
CHUNK_OVERLAP = 250  # Better continuity
```

#### Improve Search
```python
# In Streamlit UI
max_results = 12  # More context
score_threshold = 0.4  # Lower threshold
```

### 3. Speed Optimization

#### Faster Models
```python
# In config.py
LLM_MODEL = "gpt-3.5-turbo"  # Fastest
# or
LLM_MODEL = "claude-3-haiku-20240307"  # Fast Anthropic
```

#### Reduce Processing
```python
# In config.py
MAX_TOKENS = 800  # Shorter responses
```

## Monitoring and Adjustment

### 1. Usage Monitoring

Run the monitoring script:
```bash
python monitor_usage.py
```

This shows:
- Total queries and costs
- Average tokens per query
- Cost projections
- Recent query history

### 2. Performance Testing

Test different configurations:
```bash
python test_queries.py
```

Monitor:
- Response quality
- Response time
- Cost per query

### 3. A/B Testing

Test different parameter sets:
1. Create multiple config files
2. Run tests with each configuration
3. Compare results and costs
4. Choose optimal settings

## Recommended Configurations

### 1. Cost-Conscious Setup
```python
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 500
TEMPERATURE = 0.1
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
```

### 2. Quality-Focused Setup
```python
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4"
MAX_TOKENS = 1500
TEMPERATURE = 0.0
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
```

### 3. Balanced Setup
```python
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1000
TEMPERATURE = 0.1
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

### 4. Testing Setup
```python
LLM_PROVIDER = "local"  # No API costs
MAX_TOKENS = 1000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## Troubleshooting

### Common Issues

#### 1. High Costs
- **Cause**: High MAX_TOKENS or too many queries
- **Solution**: Reduce MAX_TOKENS, use cheaper model

#### 2. Poor Response Quality
- **Cause**: Low MAX_TOKENS or poor context
- **Solution**: Increase MAX_TOKENS, improve chunking

#### 3. Slow Responses
- **Cause**: Large model or high MAX_TOKENS
- **Solution**: Use faster model, reduce MAX_TOKENS

#### 4. Irrelevant Results
- **Cause**: Low score_threshold
- **Solution**: Increase score_threshold

### Performance Metrics

Monitor these metrics:
- **Response Time**: < 5 seconds
- **Cost per Query**: < $0.01
- **Relevance Score**: > 0.6
- **User Satisfaction**: High

## Advanced Tuning

### 1. Dynamic Parameters

Adjust parameters based on query type:
- Simple questions: Lower MAX_TOKENS
- Complex analysis: Higher MAX_TOKENS
- Legal queries: Lower TEMPERATURE

### 2. Context Optimization

- Analyze which chunks are most relevant
- Adjust CHUNK_SIZE based on content type
- Optimize CHUNK_OVERLAP for better continuity

### 3. Model Selection

- Use different models for different query types
- Implement model fallback for cost optimization
- Monitor model performance and costs

## Best Practices

1. **Start Conservative**: Begin with lower limits
2. **Monitor Closely**: Track usage and costs
3. **Test Regularly**: Validate performance
4. **Adjust Gradually**: Make small changes
5. **Document Changes**: Keep track of modifications
6. **Backup Configs**: Save working configurations

## Support

For issues with parameter tuning:
1. Check the monitoring output
2. Test with different configurations
3. Review the troubleshooting section
4. Consult the LLM provider documentation

Happy tuning! 🚀
