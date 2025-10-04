# LLM Setup Guide for RAG System

This guide explains how to configure Large Language Model (LLM) integration for your RAG system.

## Overview

The RAG system supports multiple LLM providers:
- **OpenAI** (GPT-3.5-turbo, GPT-4)
- **Anthropic** (Claude-3-haiku, Claude-3-sonnet)
- **Fallback Mode** (No LLM configured)

## Configuration Steps

### 1. OpenAI Setup

#### Get API Key
1. Visit [OpenAI Platform](https://platform.openai.com)
2. Sign up or log in
3. Go to API Keys section
4. Create a new secret key
5. Copy the API key (starts with `sk-`)

#### Configure in `config.py`
```python
# LLM Configuration
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-your-api-key-here"
LLM_MODEL = "gpt-3.5-turbo"  # or "gpt-4"
MAX_TOKENS = 1000
TEMPERATURE = 0.1
```

### 2. Anthropic Setup

#### Get API Key
1. Visit [Anthropic Console](https://console.anthropic.com)
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the API key (starts with `sk-ant-`)

#### Configure in `config.py`
```python
# LLM Configuration
LLM_PROVIDER = "anthropic"
ANTHROPIC_API_KEY = "sk-ant-your-api-key-here"
LLM_MODEL = "claude-3-haiku-20240307"  # or "claude-3-sonnet-20240229"
MAX_TOKENS = 1000
TEMPERATURE = 0.1
```

### 3. Model Options

#### OpenAI Models
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-4` - More capable but slower and more expensive
- `gpt-4-turbo-preview` - Latest GPT-4 with improvements

#### Anthropic Models
- `claude-3-haiku-20240307` - Fast and cost-effective
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-opus-20240229` - Most capable but expensive

### 4. Configuration Parameters

#### `MAX_TOKENS`
- Maximum tokens in the LLM response
- Recommended: 500-2000 for summaries
- Default: 1000

#### `TEMPERATURE`
- Controls randomness in responses
- 0.0 = Most deterministic
- 1.0 = Most random
- Recommended: 0.1-0.3 for factual responses
- Default: 0.1

## Usage in Streamlit App

### Response Modes
1. **AI Summary** - Only LLM-generated response
2. **Raw Chunks** - Only retrieved text chunks
3. **Both** - LLM response + raw chunks

### Features
- **Semantic Search** - Find relevant chunks using embeddings
- **AI Summarization** - Generate comprehensive answers
- **Source Attribution** - See which chunks were used
- **Similarity Scores** - Understand relevance

## Example Queries

Try these sample queries in the Streamlit app:

### GDPR-Specific Questions
- "What are the data subject rights under GDPR?"
- "How should data breaches be reported?"
- "What are the penalties for non-compliance?"
- "When is consent required for data processing?"
- "What are the principles of data protection?"

### General Legal Questions
- "What is the purpose of this regulation?"
- "Who is responsible for compliance?"
- "What are the key requirements?"
- "How is enforcement handled?"

## Troubleshooting

### Common Issues

#### 1. "LLM not configured" Warning
- **Cause**: No API key provided or wrong provider
- **Solution**: Add API key to `config.py` and restart app

#### 2. "Error generating response" Message
- **Cause**: Invalid API key or network issues
- **Solution**: Verify API key and check internet connection

#### 3. Fallback Response Only
- **Cause**: LLM provider not properly configured
- **Solution**: Check `LLM_PROVIDER` setting in `config.py`

#### 4. Empty or Incomplete Responses
- **Cause**: Low `MAX_TOKENS` or poor context
- **Solution**: Increase `MAX_TOKENS` or adjust search parameters

### Testing Configuration

Run the test script to verify setup:
```bash
python test_rag.py
```

This will show:
- LLM provider status
- Available models
- Sample response generation

## Cost Considerations

### OpenAI Pricing (per 1K tokens)
- GPT-3.5-turbo: ~$0.002 input, ~$0.002 output
- GPT-4: ~$0.03 input, ~$0.06 output

### Anthropic Pricing (per 1K tokens)
- Claude-3-haiku: ~$0.00025 input, ~$0.00125 output
- Claude-3-sonnet: ~$0.003 input, ~$0.015 output

### Cost Optimization Tips
1. Use smaller models for simple queries
2. Limit `MAX_TOKENS` appropriately
3. Adjust search parameters to get better context
4. Use fallback mode for testing

## Security Notes

### API Key Security
- Never commit API keys to version control
- Use environment variables in production
- Rotate keys regularly
- Monitor usage and costs

### Data Privacy
- API calls send data to external services
- Consider data sensitivity before using
- Check provider privacy policies
- Use local models for sensitive data

## Next Steps

1. **Configure your preferred LLM provider**
2. **Test with sample queries**
3. **Adjust parameters as needed**
4. **Monitor usage and costs**
5. **Explore advanced features**

## Support

For issues with:
- **OpenAI**: Check [OpenAI Documentation](https://platform.openai.com/docs)
- **Anthropic**: Check [Anthropic Documentation](https://docs.anthropic.com)
- **RAG System**: Review this guide and check logs

Happy querying! 🚀
