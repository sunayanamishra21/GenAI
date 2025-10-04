"""
Enhanced RAG (Retrieval-Augmented Generation) Service
Integrates vector database retrieval with LLM generation, context awareness, and memory
"""

import openai
import anthropic
from typing import List, Dict, Optional
import config
from context_aware_service import ContextAwareService
from memory_aware_service import MemoryAwareService

class RAGService:
    def __init__(self):
        self.llm_provider = config.LLM_PROVIDER
        self.openai_api_key = config.OPENAI_API_KEY
        self.anthropic_api_key = config.ANTHROPIC_API_KEY
        self.model = config.LLM_MODEL
        self.max_tokens = config.MAX_TOKENS
        self.temperature = config.TEMPERATURE
        
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        
        if self.llm_provider == "openai" and self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        elif self.llm_provider == "anthropic" and self.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        
        # Initialize enhanced services
        self.context_aware_service = ContextAwareService()
        self.memory_aware_service = MemoryAwareService()
    
    def generate_enhanced_response(self, query: str, context_chunks: List[Dict]) -> Dict:
        """Generate an enhanced response with context awareness and memory"""
        
        if not context_chunks:
            return {
                "response": "No relevant information found to answer your question.",
                "query_analysis": {},
                "conversation_context": {},
                "enhancement_info": {}
            }
        
        # 1. Analyze query intent and context
        query_analysis = self.context_aware_service.analyze_query_intent(query)
        
        # 2. Enhance query with conversation memory
        memory_enhancement = self.memory_aware_service.enhance_query_with_memory(query)
        
        # 3. Get conversation context
        conversation_context = self.memory_aware_service.get_conversation_context()
        
        # 4. Prepare enhanced context from retrieved chunks
        context_text = self._prepare_enhanced_context(context_chunks, query_analysis, conversation_context)
        
        # 5. Create enhanced prompt
        prompt = self._create_enhanced_prompt(
            query, 
            context_text, 
            query_analysis, 
            conversation_context,
            memory_enhancement
        )
        
        # 6. Generate response
        if self.llm_provider == "openai" and self.openai_client:
            response = self._generate_enhanced_openai_response(prompt)
        elif self.llm_provider == "anthropic" and self.anthropic_client:
            response = self._generate_enhanced_anthropic_response(prompt)
        else:
            response = self._generate_fallback_response(query, context_chunks)
        
        # 7. Store turn in memory
        turn_id = self.memory_aware_service.add_turn_to_session(
            user_query=query,
            ai_response=response,
            retrieved_chunks=context_chunks,
            metadata={
                "query_analysis": query_analysis,
                "conversation_context": conversation_context
            }
        )
        
        return {
            "response": response,
            "query_analysis": query_analysis,
            "conversation_context": conversation_context,
            "enhancement_info": {
                "memory_enhancement": memory_enhancement,
                "turn_id": turn_id,
                "enhanced_query": memory_enhancement.get("enhanced_query", query)
            }
        }
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Legacy method for backward compatibility"""
        enhanced_response = self.generate_enhanced_response(query, context_chunks)
        return enhanced_response["response"]
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context text from retrieved chunks"""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            payload = chunk.get('payload', {})
            text = payload.get('text', '')
            score = chunk.get('score', 0)
            
            # Add chunk with metadata
            context_parts.append(
                f"[Chunk {i} - Relevance: {score:.3f}]\n"
                f"{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _prepare_enhanced_context(self, chunks: List[Dict], query_analysis: Dict, conversation_context: Dict) -> str:
        """Prepare enhanced context text with query analysis and conversation history"""
        
        context_parts = []
        
        # Add conversation context if available
        if conversation_context.get("recent_turns"):
            context_parts.append("=== CONVERSATION CONTEXT ===")
            for turn in conversation_context["recent_turns"]:
                context_parts.append(f"Previous Q: {turn['query']}")
                context_parts.append(f"Previous A: {turn['response']}")
            context_parts.append("")
        
        # Add query analysis context
        if query_analysis.get("gdpr_concepts"):
            context_parts.append("=== RELEVANT CONCEPTS ===")
            context_parts.append(f"Query Type: {query_analysis.get('query_type', 'general')}")
            context_parts.append(f"GDPR Concepts: {', '.join(query_analysis.get('gdpr_concepts', []))}")
            context_parts.append(f"Complexity: {query_analysis.get('complexity', 'medium')}")
            context_parts.append("")
        
        # Add retrieved chunks
        context_parts.append("=== RETRIEVED DOCUMENT CONTENT ===")
        for i, chunk in enumerate(chunks, 1):
            payload = chunk.get('payload', {})
            text = payload.get('text', '')
            score = chunk.get('score', 0)
            context_score = chunk.get('context_score', score)
            
            # Add chunk with enhanced metadata
            context_parts.append(
                f"[Chunk {i} - Relevance: {score:.3f}, Context: {context_score:.3f}]\n"
                f"{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM"""
        
        return f"""You are an AI assistant that answers questions based on the provided context from a legal document (GDPR regulation).

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based ONLY on the information provided in the context above.
2. If the context doesn't contain enough information to answer the question, say so clearly.
3. Be precise and cite specific parts of the text when possible.
4. Structure your response clearly with bullet points or numbered lists when appropriate.
5. Keep your answer concise but comprehensive.

ANSWER:"""
    
    def _create_enhanced_prompt(self, query: str, context: str, query_analysis: Dict, 
                               conversation_context: Dict, memory_enhancement: Dict) -> Dict:
        """Create an enhanced prompt with context awareness and memory"""
        
        # Build enhanced system message
        system_parts = [
            "You are an AI assistant that answers questions based on provided context from a legal document (GDPR regulation).",
            "You have access to conversation history and can provide contextually aware responses."
        ]
        
        # Add conversation context guidance
        if conversation_context.get("recent_turns"):
            system_parts.append("Consider the conversation history when answering to provide coherent and contextually relevant responses.")
        
        # Add query type guidance
        query_type = query_analysis.get("query_type", "general")
        if query_type == "definition":
            system_parts.append("Focus on providing clear, comprehensive definitions with examples.")
        elif query_type == "procedure":
            system_parts.append("Provide step-by-step procedures and clear instructions.")
        elif query_type == "comparison":
            system_parts.append("Compare and contrast the different aspects clearly.")
        elif query_type == "requirement":
            system_parts.append("List all requirements and obligations clearly.")
        
        system_message = " ".join(system_parts)
        
        # Build enhanced user prompt
        user_parts = [f"CONTEXT:\n{context}\n\nQUESTION: {query}"]
        
        # Add memory enhancement info
        if memory_enhancement.get("is_follow_up"):
            user_parts.append("\nNote: This appears to be a follow-up question. Consider the conversation context.")
        
        if memory_enhancement.get("has_reference"):
            user_parts.append("\nNote: This question references previous context. Build upon earlier responses.")
        
        # Add related topics
        related_topics = conversation_context.get("related_topics", [])
        if related_topics:
            user_parts.append(f"\nRelated topics from conversation: {', '.join(related_topics)}")
        
        user_parts.append("""
INSTRUCTIONS:
1. Answer the question based on the provided context and conversation history.
2. If this is a follow-up question, build upon previous responses coherently.
3. If the context doesn't contain enough information, say so clearly.
4. Be precise and cite specific parts of the text when possible.
5. Structure your response clearly with bullet points or numbered lists when appropriate.
6. Consider the conversation flow and provide contextually relevant information.
7. Keep your answer concise but comprehensive.

ANSWER:""")
        
        return {
            "system": system_message,
            "user": "".join(user_parts)
        }
    
    def _generate_openai_response(self, prompt: str) -> str:
        """Generate response using OpenAI API"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating response with OpenAI: {str(e)}"
    
    def _generate_anthropic_response(self, prompt: str) -> str:
        """Generate response using Anthropic API"""
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            return f"Error generating response with Anthropic: {str(e)}"
    
    def _generate_fallback_response(self, query: str, chunks: List[Dict]) -> str:
        """Generate a fallback response when no LLM is configured"""
        
        if not chunks:
            return "No relevant information found to answer your question."
        
        # Simple fallback: return the most relevant chunk
        best_chunk = chunks[0]
        payload = best_chunk.get('payload', {})
        text = payload.get('text', '')
        score = best_chunk.get('score', 0)
        
        return f"""Based on the retrieved information (relevance score: {score:.3f}):

**Most Relevant Content:**
{text}

**Note:** This is a basic retrieval response. To get a more comprehensive answer, please configure an LLM provider (OpenAI or Anthropic) in the config.py file."""
    
    def _generate_enhanced_openai_response(self, prompt_dict: Dict) -> str:
        """Generate enhanced response using OpenAI API with structured prompt"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_dict["system"]},
                    {"role": "user", "content": prompt_dict["user"]}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating enhanced response with OpenAI: {str(e)}"
    
    def _generate_enhanced_anthropic_response(self, prompt_dict: Dict) -> str:
        """Generate enhanced response using Anthropic API with structured prompt"""
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": f"{prompt_dict['system']}\n\n{prompt_dict['user']}"}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            return f"Error generating enhanced response with Anthropic: {str(e)}"
    
    def is_llm_configured(self) -> bool:
        """Check if LLM is properly configured"""
        
        if self.llm_provider == "openai":
            return bool(self.openai_api_key and self.openai_client)
        elif self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key and self.anthropic_client)
        else:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models for the current provider"""
        
        if self.llm_provider == "openai":
            return [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo-preview"
            ]
        elif self.llm_provider == "anthropic":
            return [
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229"
            ]
        else:
            return []
