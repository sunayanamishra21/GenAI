"""
Self-RAG Service Implementation
Implements a Self-Reflective Retrieval-Augmented Generation system
"""

import openai
import anthropic
from typing import List, Dict, Optional, Tuple
import config
import json
import re
from dataclasses import dataclass
from enum import Enum

class RetrievalDecision(Enum):
    """Decision types for retrieval"""
    RETRIEVE = "retrieve"
    NO_RETRIEVE = "no_retrieve"

class ResponseQuality(Enum):
    """Response quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class RetrievalDecision:
    """Structured decision about retrieval"""
    decision: str
    reasoning: str
    confidence: float
    query_type: str

@dataclass
class EvidenceItem:
    """Evidence item with relevance score"""
    text: str
    relevance_score: float
    source: str
    chunk_id: str
    point_id: str

@dataclass
class ResponseReflection:
    """Self-reflection on response quality"""
    quality: str
    reasoning: str
    issues: List[str]
    suggestions: List[str]
    confidence: float

@dataclass
class SelfRAGResponse:
    """Complete Self-RAG response"""
    response: str
    retrieval_decision: RetrievalDecision
    evidence_items: List[EvidenceItem]
    reflection: ResponseReflection
    corrected_response: Optional[str] = None
    correction_reasoning: Optional[str] = None

class SelfRAGService:
    """Self-Reflective Retrieval-Augmented Generation Service"""
    
    def __init__(self, qdrant_interface, context_aware_service, memory_aware_service):
        self.qdrant_interface = qdrant_interface
        self.context_aware_service = context_aware_service
        self.memory_aware_service = memory_aware_service
        
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        
        if config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY != "your-openai-api-key-here":
            openai.api_key = config.OPENAI_API_KEY
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        if config.LLM_PROVIDER == "anthropic" and config.ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        
        self.model = config.LLM_MODEL
        self.max_tokens = config.MAX_TOKENS
        self.temperature = config.TEMPERATURE

    def generate_self_rag_response(self, query: str, session_id: str = "default") -> SelfRAGResponse:
        """Generate a complete Self-RAG response"""
        
        # Step 1: Analyze query and decide on retrieval
        retrieval_decision = self._analyze_retrieval_need(query)
        
        evidence_items = []
        if retrieval_decision.decision == "retrieve":
            # Step 2: Retrieve relevant evidence
            evidence_items = self._retrieve_evidence(query)
        
        # Step 3: Generate initial response
        response = self._generate_response(query, evidence_items, retrieval_decision)
        
        # Step 4: Self-reflect on response quality
        reflection = self._reflect_on_response(query, response, evidence_items)
        
        # Step 5: Self-correct if needed
        corrected_response = None
        correction_reasoning = None
        
        if reflection.quality in ["low", "medium"]:
            corrected_response, correction_reasoning = self._self_correct(
                query, response, evidence_items, reflection
            )
        
        # Step 6: Store in memory
        self._store_self_rag_turn(session_id, query, response, evidence_items, reflection)
        
        return SelfRAGResponse(
            response=corrected_response or response,
            retrieval_decision=retrieval_decision,
            evidence_items=evidence_items,
            reflection=reflection,
            corrected_response=corrected_response,
            correction_reasoning=correction_reasoning
        )

    def _analyze_retrieval_need(self, query: str) -> RetrievalDecision:
        """Analyze if retrieval is needed and why"""
        
        prompt = f"""Analyze the following query and decide whether retrieval from a knowledge base is needed.

Query: "{query}"

Consider these factors:
1. Does the query ask for factual information that might be in documents?
2. Does the query require specific details, examples, or citations?
3. Is the query asking for general knowledge that might not need retrieval?
4. Does the query ask for creative or opinion-based responses?

Respond in JSON format:
{{
    "decision": "retrieve" or "no_retrieve",
    "reasoning": "explanation of decision",
    "confidence": 0.0-1.0,
    "query_type": "factual", "creative", "analytical", "general"
}}"""

        try:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.1
                )
                result = json.loads(response.choices[0].message.content.strip())
            else:
                # Fallback decision logic
                result = self._fallback_retrieval_decision(query)
            
            return RetrievalDecision(
                decision=result.get("decision", "retrieve"),
                reasoning=result.get("reasoning", "Fallback decision"),
                confidence=result.get("confidence", 0.7),
                query_type=result.get("query_type", "factual")
            )
            
        except Exception as e:
            # Fallback to simple heuristics
            return self._fallback_retrieval_decision(query)

    def _fallback_retrieval_decision(self, query: str) -> Dict:
        """Fallback retrieval decision based on keywords"""
        query_lower = query.lower()
        
        # Keywords that suggest retrieval is needed
        retrieval_keywords = [
            "what", "how", "when", "where", "why", "explain", "describe",
            "requirements", "rights", "obligations", "procedures", "process",
            "data protection", "gdpr", "consent", "processing", "violation"
        ]
        
        # Keywords that suggest no retrieval needed
        no_retrieval_keywords = [
            "hello", "hi", "thanks", "thank you", "goodbye", "bye",
            "how are you", "what's up", "creative", "imagine", "write a story"
        ]
        
        has_retrieval_keywords = any(keyword in query_lower for keyword in retrieval_keywords)
        has_no_retrieval_keywords = any(keyword in query_lower for keyword in no_retrieval_keywords)
        
        if has_no_retrieval_keywords:
            return {
                "decision": "no_retrieve",
                "reasoning": "Query appears to be conversational or creative",
                "confidence": 0.8,
                "query_type": "general"
            }
        elif has_retrieval_keywords:
            return {
                "decision": "retrieve",
                "reasoning": "Query appears to seek factual information",
                "confidence": 0.7,
                "query_type": "factual"
            }
        else:
            return {
                "decision": "retrieve",
                "reasoning": "Default to retrieval for unknown query types",
                "confidence": 0.5,
                "query_type": "general"
            }

    def _retrieve_evidence(self, query: str, limit: int = 5) -> List[EvidenceItem]:
        """Retrieve and score relevant evidence"""
        
        try:
            # Create query embedding
            query_embedding = self.qdrant_interface.embedding_model.encode([query])
            if isinstance(query_embedding, list) and len(query_embedding) == 1:
                query_embedding = query_embedding[0]
            
            # Ensure it's a list of floats
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            if isinstance(query_embedding, (list, tuple)):
                # Check if elements are already floats or need conversion
                if query_embedding and isinstance(query_embedding[0], (int, float)):
                    query_embedding = [float(x) for x in query_embedding]
                elif query_embedding and isinstance(query_embedding[0], list):
                    # If it's a nested list, flatten it
                    query_embedding = [float(x) for sublist in query_embedding for x in sublist]
            
            # Search for similar vectors
            results = self.qdrant_interface.search_similar(
                query_embedding, 
                limit=limit, 
                score_threshold=0.0
            )
            
            evidence_items = []
            for result in results:
                payload = result.get('payload', {})
                text = payload.get('text', '')
                score = result.get('score', 0.0)
                
                # Calculate relevance score using LLM
                relevance_score = self._calculate_relevance_score(query, text)
                
                evidence_items.append(EvidenceItem(
                    text=text,
                    relevance_score=relevance_score,
                    source=payload.get('source', 'Unknown'),
                    chunk_id=payload.get('chunk_id', 'Unknown'),
                    point_id=result.get('id', 'Unknown')
                ))
            
            # Sort by relevance score
            evidence_items.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return evidence_items
            
        except Exception as e:
            print(f"Error retrieving evidence: {e}")
            return []

    def _calculate_relevance_score(self, query: str, text: str) -> float:
        """Calculate relevance score using LLM"""
        
        prompt = f"""Rate the relevance of the following text to the query on a scale of 0.0 to 1.0.

Query: "{query}"

Text: "{text[:500]}..."  # Truncated for token limits

Consider:
- How directly does the text answer the query?
- How specific and detailed is the information?
- How authoritative is the source?

Respond with just a number between 0.0 and 1.0:"""

        try:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0
                )
                score_text = response.choices[0].message.content.strip()
                
                # Extract number from response
                score_match = re.search(r'0\.\d+|1\.0', score_text)
                if score_match:
                    return float(score_match.group())
            
            # Fallback: use similarity score as proxy
            return 0.7
            
        except Exception as e:
            print(f"Error calculating relevance score: {e}")
            return 0.5

    def _generate_response(self, query: str, evidence_items: List[EvidenceItem], 
                          retrieval_decision: RetrievalDecision) -> str:
        """Generate response using retrieved evidence"""
        
        # Prepare context
        if evidence_items:
            context = "\n\n".join([
                f"Source {i+1} (Relevance: {item.relevance_score:.2f}):\n{item.text}"
                for i, item in enumerate(evidence_items[:3])  # Top 3 most relevant
            ])
        else:
            context = "No specific evidence retrieved."
        
        # Create prompt
        if retrieval_decision.decision == "retrieve":
            system_prompt = f"""You are an expert assistant that provides accurate, evidence-based answers using retrieved information.

Query Type: {retrieval_decision.query_type}
Retrieval Decision: {retrieval_decision.reasoning}

Available Evidence:
{context}

Instructions:
1. Base your answer primarily on the provided evidence
2. If evidence is insufficient, clearly state limitations
3. Cite specific sources when making claims
4. Be precise and factual
5. If evidence contradicts the query, explain the discrepancy"""
        else:
            system_prompt = f"""You are a helpful assistant. The query does not require specific document retrieval.

Query Type: {retrieval_decision.query_type}
Reasoning: {retrieval_decision.reasoning}

Provide a helpful response based on your general knowledge."""

        user_prompt = f"Query: {query}"

        try:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
            else:
                return f"Response generated for: {query}\n[LLM not configured]"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _reflect_on_response(self, query: str, response: str, 
                           evidence_items: List[EvidenceItem]) -> ResponseReflection:
        """Self-reflect on response quality"""
        
        evidence_summary = ""
        if evidence_items:
            evidence_summary = f"Evidence used: {len(evidence_items)} items, avg relevance: {sum(item.relevance_score for item in evidence_items) / len(evidence_items):.2f}"
        else:
            evidence_summary = "No evidence used"
        
        prompt = f"""Evaluate the quality of this response to the query.

Query: "{query}"

Response: "{response}"

{evidence_summary}

Rate the response quality and identify issues:

1. Accuracy: Is the information correct and well-supported by evidence?
2. Completeness: Does it fully address the query?
3. Clarity: Is it well-structured and easy to understand?
4. Evidence Use: Is evidence properly integrated and cited?

Respond in JSON format:
{{
    "quality": "high", "medium", or "low",
    "reasoning": "explanation of quality assessment",
    "issues": ["list of specific issues found"],
    "suggestions": ["list of improvement suggestions"],
    "confidence": 0.0-1.0
}}"""

        try:
            if self.openai_client:
                llm_response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.1
                )
                result = json.loads(llm_response.choices[0].message.content.strip())
            else:
                # Fallback reflection
                result = self._fallback_reflection(query, response, evidence_items)
            
            return ResponseReflection(
                quality=result.get("quality", "medium"),
                reasoning=result.get("reasoning", "Fallback assessment"),
                issues=result.get("issues", []),
                suggestions=result.get("suggestions", []),
                confidence=result.get("confidence", 0.6)
            )
            
        except Exception as e:
            return self._fallback_reflection(query, response, evidence_items)

    def _fallback_reflection(self, query: str, response: str, evidence_items: List[EvidenceItem]) -> ResponseReflection:
        """Fallback reflection logic"""
        
        issues = []
        suggestions = []
        quality = "medium"
        
        # Simple heuristics
        if len(response) < 50:
            issues.append("Response is too short")
            suggestions.append("Provide more detailed information")
            quality = "low"
        
        if evidence_items and len(response) < 100:
            issues.append("Response doesn't fully utilize available evidence")
            suggestions.append("Integrate more evidence into the response")
            quality = "low"
        
        if not evidence_items and "data protection" in query.lower():
            issues.append("No evidence retrieved for factual query")
            suggestions.append("Retrieve relevant evidence from knowledge base")
            quality = "medium"
        
        return ResponseReflection(
            quality=quality,
            reasoning="Fallback assessment using simple heuristics",
            issues=issues,
            suggestions=suggestions,
            confidence=0.6
        )

    def _self_correct(self, query: str, original_response: str, 
                     evidence_items: List[EvidenceItem], reflection: ResponseReflection) -> Tuple[str, str]:
        """Self-correct the response based on reflection"""
        
        if not reflection.issues:
            return None, None
        
        issues_text = "\n".join([f"- {issue}" for issue in reflection.issues])
        suggestions_text = "\n".join([f"- {suggestion}" for suggestion in reflection.suggestions])
        
        prompt = f"""Correct the following response based on the identified issues and suggestions.

Query: "{query}"

Original Response: "{original_response}"

Issues Found:
{issues_text}

Suggestions:
{suggestions_text}

Available Evidence:
{self._format_evidence_for_correction(evidence_items)}

Generate a corrected response that addresses the identified issues:"""

        try:
            if self.openai_client:
                correction_response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                corrected_response = correction_response.choices[0].message.content.strip()
                
                correction_reasoning = f"Corrected based on: {reflection.reasoning}"
                
                return corrected_response, correction_reasoning
            else:
                return None, "LLM not configured for correction"
                
        except Exception as e:
            return None, f"Error in self-correction: {str(e)}"

    def _format_evidence_for_correction(self, evidence_items: List[EvidenceItem]) -> str:
        """Format evidence for correction prompt"""
        if not evidence_items:
            return "No evidence available"
        
        return "\n\n".join([
            f"Evidence {i+1} (Relevance: {item.relevance_score:.2f}):\n{item.text[:300]}..."
            for i, item in enumerate(evidence_items[:3])
        ])

    def _store_self_rag_turn(self, session_id: str, query: str, response: str, 
                           evidence_items: List[EvidenceItem], reflection: ResponseReflection):
        """Store Self-RAG turn in memory"""
        try:
            turn_data = {
                "query": query,
                "response": response,
                "evidence_count": len(evidence_items),
                "reflection_quality": reflection.quality,
                "reflection_issues": reflection.issues,
                "timestamp": self._get_timestamp()
            }
            
            # Store in conversation memory
            if hasattr(self.memory_aware_service, 'add_self_rag_turn'):
                self.memory_aware_service.add_self_rag_turn(session_id, turn_data)
            
        except Exception as e:
            print(f"Error storing Self-RAG turn: {e}")

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_self_rag_stats(self) -> Dict:
        """Get Self-RAG system statistics"""
        return {
            "service_type": "Self-RAG",
            "features": [
                "Retrieval decision analysis",
                "Evidence relevance scoring",
                "Response quality reflection",
                "Self-correction mechanism",
                "Memory-aware conversations"
            ],
            "llm_provider": config.LLM_PROVIDER,
            "model": config.LLM_MODEL,
            "max_tokens": config.MAX_TOKENS,
            "temperature": config.TEMPERATURE
        }
