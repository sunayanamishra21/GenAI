"""
Context-Aware RAG Service
Enhances retrieval with query understanding and context expansion
"""

import re
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import config
import torch

class ContextAwareService:
    def __init__(self):
        try:
            # Initialize with explicit device handling
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Initializing embedding model '{config.EMBEDDING_MODEL}' on device: {device}")
            
            # Try different initialization approaches
            try:
                self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
                print("Model initialized successfully")
            except Exception as e1:
                print(f"Primary initialization failed: {e1}")
                # Try with CPU and trust_remote_code
                try:
                    self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device='cpu', trust_remote_code=True)
                    print("Model initialized with trust_remote_code=True")
                except Exception as e2:
                    print(f"Trust remote code failed: {e2}")
                    # Try without device specification
                    try:
                        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
                        print("Model initialized without device specification")
                    except Exception as e3:
                        print(f"All initialization attempts failed: {e3}")
                        self.embedding_model = None
        except Exception as e:
            print(f"Critical error in embedding model initialization: {e}")
            self.embedding_model = None
        
        # Context expansion patterns
        self.query_types = {
            "definition": ["what is", "define", "meaning of", "explain"],
            "procedure": ["how to", "steps", "process", "procedure", "method"],
            "comparison": ["difference", "compare", "versus", "vs", "better than"],
            "list": ["list", "all", "types", "kinds", "examples"],
            "requirement": ["required", "must", "need", "necessary", "obligation"],
            "penalty": ["penalty", "fine", "punishment", "consequence", "violation"],
            "right": ["right", "entitlement", "permission", "allowed"],
            "deadline": ["when", "deadline", "time limit", "due", "within"]
        }
        
        # GDPR-specific context expansions
        self.gdpr_contexts = {
            "data_subject_rights": [
                "right of access", "right to rectification", "right to erasure",
                "right to restriction", "right to data portability", "right to object"
            ],
            "controller_obligations": [
                "data protection by design", "data protection by default",
                "privacy impact assessment", "data protection officer"
            ],
            "breach_notification": [
                "72 hours", "supervisory authority", "data subjects",
                "high risk", "without undue delay"
            ],
            "consent_requirements": [
                "freely given", "specific", "informed", "unambiguous",
                "clear affirmative action", "withdraw consent"
            ]
        }
    
    def analyze_query_intent(self, query: str) -> Dict[str, any]:
        """Analyze query to understand intent and extract key concepts"""
        
        query_lower = query.lower()
        
        # Determine query type
        query_type = "general"
        for qtype, patterns in self.query_types.items():
            if any(pattern in query_lower for pattern in patterns):
                query_type = qtype
                break
        
        # Extract key entities and concepts
        entities = self._extract_entities(query)
        
        # Identify GDPR-specific concepts
        gdpr_concepts = []
        for concept, keywords in self.gdpr_contexts.items():
            if any(keyword in query_lower for keyword in keywords):
                gdpr_concepts.append(concept)
        
        # Determine complexity
        complexity = self._assess_complexity(query)
        
        return {
            "query_type": query_type,
            "entities": entities,
            "gdpr_concepts": gdpr_concepts,
            "complexity": complexity,
            "original_query": query
        }
    
    def expand_query_context(self, query_analysis: Dict) -> List[str]:
        """Generate expanded queries for better context retrieval"""
        
        original_query = query_analysis["original_query"]
        query_type = query_analysis["query_type"]
        gdpr_concepts = query_analysis["gdpr_concepts"]
        
        expanded_queries = [original_query]
        
        # Add context-specific expansions based on query type
        if query_type == "definition":
            expanded_queries.extend([
                f"definition of {original_query}",
                f"what does {original_query} mean",
                f"explanation of {original_query}"
            ])
        
        elif query_type == "procedure":
            expanded_queries.extend([
                f"steps for {original_query}",
                f"how to implement {original_query}",
                f"process for {original_query}"
            ])
        
        elif query_type == "requirement":
            expanded_queries.extend([
                f"obligations for {original_query}",
                f"compliance requirements for {original_query}",
                f"legal requirements for {original_query}"
            ])
        
        elif query_type == "penalty":
            expanded_queries.extend([
                f"administrative fines for {original_query}",
                f"consequences of {original_query}",
                f"enforcement of {original_query}"
            ])
        
        # Add GDPR-specific context expansions
        for concept in gdpr_concepts:
            if concept == "data_subject_rights":
                expanded_queries.extend([
                    "data subject rights under GDPR",
                    "individual rights in data protection",
                    "user rights in personal data processing"
                ])
            
            elif concept == "controller_obligations":
                expanded_queries.extend([
                    "controller responsibilities under GDPR",
                    "data controller obligations",
                    "compliance requirements for controllers"
                ])
            
            elif concept == "breach_notification":
                expanded_queries.extend([
                    "data breach notification procedures",
                    "72 hour breach notification rule",
                    "reporting data breaches to authorities"
                ])
            
            elif concept == "consent_requirements":
                expanded_queries.extend([
                    "valid consent under GDPR",
                    "consent requirements for data processing",
                    "freely given specific informed consent"
                ])
        
        # Add general context expansions
        expanded_queries.extend([
            f"GDPR {original_query}",
            f"European Union {original_query}",
            f"data protection {original_query}"
        ])
        
        return list(set(expanded_queries))  # Remove duplicates
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from the query"""
        
        # Simple entity extraction (can be enhanced with NER)
        entities = []
        
        # Common GDPR entities
        gdpr_entities = [
            "personal data", "data subject", "controller", "processor",
            "consent", "legitimate interest", "data breach", "supervisory authority",
            "data protection officer", "privacy by design", "right to be forgotten",
            "data portability", "pseudonymization", "anonymization"
        ]
        
        query_lower = query.lower()
        for entity in gdpr_entities:
            if entity in query_lower:
                entities.append(entity)
        
        return entities
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        
        word_count = len(query.split())
        question_words = ["what", "how", "when", "where", "why", "who"]
        complex_indicators = ["compare", "difference", "relationship", "impact", "implication"]
        
        query_lower = query.lower()
        
        if word_count > 15 or any(indicator in query_lower for indicator in complex_indicators):
            return "complex"
        elif word_count > 8 or len([w for w in question_words if w in query_lower]) > 1:
            return "medium"
        else:
            return "simple"
    
    def create_contextual_embeddings(self, expanded_queries: List[str]) -> List[List[float]]:
        """Create embeddings for expanded queries"""
        
        if self.embedding_model is None:
            # Return dummy embeddings if model failed to initialize
            print("Warning: Embedding model not available, returning dummy embeddings")
            return [[0.0] * 384] * len(expanded_queries)
        
        try:
            embeddings = self.embedding_model.encode(expanded_queries)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            # Return dummy embeddings as fallback
            return [[0.0] * 384] * len(expanded_queries)
    
    def rank_results_by_context(self, results: List[Dict], query_analysis: Dict) -> List[Dict]:
        """Rank results based on context relevance"""
        
        query_type = query_analysis["query_type"]
        gdpr_concepts = query_analysis["gdpr_concepts"]
        
        for result in results:
            payload = result.get("payload", {})
            text = payload.get("text", "").lower()
            
            # Base score from vector similarity
            context_score = result.get("score", 0)
            
            # Boost score based on query type relevance
            if query_type == "definition" and any(word in text for word in ["definition", "means", "refers to"]):
                context_score *= 1.2
            
            elif query_type == "procedure" and any(word in text for word in ["steps", "process", "procedure", "how to"]):
                context_score *= 1.2
            
            elif query_type == "requirement" and any(word in text for word in ["must", "shall", "required", "obligation"]):
                context_score *= 1.2
            
            elif query_type == "penalty" and any(word in text for word in ["fine", "penalty", "sanction", "consequence"]):
                context_score *= 1.2
            
            # Boost score based on GDPR concepts
            for concept in gdpr_concepts:
                if concept in text:
                    context_score *= 1.1
            
            # Update the result with context score
            result["context_score"] = context_score
        
        # Sort by context score
        return sorted(results, key=lambda x: x.get("context_score", 0), reverse=True)
