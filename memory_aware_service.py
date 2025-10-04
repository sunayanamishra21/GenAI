"""
Memory-Aware RAG Service
Maintains conversation history and context across queries
"""

import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import config

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    timestamp: str
    user_query: str
    ai_response: str
    retrieved_chunks: List[Dict]
    metadata: Dict
    turn_id: str

@dataclass
class ConversationSession:
    """Represents an entire conversation session"""
    session_id: str
    start_time: str
    last_activity: str
    turns: List[ConversationTurn]
    context_summary: str
    user_preferences: Dict

class MemoryAwareService:
    def __init__(self, memory_file: str = "conversation_memory.json"):
        self.memory_file = memory_file
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Memory service: Initializing embedding model '{config.EMBEDDING_MODEL}' on device: {device}")
            
            # Try different initialization approaches
            try:
                self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
                print("Memory service: Model initialized successfully")
            except Exception as e1:
                print(f"Memory service: Primary initialization failed: {e1}")
                # Try with CPU and trust_remote_code
                try:
                    self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device='cpu', trust_remote_code=True)
                    print("Memory service: Model initialized with trust_remote_code=True")
                except Exception as e2:
                    print(f"Memory service: Trust remote code failed: {e2}")
                    # Try without device specification
                    try:
                        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
                        print("Memory service: Model initialized without device specification")
                    except Exception as e3:
                        print(f"Memory service: All initialization attempts failed: {e3}")
                        self.embedding_model = None
        except Exception as e:
            print(f"Memory service: Critical error in embedding model initialization: {e}")
            self.embedding_model = None
        self.current_session = None
        self.load_memory()
        
        # Memory settings
        self.max_session_duration = 3600  # 1 hour
        self.max_turns_per_session = 50
        self.max_context_length = 2000  # characters
        
        # Context window for related queries
        self.context_window_size = 3  # last 3 turns
    
    def load_memory(self):
        """Load conversation memory from file"""
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.sessions = [ConversationSession(**session) for session in data.get("sessions", [])]
                self.global_context = data.get("global_context", "")
        except FileNotFoundError:
            self.sessions = []
            self.global_context = ""
    
    def save_memory(self):
        """Save conversation memory to file"""
        data = {
            "sessions": [asdict(session) for session in self.sessions if hasattr(session, '__dataclass_fields__')],
            "global_context": self.global_context,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def start_new_session(self, user_id: str = "default") -> ConversationSession:
        """Start a new conversation session"""
        session_id = f"{user_id}_{int(time.time())}"
        
        self.current_session = ConversationSession(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            last_activity=datetime.now().isoformat(),
            turns=[],
            context_summary="",
            user_preferences={}
        )
        
        self.sessions.append(self.current_session)
        self.save_memory()
        
        return self.current_session

    def add_self_rag_turn(self, session_id: str, turn_data: Dict):
        """Add a Self-RAG turn to the session"""
        try:
            session = self.get_session_by_id(session_id)
            if session:
                # Create a Self-RAG specific turn
                self_rag_turn = ConversationTurn(
                    turn_id=f"self_rag_{int(time.time())}",
                    timestamp=datetime.now().isoformat(),
                    user_input=turn_data.get("query", ""),
                    ai_response=turn_data.get("response", ""),
                    context={},
                    metadata={
                        "type": "self_rag",
                        "evidence_count": turn_data.get("evidence_count", 0),
                        "reflection_quality": turn_data.get("reflection_quality", "medium"),
                        "reflection_issues": turn_data.get("reflection_issues", []),
                        "timestamp": turn_data.get("timestamp", datetime.now().isoformat())
                    }
                )
                
                session.turns.append(self_rag_turn)
                session.last_activity = datetime.now().isoformat()
                self.save_memory()
                
        except Exception as e:
            print(f"Error adding Self-RAG turn: {e}")

    def get_session_by_id(self, session_id: str) -> Optional[ConversationSession]:
        """Get session by ID"""
        for session in self.sessions:
            if session.session_id == session_id:
                return session
        return None
    
    def get_current_session(self) -> Optional[ConversationSession]:
        """Get the current active session"""
        if self.current_session is None:
            return None
        
        # Check if session is still valid
        last_activity = datetime.fromisoformat(self.current_session.last_activity)
        if datetime.now() - last_activity > timedelta(seconds=self.max_session_duration):
            # Session expired, start new one
            return self.start_new_session()
        
        return self.current_session
    
    def add_turn_to_session(self, user_query: str, ai_response: str, 
                           retrieved_chunks: List[Dict], metadata: Dict = None) -> str:
        """Add a new turn to the current session"""
        
        session = self.get_current_session()
        if session is None:
            session = self.start_new_session()
        
        turn_id = f"turn_{len(session.turns) + 1}_{int(time.time())}"
        
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            ai_response=ai_response,
            retrieved_chunks=retrieved_chunks,
            metadata=metadata or {},
            turn_id=turn_id
        )
        
        session.turns.append(turn)
        session.last_activity = datetime.now().isoformat()
        
        # Update context summary
        session.context_summary = self._update_context_summary(session)
        
        # Trim session if too long
        if len(session.turns) > self.max_turns_per_session:
            session.turns = session.turns[-self.max_turns_per_session:]
        
        self.save_memory()
        return turn_id
    
    def _update_context_summary(self, session: ConversationSession) -> str:
        """Update the context summary for the session"""
        
        if not session.turns:
            return ""
        
        # Get recent turns for context
        recent_turns = session.turns[-self.context_window_size:]
        
        # Extract key topics and entities from recent turns
        topics = []
        for turn in recent_turns:
            # Simple topic extraction (can be enhanced)
            query_words = turn.user_query.lower().split()
            gdpr_keywords = ["gdpr", "data protection", "personal data", "consent", "breach", "rights"]
            
            for keyword in gdpr_keywords:
                if keyword in turn.user_query.lower():
                    topics.append(keyword)
        
        # Create context summary
        context_parts = []
        if topics:
            context_parts.append(f"Recent topics: {', '.join(set(topics))}")
        
        # Add recent query patterns
        recent_queries = [turn.user_query for turn in recent_turns[-2:]]
        if recent_queries:
            context_parts.append(f"Recent queries: {'; '.join(recent_queries)}")
        
        return ". ".join(context_parts)
    
    def get_conversation_context(self, include_recent: bool = True) -> Dict:
        """Get relevant conversation context for the current query"""
        
        session = self.get_current_session()
        if session is None or not session.turns:
            return {
                "recent_turns": [],
                "context_summary": "",
                "related_topics": [],
                "user_preferences": {}
            }
        
        # Get recent turns for context
        recent_turns = session.turns[-self.context_window_size:] if include_recent else []
        
        # Extract related topics
        related_topics = self._extract_related_topics(session.turns)
        
        return {
            "recent_turns": [
                {
                    "query": turn.user_query,
                    "response": turn.ai_response[:200] + "..." if len(turn.ai_response) > 200 else turn.ai_response,
                    "timestamp": turn.timestamp
                }
                for turn in recent_turns
            ],
            "context_summary": session.context_summary,
            "related_topics": related_topics,
            "user_preferences": session.user_preferences,
            "session_id": session.session_id
        }
    
    def _extract_related_topics(self, turns: List[ConversationTurn]) -> List[str]:
        """Extract topics that appear frequently in the conversation"""
        
        topic_counts = {}
        
        for turn in turns:
            query_lower = turn.user_query.lower()
            
            # GDPR-related topics
            gdpr_topics = {
                "data_subject_rights": ["right", "access", "rectification", "erasure", "portability"],
                "consent": ["consent", "freely given", "specific", "informed"],
                "data_breach": ["breach", "notification", "72 hours", "supervisory authority"],
                "penalties": ["fine", "penalty", "administrative", "consequence"],
                "compliance": ["compliance", "obligation", "requirement", "controller"],
                "privacy": ["privacy", "protection", "personal data", "processing"]
            }
            
            for topic, keywords in gdpr_topics.items():
                if any(keyword in query_lower for keyword in keywords):
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Return topics that appear more than once
        return [topic for topic, count in topic_counts.items() if count > 1]
    
    def enhance_query_with_memory(self, query: str) -> Dict:
        """Enhance the current query with conversation memory"""
        
        context = self.get_conversation_context()
        
        # Check if this is a follow-up question
        follow_up_indicators = ["also", "additionally", "furthermore", "what about", "how about", "can you"]
        is_follow_up = any(indicator in query.lower() for indicator in follow_up_indicators)
        
        # Check for references to previous context
        reference_indicators = ["that", "this", "it", "the above", "previously mentioned"]
        has_reference = any(indicator in query.lower() for indicator in reference_indicators)
        
        # Enhanced query with context
        enhanced_query = query
        
        if is_follow_up or has_reference:
            # Add context from recent turns
            if context["recent_turns"]:
                recent_context = " ".join([turn["query"] for turn in context["recent_turns"]])
                enhanced_query = f"{recent_context} {query}"
        
        # Add related topics
        if context["related_topics"]:
            topic_context = f"Related to: {', '.join(context['related_topics'])}. {query}"
            enhanced_query = topic_context
        
        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "context": context,
            "is_follow_up": is_follow_up,
            "has_reference": has_reference
        }
    
    def update_user_preferences(self, preferences: Dict):
        """Update user preferences based on conversation patterns"""
        
        session = self.get_current_session()
        if session is None:
            session = self.start_new_session()
        
        # Update preferences
        session.user_preferences.update(preferences)
        self.save_memory()
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about conversation memory"""
        
        total_sessions = len(self.sessions)
        total_turns = sum(len(session.turns) for session in self.sessions)
        
        if self.current_session:
            current_turns = len(self.current_session.turns)
            session_duration = (datetime.now() - datetime.fromisoformat(self.current_session.start_time)).total_seconds()
        else:
            current_turns = 0
            session_duration = 0
        
        # Calculate memory file size safely
        try:
            memory_file_size = len(json.dumps([asdict(session) for session in self.sessions if hasattr(session, '__dataclass_fields__')], ensure_ascii=False))
        except:
            memory_file_size = 0
        
        return {
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "current_session_turns": current_turns,
            "current_session_duration": session_duration,
            "memory_file_size": memory_file_size
        }
