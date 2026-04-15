"""
Session Manager - Conversation memory and context management
Maintains stateful conversations with history and context caching
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages conversation sessions with memory and context.
    Supports multi-turn conversations and context continuity.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize session manager.

        Args:
            storage_path: Optional path for persistent storage
        """
        self.sessions: Dict[str, Dict] = {}
        self.storage_path = storage_path
        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)

    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.

        Args:
            session_id: Optional custom session ID

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        self.sessions[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "conversation_history": [],
            "context_cache": {},
            "metadata": {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0
            }
        }

        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve session data.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None if not found
        """
        if session_id not in self.sessions:
            # Try loading from storage
            if self.storage_path:
                self._load_session(session_id)

        return self.sessions.get(session_id)

    def add_interaction(
        self,
        session_id: str,
        query: str,
        intent: str,
        response: str,
        confidence: float,
        retrieved_context: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add interaction to session history.

        Args:
            session_id: Session identifier
            query: User query
            intent: Classified intent
            response: System response
            confidence: Response confidence
            retrieved_context: Retrieved documents
            metadata: Additional metadata
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "intent": intent,
            "response": response,
            "confidence": confidence,
            "retrieved_context": retrieved_context or [],
            "metadata": metadata or {}
        }

        session["conversation_history"].append(interaction)
        session["last_activity"] = datetime.now().isoformat()
        session["metadata"]["total_queries"] += 1

        if confidence > 0.5:
            session["metadata"]["successful_queries"] += 1
        else:
            session["metadata"]["failed_queries"] += 1

        # Update context cache
        self._update_context_cache(session, query, intent, retrieved_context)

        # Persist if storage enabled
        if self.storage_path:
            self._save_session(session_id)

        logger.info(f"Added interaction to session {session_id}")

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get conversation history for session.

        Args:
            session_id: Session identifier
            limit: Optional limit on number of interactions

        Returns:
            List of interactions
        """
        session = self.get_session(session_id)
        if not session:
            return []

        history = session["conversation_history"]
        if limit:
            return history[-limit:]
        return history

    def get_context(self, session_id: str) -> Dict:
        """
        Get cached context for session.

        Args:
            session_id: Session identifier

        Returns:
            Context cache dictionary
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        return session.get("context_cache", {})

    def update_context(self, session_id: str, key: str, value: any) -> None:
        """
        Update context cache.

        Args:
            session_id: Session identifier
            key: Context key
            value: Context value
        """
        session = self.get_session(session_id)
        if not session:
            return

        session["context_cache"][key] = value

        if self.storage_path:
            self._save_session(session_id)

    def _update_context_cache(
        self,
        session: Dict,
        query: str,
        intent: str,
        retrieved_context: Optional[List[str]]
    ) -> None:
        """
        Update context cache based on interaction.

        Args:
            session: Session data
            query: User query
            intent: Query intent
            retrieved_context: Retrieved documents
        """
        cache = session["context_cache"]

        # Update last query and intent
        cache["last_query"] = query
        cache["last_intent"] = intent

        # Extract and cache topic
        topic = self._extract_topic(query)
        if topic:
            cache["last_topic"] = topic

        # Cache retrieved documents for potential reuse
        if retrieved_context:
            cache["last_retrieved_docs"] = retrieved_context[:3]  # Keep top 3

        # Track query patterns
        if "query_patterns" not in cache:
            cache["query_patterns"] = []
        cache["query_patterns"].append({
            "query": query,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 10 patterns
        cache["query_patterns"] = cache["query_patterns"][-10:]

    def _extract_topic(self, query: str) -> Optional[str]:
        """
        Extract main topic from query.

        Args:
            query: User query

        Returns:
            Extracted topic or None
        """
        # Simple keyword extraction
        keywords = {
            "password": "password_reset",
            "audio": "audio_input",
            "account": "account_management",
            "support": "customer_support",
            "troubleshoot": "troubleshooting"
        }

        query_lower = query.lower()
        for keyword, topic in keywords.items():
            if keyword in query_lower:
                return topic

        return None

    def get_session_stats(self, session_id: str) -> Dict:
        """
        Get statistics for session.

        Args:
            session_id: Session identifier

        Returns:
            Session statistics
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "last_activity": session["last_activity"],
            "total_interactions": len(session["conversation_history"]),
            **session["metadata"]
        }

    def clear_session(self, session_id: str) -> bool:
        """
        Clear session data.

        Args:
            session_id: Session identifier

        Returns:
            True if cleared successfully
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

            # Remove from storage
            if self.storage_path:
                session_file = self.storage_path / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()

            logger.info(f"Cleared session: {session_id}")
            return True

        return False

    def _save_session(self, session_id: str) -> None:
        """Save session to persistent storage."""
        if not self.storage_path:
            return

        session = self.sessions.get(session_id)
        if not session:
            return

        session_file = self.storage_path / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session, f, indent=2)

    def _load_session(self, session_id: str) -> None:
        """Load session from persistent storage."""
        if not self.storage_path:
            return

        session_file = self.storage_path / f"{session_id}.json"
        if not session_file.exists():
            return

        try:
            with open(session_file, 'r') as f:
                session = json.load(f)
                self.sessions[session_id] = session
                logger.info(f"Loaded session from storage: {session_id}")
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    # Initialize session manager
    storage_path = Path("data/sessions")
    session_manager = SessionManager(storage_path=storage_path)

    # Create session
    session_id = session_manager.create_session()
    print(f"Created session: {session_id}")

    # Add interactions
    session_manager.add_interaction(
        session_id=session_id,
        query="How do I reset my password?",
        intent="rag_query",
        response="To reset your password, click the Forgot Password link.",
        confidence=0.92,
        retrieved_context=["Password reset procedure..."],
        metadata={"latency_ms": 1200}
    )

    session_manager.add_interaction(
        session_id=session_id,
        query="What about audio input?",
        intent="follow_up",
        response="VoiceAssist Pro supports WAV, MP3, and other formats.",
        confidence=0.88,
        retrieved_context=["Audio input documentation..."]
    )

    # Get history
    history = session_manager.get_conversation_history(session_id)
    print(f"\nConversation history ({len(history)} interactions):")
    for interaction in history:
        print(f"- {interaction['query'][:50]}... (confidence: {interaction['confidence']})")

    # Get context
    context = session_manager.get_context(session_id)
    print(f"\nContext cache:")
    print(f"- Last topic: {context.get('last_topic')}")
    print(f"- Last intent: {context.get('last_intent')}")

    # Get stats
    stats = session_manager.get_session_stats(session_id)
    print(f"\nSession stats:")
    print(json.dumps(stats, indent=2))
