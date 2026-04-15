"""
Intent Router - Intelligent query classification and agent routing
Routes queries to appropriate agents based on intent, confidence, and context
"""

import logging
from typing import Dict, Optional, Tuple
from enum import Enum
import re

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Query intent types"""
    RAG_QUERY = "rag_query"  # Document retrieval needed
    TOOL_EXECUTION = "tool_execution"  # Structured operation
    CLARIFICATION = "clarification"  # Ambiguous query
    REJECTION = "rejection"  # Out of scope
    FOLLOW_UP = "follow_up"  # Continuation of previous query


class IntentRouter:
    """
    Intelligent router that analyzes queries and routes to appropriate agents.
    Uses hybrid approach: rule-based + LLM-based classification.
    """

    def __init__(self, llm_classifier=None, confidence_threshold: float = 0.5):
        """
        Initialize router with optional LLM classifier.

        Args:
            llm_classifier: Optional LLM for intent classification
            confidence_threshold: Minimum confidence for routing
        """
        self.llm_classifier = llm_classifier
        self.confidence_threshold = confidence_threshold
        self.tool_keywords = {
            "summarize", "summary", "list all", "show all", "count",
            "filter", "search for", "find all", "get all"
        }
        self.clarification_indicators = {
            "it", "that", "this", "how does it work", "what about",
            "tell me more", "explain"
        }

    def route(
        self,
        query: str,
        session_context: Optional[Dict] = None
    ) -> Tuple[Intent, float, Dict]:
        """
        Route query to appropriate agent.

        Args:
            query: User query text
            session_context: Optional conversation context

        Returns:
            Tuple of (intent, confidence, routing_metadata)
        """
        logger.info(f"Routing query: {query[:100]}...")

        # Step 1: Rule-based pre-classification
        rule_intent, rule_confidence = self._rule_based_classification(query)

        # Step 2: Check for follow-up queries
        if session_context and self._is_follow_up(query, session_context):
            return Intent.FOLLOW_UP, 0.9, {
                "previous_intent": session_context.get("last_intent"),
                "context_available": True
            }

        # Step 3: LLM-based classification (if available and needed)
        if self.llm_classifier and rule_confidence < 0.8:
            llm_intent, llm_confidence = self._llm_based_classification(
                query, session_context
            )
            # Use LLM result if more confident
            if llm_confidence > rule_confidence:
                intent, confidence = llm_intent, llm_confidence
            else:
                intent, confidence = rule_intent, rule_confidence
        else:
            intent, confidence = rule_intent, rule_confidence

        # Step 4: Apply confidence threshold
        if confidence < self.confidence_threshold:
            intent = Intent.CLARIFICATION
            confidence = 0.3

        # Step 5: Build routing metadata
        metadata = self._build_routing_metadata(query, intent, confidence)

        logger.info(f"Routed to {intent.value} with confidence {confidence:.2f}")

        return intent, confidence, metadata

    def _rule_based_classification(self, query: str) -> Tuple[Intent, float]:
        """
        Rule-based intent classification using patterns and keywords.

        Args:
            query: User query text

        Returns:
            Tuple of (intent, confidence)
        """
        query_lower = query.lower().strip()

        # Check for tool execution patterns
        if any(keyword in query_lower for keyword in self.tool_keywords):
            return Intent.TOOL_EXECUTION, 0.85

        # Check for clarification needs
        if (
            len(query.split()) < 3
            or any(indicator in query_lower for indicator in self.clarification_indicators)
        ):
            return Intent.CLARIFICATION, 0.7

        # Check for out-of-scope patterns
        out_of_scope_patterns = [
            r"weather", r"news", r"stock", r"sports", r"recipe",
            r"movie", r"music", r"game", r"joke"
        ]
        if any(re.search(pattern, query_lower) for pattern in out_of_scope_patterns):
            return Intent.REJECTION, 0.9

        # Check for question patterns (likely RAG)
        question_patterns = [
            r"^(how|what|why|when|where|who|which|can|does|is|are)",
            r"\?$"
        ]
        if any(re.search(pattern, query_lower) for pattern in question_patterns):
            return Intent.RAG_QUERY, 0.75

        # Default to RAG with medium confidence
        return Intent.RAG_QUERY, 0.6

    def _llm_based_classification(
        self,
        query: str,
        session_context: Optional[Dict]
    ) -> Tuple[Intent, float]:
        """
        LLM-based intent classification for complex queries.

        Args:
            query: User query text
            session_context: Optional conversation context

        Returns:
            Tuple of (intent, confidence)
        """
        if not self.llm_classifier:
            return Intent.RAG_QUERY, 0.5

        try:
            # Build classification prompt
            context_str = ""
            if session_context and session_context.get("last_query"):
                context_str = f"\nPrevious query: {session_context['last_query']}"

            prompt = f"""Classify the following user query into one of these intents:
1. rag_query - User wants information from documents
2. tool_execution - User wants structured operation (summarize, list, filter)
3. clarification - Query is ambiguous or needs more context
4. rejection - Query is out of scope (weather, news, etc.)

Query: {query}{context_str}

Respond with ONLY the intent name and confidence (0-1) in format: intent|confidence
Example: rag_query|0.85"""

            # Call LLM (simplified - actual implementation would use proper LLM call)
            response = self.llm_classifier.classify(prompt)

            # Parse response
            parts = response.strip().split("|")
            if len(parts) == 2:
                intent_str, conf_str = parts
                intent = Intent(intent_str.strip())
                confidence = float(conf_str.strip())
                return intent, confidence

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

        # Fallback to rule-based
        return self._rule_based_classification(query)

    def _is_follow_up(self, query: str, session_context: Dict) -> bool:
        """
        Determine if query is a follow-up to previous conversation.

        Args:
            query: Current query
            session_context: Session context with history

        Returns:
            True if follow-up query
        """
        if not session_context or not session_context.get("last_query"):
            return False

        query_lower = query.lower()

        # Check for follow-up indicators
        follow_up_indicators = [
            "also", "and", "what about", "how about", "tell me more",
            "explain", "continue", "go on", "more details"
        ]

        # Check for pronouns without clear antecedents
        pronouns = ["it", "that", "this", "they", "them"]

        has_follow_up_indicator = any(
            indicator in query_lower for indicator in follow_up_indicators
        )
        has_pronoun = any(pronoun in query_lower.split() for pronoun in pronouns)

        return has_follow_up_indicator or (has_pronoun and len(query.split()) < 10)

    def _build_routing_metadata(
        self,
        query: str,
        intent: Intent,
        confidence: float
    ) -> Dict:
        """
        Build metadata for routing decision.

        Args:
            query: User query
            intent: Classified intent
            confidence: Classification confidence

        Returns:
            Routing metadata dictionary
        """
        metadata = {
            "query_length": len(query),
            "word_count": len(query.split()),
            "has_question_mark": "?" in query,
            "intent": intent.value,
            "confidence": confidence,
            "routing_method": "hybrid"
        }

        # Add intent-specific metadata
        if intent == Intent.TOOL_EXECUTION:
            metadata["suggested_tools"] = self._extract_tool_hints(query)
        elif intent == Intent.CLARIFICATION:
            metadata["clarification_reason"] = self._get_clarification_reason(query)

        return metadata

    def _extract_tool_hints(self, query: str) -> list:
        """Extract potential tool names from query."""
        tools = []
        query_lower = query.lower()

        if any(word in query_lower for word in ["summarize", "summary"]):
            tools.append("summarization")
        if any(word in query_lower for word in ["search", "find", "filter"]):
            tools.append("document_search")
        if any(word in query_lower for word in ["count", "how many", "list all"]):
            tools.append("database_query")

        return tools

    def _get_clarification_reason(self, query: str) -> str:
        """Determine why clarification is needed."""
        if len(query.split()) < 3:
            return "query_too_short"
        elif any(word in query.lower() for word in ["it", "that", "this"]):
            return "ambiguous_reference"
        else:
            return "insufficient_context"


# Example usage
if __name__ == "__main__":
    router = IntentRouter()

    test_queries = [
        "How do I reset my password?",
        "Summarize all password-related documents",
        "What about it?",
        "What's the weather today?",
        "Tell me more"
    ]

    for query in test_queries:
        intent, confidence, metadata = router.route(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {intent.value}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Metadata: {metadata}")
