"""
Intent Router - LLM-Driven Intelligent Query Classification and Agent Routing
Routes queries to appropriate agents using LLM-based reasoning with rule-based fallback
"""

import logging
from typing import Dict, Optional, Tuple
from enum import Enum
import re
import json

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
    LLM-driven intelligent router that analyzes queries and routes to appropriate agents.
    Primary: LLM-based classification with structured JSON output
    Fallback: Rule-based classification (only on LLM failure)
    """

    def __init__(self, llm_classifier=None, confidence_threshold: float = 0.5):
        """
        Initialize router with LLM classifier.

        Args:
            llm_classifier: LLM for intent classification (required for intelligent routing)
            confidence_threshold: Minimum confidence for routing
        """
        self.llm_classifier = llm_classifier
        self.confidence_threshold = confidence_threshold
        
        # Fallback keywords (only used if LLM fails)
        self.tool_keywords = {
            "summarize", "summary", "list all", "show all", "count",
            "filter", "search for", "find all", "get all"
        }
        self.clarification_indicators = {
            "it", "that", "this", "how does it work", "what about",
            "tell me more", "explain"
        }
        
        # Routing statistics
        self.routing_stats = {
            "llm_success": 0,
            "llm_failure": 0,
            "fallback_used": 0
        }

    def route(
        self,
        query: str,
        session_context: Optional[Dict] = None
    ) -> Tuple[Intent, float, Dict]:
        """
        Route query to appropriate agent using LLM-driven decision making.

        Args:
            query: User query text
            session_context: Optional conversation context

        Returns:
            Tuple of (intent, confidence, routing_metadata)
        """
        logger.info(f"Routing query: {query[:100]}...")

        # Step 1: Quick follow-up detection (pre-LLM optimization)
        if session_context and self._is_obvious_follow_up(query, session_context):
            logger.info("Detected obvious follow-up query")
            return Intent.FOLLOW_UP, 0.95, {
                "previous_intent": session_context.get("last_intent"),
                "context_available": True,
                "routing_method": "follow_up_detection"
            }

        # Step 2: LLM-based classification (PRIMARY METHOD)
        if self.llm_classifier:
            try:
                llm_result = self._llm_based_classification(query, session_context)
                
                if llm_result:
                    intent, confidence, reasoning, entities = llm_result
                    self.routing_stats["llm_success"] += 1
                    
                    # Apply confidence threshold
                    if confidence < self.confidence_threshold:
                        logger.warning(f"LLM confidence {confidence:.2f} below threshold {self.confidence_threshold}")
                        intent = Intent.CLARIFICATION
                        confidence = max(confidence, 0.3)
                    
                    # Build metadata with LLM reasoning
                    metadata = self._build_routing_metadata(
                        query, intent, confidence,
                        reasoning=reasoning,
                        entities=entities,
                        routing_method="llm_driven"
                    )
                    
                    logger.info(f"LLM routed to {intent.value} with confidence {confidence:.2f}")
                    logger.info(f"Reasoning: {reasoning}")
                    
                    return intent, confidence, metadata
                    
            except Exception as e:
                logger.error(f"LLM routing failed: {e}, falling back to rule-based")
                self.routing_stats["llm_failure"] += 1

        # Step 3: Fallback to rule-based classification (ONLY ON LLM FAILURE)
        logger.warning("Using rule-based fallback routing")
        self.routing_stats["fallback_used"] += 1
        
        intent, confidence = self._rule_based_classification(query)
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            intent = Intent.CLARIFICATION
            confidence = 0.3
        
        metadata = self._build_routing_metadata(
            query, intent, confidence,
            routing_method="rule_based_fallback"
        )
        
        logger.info(f"Fallback routed to {intent.value} with confidence {confidence:.2f}")
        
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
    ) -> Optional[Tuple[Intent, float, str, Dict]]:
        """
        LLM-based intent classification with structured JSON output.

        Args:
            query: User query text
            session_context: Optional conversation context

        Returns:
            Tuple of (intent, confidence, reasoning, entities) or None on failure
        """
        if not self.llm_classifier:
            return None

        try:
            # Build context-aware prompt
            prompt = self._build_routing_prompt(query, session_context)
            
            # Call LLM
            logger.info("Calling LLM for routing decision...")
            result = self.llm_classifier.generate_response(
                query=prompt,
                context_chunks=[]  # Routing doesn't need document context
            )
            
            response_text = result.get("response", "")
            
            # Parse structured JSON output
            routing_decision = self._parse_routing_response(response_text)
            
            if not routing_decision:
                logger.warning("Failed to parse LLM routing response")
                return None
            
            # Extract and validate intent
            intent_str = routing_decision.get("intent", "").lower()
            try:
                intent = Intent(intent_str)
            except ValueError:
                logger.warning(f"Invalid intent from LLM: {intent_str}")
                return None
            
            # Extract confidence
            confidence = float(routing_decision.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            
            # Extract reasoning and entities
            reasoning = routing_decision.get("reasoning", "No reasoning provided")
            entities = routing_decision.get("entities", {})
            
            logger.info(f"LLM classification successful: {intent.value} ({confidence:.2f})")
            
            return intent, confidence, reasoning, entities
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return None
    
    def _build_routing_prompt(self, query: str, session_context: Optional[Dict]) -> str:
        """
        Build comprehensive prompt for LLM routing decision.
        
        Args:
            query: User query
            session_context: Optional session context
            
        Returns:
            Formatted prompt string
        """
        # Build context section
        context_section = ""
        if session_context:
            last_query = session_context.get("last_query", "")
            last_intent = session_context.get("last_intent", "")
            last_topic = session_context.get("last_topic", "")
            
            if last_query or last_intent:
                context_section = f"""
## CONVERSATION CONTEXT
- Previous Query: "{last_query}"
- Previous Intent: {last_intent}
- Previous Topic: {last_topic}
"""
        
        # Build the routing prompt
        prompt = f"""You are an intelligent routing system for VoiceAssist Pro, a document-based Q&A assistant.

Your task: Analyze the user query and determine the correct intent for routing.

## AVAILABLE INTENTS

1. **rag_query** - User wants information from documents
   - Factual questions about product features, procedures, policies
   - "How do I...", "What is...", "Where can I find..."
   - Requires document retrieval and grounded response

2. **tool_execution** - User wants structured operations
   - Summarization: "Summarize all...", "Give me an overview of..."
   - Search/Filter: "Find all documents about...", "Show me..."
   - List/Count: "How many...", "List all..."
   - Requires tool invocation (summarization, document_search)

3. **follow_up** - Continuation of previous conversation
   - References previous context: "What about that?", "Tell me more", "And also..."
   - Uses pronouns without clear antecedents: "it", "that", "this"
   - Builds on previous topic

4. **clarification** - Query is ambiguous or needs more context
   - Too vague: "help", "info", "tell me"
   - Unclear reference: "How does it work?" (without context)
   - Insufficient information to route confidently

5. **rejection** - Query is out of scope
   - Not related to VoiceAssist Pro documentation
   - General knowledge questions: weather, news, recipes, entertainment
   - Personal advice, opinions, creative writing

## SYSTEM CAPABILITIES
- Document retrieval from VoiceAssist Pro knowledge base
- Topics: password reset, audio input, account management, troubleshooting, policies
- Tools: summarization, document_search
- Does NOT have: internet access, general knowledge, real-time data
{context_section}
## USER QUERY
"{query}"

## INSTRUCTIONS
1. Analyze the query carefully
2. Consider conversation context if available
3. Choose the most appropriate intent
4. Provide confidence score (0.0 to 1.0)
5. Explain your reasoning briefly
6. Extract key entities (topic, action, etc.)

## OUTPUT FORMAT (JSON ONLY)
Respond with ONLY valid JSON in this exact format:
{{
  "intent": "rag_query|tool_execution|follow_up|clarification|rejection",
  "confidence": 0.85,
  "reasoning": "Brief explanation of why this intent was chosen",
  "entities": {{
    "topic": "extracted topic if any",
    "action": "extracted action if any"
  }}
}}

IMPORTANT: Return ONLY the JSON object, no additional text.

Routing Decision:"""
        
        return prompt
    
    def _parse_routing_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse LLM response to extract routing decision JSON.
        
        Args:
            response_text: LLM response text
            
        Returns:
            Parsed routing decision dict or None on failure
        """
        # Try to find JSON object in response
        # Pattern 1: Look for complete JSON object
        json_match = re.search(r'\{[\s\S]*?\}', response_text)
        
        if json_match:
            try:
                json_str = json_match.group(0)
                routing_decision = json.loads(json_str)
                
                # Validate required fields
                if "intent" in routing_decision and "confidence" in routing_decision:
                    return routing_decision
                else:
                    logger.warning("JSON missing required fields: intent or confidence")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
                logger.debug(f"Attempted to parse: {json_str[:200]}")
        
        # Pattern 2: Try to extract fields individually (more robust)
        try:
            intent_match = re.search(r'"intent"\s*:\s*"([^"]+)"', response_text)
            confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', response_text)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response_text)
            
            if intent_match and confidence_match:
                return {
                    "intent": intent_match.group(1),
                    "confidence": float(confidence_match.group(1)),
                    "reasoning": reasoning_match.group(1) if reasoning_match else "Extracted from partial JSON",
                    "entities": {}
                }
        except Exception as e:
            logger.warning(f"Field extraction failed: {e}")
        
        logger.error("Could not parse routing response")
        logger.debug(f"Response text: {response_text[:300]}")
        return None

    def _is_obvious_follow_up(self, query: str, session_context: Dict) -> bool:
        """
        Quick detection of obvious follow-up queries (pre-LLM optimization).
        
        Args:
            query: Current query
            session_context: Session context with history
            
        Returns:
            True if obviously a follow-up query
        """
        if not session_context or not session_context.get("last_query"):
            return False
        
        query_lower = query.lower().strip()
        
        # Very short queries with pronouns are likely follow-ups
        if len(query.split()) <= 5:
            pronouns = ["it", "that", "this", "they", "them", "those"]
            if any(f" {pronoun} " in f" {query_lower} " or query_lower.startswith(f"{pronoun} ") 
                   for pronoun in pronouns):
                return True
        
        # Explicit continuation phrases
        continuation_phrases = [
            "tell me more", "what about", "how about", "and also",
            "continue", "go on", "more details", "explain that"
        ]
        if any(phrase in query_lower for phrase in continuation_phrases):
            return True
        
        return False

    def _build_routing_metadata(
        self,
        query: str,
        intent: Intent,
        confidence: float,
        reasoning: Optional[str] = None,
        entities: Optional[Dict] = None,
        routing_method: str = "unknown"
    ) -> Dict:
        """
        Build metadata for routing decision.

        Args:
            query: User query
            intent: Classified intent
            confidence: Classification confidence
            reasoning: Optional LLM reasoning
            entities: Optional extracted entities
            routing_method: Method used for routing

        Returns:
            Routing metadata dictionary
        """
        metadata = {
            "query_length": len(query),
            "word_count": len(query.split()),
            "has_question_mark": "?" in query,
            "intent": intent.value,
            "confidence": confidence,
            "routing_method": routing_method
        }
        
        # Add LLM-specific metadata
        if reasoning:
            metadata["reasoning"] = reasoning
        
        if entities:
            metadata["entities"] = entities
        
        # Add intent-specific metadata
        if intent == Intent.TOOL_EXECUTION:
            metadata["suggested_tools"] = self._extract_tool_hints(query, entities)
        elif intent == Intent.CLARIFICATION:
            metadata["clarification_reason"] = self._get_clarification_reason(query)
        
        return metadata

    def _extract_tool_hints(self, query: str, entities: Optional[Dict] = None) -> list:
        """
        Extract potential tool names from query and entities.
        
        Args:
            query: User query
            entities: Optional extracted entities from LLM
            
        Returns:
            List of suggested tool names
        """
        tools = []
        query_lower = query.lower()
        
        # Check entities first (from LLM)
        if entities:
            action = entities.get("action", "").lower()
            if "summarize" in action or "summary" in action:
                tools.append("summarization")
            if "search" in action or "find" in action:
                tools.append("document_search")
        
        # Fallback to keyword matching
        if not tools:
            if any(word in query_lower for word in ["summarize", "summary", "overview"]):
                tools.append("summarization")
            if any(word in query_lower for word in ["search", "find", "filter", "show"]):
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
    
    def get_routing_stats(self) -> Dict:
        """
        Get routing statistics for monitoring and debugging.
        
        Returns:
            Dictionary with routing statistics
        """
        total_routes = (
            self.routing_stats["llm_success"] + 
            self.routing_stats["llm_failure"]
        )
        
        if total_routes == 0:
            return {
                **self.routing_stats,
                "llm_success_rate": 0.0,
                "fallback_rate": 0.0
            }
        
        return {
            **self.routing_stats,
            "total_routes": total_routes,
            "llm_success_rate": self.routing_stats["llm_success"] / total_routes,
            "fallback_rate": self.routing_stats["fallback_used"] / total_routes
        }
    
    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self.routing_stats = {
            "llm_success": 0,
            "llm_failure": 0,
            "fallback_used": 0
        }


# Example usage
if __name__ == "__main__":
    from llm import MistralInstructModel
    
    # Initialize router with LLM
    llm = MistralInstructModel()
    router = IntentRouter(llm_classifier=llm, confidence_threshold=0.5)

    print("=" * 70)
    print("LLM-DRIVEN INTENT ROUTER - TEST SUITE")
    print("=" * 70)
    
    test_cases = [
        {
            "query": "How do I reset my password?",
            "context": None,
            "expected": "rag_query"
        },
        {
            "query": "Summarize all password-related documents",
            "context": None,
            "expected": "tool_execution"
        },
        {
            "query": "What about it?",
            "context": {"last_query": "How do I reset my password?", "last_intent": "rag_query"},
            "expected": "follow_up"
        },
        {
            "query": "What's the weather today?",
            "context": None,
            "expected": "rejection"
        },
        {
            "query": "help",
            "context": None,
            "expected": "clarification"
        },
        {
            "query": "Find all documents about audio input",
            "context": None,
            "expected": "tool_execution"
        }
    ]

    print("\nRunning test cases...\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['query']}")
        print(f"Expected: {test['expected']}")
        
        intent, confidence, metadata = router.route(test["query"], test["context"])
        
        print(f"Result: {intent.value} (confidence: {confidence:.2f})")
        print(f"Method: {metadata.get('routing_method', 'unknown')}")
        
        if metadata.get("reasoning"):
            print(f"Reasoning: {metadata['reasoning']}")
        
        if metadata.get("entities"):
            print(f"Entities: {metadata['entities']}")
        
        match = "✅ MATCH" if intent.value == test["expected"] else "❌ MISMATCH"
        print(f"{match}\n")
        print("-" * 70)
    
    # Print routing statistics
    print("\nROUTING STATISTICS:")
    print("=" * 70)
    stats = router.get_routing_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 70)
    print("Test suite completed!")
