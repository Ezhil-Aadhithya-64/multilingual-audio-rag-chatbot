"""
Test Script for LLM-Driven Router Upgrade
Demonstrates the improved routing capabilities
"""

import logging
from core.router import IntentRouter, Intent
from llm import MistralInstructModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_result(query, intent, confidence, metadata, expected=None):
    """Print formatted routing result"""
    print(f"Query: \"{query}\"")
    print(f"Intent: {intent.value}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Method: {metadata.get('routing_method', 'unknown')}")
    
    if metadata.get('reasoning'):
        print(f"Reasoning: {metadata['reasoning']}")
    
    if metadata.get('entities'):
        print(f"Entities: {metadata['entities']}")
    
    if metadata.get('suggested_tools'):
        print(f"Suggested Tools: {metadata['suggested_tools']}")
    
    if expected:
        match = "✅ MATCH" if intent.value == expected else "❌ MISMATCH"
        print(f"Expected: {expected} → {match}")
    
    print("-" * 80)


def test_basic_routing():
    """Test basic routing scenarios"""
    print_section("TEST 1: Basic Routing Scenarios")
    
    # Initialize router with LLM
    llm = MistralInstructModel()
    router = IntentRouter(llm_classifier=llm, confidence_threshold=0.5)
    
    test_cases = [
        {
            "query": "How do I reset my password?",
            "context": None,
            "expected": "rag_query",
            "description": "Factual question requiring document retrieval"
        },
        {
            "query": "What is VoiceAssist Pro?",
            "context": None,
            "expected": "rag_query",
            "description": "Product information query"
        },
        {
            "query": "Can I use audio input?",
            "context": None,
            "expected": "rag_query",
            "description": "Feature availability question"
        }
    ]
    
    for test in test_cases:
        print(f"Description: {test['description']}")
        intent, confidence, metadata = router.route(test["query"], test["context"])
        print_result(test["query"], intent, confidence, metadata, test["expected"])
        print()


def test_tool_execution():
    """Test tool execution routing"""
    print_section("TEST 2: Tool Execution Routing")
    
    llm = MistralInstructModel()
    router = IntentRouter(llm_classifier=llm)
    
    test_cases = [
        {
            "query": "Summarize all documents about password reset",
            "context": None,
            "expected": "tool_execution",
            "description": "Summarization request"
        },
        {
            "query": "Find all documents related to audio input",
            "context": None,
            "expected": "tool_execution",
            "description": "Document search request"
        },
        {
            "query": "Give me an overview of account management features",
            "context": None,
            "expected": "tool_execution",
            "description": "Overview/summary request"
        }
    ]
    
    for test in test_cases:
        print(f"Description: {test['description']}")
        intent, confidence, metadata = router.route(test["query"], test["context"])
        print_result(test["query"], intent, confidence, metadata, test["expected"])
        print()


def test_context_awareness():
    """Test context-aware routing"""
    print_section("TEST 3: Context-Aware Routing")
    
    llm = MistralInstructModel()
    router = IntentRouter(llm_classifier=llm)
    
    # Simulate conversation
    print("Conversation Flow:\n")
    
    # First query
    query1 = "How do I reset my password?"
    context1 = None
    
    print("User: " + query1)
    intent1, conf1, meta1 = router.route(query1, context1)
    print(f"System: Routed to {intent1.value} (confidence: {conf1:.2f})")
    print(f"Method: {meta1.get('routing_method')}\n")
    
    # Build context from first interaction
    context2 = {
        "last_query": query1,
        "last_intent": intent1.value,
        "last_topic": "password_reset"
    }
    
    # Follow-up query
    query2 = "What about it?"
    print("User: " + query2)
    intent2, conf2, meta2 = router.route(query2, context2)
    print(f"System: Routed to {intent2.value} (confidence: {conf2:.2f})")
    print(f"Method: {meta2.get('routing_method')}")
    print(f"Context Used: {meta2.get('context_available', False)}\n")
    
    # Another follow-up
    query3 = "Tell me more about that"
    print("User: " + query3)
    intent3, conf3, meta3 = router.route(query3, context2)
    print(f"System: Routed to {intent3.value} (confidence: {conf3:.2f})")
    print(f"Method: {meta3.get('routing_method')}\n")


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print_section("TEST 4: Edge Cases")
    
    llm = MistralInstructModel()
    router = IntentRouter(llm_classifier=llm)
    
    test_cases = [
        {
            "query": "help",
            "context": None,
            "expected": "clarification",
            "description": "Vague query requiring clarification"
        },
        {
            "query": "What's the weather today?",
            "context": None,
            "expected": "rejection",
            "description": "Out-of-scope query"
        },
        {
            "query": "Tell me a joke",
            "context": None,
            "expected": "rejection",
            "description": "Entertainment request (out of scope)"
        },
        {
            "query": "info",
            "context": None,
            "expected": "clarification",
            "description": "Too short and ambiguous"
        }
    ]
    
    for test in test_cases:
        print(f"Description: {test['description']}")
        intent, confidence, metadata = router.route(test["query"], test["context"])
        print_result(test["query"], intent, confidence, metadata, test["expected"])
        print()


def test_entity_extraction():
    """Test entity extraction from queries"""
    print_section("TEST 5: Entity Extraction")
    
    llm = MistralInstructModel()
    router = IntentRouter(llm_classifier=llm)
    
    test_cases = [
        "How do I reset my password?",
        "Summarize all documents about audio input",
        "Find information on account management",
        "What are the troubleshooting steps?"
    ]
    
    for query in test_cases:
        intent, confidence, metadata = router.route(query, None)
        
        print(f"Query: \"{query}\"")
        print(f"Intent: {intent.value}")
        
        entities = metadata.get('entities', {})
        if entities:
            print(f"Extracted Entities:")
            for key, value in entities.items():
                print(f"  - {key}: {value}")
        else:
            print("No entities extracted")
        
        print("-" * 80)
        print()


def test_routing_statistics():
    """Test routing statistics tracking"""
    print_section("TEST 6: Routing Statistics")
    
    llm = MistralInstructModel()
    router = IntentRouter(llm_classifier=llm)
    
    # Run multiple queries
    queries = [
        "How do I reset my password?",
        "Summarize all documents",
        "What about it?",
        "What's the weather?",
        "help",
        "Find documents about audio",
        "Tell me about account management",
        "What is VoiceAssist Pro?"
    ]
    
    print("Running multiple queries to collect statistics...\n")
    
    for query in queries:
        intent, confidence, metadata = router.route(query, None)
        print(f"✓ Routed: \"{query[:40]}...\" → {intent.value}")
    
    # Get statistics
    print("\n" + "=" * 80)
    print("ROUTING STATISTICS")
    print("=" * 80 + "\n")
    
    stats = router.get_routing_stats()
    
    print(f"Total Routes: {stats.get('total_routes', 0)}")
    print(f"LLM Success: {stats.get('llm_success', 0)}")
    print(f"LLM Failures: {stats.get('llm_failure', 0)}")
    print(f"Fallback Used: {stats.get('fallback_used', 0)}")
    print(f"\nLLM Success Rate: {stats.get('llm_success_rate', 0):.1%}")
    print(f"Fallback Rate: {stats.get('fallback_rate', 0):.1%}")
    
    # Evaluate performance
    print("\n" + "-" * 80)
    success_rate = stats.get('llm_success_rate', 0)
    if success_rate >= 0.95:
        print("✅ EXCELLENT: LLM routing is working very well!")
    elif success_rate >= 0.85:
        print("✓ GOOD: LLM routing is working well")
    elif success_rate >= 0.70:
        print("⚠ FAIR: LLM routing needs improvement")
    else:
        print("❌ POOR: LLM routing has issues, check logs")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("  LLM-DRIVEN ROUTER UPGRADE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    try:
        # Run test suites
        test_basic_routing()
        test_tool_execution()
        test_context_awareness()
        test_edge_cases()
        test_entity_extraction()
        test_routing_statistics()
        
        print_section("TEST SUITE COMPLETED SUCCESSFULLY ✅")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        print(f"\n❌ TEST SUITE FAILED: {e}")


if __name__ == "__main__":
    main()
