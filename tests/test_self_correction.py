"""
Test Script for Self-Correction Loop
Demonstrates intelligent retry and improvement capabilities
"""

import logging
from services.orchestrator import AgenticOrchestrator
from core.session import SessionManager
from evaluation.metrics import MetricsCollector
from pathlib import Path

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


def print_result(result, show_history=True):
    """Print formatted query result"""
    print(f"Response: {result['response'][:200]}...")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Intent: {result['intent']}")
    print(f"Agent: {result['agent']}")
    
    metadata = result.get('metadata', {})
    
    # Self-correction info
    if metadata.get('self_corrected'):
        print(f"\n🔄 SELF-CORRECTION OCCURRED")
        print(f"Retry Count: {metadata.get('retry_count', 0)}")
        print(f"Initial Confidence: {metadata.get('initial_confidence', 0):.2f}")
        print(f"Final Confidence: {metadata.get('final_confidence', 0):.2f}")
        print(f"Improvement: {metadata.get('improvement_delta', 0):+.2f}")
        
        if show_history and metadata.get('correction_history'):
            print(f"\nCorrection History:")
            for attempt in metadata['correction_history']:
                print(f"\n  Attempt {attempt['attempt']}:")
                print(f"    Failure: {attempt['failure_reason']}")
                print(f"    Previous Confidence: {attempt['confidence']:.2f}")
                print(f"    Feedback: {attempt['feedback'][:150]}...")
    else:
        print(f"\n✅ NO CORRECTION NEEDED (passed on first attempt)")
    
    # Guardrails info
    guardrails = metadata.get('guardrails_metadata', {})
    if guardrails:
        print(f"\nGuardrails:")
        print(f"  Passed: {guardrails.get('passed', False)}")
        if guardrails.get('grounding_score'):
            print(f"  Grounding Score: {guardrails['grounding_score']:.2f}")
        if guardrails.get('enhanced_confidence'):
            print(f"  Enhanced Confidence: {guardrails['enhanced_confidence']:.2f}")
    
    print("-" * 80)


def test_basic_self_correction():
    """Test basic self-correction functionality"""
    print_section("TEST 1: Basic Self-Correction")
    
    # Initialize orchestrator with retries enabled
    session_manager = SessionManager(storage_path=Path("data/test_sessions"))
    metrics_collector = MetricsCollector(storage_path=Path("data/test_metrics"))
    
    orchestrator = AgenticOrchestrator(
        session_manager=session_manager,
        metrics_collector=metrics_collector,
        max_retries=2
    )
    
    # Create test session
    session_id = session_manager.create_session()
    
    # Test query
    query = "How do I reset my password?"
    
    print(f"Query: {query}\n")
    print("Processing...")
    
    result = orchestrator.process_query(
        query=query,
        session_id=session_id
    )
    
    print_result(result)


def test_multiple_queries():
    """Test self-correction across multiple queries"""
    print_section("TEST 2: Multiple Queries with Self-Correction")
    
    orchestrator = AgenticOrchestrator(max_retries=2)
    session_id = orchestrator.session_manager.create_session()
    
    test_queries = [
        "How do I reset my password?",
        "What audio formats are supported?",
        "Tell me about account management",
        "How do I troubleshoot audio issues?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        
        result = orchestrator.process_query(
            query=query,
            session_id=session_id
        )
        
        # Print summary
        metadata = result.get('metadata', {})
        if metadata.get('self_corrected'):
            print(f"✓ Self-corrected (retries: {metadata.get('retry_count', 0)})")
            print(f"  Improvement: {metadata.get('improvement_delta', 0):+.2f} confidence")
        else:
            print(f"✓ Passed on first attempt")
        
        print(f"  Final Confidence: {result['confidence']:.2f}")


def test_correction_statistics():
    """Test correction statistics tracking"""
    print_section("TEST 3: Correction Statistics")
    
    orchestrator = AgenticOrchestrator(max_retries=2)
    session_id = orchestrator.session_manager.create_session()
    
    # Process multiple queries
    queries = [
        "How do I reset my password?",
        "What audio formats are supported?",
        "Tell me about VoiceAssist Pro",
        "How do I troubleshoot issues?",
        "What is the account verification process?"
    ]
    
    print("Processing multiple queries to collect statistics...\n")
    
    for query in queries:
        result = orchestrator.process_query(
            query=query,
            session_id=session_id
        )
        
        metadata = result.get('metadata', {})
        status = "🔄 Corrected" if metadata.get('self_corrected') else "✅ First attempt"
        print(f"{status}: {query[:50]}...")
    
    # Get statistics
    print("\n" + "=" * 80)
    print("SELF-CORRECTION STATISTICS")
    print("=" * 80 + "\n")
    
    stats = orchestrator.get_correction_stats()
    
    print(f"Total Retries: {stats['total_retries']}")
    print(f"Successful Corrections: {stats['successful_corrections']}")
    print(f"Failed Corrections: {stats['failed_corrections']}")
    print(f"Total Correction Attempts: {stats['total_correction_attempts']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    
    # Evaluate performance
    print("\n" + "-" * 80)
    if stats['success_rate'] >= 0.80:
        print("✅ EXCELLENT: Self-correction is working very well!")
    elif stats['success_rate'] >= 0.60:
        print("✓ GOOD: Self-correction is working well")
    else:
        print("⚠ NEEDS IMPROVEMENT: Self-correction success rate is low")


def test_retry_with_different_thresholds():
    """Test self-correction with different guardrail thresholds"""
    print_section("TEST 4: Different Guardrail Thresholds")
    
    query = "How do I reset my password?"
    
    # Test with strict thresholds
    print("Testing with STRICT thresholds (more retries expected):")
    print("-" * 80)
    
    orchestrator_strict = AgenticOrchestrator(max_retries=2)
    orchestrator_strict.guardrails.confidence_threshold = 0.7
    orchestrator_strict.guardrails.grounding_threshold = 0.8
    
    session_id = orchestrator_strict.session_manager.create_session()
    result = orchestrator_strict.process_query(query, session_id)
    
    metadata = result.get('metadata', {})
    print(f"Retry Count: {metadata.get('retry_count', 0)}")
    print(f"Final Confidence: {result['confidence']:.2f}")
    print(f"Self-Corrected: {metadata.get('self_corrected', False)}")
    
    # Test with lenient thresholds
    print("\n\nTesting with LENIENT thresholds (fewer retries expected):")
    print("-" * 80)
    
    orchestrator_lenient = AgenticOrchestrator(max_retries=2)
    orchestrator_lenient.guardrails.confidence_threshold = 0.4
    orchestrator_lenient.guardrails.grounding_threshold = 0.6
    
    session_id = orchestrator_lenient.session_manager.create_session()
    result = orchestrator_lenient.process_query(query, session_id)
    
    metadata = result.get('metadata', {})
    print(f"Retry Count: {metadata.get('retry_count', 0)}")
    print(f"Final Confidence: {result['confidence']:.2f}")
    print(f"Self-Corrected: {metadata.get('self_corrected', False)}")


def test_correction_history():
    """Test detailed correction history tracking"""
    print_section("TEST 5: Detailed Correction History")
    
    orchestrator = AgenticOrchestrator(max_retries=2)
    session_id = orchestrator.session_manager.create_session()
    
    query = "What audio formats are supported?"
    
    print(f"Query: {query}\n")
    print("Processing with detailed history tracking...\n")
    
    result = orchestrator.process_query(
        query=query,
        session_id=session_id
    )
    
    metadata = result.get('metadata', {})
    
    if metadata.get('correction_history'):
        print("CORRECTION HISTORY:")
        print("=" * 80)
        
        for attempt in metadata['correction_history']:
            print(f"\nAttempt {attempt['attempt']}:")
            print(f"  Failure Reason: {attempt['failure_reason']}")
            print(f"  Previous Confidence: {attempt['confidence']:.2f}")
            print(f"\n  Previous Response:")
            print(f"    {attempt['previous_response'][:200]}...")
            print(f"\n  Feedback:")
            print(f"    {attempt['feedback'][:300]}...")
            print("-" * 80)
        
        print(f"\nFinal Result:")
        print(f"  Response: {result['response'][:200]}...")
        print(f"  Final Confidence: {result['confidence']:.2f}")
        print(f"  Improvement: {metadata.get('improvement_delta', 0):+.2f}")
    else:
        print("No correction needed - passed on first attempt!")


def test_max_retries_limit():
    """Test that max retries limit is respected"""
    print_section("TEST 6: Max Retries Limit")
    
    # Test with different max_retries values
    for max_retries in [0, 1, 2, 3]:
        print(f"\nTesting with max_retries={max_retries}:")
        print("-" * 80)
        
        orchestrator = AgenticOrchestrator(max_retries=max_retries)
        
        # Use very strict thresholds to force retries
        orchestrator.guardrails.confidence_threshold = 0.9
        orchestrator.guardrails.grounding_threshold = 0.9
        
        session_id = orchestrator.session_manager.create_session()
        result = orchestrator.process_query(
            query="How do I reset my password?",
            session_id=session_id
        )
        
        metadata = result.get('metadata', {})
        actual_retries = metadata.get('retry_count', 0)
        
        print(f"  Max Retries Setting: {max_retries}")
        print(f"  Actual Retries: {actual_retries}")
        print(f"  Limit Respected: {actual_retries <= max_retries}")
        
        if actual_retries <= max_retries:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL - Exceeded max retries!")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("  SELF-CORRECTION LOOP - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    try:
        # Run test suites
        test_basic_self_correction()
        test_multiple_queries()
        test_correction_statistics()
        test_retry_with_different_thresholds()
        test_correction_history()
        test_max_retries_limit()
        
        print_section("TEST SUITE COMPLETED SUCCESSFULLY ✅")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        print(f"\n❌ TEST SUITE FAILED: {e}")


if __name__ == "__main__":
    main()
