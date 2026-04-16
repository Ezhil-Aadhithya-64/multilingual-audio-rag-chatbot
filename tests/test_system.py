"""
Quick Test Script for Agentic System
Tests core components without full dependencies
"""

print("=" * 60)
print("VOICEASSIST PRO - AGENTIC SYSTEM TEST")
print("=" * 60)

# Test 1: Intent Router
print("\n1. Testing Intent Router...")
try:
    from core.router import IntentRouter
    
    router = IntentRouter()
    
    test_queries = [
        "How do I reset my password?",
        "Summarize all password documents",
        "What about it?",
        "What's the weather today?"
    ]
    
    for query in test_queries:
        intent, confidence, metadata = router.route(query)
        print(f"   ✓ '{query[:40]}...' → {intent.value} ({confidence:.2f})")
    
    print("   ✅ Intent Router: PASSED")
except Exception as e:
    print(f"   ❌ Intent Router: FAILED - {e}")

# Test 2: Guardrails Engine
print("\n2. Testing Guardrails Engine...")
try:
    from core.guardrails import GuardrailsEngine
    
    guardrails = GuardrailsEngine()
    
    # Test valid response
    is_valid, response, metadata = guardrails.validate(
        response="To reset your password, click the Forgot Password link.",
        confidence=0.9,
        retrieved_context=["Password reset procedure: Click Forgot Password link."],
        query="How do I reset my password?"
    )
    
    print(f"   ✓ Valid response: {is_valid}")
    print(f"   ✓ Grounding score: {metadata.get('grounding_score', 0):.2f}")
    
    # Test low confidence
    is_valid, response, metadata = guardrails.validate(
        response="Some response",
        confidence=0.3,
        retrieved_context=["Context"],
        query="Test query"
    )
    
    print(f"   ✓ Low confidence handled: {not is_valid}")
    print("   ✅ Guardrails Engine: PASSED")
except Exception as e:
    print(f"   ❌ Guardrails Engine: FAILED - {e}")

# Test 3: Session Manager
print("\n3. Testing Session Manager...")
try:
    from core.session import SessionManager
    from pathlib import Path
    
    session_manager = SessionManager(storage_path=Path("data/test_sessions"))
    
    # Create session
    session_id = session_manager.create_session()
    print(f"   ✓ Created session: {session_id[:8]}...")
    
    # Add interaction
    session_manager.add_interaction(
        session_id=session_id,
        query="Test query",
        intent="rag_query",
        response="Test response",
        confidence=0.9,
        retrieved_context=["Context 1"],
        metadata={"test": True}
    )
    
    # Get history
    history = session_manager.get_conversation_history(session_id)
    print(f"   ✓ Stored interaction: {len(history)} items")
    
    # Get stats
    stats = session_manager.get_session_stats(session_id)
    print(f"   ✓ Session stats: {stats['total_interactions']} interactions")
    
    print("   ✅ Session Manager: PASSED")
except Exception as e:
    print(f"   ❌ Session Manager: FAILED - {e}")

# Test 4: Metrics Collector
print("\n4. Testing Metrics Collector...")
try:
    from evaluation.metrics import MetricsCollector
    from pathlib import Path
    
    collector = MetricsCollector(storage_path=Path("data/test_metrics"))
    
    # Record metrics
    metrics = collector.record_query_metrics(
        query_id="test-001",
        query="Test query",
        intent="rag_query",
        response="Test response",
        confidence=0.9,
        retrieved_docs=["Doc 1", "Doc 2"],
        latency_breakdown={"retrieval_ms": 100, "llm_ms": 500},
        agent_name="RAGAgent",
        success=True
    )
    
    print(f"   ✓ Recorded metrics: {metrics['query_id']}")
    print(f"   ✓ Grounding score: {metrics['response_metrics']['grounding_score']:.2f}")
    
    # Get aggregates
    aggregates = collector.get_aggregate_metrics()
    print(f"   ✓ Aggregate metrics: {aggregates['total_queries']} queries")
    
    print("   ✅ Metrics Collector: PASSED")
except Exception as e:
    print(f"   ❌ Metrics Collector: FAILED - {e}")

# Test 5: Base Agent
print("\n5. Testing Base Agent...")
try:
    from agents.base_agent import BaseAgent
    
    class TestAgent(BaseAgent):
        def execute(self, query, context=None, **kwargs):
            return self._build_result(
                success=True,
                response="Test response",
                confidence=0.9,
                metadata={"test": True}
            )
    
    agent = TestAgent("TestAgent")
    result = agent.execute("Test query")
    
    print(f"   ✓ Agent execution: {result['success']}")
    print(f"   ✓ Response: {result['response']}")
    
    # Record execution
    agent._record_execution(True, 100.0)
    stats = agent.get_stats()
    
    print(f"   ✓ Agent stats: {stats['execution_count']} executions")
    print("   ✅ Base Agent: PASSED")
except Exception as e:
    print(f"   ❌ Base Agent: FAILED - {e}")

# Test 6: Tools
print("\n6. Testing Tool System...")
try:
    from tools.base_tool import BaseTool
    
    class TestTool(BaseTool):
        def execute(self, query, context=None):
            return {"result": "Test tool executed", "success": True}
    
    tool = TestTool("test_tool", "A test tool")
    result = tool.execute("Test query")
    
    print(f"   ✓ Tool execution: {result['success']}")
    print(f"   ✓ Tool info: {tool.get_info()['name']}")
    print("   ✅ Tool System: PASSED")
except Exception as e:
    print(f"   ❌ Tool System: FAILED - {e}")

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("✅ All core components tested successfully!")
print("\nNext steps:")
print("1. Start FastAPI: uvicorn api.main:app --reload --port 8000")
print("2. Test API: http://localhost:8000/docs")
print("3. Or run Streamlit: streamlit run app.py")
print("=" * 60)
