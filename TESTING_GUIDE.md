# VoiceAssist Pro - Testing Guide

## ✅ Quick Test (Just Completed)

All core components tested successfully:
- ✅ Intent Router
- ✅ Guardrails Engine  
- ✅ Session Manager
- ✅ Metrics Collector
- ✅ Base Agent
- ✅ Tool System

---

## 🧪 Testing Options

### Option 1: Test Core Components (No Dependencies)

```bash
python test_system.py
```

This tests:
- Intent classification
- Guardrails validation
- Session management
- Metrics collection
- Agent system
- Tool system

**No external dependencies required!**

---

### Option 2: Test FastAPI Backend

#### Start the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

#### Test with Browser

Open: http://localhost:8000/docs

You'll see interactive API documentation with all endpoints.

#### Test with Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/api/v1/health")
print(response.json())

# Simple query (will fail without full setup, but tests routing)
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={"query": "How do I reset my password?"}
)
print(response.json())
```

#### Test with curl

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Query endpoint
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'
```

---

### Option 3: Test Streamlit Interface

```bash
streamlit run app.py
```

Open: http://localhost:8501

Test features:
- Text query input
- Live microphone (if available)
- Configuration options
- Database stats

---

## 🔧 Component-Specific Tests

### Test Intent Router

```python
from core.router import IntentRouter

router = IntentRouter()

# Test different query types
queries = {
    "rag_query": "How do I reset my password?",
    "tool_execution": "Summarize all documents",
    "clarification": "What about it?",
    "rejection": "What's the weather?"
}

for expected_intent, query in queries.items():
    intent, confidence, metadata = router.route(query)
    print(f"{query} → {intent.value} ({confidence:.2f})")
    assert intent.value == expected_intent or confidence < 0.8
```

### Test Guardrails

```python
from core.guardrails import GuardrailsEngine

guardrails = GuardrailsEngine()

# Test valid response
is_valid, response, metadata = guardrails.validate(
    response="To reset your password, click Forgot Password.",
    confidence=0.9,
    retrieved_context=["Password reset: Click Forgot Password link."],
    query="How do I reset my password?"
)

print(f"Valid: {is_valid}")
print(f"Grounding: {metadata['grounding_score']:.2f}")

# Test low confidence
is_valid, response, metadata = guardrails.validate(
    response="Some answer",
    confidence=0.3,
    retrieved_context=["Context"],
    query="Test"
)

print(f"Low confidence rejected: {not is_valid}")
```

### Test Session Memory

```python
from core.session import SessionManager
from pathlib import Path

session_manager = SessionManager(storage_path=Path("data/test_sessions"))

# Create session
session_id = session_manager.create_session()

# Add multiple interactions
for i in range(3):
    session_manager.add_interaction(
        session_id=session_id,
        query=f"Query {i+1}",
        intent="rag_query",
        response=f"Response {i+1}",
        confidence=0.9,
        retrieved_context=[f"Context {i+1}"]
    )

# Get history
history = session_manager.get_conversation_history(session_id)
print(f"Conversation history: {len(history)} interactions")

# Get context
context = session_manager.get_context(session_id)
print(f"Last query: {context.get('last_query')}")
print(f"Last topic: {context.get('last_topic')}")
```

### Test Metrics Collection

```python
from evaluation.metrics import MetricsCollector
from pathlib import Path

collector = MetricsCollector(storage_path=Path("data/test_metrics"))

# Record multiple queries
for i in range(5):
    collector.record_query_metrics(
        query_id=f"test-{i:03d}",
        query=f"Test query {i+1}",
        intent="rag_query",
        response=f"Test response {i+1}",
        confidence=0.8 + (i * 0.02),
        retrieved_docs=[f"Doc {j}" for j in range(3)],
        latency_breakdown={"retrieval_ms": 100, "llm_ms": 500},
        agent_name="RAGAgent",
        success=True
    )

# Get aggregates
aggregates = collector.get_aggregate_metrics()
print(f"Total queries: {aggregates['total_queries']}")
print(f"Success rate: {aggregates['success_rate']:.2%}")
print(f"Avg confidence: {aggregates['avg_confidence']:.2f}")

# Generate report
report = collector.generate_report()
print(report)
```

---

## 🚀 Full System Test (Requires All Dependencies)

### 1. Ingest Documents

```bash
python ingest_docs.py
```

### 2. Start API

```bash
uvicorn api.main:app --reload --port 8000
```

### 3. Test Complete Flow

```python
import requests

# Create session
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query": "How do I reset my password?",
        "return_audio": False
    }
)

result = response.json()
session_id = result['session_id']

print(f"Query ID: {result['query_id']}")
print(f"Intent: {result['intent']}")
print(f"Agent: {result['agent']}")
print(f"Confidence: {result['confidence']}")
print(f"Response: {result['response'][:100]}...")

# Get metrics
metrics_response = requests.get(
    f"http://localhost:8000/api/v1/metrics/{result['query_id']}"
)
metrics = metrics_response.json()

print(f"\nMetrics:")
print(f"- Latency: {metrics['performance_metrics']['total_latency_ms']}ms")
print(f"- Grounding: {metrics['response_metrics']['grounding_score']:.2f}")

# Follow-up query (tests memory)
response2 = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query": "What if I don't receive the email?",
        "session_id": session_id
    }
)

result2 = response2.json()
print(f"\nFollow-up Intent: {result2['intent']}")
print(f"Follow-up Response: {result2['response'][:100]}...")

# Get session history
session_response = requests.get(
    f"http://localhost:8000/api/v1/session/{session_id}"
)
session = session_response.json()

print(f"\nSession has {len(session['conversation_history'])} interactions")
```

---

## 📊 Expected Results

### Intent Router
- "How do I...?" → rag_query (0.75+)
- "Summarize all..." → tool_execution (0.85+)
- "What about it?" → clarification (0.70+)
- "What's the weather?" → rejection (0.90+)

### Guardrails
- High confidence + good grounding → Pass
- Low confidence (<0.5) → Reject with clarification
- Poor grounding (<0.7) → Reject with fallback

### Session Memory
- Stores all interactions
- Maintains context across queries
- Tracks query patterns

### Metrics
- Records latency per stage
- Calculates grounding scores
- Tracks success rates
- Generates reports

---

## 🐛 Troubleshooting

### Issue: Import errors

```bash
# Make sure you're in the project directory
cd /path/to/voiceassist-pro

# Check Python path
python -c "import sys; print(sys.path)"
```

### Issue: API won't start

```bash
# Check if port is in use
netstat -ano | findstr :8000

# Use different port
uvicorn api.main:app --port 8001
```

### Issue: Module not found

```bash
# Install missing dependencies
pip install fastapi uvicorn pydantic --user
```

---

## ✅ Test Checklist

- [ ] Core components test passes (`python test_system.py`)
- [ ] Intent router classifies correctly
- [ ] Guardrails validate responses
- [ ] Session manager stores history
- [ ] Metrics collector records data
- [ ] API health check responds
- [ ] API query endpoint works
- [ ] Streamlit interface loads
- [ ] Full flow test completes

---

## 📝 Notes

- Core components work **without** full dependencies
- FastAPI requires: `fastapi`, `uvicorn`, `pydantic`
- Full system requires: All dependencies in `requirements_agentic.txt`
- Tests create data in `data/test_sessions/` and `data/test_metrics/`

---

**Status:** ✅ Core system tested and working!

**Next:** Start FastAPI or Streamlit to test full system.
