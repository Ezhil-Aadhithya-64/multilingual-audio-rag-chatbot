# VoiceAssist Pro - Deployment Guide

## Quick Start

### 1. Install Dependencies

```bash
# Install agentic system dependencies
pip install -r requirements_agentic.txt
```

### 2. Configure Environment

```bash
# Create .env file
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

### 3. Ingest Documents

```bash
# Load sample documents into vector database
python ingest_docs.py
```

### 4. Start Services

#### Option A: FastAPI Backend (Production)

```bash
# Start API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Access API docs: http://localhost:8000/docs
```

#### Option B: Streamlit Interface (Interactive)

```bash
# Start web interface
streamlit run app.py

# Access UI: http://localhost:8501
```

---

## API Usage Examples

### Text Query

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query": "How do I reset my password?",
        "return_audio": False
    }
)

result = response.json()
print(f"Intent: {result['intent']}")
print(f"Agent: {result['agent']}")
print(f"Response: {result['response']}")
print(f"Confidence: {result['confidence']}")
```

### Get Metrics

```python
metrics = requests.get(
    f"http://localhost:8000/api/v1/metrics/{result['query_id']}"
).json()

print(f"Latency: {metrics['performance_metrics']['total_latency_ms']}ms")
print(f"Grounding: {metrics['response_metrics']['grounding_score']}")
```

### Session Management

```python
# Get session history
session = requests.get(
    f"http://localhost:8000/api/v1/session/{result['session_id']}"
).json()

print(f"Interactions: {len(session['conversation_history'])}")

# Clear session
requests.delete(f"http://localhost:8000/api/v1/session/{result['session_id']}")
```

---

## Testing

### Test Intent Router

```python
from core.router import IntentRouter

router = IntentRouter()

queries = [
    "How do I reset my password?",
    "Summarize all documents",
    "What about it?",
    "What's the weather?"
]

for query in queries:
    intent, confidence, metadata = router.route(query)
    print(f"{query} → {intent.value} ({confidence:.2f})")
```

### Test Orchestrator

```python
from services.orchestrator import AgenticOrchestrator

orchestrator = AgenticOrchestrator()
session_id = orchestrator.session_manager.create_session()

result = orchestrator.process_query(
    query="How do I reset my password?",
    session_id=session_id
)

print(f"Response: {result['response']}")
print(f"Confidence: {result['confidence']}")
```

---

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_agentic.txt .
RUN pip install --no-cache-dir -r requirements_agentic.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t voiceassist-pro .
docker run -p 8000:8000 --env-file .env voiceassist-pro
```

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_key_here

# Optional
OPENAI_API_KEY=your_key_here  # For OpenAI Whisper API
HUGGINGFACE_TOKEN=your_token  # For HuggingFace models
```

---

## Monitoring

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Aggregate Metrics

```bash
# Last hour
curl "http://localhost:8000/api/v1/metrics?time_window=3600"
```

### Generate Report

```python
from evaluation.metrics import MetricsCollector
from pathlib import Path

collector = MetricsCollector(storage_path=Path("data/metrics"))
report = collector.generate_report(output_path=Path("evaluation_report.txt"))
print(report)
```

---

## Troubleshooting

### Issue: API not starting

```bash
# Check if port is in use
lsof -i :8000

# Use different port
uvicorn api.main:app --port 8001
```

### Issue: Low confidence responses

- Check if documents are ingested: `python ingest_docs.py`
- Verify vector database: Check `data/chroma_db/`
- Review guardrails thresholds in `core/guardrails.py`

### Issue: Slow responses

- Check latency breakdown in metrics
- Consider using local Whisper (faster than API)
- Enable caching: `use_cache=True` in queries

---

## Key Files

- `ARCHITECTURE.md` — Detailed system architecture
- `TRANSFORMATION_SUMMARY.md` — Transformation overview
- `requirements_agentic.txt` — Full dependencies
- `api/main.py` — FastAPI endpoints
- `services/orchestrator.py` — Main coordinator

---

## Support

For issues or questions:
- Check `ARCHITECTURE.md` for system design
- Review `TRANSFORMATION_SUMMARY.md` for workflows
- Examine logs in `chatbot.log`
