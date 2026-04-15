# VoiceAssist Pro - Production Architecture

## System Transformation Overview

This document describes the transformation from a linear RAG pipeline to a production-grade agentic AI system.

---

## 🏗️ Architecture Evolution

### Before: Linear Pipeline
```
User Query → STT → Embedding → Vector Search → Rerank → LLM → TTS → Response
```

### After: Agentic Decision System
```
User Query
    ↓
[Intent Router] ← Conversation Memory
    ↓
Decision Tree:
├─ RAG Agent (document retrieval + grounded response)
├─ Tool Agent (structured queries, summarization, API calls)
├─ Clarification Agent (ambiguous queries)
└─ Rejection Agent (out-of-scope, low confidence)
    ↓
[Guardrails & Evaluation]
    ↓
Response + Metrics + Memory Update
```

---

## 🧠 Core Components

### 1. Intent Router (`core/router.py`)
**Purpose:** Intelligent query classification and routing

**Decision Logic:**
- Analyzes query intent using LLM-based classification
- Routes to appropriate agent based on:
  - Query type (factual, procedural, clarification)
  - Confidence score
  - Context availability
  - Tool requirements

**Routing Rules:**
```python
if confidence < 0.3:
    → ClarificationAgent
elif requires_structured_data:
    → ToolAgent
elif requires_document_retrieval:
    → RAGAgent
elif out_of_scope:
    → RejectionAgent
```

---

### 2. Agent System (`agents/`)

#### RAGAgent (`agents/rag_agent.py`)
- Handles document-based queries
- Performs semantic search + reranking
- Generates grounded responses
- Tracks retrieval quality metrics

#### ToolAgent (`agents/tool_agent.py`)
- Executes structured operations
- Available tools:
  - `document_search`: Metadata-based filtering
  - `summarize_documents`: Multi-document summarization
  - `query_database`: Mock structured data queries
- Maintains tool execution logs

#### ClarificationAgent (`agents/clarification_agent.py`)
- Handles ambiguous queries
- Generates clarifying questions
- Tracks clarification history

#### RejectionAgent (`agents/rejection_agent.py`)
- Handles out-of-scope queries
- Provides helpful rejection messages
- Suggests alternative approaches

---

### 3. Memory System (`core/memory.py`)

**Session Memory:**
```python
{
    "session_id": "uuid",
    "conversation_history": [
        {
            "timestamp": "2026-04-15T10:30:00",
            "query": "How do I reset my password?",
            "intent": "rag",
            "response": "...",
            "retrieved_context": [...],
            "confidence": 0.92
        }
    ],
    "context_cache": {
        "last_topic": "password_reset",
        "retrieved_docs": [...]
    }
}
```

**Benefits:**
- Avoid redundant retrieval
- Maintain conversation context
- Enable follow-up queries
- Track user patterns

---

### 4. Tool System (`tools/`)

#### DocumentSearchTool
```python
def execute(query: str, filters: Dict) -> List[Dict]:
    """
    Structured document search with metadata filtering
    - Filter by: language, source, topic, date_range
    - Returns: Ranked documents with metadata
    """
```

#### SummarizationTool
```python
def execute(doc_ids: List[str], style: str) -> str:
    """
    Multi-document summarization
    - Styles: brief, detailed, bullet_points
    - Handles: Cross-document synthesis
    """
```

#### DatabaseQueryTool
```python
def execute(query_type: str, params: Dict) -> Dict:
    """
    Mock structured data queries
    - Types: user_stats, system_metrics, audit_logs
    - Returns: Structured JSON responses
    """
```

---

### 5. Evaluation System (`evaluation/`)

**Metrics Tracked:**

1. **Retrieval Quality**
   - Top-K relevance scores
   - Reranking improvement delta
   - Context coverage

2. **Response Quality**
   - Grounding score (context overlap)
   - Confidence score
   - Hallucination detection

3. **System Performance**
   - Latency per stage (STT, retrieval, LLM, TTS)
   - Token usage
   - Cache hit rate

4. **Agent Behavior**
   - Intent classification accuracy
   - Tool selection correctness
   - Clarification effectiveness

**Output:**
```json
{
    "query_id": "uuid",
    "metrics": {
        "retrieval": {
            "top_k_scores": [0.92, 0.87, 0.81],
            "rerank_improvement": 0.15
        },
        "response": {
            "confidence": 0.89,
            "grounding_score": 0.94
        },
        "latency": {
            "total_ms": 1250,
            "breakdown": {
                "stt": 200,
                "retrieval": 150,
                "rerank": 100,
                "llm": 700,
                "tts": 100
            }
        }
    }
}
```

---

### 6. Guardrails System (`core/guardrails.py`)

**Pre-Response Checks:**
- Confidence threshold validation
- Context sufficiency check
- Hallucination detection
- PII/sensitive data filtering

**Decision Logic:**
```python
if confidence < CONFIDENCE_THRESHOLD:
    return "I don't have enough information to answer confidently."

if context_overlap < GROUNDING_THRESHOLD:
    return "I cannot find relevant information in the knowledge base."

if contains_hallucination:
    return fallback_response
```

---

### 7. FastAPI Backend (`api/`)

**Endpoints:**

```python
POST /api/v1/query
{
    "query": "How do I reset my password?",
    "session_id": "optional-uuid",
    "language": "en",
    "return_audio": true
}

POST /api/v1/audio-query
{
    "audio_file": "base64_encoded_audio",
    "session_id": "optional-uuid",
    "language": "auto"
}

POST /api/v1/ingest
{
    "documents": [...],
    "metadata": [...],
    "chunk": true
}

GET /api/v1/metrics/{query_id}
GET /api/v1/session/{session_id}
GET /api/v1/health
```

---

## 🔄 Execution Flow Examples

### Example 1: Standard RAG Query

**Query:** "How do I reset my password?"

```
1. Router Analysis
   - Intent: factual_query
   - Confidence: 0.95
   - Route: RAGAgent

2. RAGAgent Execution
   - Retrieve top-10 documents
   - Rerank to top-5
   - Generate grounded response
   - Confidence: 0.92

3. Guardrails Check
   - Context overlap: 0.94 ✓
   - Confidence: 0.92 ✓
   - Pass

4. Memory Update
   - Store query + response
   - Cache retrieved context

5. Metrics Logging
   - Retrieval quality: 0.89
   - Latency: 1.2s
```

---

### Example 2: Tool-Based Query

**Query:** "Summarize all password-related documents"

```
1. Router Analysis
   - Intent: tool_execution
   - Required tool: summarization
   - Route: ToolAgent

2. ToolAgent Execution
   - Tool: DocumentSearchTool
     - Filter: topic="password"
     - Results: 5 documents
   
   - Tool: SummarizationTool
     - Input: 5 doc_ids
     - Style: detailed
     - Output: Multi-doc summary

3. Response Generation
   - Combine tool outputs
   - Format for user

4. Tool Execution Log
   - Tools used: [search, summarize]
   - Success: true
   - Latency: 2.1s
```

---

### Example 3: Ambiguous Query

**Query:** "How does it work?"

```
1. Router Analysis
   - Intent: ambiguous
   - Confidence: 0.25
   - Route: ClarificationAgent

2. ClarificationAgent Execution
   - Generate clarifying questions:
     * "Are you asking about password reset?"
     * "Are you asking about audio input?"
     * "Are you asking about the system architecture?"

3. Memory Update
   - Store clarification request
   - Await user response

4. Follow-up Handling
   - User: "Password reset"
   - Router: → RAGAgent with context
```

---

### Example 4: Low Confidence / Out of Scope

**Query:** "What's the weather today?"

```
1. Router Analysis
   - Intent: out_of_scope
   - Confidence: 0.15
   - Route: RejectionAgent

2. RejectionAgent Execution
   - Response: "I'm designed to help with VoiceAssist Pro 
     documentation. I cannot answer questions about weather."
   - Suggestion: "Try asking about password reset, audio 
     input, or system features."

3. Metrics Logging
   - Rejection reason: out_of_scope
   - Suggested alternatives: true
```

---

## 📊 System Intelligence Features

### 1. Adaptive Retrieval
- Adjusts top-K based on query complexity
- Uses conversation history to refine search
- Caches frequently accessed documents

### 2. Confidence-Based Routing
- High confidence (>0.8): Direct response
- Medium confidence (0.5-0.8): Response with caveats
- Low confidence (<0.5): Clarification or rejection

### 3. Context Continuity
- Tracks conversation topics
- Resolves pronouns and references
- Maintains multi-turn coherence

### 4. Failure Recovery
- LLM failure → Fallback to retrieval-only
- TTS failure → Text-only response
- Low retrieval quality → Clarification request

---

## 🔐 Production-Ready Features

### Security
- API key management via environment variables
- Rate limiting on endpoints
- Input sanitization and validation
- PII detection and filtering

### Scalability
- Stateless API design
- Session management via external store (Redis-ready)
- Async processing for long-running tasks
- Horizontal scaling support

### Observability
- Structured logging (JSON format)
- Distributed tracing (OpenTelemetry-ready)
- Metrics export (Prometheus-compatible)
- Health check endpoints

### Reliability
- Graceful degradation
- Circuit breakers for external services
- Retry logic with exponential backoff
- Comprehensive error handling

---

## 🎯 Key Design Decisions

### Why Agent-Based Architecture?
- **Modularity:** Each agent has single responsibility
- **Extensibility:** Add new agents without modifying core
- **Testability:** Test agents independently
- **Maintainability:** Clear separation of concerns

### Why LLM-Based Router?
- **Flexibility:** Handles nuanced intent classification
- **Adaptability:** Learns from conversation context
- **Accuracy:** Better than rule-based for complex queries
- **Fallback:** Rule-based backup for reliability

### Why Session Memory?
- **Context:** Enables multi-turn conversations
- **Efficiency:** Reduces redundant retrieval
- **Personalization:** Adapts to user patterns
- **Analytics:** Tracks conversation flows

### Why Evaluation Layer?
- **Quality:** Measurable system performance
- **Debugging:** Identify failure points
- **Optimization:** Data-driven improvements
- **Compliance:** Audit trail for responses

---

## 📈 Performance Benchmarks

**Target Metrics:**
- Query latency: <2s (p95)
- Retrieval accuracy: >85%
- Grounding score: >90%
- System uptime: >99.5%

**Optimization Strategies:**
- Embedding cache for common queries
- Async tool execution
- Batch processing for ingestion
- Model quantization for inference

---

## 🚀 Future Enhancements

### Short-Term
- Redis-backed session store
- Async FastAPI endpoints
- Prometheus metrics export
- Docker containerization

### Medium-Term
- Fine-tuned intent classifier
- Custom reranking model
- Multi-agent collaboration
- Advanced tool orchestration

### Long-Term
- Reinforcement learning from feedback
- Automated knowledge base updates
- Multi-modal support (images, tables)
- Federated learning across deployments

---

This architecture transforms VoiceAssist Pro from a simple RAG pipeline into a production-grade agentic AI system suitable for enterprise deployment.
