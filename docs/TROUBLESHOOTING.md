# Troubleshooting Guide

## Common Issues and Solutions

### 1. ChromaDB Corruption Error

**Error:**
```
thread '<unnamed>' panicked at rust\sqlite\src\db.rs:157:42:
range start index 10 out of range for slice of length 9
pyo3_runtime.PanicException: range start index 10 out of range for slice of length 9
```

**Cause:** ChromaDB SQLite database is corrupted

**Solution:**

**Option A: Automatic Recovery (NEW - Recommended)**

The system now includes automatic recovery. Simply restart the application:

```bash
streamlit run app.py
```

The system will:
1. Detect database corruption automatically
2. Delete the corrupted database
3. Create a fresh database
4. Continue initialization

**Note:** You'll need to re-ingest documents after auto-recovery.

**Option B: Health Check (Diagnostic)**

Run the health check script to diagnose the issue:

```bash
python check_db_health.py
```

This will verify database integrity and provide specific recommendations.

**Option C: Quick Fix (Manual Reset)**

If auto-recovery fails, manually reset the database:

```bash
python quick_fix_db.py
```

**Option D: Full Reinitialization (With Sample Data)**

For a complete reset with sample data:

```bash
python reinitialize_db.py
```

This will recreate the database and populate it with sample data (takes 2-3 minutes).

**What Changed:**
- `vector_db.py`: Added health check and auto-recovery during initialization
- `app.py`: Changed from `@st.cache_resource` to session state to prevent caching conflicts
- New script: `check_db_health.py` for diagnostics

---

### 2. Missing API Key

**Error:**
```
ValueError: GROQ_KEY must be set in environment or passed explicitly
```

**Solution:**
1. Create/edit `.env` file in project root
2. Add your Groq API key:
   ```
   GROQ_KEY=your_api_key_here
   ```
3. Restart the application

---

### 3. Model Loading Issues

**Error:**
```
OSError: Can't load tokenizer for 'sentence-transformers/...'
```

**Solution:**
1. Check internet connection
2. Clear HuggingFace cache:
   ```bash
   # Windows
   Remove-Item -Recurse -Force $env:USERPROFILE\.cache\huggingface
   
   # Linux/Mac
   rm -rf ~/.cache/huggingface
   ```
3. Restart the application

---

### 4. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'X'
```

**Solution:**
```bash
# Install all dependencies
pip install -r requirements_agentic.txt

# Or install specific package
pip install package_name
```

---

### 5. Streamlit Port Already in Use

**Error:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process (Windows)
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Or kill existing process (Linux/Mac)
lsof -ti:8501 | xargs kill -9
```

---

### 6. Slow First Startup

**Issue:** Application takes 1-2 minutes to start

**Explanation:** This is normal! The system needs to:
- Load Whisper model (~500MB)
- Load embedding model (~400MB)
- Initialize ChromaDB
- Load LLM connection

**Solution:** Be patient on first startup. Subsequent startups will be faster due to caching.

---

### 7. Low Quality Responses

**Issue:** Responses are generic or incorrect

**Possible Causes:**
1. Empty or corrupted knowledge base
2. LLM API issues
3. Low confidence threshold

**Solutions:**

**Check Knowledge Base:**
```python
from vector_db import VectorDatabase
import config

db = VectorDatabase(embedding_model=config.EMBEDDING_MODEL)
count = db.collection.count()
print(f"Documents in database: {count}")
```

If count is 0, reinitialize:
```bash
python reinitialize_db.py
```

**Check LLM Connection:**
```python
from llm import MistralInstructModel

llm = MistralInstructModel()
result = llm.generate_response("Test query", ["Test context"])
print(result)
```

**Adjust Thresholds:**
Edit `services/orchestrator.py`:
```python
self.guardrails = GuardrailsEngine(
    confidence_threshold=0.4,  # Lower = more lenient
    grounding_threshold=0.6    # Lower = more lenient
)
```

---

### 8. Self-Correction Not Working

**Issue:** Responses fail guardrails but no retry occurs

**Check:**
1. Verify `max_retries` is set:
   ```python
   orchestrator = AgenticOrchestrator(max_retries=2)
   ```

2. Check if retry is recommended:
   ```python
   # In response metadata
   metadata = result["metadata"]
   print(f"Retry recommended: {metadata.get('guardrails_metadata', {}).get('retry_recommended')}")
   ```

3. View correction statistics:
   ```python
   stats = orchestrator.get_correction_stats()
   print(stats)
   ```

---

### 9. Memory/Performance Issues

**Issue:** High memory usage or slow responses

**Solutions:**

**Reduce Model Size:**
Edit `config.py`:
```python
# Use smaller Whisper model
WHISPER_MODEL = "tiny"  # Instead of "base"

# Use smaller embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

**Disable Reranker:**
```python
orchestrator = AgenticOrchestrator(use_reranker=False)
```

**Reduce Retrieval:**
Edit `agents/rag_agent.py`:
```python
self.top_k = 5  # Instead of 10
self.rerank_top_n = 3  # Instead of 5
```

---

### 10. Router Not Using LLM

**Issue:** Router falls back to rule-based routing

**Check:**
1. Verify LLM is passed to router:
   ```python
   from llm import MistralInstructModel
   
   llm = MistralInstructModel()
   router = IntentRouter(llm_classifier=llm)
   ```

2. Check routing statistics:
   ```python
   stats = router.get_routing_stats()
   print(f"LLM Success Rate: {stats['llm_success_rate']:.1%}")
   print(f"Fallback Rate: {stats['fallback_rate']:.1%}")
   ```

3. If fallback rate > 10%, check:
   - LLM API connectivity
   - API key validity
   - Prompt clarity

---

## Quick Diagnostic Script

Save as `diagnose.py`:

```python
"""Quick diagnostic script"""

print("=" * 80)
print("  SYSTEM DIAGNOSTICS")
print("=" * 80)

# 1. Check imports
print("\n1. Checking imports...")
try:
    import streamlit
    import chromadb
    import sentence_transformers
    import whisper
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ❌ Import error: {e}")

# 2. Check database
print("\n2. Checking database...")
try:
    from vector_db import VectorDatabase
    import config
    db = VectorDatabase(embedding_model=config.EMBEDDING_MODEL)
    count = db.collection.count()
    print(f"   ✓ Database accessible ({count} documents)")
except Exception as e:
    print(f"   ❌ Database error: {e}")

# 3. Check LLM
print("\n3. Checking LLM...")
try:
    from llm import MistralInstructModel
    llm = MistralInstructModel()
    print("   ✓ LLM initialized")
except Exception as e:
    print(f"   ❌ LLM error: {e}")

# 4. Check API key
print("\n4. Checking API key...")
import os
from dotenv import load_dotenv
load_dotenv()
if os.getenv("GROQ_KEY"):
    print("   ✓ API key found")
else:
    print("   ❌ API key not found in .env")

print("\n" + "=" * 80)
print("  DIAGNOSTICS COMPLETE")
print("=" * 80)
```

Run with:
```bash
python diagnose.py
```

---

## Getting Help

If issues persist:

1. **Check logs:** Look for error messages in console output
2. **Enable debug logging:** Set `LOG_LEVEL=DEBUG` in `config.py`
3. **Run diagnostics:** `python diagnose.py`
4. **Check documentation:** Review README.md and ARCHITECTURE.md
5. **Reset system:** 
   ```bash
   python quick_fix_db.py
   python reinitialize_db.py
   ```

---

## Prevention Tips

1. **Regular backups:** Backup `data/chroma_db` periodically
2. **Graceful shutdown:** Use Ctrl+C to stop Streamlit (don't force kill)
3. **Keep dependencies updated:** `pip install --upgrade -r requirements_agentic.txt`
4. **Monitor disk space:** ChromaDB needs adequate disk space
5. **Check API limits:** Monitor Groq API usage

---

## System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB disk space
- Internet connection (for model downloads)

**Recommended:**
- Python 3.10+
- 8GB RAM
- 5GB disk space
- Stable internet connection

---

## Status Indicators

**Healthy System:**
- ✅ All imports successful
- ✅ Database accessible
- ✅ LLM initialized
- ✅ API key configured
- ✅ Models loaded

**Needs Attention:**
- ⚠️ High fallback rate (>10%)
- ⚠️ Low correction success rate (<80%)
- ⚠️ Slow response times (>5s)

**Critical Issues:**
- ❌ Database corruption
- ❌ Missing API key
- ❌ Import errors
- ❌ Model loading failures
