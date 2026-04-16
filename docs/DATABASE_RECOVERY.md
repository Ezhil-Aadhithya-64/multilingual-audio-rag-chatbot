# ChromaDB Corruption Recovery Guide

## 🚨 Understanding the Issue

### Root Cause: HNSW Index Corruption

The error `pyo3_runtime.PanicException: range start index 10 out of range for slice of length 9` occurs when ChromaDB's HNSW (Hierarchical Navigable Small World) index files become corrupted.

**Specific cause:** The `link_lists.bin` file (which stores the graph structure for vector search) becomes empty or truncated while other index files still contain data. When ChromaDB tries to read position 10 from a file that only has 9 bytes, the Rust backend panics.

### Common Triggers

1. **Streamlit hot-reload** - Interrupts ChromaDB write operations
2. **Multiple ChromaDB instances** - Concurrent access to the same database
3. **Forced termination** - Ctrl+C during database operations
4. **System crashes** - Power loss or system freeze during writes
5. **Disk space issues** - Incomplete writes due to full disk

## 🔧 Automatic Recovery (Built-in)

The system now includes **automatic corruption detection and recovery**:

### How It Works

1. **Pre-initialization check** - Scans database files BEFORE loading ChromaDB
2. **Corruption detection** - Identifies empty or inconsistent index files
3. **Automatic cleanup** - Backs up and removes corrupted data
4. **Fresh start** - Creates clean database directory

### What You'll See

```
🚨 CHROMADB CORRUPTION DETECTED
================================================================================
  - HNSW index corruption: link_lists.bin is empty but data exists (data_level0.bin: 167600 bytes)
================================================================================
🔧 Automatically cleaning up corrupted database...
✅ Cleanup complete. Fresh database will be created.
⚠️  IMPORTANT: You need to repopulate the database!
   Run: python init_db_simple.py
```

## 🛠️ Manual Recovery Options

### Option 1: Quick Fix (Recommended)

```bash
# Step 1: Delete corrupted database
python scripts/quick_fix_db.py

# Step 2: Reinitialize with sample data
python init_db_simple.py

# Step 3: Restart application
streamlit run app.py
```

### Option 2: Health Check + Repair

```bash
# Check database health
python scripts/check_db_corruption.py

# Check and auto-repair if corrupted
python scripts/check_db_corruption.py --repair

# Reinitialize database
python init_db_simple.py
```

### Option 3: Manual Cleanup

```bash
# Windows PowerShell
Remove-Item -Recurse -Force data/chroma_db
New-Item -ItemType Directory -Path data/chroma_db

# Linux/Mac
rm -rf data/chroma_db
mkdir -p data/chroma_db

# Then reinitialize
python init_db_simple.py
```

## 🔍 Diagnostic Tools

### Check Database Health

```bash
python scripts/check_db_corruption.py --verbose
```

**Output example:**
```
================================================================================
  CHROMADB HEALTH CHECK
================================================================================

Database path: data/chroma_db

1. Checking database directory...
   ✅ PASS

2. Checking SQLite database file...
   File size: 450,560 bytes
   ✅ PASS

3. Checking SQLite integrity...
   ✅ PASS

4. Checking collection directories...
   Found 1 collection(s)
     - 312872cd-8373-4be1-a91f-84b1575ac8ee
   ✅ PASS

5. Checking HNSW index files...

   Collection: 312872cd-8373-4be1-a91f-84b1575ac8ee
     ✅ data_level0.bin: 167,600 bytes
     ✅ header.bin: 100 bytes
     ✅ length.bin: 400 bytes
     ❌ link_lists.bin: CORRUPTED (0 bytes, but data exists)

   ❌ FAIL: Index corruption detected

================================================================================
  HEALTH CHECK SUMMARY
================================================================================

❌ ISSUES FOUND: 1
  1. HNSW index corruption: link_lists.bin is empty but data_level0.bin has 167600 bytes

❌ DATABASE IS CORRUPTED
================================================================================
```

### View Database Stats

```python
from vector_db import VectorDatabase

# Initialize (will auto-recover if corrupted)
db = VectorDatabase()

# Get stats
stats = db.get_collection_stats()
print(f"Collection: {stats['name']}")
print(f"Documents: {stats['count']}")
```

## 🛡️ Prevention Measures

### 1. Singleton Pattern (Implemented)

The `VectorDatabase` class now uses a singleton pattern to prevent multiple instances:

```python
# First initialization
db1 = VectorDatabase()  # Creates new instance

# Subsequent calls reuse the same instance
db2 = VectorDatabase()  # Reuses db1 (same instance)
```

### 2. Session State (Not Cache Resource)

In Streamlit apps, use `session_state` instead of `@st.cache_resource`:

```python
# ❌ BAD: Can cause multiple ChromaDB instances
@st.cache_resource
def load_db():
    return VectorDatabase()

# ✅ GOOD: Single instance per session
if "db" not in st.session_state:
    st.session_state.db = VectorDatabase()
```

### 3. Graceful Shutdown

Always allow the application to shut down gracefully:

- Use Ctrl+C once and wait
- Don't force kill (Ctrl+C multiple times)
- Let Streamlit finish cleanup

### 4. Embedding Validation

The system now validates embeddings before insertion:

```python
# Automatic validation
db.add_documents(
    documents=["text"],
    embeddings=[embedding]  # Validates dimension, NaN, Inf
)
```

## 🔄 Backup and Restore

### Automatic Backups

When corruption is detected, the system automatically creates backups:

```
data/
  chroma_db/              # Current database
  chroma_db_backups/      # Automatic backups
    corrupted_backup_20260416_143022/
    corrupted_backup_20260416_150315/
    corrupted_backup_20260416_152847/
```

Only the last 3 backups are kept to save space.

### Manual Backup

```bash
# Windows PowerShell
Copy-Item -Recurse data/chroma_db data/chroma_db_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')

# Linux/Mac
cp -r data/chroma_db data/chroma_db_backup_$(date +%Y%m%d_%H%M%S)
```

### Restore from Backup

```bash
# Windows PowerShell
Remove-Item -Recurse -Force data/chroma_db
Copy-Item -Recurse data/chroma_db_backups/corrupted_backup_20260416_143022 data/chroma_db

# Linux/Mac
rm -rf data/chroma_db
cp -r data/chroma_db_backups/corrupted_backup_20260416_143022 data/chroma_db
```

## 📊 Monitoring

### Log Messages to Watch

**Healthy initialization:**
```
INFO - Initializing ChromaDB at data/chroma_db
INFO - ✅ Pre-initialization check: Database appears healthy
INFO - Loaded existing collection: multilingual_audio_docs
INFO - ✅ Health check passed: 42 documents in collection
```

**Corruption detected:**
```
ERROR - 🚨 CHROMADB CORRUPTION DETECTED
ERROR -   - HNSW index corruption: link_lists.bin is empty but data exists
WARNING - 🔧 Automatically cleaning up corrupted database...
INFO - ✅ Cleanup complete. Fresh database will be created.
```

### Health Check in Code

```python
from vector_db import VectorDatabase

try:
    db = VectorDatabase(auto_recover=True)
    stats = db.get_collection_stats()
    print(f"✅ Database healthy: {stats['count']} documents")
except RuntimeError as e:
    print(f"❌ Database corrupted: {e}")
```

## 🆘 Troubleshooting

### Issue: "range start index 10 out of range"

**Solution:** Run automatic recovery
```bash
python scripts/check_db_corruption.py --repair
python init_db_simple.py
```

### Issue: Database keeps getting corrupted

**Possible causes:**
1. Multiple Streamlit instances running
2. Using `@st.cache_resource` for VectorDatabase
3. Disk space issues
4. Antivirus interfering with file writes

**Solutions:**
1. Kill all Streamlit processes before starting
2. Use session_state pattern (already implemented)
3. Check disk space: `df -h` (Linux/Mac) or `Get-PSDrive` (Windows)
4. Add ChromaDB directory to antivirus exclusions

### Issue: Slow initialization after recovery

**Explanation:** First initialization downloads embedding models (~100MB)

**Solution:** Wait for model download to complete (one-time only)

### Issue: Empty database after recovery

**Expected behavior:** Recovery deletes corrupted data

**Solution:** Repopulate database
```bash
python init_db_simple.py
```

## 📝 Best Practices

1. **Always use auto_recover=True** (default)
   ```python
   db = VectorDatabase(auto_recover=True)
   ```

2. **Check health before critical operations**
   ```python
   stats = db.get_collection_stats()
   if stats['count'] == 0:
       print("Warning: Database is empty!")
   ```

3. **Monitor logs** - Watch for corruption warnings

4. **Regular backups** - Backup before major changes

5. **Graceful shutdown** - Don't force kill the application

## 🔗 Related Files

- `vector_db.py` - Main database module with auto-recovery
- `scripts/check_db_corruption.py` - Health check tool
- `scripts/quick_fix_db.py` - Quick cleanup script
- `init_db_simple.py` - Database initialization
- `app.py` - Streamlit app with proper error handling

## 📞 Support

If issues persist after following this guide:

1. Check logs in `chatbot.log`
2. Run health check with `--verbose` flag
3. Create a backup of the corrupted database
4. Open an issue with:
   - Error message
   - Health check output
   - Log file excerpt
   - Steps to reproduce
