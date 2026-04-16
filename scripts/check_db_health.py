"""
ChromaDB Health Check Script
Verifies database integrity and provides diagnostics
"""

import logging
import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings

import config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_database_health():
    """
    Perform comprehensive health check on ChromaDB.
    
    Returns:
        bool: True if healthy, False if corrupted
    """
    db_path = config.CHROMA_DB_DIR
    
    logger.info("=" * 60)
    logger.info("ChromaDB Health Check")
    logger.info("=" * 60)
    
    # Check 1: Directory exists
    logger.info(f"\n1. Checking database directory: {db_path}")
    if not db_path.exists():
        logger.warning("❌ Database directory does not exist")
        logger.info("   → Run 'python reinitialize_db.py' to create and populate")
        return False
    else:
        logger.info("✅ Database directory exists")
    
    # Check 2: Directory not empty
    logger.info("\n2. Checking directory contents")
    contents = list(db_path.iterdir())
    if not contents:
        logger.warning("❌ Database directory is empty")
        logger.info("   → Run 'python reinitialize_db.py' to populate")
        return False
    else:
        logger.info(f"✅ Found {len(contents)} items in database directory")
    
    # Check 3: Initialize client
    logger.info("\n3. Testing ChromaDB client initialization")
    try:
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        logger.info("✅ Client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Client initialization failed: {e}")
        logger.info("   → Database is corrupted")
        logger.info("   → Run 'python quick_fix_db.py' to reset")
        return False
    
    # Check 4: Get collection
    logger.info("\n4. Testing collection access")
    try:
        collection = client.get_collection(name=config.CHROMA_COLLECTION_NAME)
        logger.info(f"✅ Collection '{config.CHROMA_COLLECTION_NAME}' accessed successfully")
    except Exception as e:
        logger.warning(f"❌ Collection access failed: {e}")
        logger.info("   → Collection may not exist")
        logger.info("   → Run 'python reinitialize_db.py' to create")
        return False
    
    # Check 5: Count documents
    logger.info("\n5. Counting documents")
    try:
        count = collection.count()
        logger.info(f"✅ Collection contains {count} documents")
        
        if count == 0:
            logger.warning("⚠️  Collection is empty")
            logger.info("   → Run 'python reinitialize_db.py' to populate")
            return True  # Not corrupted, just empty
    except Exception as e:
        logger.error(f"❌ Document count failed: {e}")
        logger.info("   → Database is corrupted")
        logger.info("   → Run 'python quick_fix_db.py' to reset")
        return False
    
    # Check 6: Test query
    logger.info("\n6. Testing query functionality")
    try:
        results = collection.query(
            query_texts=["test query"],
            n_results=min(3, count)
        )
        num_results = len(results['documents'][0]) if results['documents'] else 0
        logger.info(f"✅ Query successful, returned {num_results} results")
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        logger.info("   → Database may be corrupted")
        logger.info("   → Run 'python quick_fix_db.py' to reset")
        return False
    
    # All checks passed
    logger.info("\n" + "=" * 60)
    logger.info("✅ DATABASE HEALTH CHECK PASSED")
    logger.info("=" * 60)
    logger.info(f"Database is healthy with {count} documents")
    logger.info("You can safely run the application")
    
    return True


def main():
    """Main execution"""
    try:
        is_healthy = check_database_health()
        
        if not is_healthy:
            logger.info("\n" + "=" * 60)
            logger.info("❌ DATABASE HEALTH CHECK FAILED")
            logger.info("=" * 60)
            logger.info("\nRecommended actions:")
            logger.info("1. Quick fix (deletes and recreates): python quick_fix_db.py")
            logger.info("2. Full reset (with sample data): python reinitialize_db.py")
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"\n❌ Health check failed with error: {e}")
        logger.info("\nRun 'python quick_fix_db.py' to reset the database")
        sys.exit(1)


if __name__ == "__main__":
    main()
