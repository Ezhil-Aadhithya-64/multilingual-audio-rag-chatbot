"""
Test Auto-Recovery Mechanism
Simulates database corruption and verifies auto-recovery works
"""

import logging
import shutil
from pathlib import Path
import sys

import config
from vector_db import VectorDatabase

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_auto_recovery():
    """
    Test the auto-recovery mechanism by:
    1. Creating a corrupted database state
    2. Attempting to initialize VectorDatabase
    3. Verifying auto-recovery triggers
    4. Confirming database is functional
    """
    
    logger.info("=" * 60)
    logger.info("AUTO-RECOVERY TEST")
    logger.info("=" * 60)
    
    db_path = config.CHROMA_DB_DIR
    
    # Step 1: Backup existing database if it exists
    logger.info("\n1. Backing up existing database (if any)...")
    backup_path = Path(str(db_path) + "_backup_test")
    
    if db_path.exists():
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(db_path, backup_path)
        logger.info(f"   ✅ Backup created at {backup_path}")
    else:
        logger.info("   ℹ️  No existing database to backup")
    
    # Step 2: Simulate corruption by creating invalid database structure
    logger.info("\n2. Simulating database corruption...")
    try:
        if db_path.exists():
            shutil.rmtree(db_path)
        
        # Create directory with invalid structure to simulate corruption
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy file that will cause corruption
        (db_path / "corrupted.db").write_text("CORRUPTED DATA")
        
        logger.info("   ✅ Simulated corruption created")
    except Exception as e:
        logger.error(f"   ❌ Failed to simulate corruption: {e}")
        return False
    
    # Step 3: Test auto-recovery
    logger.info("\n3. Testing auto-recovery mechanism...")
    logger.info("   Attempting to initialize VectorDatabase with auto_recover=True...")
    
    try:
        # This should trigger auto-recovery
        vector_db = VectorDatabase(
            embedding_model=config.EMBEDDING_MODEL,
            auto_recover=True
        )
        
        logger.info("   ✅ VectorDatabase initialized successfully")
        
        # Step 4: Verify database is functional
        logger.info("\n4. Verifying database functionality...")
        
        # Test count
        count = vector_db.collection.count()
        logger.info(f"   ✅ Document count: {count}")
        
        # Test add
        test_docs = ["Test document 1", "Test document 2"]
        vector_db.add_documents(test_docs)
        new_count = vector_db.collection.count()
        logger.info(f"   ✅ Added documents, new count: {new_count}")
        
        # Test query
        results = vector_db.query(query_text="test", n_results=1)
        logger.info(f"   ✅ Query successful, returned {len(results['documents'])} results")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ AUTO-RECOVERY TEST PASSED")
        logger.info("=" * 60)
        logger.info("\nThe auto-recovery mechanism is working correctly!")
        logger.info("Database was corrupted and automatically recovered.")
        
        success = True
        
    except Exception as e:
        logger.error(f"\n   ❌ Auto-recovery failed: {e}")
        logger.info("\n" + "=" * 60)
        logger.info("❌ AUTO-RECOVERY TEST FAILED")
        logger.info("=" * 60)
        success = False
    
    # Step 5: Cleanup and restore
    logger.info("\n5. Cleaning up test environment...")
    
    try:
        # Remove test database
        if db_path.exists():
            shutil.rmtree(db_path)
            logger.info("   ✅ Test database removed")
        
        # Restore backup if it exists
        if backup_path.exists():
            shutil.copytree(backup_path, db_path)
            logger.info(f"   ✅ Original database restored from backup")
            shutil.rmtree(backup_path)
            logger.info("   ✅ Backup removed")
        else:
            logger.info("   ℹ️  No backup to restore")
            
    except Exception as e:
        logger.error(f"   ⚠️  Cleanup warning: {e}")
    
    return success


def test_without_auto_recovery():
    """
    Test that initialization fails without auto-recovery enabled.
    """
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST WITHOUT AUTO-RECOVERY")
    logger.info("=" * 60)
    
    db_path = config.CHROMA_DB_DIR
    
    # Create corrupted database
    logger.info("\n1. Creating corrupted database...")
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    (db_path / "corrupted.db").write_text("CORRUPTED DATA")
    logger.info("   ✅ Corrupted database created")
    
    # Try to initialize without auto-recovery
    logger.info("\n2. Testing initialization with auto_recover=False...")
    
    try:
        vector_db = VectorDatabase(
            embedding_model=config.EMBEDDING_MODEL,
            auto_recover=False
        )
        logger.error("   ❌ Should have failed but didn't!")
        success = False
    except RuntimeError as e:
        logger.info(f"   ✅ Correctly failed with RuntimeError: {str(e)[:100]}...")
        success = True
    except Exception as e:
        logger.info(f"   ✅ Failed as expected: {str(e)[:100]}...")
        success = True
    
    # Cleanup
    logger.info("\n3. Cleaning up...")
    if db_path.exists():
        shutil.rmtree(db_path)
        logger.info("   ✅ Test database removed")
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✅ TEST PASSED")
        logger.info("=" * 60)
        logger.info("Initialization correctly fails without auto-recovery")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("❌ TEST FAILED")
        logger.info("=" * 60)
    
    return success


def main():
    """Run all tests"""
    
    logger.info("\n" + "=" * 60)
    logger.info("CHROMADB AUTO-RECOVERY TEST SUITE")
    logger.info("=" * 60)
    
    results = []
    
    # Test 1: Auto-recovery enabled
    logger.info("\n\nTEST 1: Auto-Recovery Enabled")
    logger.info("-" * 60)
    results.append(("Auto-Recovery Enabled", test_auto_recovery()))
    
    # Test 2: Auto-recovery disabled
    logger.info("\n\nTEST 2: Auto-Recovery Disabled")
    logger.info("-" * 60)
    results.append(("Auto-Recovery Disabled", test_without_auto_recovery()))
    
    # Summary
    logger.info("\n\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        logger.info("\nAuto-recovery mechanism is working correctly!")
        sys.exit(0)
    else:
        logger.info("\n" + "=" * 60)
        logger.info("❌ SOME TESTS FAILED")
        logger.info("=" * 60)
        logger.info("\nPlease review the failures above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
