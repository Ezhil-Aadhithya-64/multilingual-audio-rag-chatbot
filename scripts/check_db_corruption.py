"""
ChromaDB Corruption Detection and Repair Tool

This script performs comprehensive health checks on the ChromaDB database
and can automatically repair corruption issues.

Usage:
    python scripts/check_db_corruption.py [--repair] [--verbose]
"""

import sys
import argparse
import logging
from pathlib import Path
import sqlite3
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaDBHealthChecker:
    """Comprehensive health checker for ChromaDB"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.issues = []
        self.warnings = []
    
    def check_all(self) -> bool:
        """
        Run all health checks.
        
        Returns:
            True if database is healthy, False if corruption detected
        """
        print("=" * 80)
        print("  CHROMADB HEALTH CHECK")
        print("=" * 80)
        print(f"\nDatabase path: {self.db_path}")
        print()
        
        # Check 1: Directory exists
        if not self._check_directory_exists():
            return False
        
        # Check 2: SQLite file
        if not self._check_sqlite_file():
            return False
        
        # Check 3: SQLite integrity
        if not self._check_sqlite_integrity():
            return False
        
        # Check 4: Collection directories
        if not self._check_collection_directories():
            return False
        
        # Check 5: HNSW index files
        if not self._check_hnsw_indices():
            return False
        
        # Print summary
        self._print_summary()
        
        return len(self.issues) == 0
    
    def _check_directory_exists(self) -> bool:
        """Check if database directory exists"""
        print("1. Checking database directory...")
        
        if not self.db_path.exists():
            self.issues.append("Database directory does not exist")
            print("   ❌ FAIL: Directory not found")
            return False
        
        if not self.db_path.is_dir():
            self.issues.append("Database path is not a directory")
            print("   ❌ FAIL: Not a directory")
            return False
        
        print("   ✅ PASS")
        return True
    
    def _check_sqlite_file(self) -> bool:
        """Check SQLite database file"""
        print("\n2. Checking SQLite database file...")
        
        sqlite_path = self.db_path / "chroma.sqlite3"
        
        if not sqlite_path.exists():
            self.issues.append("SQLite database file missing")
            print("   ❌ FAIL: chroma.sqlite3 not found")
            return False
        
        # Check file size
        file_size = sqlite_path.stat().st_size
        print(f"   File size: {file_size:,} bytes")
        
        if file_size < 1000:
            self.issues.append(f"SQLite file suspiciously small ({file_size} bytes)")
            print("   ❌ FAIL: File too small (likely corrupted)")
            return False
        
        print("   ✅ PASS")
        return True
    
    def _check_sqlite_integrity(self) -> bool:
        """Check SQLite database integrity"""
        print("\n3. Checking SQLite integrity...")
        
        sqlite_path = self.db_path / "chroma.sqlite3"
        
        try:
            conn = sqlite3.connect(str(sqlite_path), timeout=5)
            cursor = conn.cursor()
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            conn.close()
            
            if result[0] != 'ok':
                self.issues.append(f"SQLite integrity check failed: {result[0]}")
                print(f"   ❌ FAIL: {result[0]}")
                return False
            
            print("   ✅ PASS")
            return True
            
        except Exception as e:
            self.issues.append(f"Cannot open SQLite database: {e}")
            print(f"   ❌ FAIL: {e}")
            return False
    
    def _check_collection_directories(self) -> bool:
        """Check collection directories"""
        print("\n4. Checking collection directories...")
        
        # Look for UUID-format directories (36 chars with 4 hyphens)
        collection_dirs = [
            d for d in self.db_path.iterdir()
            if d.is_dir() and len(d.name) == 36 and d.name.count('-') == 4
        ]
        
        if not collection_dirs:
            self.warnings.append("No collection directories found (database may be empty)")
            print("   ⚠️  WARNING: No collections found")
            return True
        
        print(f"   Found {len(collection_dirs)} collection(s)")
        for coll_dir in collection_dirs:
            print(f"     - {coll_dir.name}")
        
        print("   ✅ PASS")
        return True
    
    def _check_hnsw_indices(self) -> bool:
        """Check HNSW index files (CRITICAL CHECK)"""
        print("\n5. Checking HNSW index files...")
        
        collection_dirs = [
            d for d in self.db_path.iterdir()
            if d.is_dir() and len(d.name) == 36 and d.name.count('-') == 4
        ]
        
        if not collection_dirs:
            print("   ⏭️  SKIP: No collections to check")
            return True
        
        all_healthy = True
        
        for coll_dir in collection_dirs:
            print(f"\n   Collection: {coll_dir.name}")
            
            # Required index files
            index_files = {
                'data_level0.bin': 100,
                'header.bin': 50,
                'length.bin': 50,
                'link_lists.bin': 0  # Can be 0 for small collections
            }
            
            for index_file, min_size in index_files.items():
                index_path = coll_dir / index_file
                
                if not index_path.exists():
                    self.issues.append(f"Missing index file: {index_file} in {coll_dir.name}")
                    print(f"     ❌ {index_file}: MISSING")
                    all_healthy = False
                    continue
                
                file_size = index_path.stat().st_size
                
                # CRITICAL: Check for empty link_lists.bin with non-empty data
                # NOTE: link_lists.bin can be legitimately 0 bytes for small collections
                # Only flag as corruption if we have a LARGE amount of data
                if index_file == 'link_lists.bin' and file_size == 0:
                    data_level0_path = coll_dir / 'data_level0.bin'
                    if data_level0_path.exists():
                        data_size = data_level0_path.stat().st_size
                        # Only flag as corruption if data is very large (>500KB)
                        # Small collections can have empty link_lists.bin
                        if data_size > 500000:  # 500KB threshold
                            self.issues.append(
                                f"HNSW index corruption: {index_file} is empty but "
                                f"data_level0.bin has {data_size} bytes"
                            )
                            print(f"     ❌ {index_file}: CORRUPTED (0 bytes, but data exists)")
                            all_healthy = False
                            continue
                        else:
                            # Small collection, empty link_lists.bin is OK
                            self.warnings.append(
                                f"{index_file} is empty (OK for small collections with "
                                f"{data_size} bytes of data)"
                            )
                
                # Check minimum size
                if file_size < min_size:
                    self.issues.append(
                        f"Index file too small: {index_file} ({file_size} bytes, "
                        f"expected >{min_size})"
                    )
                    print(f"     ❌ {index_file}: TOO SMALL ({file_size} bytes)")
                    all_healthy = False
                    continue
                
                print(f"     ✅ {index_file}: {file_size:,} bytes")
        
        if all_healthy:
            print("\n   ✅ PASS: All index files healthy")
        else:
            print("\n   ❌ FAIL: Index corruption detected")
        
        return all_healthy
    
    def _print_summary(self):
        """Print health check summary"""
        print("\n" + "=" * 80)
        print("  HEALTH CHECK SUMMARY")
        print("=" * 80)
        
        if self.issues:
            print(f"\n❌ ISSUES FOUND: {len(self.issues)}")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.issues and not self.warnings:
            print("\n✅ DATABASE IS HEALTHY")
        elif not self.issues:
            print("\n✅ DATABASE IS HEALTHY (with warnings)")
        else:
            print("\n❌ DATABASE IS CORRUPTED")
        
        print("=" * 80)
    
    def repair(self) -> bool:
        """
        Attempt to repair the database by backing up and deleting corrupted data.
        
        Returns:
            True if repair successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("  ATTEMPTING REPAIR")
        print("=" * 80)
        
        # Create backup
        print("\n1. Creating backup...")
        backup_path = self._create_backup()
        if backup_path:
            print(f"   ✅ Backup created: {backup_path}")
        else:
            print("   ⚠️  Backup failed (continuing anyway)")
        
        # Delete corrupted database
        print("\n2. Deleting corrupted database...")
        try:
            shutil.rmtree(self.db_path)
            print("   ✅ Deleted")
        except Exception as e:
            print(f"   ❌ Failed to delete: {e}")
            return False
        
        # Recreate directory
        print("\n3. Creating fresh database directory...")
        try:
            self.db_path.mkdir(parents=True, exist_ok=True)
            print("   ✅ Created")
        except Exception as e:
            print(f"   ❌ Failed to create: {e}")
            return False
        
        print("\n" + "=" * 80)
        print("  REPAIR COMPLETE ✅")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run: python init_db_simple.py")
        print("2. Restart your application")
        print()
        
        return True
    
    def _create_backup(self) -> Path:
        """Create backup of corrupted database"""
        try:
            backup_dir = self.db_path.parent / "chroma_db_backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"corrupted_backup_{timestamp}"
            
            shutil.copytree(self.db_path, backup_path)
            
            # Keep only last 3 backups
            backups = sorted(backup_dir.glob("corrupted_backup_*"))
            if len(backups) > 3:
                for old_backup in backups[:-3]:
                    shutil.rmtree(old_backup)
            
            return backup_path
            
        except Exception as e:
            logger.warning(f"Backup failed: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Check ChromaDB health and optionally repair corruption"
    )
    parser.add_argument(
        '--repair',
        action='store_true',
        help='Automatically repair corruption if detected'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run health check
    checker = ChromaDBHealthChecker(config.CHROMA_DB_DIR)
    is_healthy = checker.check_all()
    
    # Repair if requested and corruption detected
    if not is_healthy and args.repair:
        print("\n🔧 Repair mode enabled, attempting to fix...")
        success = checker.repair()
        sys.exit(0 if success else 1)
    elif not is_healthy:
        print("\n💡 To automatically repair, run with --repair flag:")
        print(f"   python {sys.argv[0]} --repair")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
