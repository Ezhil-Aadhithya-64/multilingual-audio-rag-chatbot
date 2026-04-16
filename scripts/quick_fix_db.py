"""
Quick Fix for ChromaDB Corruption
Simply deletes and recreates the database directory
"""

import shutil
from pathlib import Path

print("=" * 80)
print("  QUICK FIX: ChromaDB Corruption")
print("=" * 80)

# Delete corrupted database
db_path = Path("data/chroma_db")
if db_path.exists():
    print(f"\n1. Removing corrupted database at {db_path}...")
    shutil.rmtree(db_path)
    print("   ✓ Removed")
else:
    print(f"\n1. Database directory doesn't exist")

# Create fresh directory
print(f"\n2. Creating fresh database directory...")
db_path.mkdir(parents=True, exist_ok=True)
print("   ✓ Created")

print("\n" + "=" * 80)
print("  DATABASE RESET COMPLETE ✅")
print("=" * 80)
print("\nNext steps:")
print("1. Run: streamlit run app.py")
print("2. The database will be automatically populated on first use")
print("3. Or run: python sample_knowledge_base.py")
print()
