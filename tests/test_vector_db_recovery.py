"""
Test suite for VectorDatabase corruption detection and recovery
"""

import sys
import pytest
import shutil
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_db import VectorDatabase
from embedding import TextEmbedder
import config


class TestVectorDBRecovery:
    """Test VectorDatabase corruption detection and recovery"""
    
    @pytest.fixture
    def temp_db_dir(self):
        """Create temporary database directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance"""
        return TextEmbedder()
    
    def test_fresh_initialization(self, temp_db_dir, embedder):
        """Test fresh database initialization"""
        db = VectorDatabase(
            persist_directory=str(temp_db_dir),
            embedding_model=config.EMBEDDING_MODEL,
            use_singleton=False  # Disable singleton for testing
        )
        
        # Should create empty collection
        stats = db.get_collection_stats()
        assert stats['count'] == 0
        assert stats['name'] == config.CHROMA_COLLECTION_NAME
    
    def test_add_documents(self, temp_db_dir, embedder):
        """Test adding documents to database"""
        db = VectorDatabase(
            persist_directory=str(temp_db_dir),
            embedding_model=config.EMBEDDING_MODEL,
            use_singleton=False
        )
        
        # Add sample documents
        docs = ["Test document 1", "Test document 2"]
        embeddings = embedder.embed_batch(docs)
        
        db.add_documents(documents=docs, embeddings=embeddings)
        
        # Verify documents added
        stats = db.get_collection_stats()
        assert stats['count'] == 2
    
    def test_embedding_validation_count_mismatch(self, temp_db_dir, embedder):
        """Test embedding validation catches count mismatch"""
        db = VectorDatabase(
            persist_directory=str(temp_db_dir),
            embedding_model=config.EMBEDDING_MODEL,
            use_singleton=False
        )
        
        docs = ["Doc 1", "Doc 2"]
        embeddings = embedder.embed_batch(["Doc 1"])  # Only 1 embedding
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Embedding count mismatch"):
            db.add_documents(documents=docs, embeddings=embeddings)
    
    def test_embedding_validation_dimension_consistency(self, temp_db_dir):
        """Test embedding validation catches dimension inconsistency"""
        import numpy as np
        
        db = VectorDatabase(
            persist_directory=str(temp_db_dir),
            embedding_model=config.EMBEDDING_MODEL,
            use_singleton=False
        )
        
        docs = ["Doc 1", "Doc 2"]
        # Create embeddings with different dimensions
        embeddings = [
            np.random.rand(384),  # Correct dimension
            np.random.rand(512)   # Wrong dimension
        ]
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Inconsistent embedding dimensions"):
            db.add_documents(documents=docs, embeddings=embeddings)
    
    def test_embedding_validation_nan_values(self, temp_db_dir):
        """Test embedding validation catches NaN values"""
        import numpy as np
        
        db = VectorDatabase(
            persist_directory=str(temp_db_dir),
            embedding_model=config.EMBEDDING_MODEL,
            use_singleton=False
        )
        
        docs = ["Doc 1"]
        embeddings = [np.array([np.nan] * 384)]
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="NaN or Inf"):
            db.add_documents(documents=docs, embeddings=embeddings)
    
    def test_singleton_pattern(self, temp_db_dir):
        """Test singleton pattern prevents multiple instances"""
        # Reset singleton first
        VectorDatabase.reset_singleton()
        
        db1 = VectorDatabase(
            persist_directory=str(temp_db_dir),
            embedding_model=config.EMBEDDING_MODEL,
            use_singleton=True
        )
        
        db2 = VectorDatabase(
            persist_directory=str(temp_db_dir),
            embedding_model=config.EMBEDDING_MODEL,
            use_singleton=True
        )
        
        # Should be the same instance
        assert db1.client is db2.client
        assert db1.collection is db2.collection
    
    def test_singleton_reset(self, temp_db_dir):
        """Test singleton reset functionality"""
        VectorDatabase.reset_singleton()
        
        db1 = VectorDatabase(
            persist_directory=str(temp_db_dir),
            embedding_model=config.EMBEDDING_MODEL,
            use_singleton=True
        )
        
        # Reset singleton
        VectorDatabase.reset_singleton()
        
        db2 = VectorDatabase(
            persist_directory=str(temp_db_dir),
            embedding_model=config.EMBEDDING_MODEL,
            use_singleton=True
        )
        
        # Should be different instances after reset
        # (but will load same collection from disk)
        assert db1 is not db2
    
    def test_auto_recovery_on_missing_directory(self, temp_db_dir):
        """Test auto-recovery when directory doesn't exist"""
        # Use non-existent directory
        non_existent = temp_db_dir / "non_existent"
        
        db = VectorDatabase(
            persist_directory=str(non_existent),
            embedding_model=config.EMBEDDING_MODEL,
            auto_recover=True,
            use_singleton=False
        )
        
        # Should create directory and initialize
        assert non_existent.exists()
        stats = db.get_collection_stats()
        assert stats['count'] == 0
    
    def test_query_empty_database(self, temp_db_dir):
        """Test querying empty database"""
        db = VectorDatabase(
            persist_directory=str(temp_db_dir),
            embedding_model=config.EMBEDDING_MODEL,
            use_singleton=False
        )
        
        # Query should return empty results
        results = db.query(query_text="test query", n_results=5)
        
        assert results['documents'] == []
        assert results['metadatas'] == []
        assert results['distances'] == []
        assert results['ids'] == []
    
    def test_query_with_results(self, temp_db_dir, embedder):
        """Test querying database with results"""
        db = VectorDatabase(
            persist_directory=str(temp_db_dir),
            embedding_model=config.EMBEDDING_MODEL,
            use_singleton=False
        )
        
        # Add documents
        docs = ["Machine learning is great", "Python is awesome"]
        embeddings = embedder.embed_batch(docs)
        db.add_documents(documents=docs, embeddings=embeddings)
        
        # Query
        results = db.query(query_text="machine learning", n_results=1)
        
        assert len(results['documents']) == 1
        assert "Machine learning" in results['documents'][0]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
