"""
Vector Database Module
Uses ChromaDB for storing embeddings and retrieving relevant chunks.
Includes metadata storage and filtering capabilities.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
import config
import threading

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Singleton lock to prevent multiple ChromaDB instances
_instance_lock = threading.Lock()
_instance = None


class VectorDatabase:
    """
    Vector database manager using ChromaDB.
    Handles document storage, retrieval, and metadata filtering.
    """

    def __init__(
            self,
            collection_name: str = config.CHROMA_COLLECTION_NAME,
            persist_directory: str = str(config.CHROMA_DB_DIR),
            embedding_model: Optional[str] = None,
            auto_recover: bool = True,
            use_singleton: bool = True
    ):
        """
        Initialize ChromaDB client and collection with health check and auto-recovery.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist database
            embedding_model: Model for generating embeddings (optional)
            auto_recover: Automatically recover from corruption (default: True)
            use_singleton: Use singleton pattern to prevent multiple instances (default: True)
        """
        # Singleton pattern to prevent multiple ChromaDB instances
        # This is CRITICAL for Streamlit apps that hot-reload
        global _instance
        
        if use_singleton:
            with _instance_lock:
                if _instance is not None:
                    logger.info("Reusing existing VectorDatabase singleton instance")
                    # Copy attributes from singleton
                    self.__dict__ = _instance.__dict__
                    return
        
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.auto_recover = auto_recover

        # Set up embedding function if provided
        if embedding_model:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        else:
            self.embedding_function = None

        # PRE-INITIALIZATION CHECK: Detect and fix corruption before attempting to load
        if auto_recover:
            self._check_and_fix_corruption_before_init(persist_directory)

        # Initialize with health check and auto-recovery
        max_attempts = 2 if auto_recover else 1
        for attempt in range(max_attempts):
            try:
                # Initialize client with persistence
                self.client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True  # Allow reset for recovery
                    )
                )

                # Get or create collection
                try:
                    self.collection = self.client.get_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    logger.info(f"Loaded existing collection: {collection_name}")
                except:
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function,
                        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                    )
                    logger.info(f"Created new collection: {collection_name}")
                
                # Health check: verify collection is accessible
                try:
                    count = self.collection.count()
                    logger.info(f"✅ Health check passed: {count} documents in collection")
                    
                    # Store singleton instance
                    if use_singleton:
                        with _instance_lock:
                            _instance = self
                    
                    break  # Success, exit retry loop
                    
                except Exception as health_error:
                    logger.error(f"Health check failed: {health_error}")
                    raise health_error
                    
            except Exception as e:
                logger.error(f"ChromaDB initialization failed (attempt {attempt + 1}/{max_attempts}): {e}")
                
                if attempt < max_attempts - 1 and auto_recover:
                    # Attempt auto-recovery
                    logger.warning("Attempting auto-recovery: deleting corrupted database")
                    self._delete_corrupted_db(persist_directory)
                else:
                    # Final attempt failed or auto-recover disabled
                    error_msg = (
                        f"ChromaDB initialization failed after {max_attempts} attempts. "
                        f"Database may be corrupted.\n\n"
                        f"To fix:\n"
                        f"1. Run: python scripts/quick_fix_db.py\n"
                        f"2. Run: python init_db_simple.py\n"
                        f"3. Restart the application"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
    
    def _check_and_fix_corruption_before_init(self, persist_directory: str) -> None:
        """
        Check for known corruption patterns BEFORE initialization and fix them.
        This prevents the Rust panic from occurring during PersistentClient creation.
        
        CRITICAL: This is the PRIMARY defense against the "range start index 10 out of range" error.
        The error occurs when HNSW index files (especially link_lists.bin) are corrupted or empty.
        
        Args:
            persist_directory: Path to database directory
        """
        from pathlib import Path
        import sqlite3
        
        db_path = Path(persist_directory)
        
        # Check if database directory exists
        if not db_path.exists():
            logger.info("Database directory doesn't exist, will create fresh")
            return
        
        # Check for the main ChromaDB SQLite file
        chroma_sqlite = db_path / "chroma.sqlite3"
        if not chroma_sqlite.exists():
            logger.info("No existing database found, will create fresh")
            return
        
        # Track corruption indicators
        corruption_detected = False
        corruption_reasons = []
        
        # ============================================================================
        # INDICATOR 1: Check if chroma.sqlite3 is too small (likely corrupted)
        # ============================================================================
        try:
            file_size = chroma_sqlite.stat().st_size
            if file_size < 1000:  # Less than 1KB is suspicious
                corruption_reasons.append(f"SQLite file too small ({file_size} bytes)")
                corruption_detected = True
        except Exception as e:
            corruption_reasons.append(f"Cannot read SQLite file: {e}")
            corruption_detected = True
        
        # ============================================================================
        # INDICATOR 2: Check for HNSW index corruption (PRIMARY ROOT CAUSE)
        # The "range start index 10 out of range" error comes from corrupted HNSW index files
        # ============================================================================
        if not corruption_detected:
            try:
                # Look for collection directories (UUID format: 36 chars with hyphens)
                collection_dirs = [
                    d for d in db_path.iterdir() 
                    if d.is_dir() and len(d.name) == 36 and d.name.count('-') == 4
                ]
                
                for coll_dir in collection_dirs:
                    # CRITICAL: Check HNSW index files
                    # These files store the vector index structure
                    index_files = {
                        'data_level0.bin': 100,      # Min size: at least 100 bytes
                        'header.bin': 50,             # Min size: at least 50 bytes
                        'length.bin': 50,             # Min size: at least 50 bytes
                        'link_lists.bin': 0           # Can be 0 for small collections, but check consistency
                    }
                    
                    for index_file, min_size in index_files.items():
                        index_path = coll_dir / index_file
                        
                        if not index_path.exists():
                            # Missing index file is a corruption indicator
                            corruption_reasons.append(f"Missing index file: {index_file}")
                            corruption_detected = True
                            break
                        
                        try:
                            size = index_path.stat().st_size
                            
                            # CRITICAL CHECK: Empty link_lists.bin with non-empty data files
                            # This is the EXACT cause of "range start index 10 out of range"
                            # NOTE: For small collections, link_lists.bin can be legitimately empty
                            if index_file == 'link_lists.bin' and size == 0:
                                # Check if other files have data
                                data_level0_size = (coll_dir / 'data_level0.bin').stat().st_size
                                # Only flag as corruption if data is very large (>500KB)
                                # The original corruption had 167KB which caused issues
                                # But fresh small collections can have empty link_lists.bin
                                if data_level0_size > 500000:  # 500KB threshold
                                    corruption_reasons.append(
                                        f"HNSW index corruption: {index_file} is empty but data exists "
                                        f"(data_level0.bin: {data_level0_size} bytes)"
                                    )
                                    corruption_detected = True
                                    break
                                else:
                                    # Small collection, this is normal
                                    logger.debug(
                                        f"{index_file} is empty but data is small "
                                        f"({data_level0_size} bytes) - this is normal for small collections"
                                    )
                            
                            # Check minimum size for other files
                            elif size < min_size:
                                corruption_reasons.append(
                                    f"Index file too small: {index_file} ({size} bytes, expected >{min_size})"
                                )
                                corruption_detected = True
                                break
                                
                        except Exception as e:
                            corruption_reasons.append(f"Cannot read index file {index_file}: {e}")
                            corruption_detected = True
                            break
                    
                    if corruption_detected:
                        break
                        
            except Exception as e:
                logger.warning(f"Could not check index files: {e}")
                # Don't mark as corrupted just because we can't check
        
        # ============================================================================
        # INDICATOR 3: SQLite integrity check
        # ============================================================================
        if not corruption_detected:
            try:
                conn = sqlite3.connect(str(chroma_sqlite), timeout=5)
                cursor = conn.cursor()
                
                # Run SQLite integrity check
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                
                if result[0] != 'ok':
                    corruption_reasons.append(f"SQLite integrity check failed: {result[0]}")
                    corruption_detected = True
                
                conn.close()
                
            except Exception as e:
                corruption_reasons.append(f"SQLite check failed: {e}")
                corruption_detected = True
        
        # ============================================================================
        # INDICATOR 4: Check for corruption marker from previous runs
        # ============================================================================
        corruption_marker = db_path / ".corruption_detected"
        if corruption_marker.exists():
            corruption_reasons.append("Previous corruption marker found")
            corruption_detected = True
            try:
                corruption_marker.unlink()
            except:
                pass
        
        # ============================================================================
        # RECOVERY ACTION: Clean up corrupted database
        # ============================================================================
        if corruption_detected:
            logger.error("=" * 80)
            logger.error("🚨 CHROMADB CORRUPTION DETECTED")
            logger.error("=" * 80)
            for reason in corruption_reasons:
                logger.error(f"  - {reason}")
            logger.error("=" * 80)
            logger.warning("🔧 Automatically cleaning up corrupted database...")
            
            # Create backup before deletion (optional)
            self._backup_corrupted_db(persist_directory)
            
            # Delete corrupted database
            self._delete_corrupted_db(persist_directory)
            
            logger.info("✅ Cleanup complete. Fresh database will be created.")
            logger.warning("⚠️  IMPORTANT: You need to repopulate the database!")
            logger.warning("   Run: python init_db_simple.py")
        else:
            logger.info("✅ Pre-initialization check: Database appears healthy")
    
    def _backup_corrupted_db(self, persist_directory: str) -> None:
        """
        Create a backup of corrupted database for forensic analysis.
        
        Args:
            persist_directory: Path to database directory
        """
        import shutil
        from pathlib import Path
        from datetime import datetime
        
        try:
            db_path = Path(persist_directory)
            if not db_path.exists():
                return
            
            # Create backup directory
            backup_dir = db_path.parent / "chroma_db_backups"
            backup_dir.mkdir(exist_ok=True)
            
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"corrupted_backup_{timestamp}"
            
            logger.info(f"Creating backup at {backup_path}")
            shutil.copytree(db_path, backup_path)
            logger.info("Backup created successfully")
            
            # Keep only last 3 backups to save space
            backups = sorted(backup_dir.glob("corrupted_backup_*"))
            if len(backups) > 3:
                for old_backup in backups[:-3]:
                    logger.info(f"Removing old backup: {old_backup}")
                    shutil.rmtree(old_backup)
                    
        except Exception as e:
            logger.warning(f"Failed to create backup (non-critical): {e}")
    
    def _delete_corrupted_db(self, persist_directory: str) -> None:
        """
        Delete corrupted database directory.
        
        Args:
            persist_directory: Path to database directory
        """
        import shutil
        from pathlib import Path
        
        try:
            db_path = Path(persist_directory)
            if db_path.exists():
                logger.info(f"Deleting corrupted database at {persist_directory}")
                shutil.rmtree(db_path)
                logger.info("Corrupted database deleted successfully")
            else:
                logger.info("Database directory does not exist, nothing to delete")
        except Exception as e:
            logger.error(f"Failed to delete corrupted database: {e}")
            raise

    def add_documents(
            self,
            documents: List[str],
            embeddings: Optional[List[np.ndarray]] = None,
            metadatas: Optional[List[Dict[str, Any]]] = None,
            ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector database with validation.

        Args:
            documents: List of document texts
            embeddings: Pre-computed embeddings (optional if using embedding_function)
            metadatas: List of metadata dictionaries
            ids: Document IDs (auto-generated if None)
        """
        if not documents:
            logger.warning("No documents to add")
            return

        # Validate embeddings if provided
        if embeddings is not None:
            self._validate_embeddings(embeddings, len(documents))

        # Generate IDs if not provided
        if ids is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            ids = [f"doc_{timestamp}_{i}" for i in range(len(documents))]

        # Add default metadata if not provided
        if metadatas is None:
            metadatas = [{"timestamp": datetime.now().isoformat()} for _ in documents]
        else:
            # Ensure timestamp in metadata
            for metadata in metadatas:
                if "timestamp" not in metadata:
                    metadata["timestamp"] = datetime.now().isoformat()

        try:
            if embeddings is not None:
                # Convert numpy arrays to lists for ChromaDB
                embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb
                                   for emb in embeddings]

                self.collection.add(
                    documents=documents,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                # Let ChromaDB generate embeddings
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )

            logger.info(f"✅ Added {len(documents)} documents to collection")

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def _validate_embeddings(self, embeddings: List[np.ndarray], expected_count: int) -> None:
        """
        Validate embeddings before insertion to prevent corruption.
        
        Args:
            embeddings: List of embedding vectors
            expected_count: Expected number of embeddings
            
        Raises:
            ValueError: If embeddings are invalid
        """
        if len(embeddings) != expected_count:
            raise ValueError(
                f"Embedding count mismatch: got {len(embeddings)}, expected {expected_count}"
            )
        
        # Check embedding dimensions
        dimensions = set()
        for i, emb in enumerate(embeddings):
            if isinstance(emb, np.ndarray):
                dim = emb.shape[0] if len(emb.shape) == 1 else emb.shape[-1]
            elif isinstance(emb, list):
                dim = len(emb)
            else:
                raise ValueError(f"Invalid embedding type at index {i}: {type(emb)}")
            
            dimensions.add(dim)
            
            # Check for NaN or Inf values
            if isinstance(emb, np.ndarray):
                if np.isnan(emb).any() or np.isinf(emb).any():
                    raise ValueError(f"Embedding at index {i} contains NaN or Inf values")
        
        # All embeddings should have the same dimension
        if len(dimensions) > 1:
            raise ValueError(
                f"Inconsistent embedding dimensions: {dimensions}. "
                f"All embeddings must have the same dimension."
            )
        
        expected_dim = config.EMBEDDING_DIMENSION
        actual_dim = dimensions.pop()
        
        if actual_dim != expected_dim:
            logger.warning(
                f"Embedding dimension mismatch: got {actual_dim}, expected {expected_dim}. "
                f"This may cause issues if the collection was created with a different model."
            )
        
        logger.debug(f"Embedding validation passed: {expected_count} embeddings, dimension {actual_dim}")

    def query(
            self,
            query_text: Optional[str] = None,
            query_embedding: Optional[np.ndarray] = None,
            n_results: int = config.TOP_K_RETRIEVAL,
            where: Optional[Dict[str, Any]] = None,
            where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector database for similar documents.

        Args:
            query_text: Query text (used if query_embedding not provided)
            query_embedding: Pre-computed query embedding
            n_results: Number of results to return
            where: Metadata filter (e.g., {"language": "en"})
            where_document: Document content filter

        Returns:
            Dictionary with results including documents, distances, metadatas
        """
        if query_text is None and query_embedding is None:
            raise ValueError("Either query_text or query_embedding must be provided")

        try:
            params = {"n_results": n_results}

            if where is not None:
                params["where"] = where

            if where_document is not None:
                params["where_document"] = where_document

            if query_embedding is not None:
                # Convert to list if numpy array
                if isinstance(query_embedding, np.ndarray):
                    query_embedding = query_embedding.tolist()
                params["query_embeddings"] = [query_embedding]
            else:
                params["query_texts"] = [query_text]

            results = self.collection.query(**params)

            logger.info(f"Retrieved {len(results['documents'][0])} results")

            return self._format_results(results)

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise

    def _format_results(self, raw_results: Dict) -> Dict[str, Any]:
        """Format ChromaDB results into a cleaner structure."""
        if not raw_results["documents"][0]:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": []
            }

        return {
            "documents": raw_results["documents"][0],
            "metadatas": raw_results["metadatas"][0] if raw_results.get("metadatas") else [],
            "distances": raw_results["distances"][0] if raw_results.get("distances") else [],
            "ids": raw_results["ids"][0]
        }

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection.name)
            logger.info(f"Deleted collection: {self.collection.name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            "name": self.collection.name,
            "count": count,
            "metadata": self.collection.metadata
        }
    
    @staticmethod
    def reset_singleton():
        """
        Reset the singleton instance.
        Useful for testing or when you need to force reinitialization.
        
        WARNING: Only call this when you're sure no other code is using the instance.
        """
        global _instance
        with _instance_lock:
            if _instance is not None:
                logger.info("Resetting VectorDatabase singleton instance")
                try:
                    # Try to close the client gracefully
                    if hasattr(_instance, 'client'):
                        # ChromaDB doesn't have an explicit close, but we can delete the reference
                        del _instance.client
                except Exception as e:
                    logger.warning(f"Error during singleton reset: {e}")
                finally:
                    _instance = None
                    logger.info("Singleton reset complete")


# Example usage
if __name__ == "__main__":
    # Initialize vector database
    vector_db = VectorDatabase()

    # Example: Add sample documents
    sample_docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Vector databases are essential for semantic search."
    ]

    sample_metadata = [
        {"language": "en", "source": "sample", "topic": "ml"},
        {"language": "en", "source": "sample", "topic": "nlp"},
        {"language": "en", "source": "sample", "topic": "database"}
    ]

    # Add documents (embedding_function will generate embeddings)
    # vector_db.add_documents(sample_docs, metadatas=sample_metadata)

    # Query example
    # results = vector_db.query(
    #     query_text="What is AI?",
    #     n_results=2,
    #     where={"language": "en"}
    # )
    # print(f"Found {len(results['documents'])} results")

    # Get stats
    stats = vector_db.get_collection_stats()
    print(f"Collection stats: {stats}")

    print("Vector database module loaded successfully")
