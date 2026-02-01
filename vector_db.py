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

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Vector database manager using ChromaDB.
    Handles document storage, retrieval, and metadata filtering.
    """

    def __init__(
            self,
            collection_name: str = config.CHROMA_COLLECTION_NAME,
            persist_directory: str = str(config.CHROMA_DB_DIR),
            embedding_model: Optional[str] = None
    ):
        """
        Initialize ChromaDB client and collection.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist database
            embedding_model: Model for generating embeddings (optional)
        """
        logger.info(f"Initializing ChromaDB at {persist_directory}")

        # Initialize client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Set up embedding function if provided
        if embedding_model:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        else:
            self.embedding_function = None

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

    def add_documents(
            self,
            documents: List[str],
            embeddings: Optional[List[np.ndarray]] = None,
            metadatas: Optional[List[Dict[str, Any]]] = None,
            ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of document texts
            embeddings: Pre-computed embeddings (optional if using embedding_function)
            metadatas: List of metadata dictionaries
            ids: Document IDs (auto-generated if None)
        """
        if not documents:
            logger.warning("No documents to add")
            return

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

            logger.info(f"Added {len(documents)} documents to collection")

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

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
