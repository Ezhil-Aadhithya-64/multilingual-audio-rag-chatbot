"""
Embedding Module
Converts text into vector embeddings using SentenceTransformers.
Includes text chunking with overlap for efficient retrieval.
"""

import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Text embedding generator using multilingual SentenceTransformers.
    Supports chunking with overlap for long documents.
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name for embeddings
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed_text(
            self,
            text: str,
            normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string
            normalize: Normalize embedding vector

        Returns:
            Numpy array of embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.dimension)

        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

        return embedding

    def embed_batch(
            self,
            texts: List[str],
            batch_size: int = config.BATCH_SIZE,
            normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            normalize: Normalize embedding vectors

        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])

        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True
        )

        return embeddings

    def chunk_text(
            self,
            text: str,
            chunk_size: int = config.CHUNK_SIZE,
            overlap: int = config.CHUNK_OVERLAP,
            separator: str = " "
    ) -> List[Dict[str, any]]:
        """
        Split text into overlapping chunks for better retrieval.

        Args:
            text: Input text to chunk
            chunk_size: Number of tokens per chunk
            overlap: Number of overlapping tokens between chunks
            separator: Token separator (default: space)

        Returns:
            List of dictionaries with chunk text and metadata
        """
        if not text or not text.strip():
            return []

        # Simple token splitting (can be enhanced with proper tokenizer)
        tokens = text.split(separator)

        if len(tokens) <= chunk_size:
            return [{
                "text": text,
                "chunk_id": 0,
                "start_token": 0,
                "end_token": len(tokens)
            }]

        chunks = []
        chunk_id = 0

        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]

            if not chunk_tokens:
                break

            chunk_text = separator.join(chunk_tokens)

            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "start_token": i,
                "end_token": min(i + chunk_size, len(tokens)),
                "token_count": len(chunk_tokens)
            })

            chunk_id += 1

            # Stop if we've covered all tokens
            if i + chunk_size >= len(tokens):
                break

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def embed_chunks(
            self,
            chunks: List[Dict[str, any]],
            add_embeddings: bool = True
    ) -> List[Dict[str, any]]:
        """
        Generate embeddings for text chunks.

        Args:
            chunks: List of chunk dictionaries from chunk_text()
            add_embeddings: Add embeddings to chunk dictionaries

        Returns:
            Updated chunks with embeddings
        """
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_batch(texts)

        if add_embeddings:
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i]

        return chunks

    def compute_similarity(
            self,
            embedding1: np.ndarray,
            embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (-1 to 1)
        """
        return np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )


# Example usage
if __name__ == "__main__":
    # Initialize embedder
    embedder = TextEmbedder()

    # Example text
    sample_text = """
    Artificial intelligence is transforming how we interact with technology.
    Machine learning models can now understand multiple languages and contexts.
    This enables powerful applications like multilingual chatbots and semantic search.
    """

    # Chunk text
    chunks = embedder.chunk_text(sample_text, chunk_size=20, overlap=5)
    print(f"Created {len(chunks)} chunks")

    # Embed chunks
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    print(f"Embedding shape: {chunks_with_embeddings[0]['embedding'].shape}")

    # Single text embedding
    query = "What is AI?"
    query_embedding = embedder.embed_text(query)
    print(f"Query embedding shape: {query_embedding.shape}")

    print("Embedding module loaded successfully")
