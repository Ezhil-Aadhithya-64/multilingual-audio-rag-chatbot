"""
Reranker Module
Uses cross-encoder models to rerank retrieved chunks for higher relevance.
Improves retrieval quality by computing pairwise relevance scores.
"""

import logging
from typing import List, Dict, Optional  # ← ADD Optional HERE
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class DocumentReranker:
    """
    Document reranker using cross-encoder models.
    Provides more accurate relevance scoring than bi-encoder retrieval.
    """

    def __init__(
            self,
            model_name: str = config.RERANKER_MODEL,
            top_n: int = config.RERANK_TOP_N
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
            top_n: Number of top documents to return after reranking
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.top_n = top_n
        logger.info("Reranker model loaded successfully")

    def rerank(
            self,
            query: str,
            documents: List[str],
            metadatas: Optional[List[Dict]] = None,
            return_scores: bool = True
    ) -> List[Dict[str, any]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Query text
            documents: List of document texts to rerank
            metadatas: Optional metadata for each document
            return_scores: Include relevance scores in output

        Returns:
            List of reranked documents with scores and metadata
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided")
            return self._format_results(documents, None, metadatas)

        logger.info(f"Reranking {len(documents)} documents")

        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]

            # Compute relevance scores
            scores = self.model.predict(pairs)

            # Sort by scores (descending)
            ranked_indices = np.argsort(scores)[::-1]

            # Select top N
            top_indices = ranked_indices[:self.top_n]

            # Format results
            results = []
            for idx in top_indices:
                result = {
                    "document": documents[idx],
                    "rank": len(results) + 1,
                    "original_rank": int(idx) + 1
                }

                if return_scores:
                    result["score"] = float(scores[idx])

                if metadatas and idx < len(metadatas):
                    result["metadata"] = metadatas[idx]

                results.append(result)

            logger.info(f"Reranked to top {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Return original order on failure
            return self._format_results(documents[:self.top_n], None, metadatas)

    def _format_results(
            self,
            documents: List[str],
            scores: Optional[np.ndarray],
            metadatas: Optional[List[Dict]]
    ) -> List[Dict[str, any]]:
        """Format reranking results into structured output."""
        results = []
        for i, doc in enumerate(documents):
            result = {
                "document": doc,
                "rank": i + 1,
                "original_rank": i + 1
            }

            if scores is not None:
                result["score"] = float(scores[i])

            if metadatas and i < len(metadatas):
                result["metadata"] = metadatas[i]

            results.append(result)

        return results

    def compute_pairwise_scores(
            self,
            query: str,
            documents: List[str]
    ) -> np.ndarray:
        """
        Compute relevance scores for all query-document pairs.

        Args:
            query: Query text
            documents: List of documents

        Returns:
            Numpy array of relevance scores
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        return scores

    def filter_by_threshold(
            self,
            ranked_results: List[Dict[str, any]],
            threshold: float = 0.5
    ) -> List[Dict[str, any]]:
        """
        Filter reranked results by minimum relevance score.

        Args:
            ranked_results: Results from rerank()
            threshold: Minimum score threshold

        Returns:
            Filtered results
        """
        if not ranked_results or "score" not in ranked_results[0]:
            return ranked_results

        filtered = [r for r in ranked_results if r.get("score", 0) >= threshold]
        logger.info(f"Filtered to {len(filtered)}/{len(ranked_results)} results above threshold {threshold}")

        return filtered


# Example usage
if __name__ == "__main__":
    # Initialize reranker
    reranker = DocumentReranker()

    # Example query and documents
    query = "What is machine learning?"

    documents = [
        "Machine learning is a branch of artificial intelligence.",
        "Python is a popular programming language.",
        "Deep learning uses neural networks with multiple layers.",
        "Databases store and manage data efficiently.",
        "Supervised learning requires labeled training data."
    ]

    # Rerank documents
    results = reranker.rerank(query, documents)

    print("Reranked Results:")
    for result in results:
        print(f"Rank {result['rank']}: {result['document'][:60]}... (score: {result.get('score', 'N/A')})")

    print("\nReranker module loaded successfully")
