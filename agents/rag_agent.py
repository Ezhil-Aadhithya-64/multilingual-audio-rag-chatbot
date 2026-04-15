"""
RAG Agent - Document retrieval and grounded response generation
Handles queries requiring knowledge base retrieval
"""

import logging
from typing import Dict, Optional, List
import time

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """
    Retrieval-Augmented Generation agent.
    Performs semantic search, reranking, and grounded response generation.
    """

    def __init__(
        self,
        embedder,
        vector_db,
        reranker,
        llm,
        top_k: int = 10,
        rerank_top_n: int = 5
    ):
        """
        Initialize RAG agent.

        Args:
            embedder: Text embedding model
            vector_db: Vector database instance
            reranker: Document reranker
            llm: Language model for generation
            top_k: Number of documents to retrieve
            rerank_top_n: Number of documents after reranking
        """
        super().__init__("RAGAgent")
        self.embedder = embedder
        self.vector_db = vector_db
        self.reranker = reranker
        self.llm = llm
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n

    def execute(
        self,
        query: str,
        context: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Execute RAG pipeline.

        Args:
            query: User query
            context: Optional session context
            **kwargs: Additional parameters (language, use_cache, etc.)

        Returns:
            Execution result with response and metadata
        """
        start_time = time.time()
        logger.info(f"RAGAgent executing query: {query[:100]}...")

        try:
            # Check if we can use cached context
            if context and kwargs.get("use_cache", False):
                cached_docs = context.get("last_retrieved_docs")
                if cached_docs and self._is_related_query(query, context):
                    logger.info("Using cached documents")
                    retrieved_docs = cached_docs
                    retrieval_latency = 0
                else:
                    retrieved_docs, retrieval_latency = self._retrieve_documents(
                        query, kwargs.get("language")
                    )
            else:
                retrieved_docs, retrieval_latency = self._retrieve_documents(
                    query, kwargs.get("language")
                )

            # Rerank documents
            reranked_docs, rerank_latency = self._rerank_documents(
                query, retrieved_docs
            )

            # Generate response
            response, llm_latency, llm_confidence = self._generate_response(
                query, reranked_docs
            )

            # Calculate total latency
            total_latency = (time.time() - start_time) * 1000

            # Calculate confidence
            confidence = self._calculate_confidence(
                llm_confidence,
                len(reranked_docs),
                reranked_docs
            )

            # Record execution
            self._record_execution(True, total_latency)

            # Build result
            return self._build_result(
                success=True,
                response=response,
                confidence=confidence,
                metadata={
                    "retrieved_documents": reranked_docs,
                    "num_documents": len(reranked_docs),
                    "latency": {
                        "total_ms": total_latency,
                        "retrieval_ms": retrieval_latency,
                        "rerank_ms": rerank_latency,
                        "llm_ms": llm_latency
                    },
                    "retrieval_scores": self._get_retrieval_scores(reranked_docs)
                }
            )

        except Exception as e:
            logger.error(f"RAGAgent execution failed: {e}")
            total_latency = (time.time() - start_time) * 1000
            self._record_execution(False, total_latency)

            return self._build_result(
                success=False,
                response=f"Failed to process query: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def _retrieve_documents(
        self,
        query: str,
        language: Optional[str] = None
    ) -> tuple[List[str], float]:
        """
        Retrieve documents from vector database.

        Args:
            query: User query
            language: Optional language filter

        Returns:
            Tuple of (documents, latency_ms)
        """
        start_time = time.time()

        # Embed query
        query_embedding = self.embedder.embed_text(query)

        # Build retrieval parameters
        retrieval_params = {
            "query_embedding": query_embedding,
            "n_results": self.top_k
        }

        if language:
            retrieval_params["where"] = {"language": language}

        # Query vector database
        results = self.vector_db.query(**retrieval_params)

        documents = results["documents"]
        latency = (time.time() - start_time) * 1000

        logger.info(f"Retrieved {len(documents)} documents in {latency:.2f}ms")

        return documents, latency

    def _rerank_documents(
        self,
        query: str,
        documents: List[str]
    ) -> tuple[List[str], float]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Tuple of (reranked_documents, latency_ms)
        """
        if not documents or not self.reranker:
            return documents, 0.0

        start_time = time.time()

        # Rerank
        reranked = self.reranker.rerank(
            query,
            documents,
            return_scores=True
        )

        # Extract top N documents
        reranked_docs = [r["document"] for r in reranked[:self.rerank_top_n]]

        latency = (time.time() - start_time) * 1000

        logger.info(f"Reranked to top {len(reranked_docs)} in {latency:.2f}ms")

        return reranked_docs, latency

    def _generate_response(
        self,
        query: str,
        documents: List[str]
    ) -> tuple[str, float, float]:
        """
        Generate grounded response using LLM.

        Args:
            query: User query
            documents: Context documents

        Returns:
            Tuple of (response, latency_ms, confidence)
        """
        if not documents:
            return (
                "I couldn't find relevant information to answer your question.",
                0.0,
                0.0
            )

        start_time = time.time()

        # Generate response
        llm_result = self.llm.generate_response(query, documents)

        response = llm_result["response"]
        latency = (time.time() - start_time) * 1000

        # Extract confidence (if available)
        confidence = llm_result.get("confidence", 0.8)

        logger.info(f"Generated response in {latency:.2f}ms")

        return response, latency, confidence

    def _calculate_confidence(
        self,
        llm_confidence: float,
        num_docs: int,
        documents: List[str]
    ) -> float:
        """
        Calculate overall confidence score.

        Args:
            llm_confidence: LLM generation confidence
            num_docs: Number of retrieved documents
            documents: Retrieved documents

        Returns:
            Overall confidence score (0-1)
        """
        # Base confidence from LLM
        confidence = llm_confidence

        # Adjust based on number of documents
        if num_docs == 0:
            confidence *= 0.1
        elif num_docs < 3:
            confidence *= 0.8

        # Adjust based on document quality (length as proxy)
        if documents:
            avg_length = sum(len(doc) for doc in documents) / len(documents)
            if avg_length < 50:
                confidence *= 0.9

        return min(confidence, 1.0)

    def _get_retrieval_scores(self, documents: List[str]) -> List[float]:
        """Get retrieval quality scores."""
        # Placeholder - would use actual similarity scores
        return [0.9 - (i * 0.1) for i in range(len(documents))]

    def _is_related_query(self, query: str, context: Dict) -> bool:
        """
        Check if query is related to previous context.

        Args:
            query: Current query
            context: Session context

        Returns:
            True if related
        """
        last_topic = context.get("last_topic")
        if not last_topic:
            return False

        # Simple keyword matching
        return last_topic.replace("_", " ") in query.lower()


# Example usage
if __name__ == "__main__":
    # This would be initialized with actual components
    print("RAGAgent module loaded")
