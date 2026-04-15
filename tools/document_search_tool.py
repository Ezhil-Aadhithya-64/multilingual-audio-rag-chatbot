"""
Document Search Tool - Structured document search with metadata filtering
"""

import logging
from typing import Dict, Optional, List
from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class DocumentSearchTool(BaseTool):
    """
    Structured document search with advanced filtering.
    Supports metadata-based filtering beyond semantic search.
    """

    def __init__(self, vector_db, embedder):
        """
        Initialize document search tool.

        Args:
            vector_db: Vector database instance
            embedder: Text embedder
        """
        super().__init__(
            name="document_search",
            description="Search documents with metadata filtering (language, source, topic, type)"
        )
        self.vector_db = vector_db
        self.embedder = embedder

    def execute(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Execute structured document search.

        Args:
            query: Search query
            context: Optional context with filters

        Returns:
            Search results with metadata
        """
        logger.info(f"DocumentSearchTool executing: {query}")

        # Extract filters from query
        filters = self._extract_filters(query, context)

        # Embed query
        query_embedding = self.embedder.embed_text(query)

        # Build search parameters
        search_params = {
            "query_embedding": query_embedding,
            "n_results": 10
        }

        if filters:
            search_params["where"] = filters

        # Execute search
        results = self.vector_db.query(**search_params)

        # Format results
        formatted_results = self._format_results(results)

        return {
            "query": query,
            "filters_applied": filters,
            "num_results": len(formatted_results),
            "results": formatted_results,
            "summary": self._generate_summary(formatted_results)
        }

    def _extract_filters(self, query: str, context: Optional[Dict]) -> Dict:
        """
        Extract metadata filters from query.

        Args:
            query: Search query
            context: Optional context

        Returns:
            Filter dictionary
        """
        filters = {}
        query_lower = query.lower()

        # Language filter
        languages = ["en", "es", "fr", "de", "it", "pt"]
        for lang in languages:
            if lang in query_lower or f"{lang} language" in query_lower:
                filters["language"] = lang
                break

        # Topic filter
        topics = ["password", "audio", "account", "support", "troubleshooting"]
        for topic in topics:
            if topic in query_lower:
                filters["topic"] = topic
                break

        # Type filter
        if "policy" in query_lower or "policies" in query_lower:
            filters["type"] = "policy"
        elif "procedure" in query_lower:
            filters["type"] = "procedure"

        return filters

    def _format_results(self, results: Dict) -> List[Dict]:
        """Format search results."""
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])

        formatted = []
        for i, doc in enumerate(documents):
            formatted.append({
                "document": doc,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "relevance_score": 1 - distances[i] if i < len(distances) else 0.0
            })

        return formatted

    def _generate_summary(self, results: List[Dict]) -> str:
        """Generate summary of search results."""
        if not results:
            return "No documents found matching the criteria."

        # Group by metadata
        by_topic = {}
        for result in results:
            topic = result["metadata"].get("topic", "unknown")
            if topic not in by_topic:
                by_topic[topic] = 0
            by_topic[topic] += 1

        summary_parts = [f"Found {len(results)} documents"]
        if by_topic:
            topic_summary = ", ".join([f"{count} on {topic}" for topic, count in by_topic.items()])
            summary_parts.append(f"Topics: {topic_summary}")

        return ". ".join(summary_parts) + "."
