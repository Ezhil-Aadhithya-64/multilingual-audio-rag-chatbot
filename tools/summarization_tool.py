"""
Summarization Tool - Multi-document summarization
"""

import logging
from typing import Dict, Optional, List
from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class SummarizationTool(BaseTool):
    """
    Multi-document summarization tool.
    Generates summaries from multiple documents.
    """

    def __init__(self, vector_db, llm):
        """
        Initialize summarization tool.

        Args:
            vector_db: Vector database instance
            llm: Language model for summarization
        """
        super().__init__(
            name="summarization",
            description="Summarize multiple documents (brief, detailed, or bullet points)"
        )
        self.vector_db = vector_db
        self.llm = llm

    def execute(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Execute summarization.

        Args:
            query: Summarization request
            context: Optional context with document IDs or filters

        Returns:
            Summarization result
        """
        logger.info(f"SummarizationTool executing: {query}")

        # Extract summarization style
        style = self._extract_style(query)

        # Get documents to summarize
        documents = self._get_documents(query, context)

        if not documents:
            return {
                "summary": "No documents found to summarize.",
                "style": style,
                "num_documents": 0
            }

        # Generate summary
        summary = self._generate_summary(documents, style)

        return {
            "summary": summary,
            "style": style,
            "num_documents": len(documents),
            "document_sources": [doc.get("metadata", {}).get("source", "unknown") for doc in documents]
        }

    def _extract_style(self, query: str) -> str:
        """Extract summarization style from query."""
        query_lower = query.lower()

        if "brief" in query_lower or "short" in query_lower:
            return "brief"
        elif "detailed" in query_lower or "comprehensive" in query_lower:
            return "detailed"
        elif "bullet" in query_lower or "points" in query_lower:
            return "bullet_points"
        else:
            return "standard"

    def _get_documents(self, query: str, context: Optional[Dict]) -> List[Dict]:
        """Get documents to summarize."""
        # Extract topic from query
        topic_keywords = ["password", "audio", "account", "support"]
        topic = None
        for keyword in topic_keywords:
            if keyword in query.lower():
                topic = keyword
                break

        # Query vector database
        if topic:
            results = self.vector_db.query(
                query_text=query,
                n_results=5,
                where={"topic": topic}
            )
        else:
            results = self.vector_db.query(
                query_text=query,
                n_results=5
            )

        # Format documents
        documents = []
        for i, doc in enumerate(results.get("documents", [])):
            documents.append({
                "text": doc,
                "metadata": results["metadatas"][i] if i < len(results.get("metadatas", [])) else {}
            })

        return documents

    def _generate_summary(self, documents: List[Dict], style: str) -> str:
        """Generate summary using LLM."""
        # Combine document texts
        combined_text = "\n\n".join([doc["text"] for doc in documents])

        # Build summarization prompt
        if style == "brief":
            prompt = f"Provide a brief 2-3 sentence summary of the following:\n\n{combined_text}"
        elif style == "detailed":
            prompt = f"Provide a comprehensive summary of the following:\n\n{combined_text}"
        elif style == "bullet_points":
            prompt = f"Summarize the following as bullet points:\n\n{combined_text}"
        else:
            prompt = f"Summarize the following:\n\n{combined_text}"

        # Generate summary (simplified - actual implementation would use LLM)
        try:
            # For now, return first 500 chars as placeholder
            summary = combined_text[:500] + "..."
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "Failed to generate summary."
