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
        parameters_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or query to summarize documents about"
                },
                "style": {
                    "type": "string",
                    "description": "Summarization style",
                    "enum": ["brief", "detailed", "bullet_points"],
                    "default": "standard"
                },
                "max_documents": {
                    "type": "integer",
                    "description": "Maximum number of documents to summarize",
                    "default": 5
                }
            },
            "required": ["query"]
        }
        
        super().__init__(
            name="summarization",
            description="Summarize multiple documents on a topic. Use this when user asks to summarize, get overview, or synthesize information from multiple sources.",
            parameters_schema=parameters_schema
        )
        self.vector_db = vector_db
        self.llm = llm

    def execute(self, query: str, context: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Execute summarization.

        Args:
            query: Summarization request
            context: Optional context with document IDs or filters

        Returns:
            Summarization result
        """
        logger.info(f"SummarizationTool executing: {query}")

        # Extract parameters
        style = kwargs.get("style") or self._extract_style(query)
        max_documents = kwargs.get("max_documents", 5)

        # Get documents to summarize
        documents = self._get_documents(query, context, max_documents)

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

    def _get_documents(self, query: str, context: Optional[Dict], max_docs: int = 5) -> List[Dict]:
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
                n_results=max_docs,
                where={"topic": topic}
            )
        else:
            results = self.vector_db.query(
                query_text=query,
                n_results=max_docs
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
        if not documents:
            return "No documents available to summarize."
        
        # Combine document texts
        combined_text = "\n\n".join([doc["text"] for doc in documents])
        
        # Truncate if too long (keep within LLM context window)
        max_context_length = 3000
        if len(combined_text) > max_context_length:
            combined_text = combined_text[:max_context_length] + "..."

        # Build summarization prompt based on style
        if style == "brief":
            prompt = (
                "Provide a brief 2-3 sentence summary of the key points from the following documents:\n\n"
                f"{combined_text}\n\n"
                "Summary:"
            )
        elif style == "detailed":
            prompt = (
                "Provide a comprehensive summary covering all main points from the following documents:\n\n"
                f"{combined_text}\n\n"
                "Detailed Summary:"
            )
        elif style == "bullet_points":
            prompt = (
                "Summarize the following documents as a list of key bullet points:\n\n"
                f"{combined_text}\n\n"
                "Key Points:\n-"
            )
        else:
            prompt = (
                "Summarize the main points from the following documents:\n\n"
                f"{combined_text}\n\n"
                "Summary:"
            )

        # Generate summary using LLM
        try:
            # Use LLM's generate_response method
            # Pass empty context since we're providing the full text in the prompt
            result = self.llm.generate_response(
                query=prompt,
                context_chunks=[]  # Context is already in the prompt
            )
            
            summary = result.get("response", "")
            
            # Clean up the response
            if style == "bullet_points" and not summary.startswith("-"):
                summary = "- " + summary
            
            logger.info(f"Generated {style} summary ({len(summary)} chars)")
            return summary
            
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            # Fallback: return first document excerpt
            return f"Summary generation failed. First document excerpt: {documents[0]['text'][:500]}..."
