"""
Metrics System - Comprehensive evaluation and monitoring
Tracks retrieval quality, response quality, and system performance
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and analyzes system metrics.
    Tracks retrieval, response, and performance metrics.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize metrics collector.

        Args:
            storage_path: Optional path for persistent storage
        """
        self.storage_path = storage_path
        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)

        self.metrics_log = []

    def record_query_metrics(
        self,
        query_id: str,
        query: str,
        intent: str,
        response: str,
        confidence: float,
        retrieved_docs: List[str],
        latency_breakdown: Dict[str, float],
        agent_name: str,
        success: bool
    ) -> Dict:
        """
        Record comprehensive metrics for a query.

        Args:
            query_id: Unique query identifier
            query: User query
            intent: Classified intent
            response: System response
            confidence: Response confidence
            retrieved_docs: Retrieved documents
            latency_breakdown: Latency by stage
            agent_name: Agent that handled query
            success: Whether query succeeded

        Returns:
            Metrics dictionary
        """
        metrics = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "intent": intent,
            "agent": agent_name,
            "success": success,
            "retrieval_metrics": self._calculate_retrieval_metrics(retrieved_docs),
            "response_metrics": self._calculate_response_metrics(
                response, retrieved_docs, confidence
            ),
            "performance_metrics": self._calculate_performance_metrics(latency_breakdown),
            "confidence": confidence
        }

        # Store metrics
        self.metrics_log.append(metrics)

        # Persist if storage enabled
        if self.storage_path:
            self._save_metrics(query_id, metrics)

        logger.info(f"Recorded metrics for query {query_id}")

        return metrics

    def _calculate_retrieval_metrics(self, retrieved_docs: List[str]) -> Dict:
        """
        Calculate retrieval quality metrics.

        Args:
            retrieved_docs: Retrieved documents

        Returns:
            Retrieval metrics dictionary
        """
        if not retrieved_docs:
            return {
                "num_documents": 0,
                "avg_document_length": 0,
                "coverage_score": 0.0
            }

        return {
            "num_documents": len(retrieved_docs),
            "avg_document_length": sum(len(doc) for doc in retrieved_docs) / len(retrieved_docs),
            "min_document_length": min(len(doc) for doc in retrieved_docs),
            "max_document_length": max(len(doc) for doc in retrieved_docs),
            "coverage_score": self._calculate_coverage(retrieved_docs)
        }

    def _calculate_response_metrics(
        self,
        response: str,
        retrieved_docs: List[str],
        confidence: float
    ) -> Dict:
        """
        Calculate response quality metrics.

        Args:
            response: Generated response
            retrieved_docs: Retrieved documents
            confidence: Response confidence

        Returns:
            Response metrics dictionary
        """
        grounding_score = self._calculate_grounding_score(response, retrieved_docs)

        return {
            "response_length": len(response),
            "word_count": len(response.split()),
            "grounding_score": grounding_score,
            "confidence": confidence,
            "quality_score": (grounding_score + confidence) / 2
        }

    def _calculate_performance_metrics(self, latency_breakdown: Dict[str, float]) -> Dict:
        """
        Calculate performance metrics.

        Args:
            latency_breakdown: Latency by stage

        Returns:
            Performance metrics dictionary
        """
        total_latency = sum(latency_breakdown.values())

        return {
            "total_latency_ms": total_latency,
            "latency_breakdown": latency_breakdown,
            "slowest_stage": max(latency_breakdown, key=latency_breakdown.get) if latency_breakdown else None
        }

    def _calculate_coverage(self, documents: List[str]) -> float:
        """
        Calculate document coverage score.

        Args:
            documents: Retrieved documents

        Returns:
            Coverage score (0-1)
        """
        if not documents:
            return 0.0

        # Simple heuristic: more documents with good length = better coverage
        avg_length = sum(len(doc) for doc in documents) / len(documents)
        length_score = min(avg_length / 500, 1.0)  # Normalize to 500 chars
        count_score = min(len(documents) / 5, 1.0)  # Normalize to 5 docs

        return (length_score + count_score) / 2

    def _calculate_grounding_score(self, response: str, context: List[str]) -> float:
        """
        Calculate how well response is grounded in context.

        Args:
            response: Generated response
            context: Retrieved context

        Returns:
            Grounding score (0-1)
        """
        if not context:
            return 0.0

        response_tokens = set(response.lower().split())
        context_tokens = set()
        for doc in context:
            context_tokens.update(doc.lower().split())

        if len(response_tokens) == 0:
            return 0.0

        overlap = len(response_tokens.intersection(context_tokens))
        score = overlap / len(response_tokens)

        return min(score, 1.0)

    def get_aggregate_metrics(self, time_window: Optional[int] = None) -> Dict:
        """
        Get aggregate metrics across queries.

        Args:
            time_window: Optional time window in seconds

        Returns:
            Aggregate metrics dictionary
        """
        if not self.metrics_log:
            return {}

        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now().timestamp() - time_window
            filtered_metrics = [
                m for m in self.metrics_log
                if datetime.fromisoformat(m["timestamp"]).timestamp() > cutoff_time
            ]
        else:
            filtered_metrics = self.metrics_log

        if not filtered_metrics:
            return {}

        # Calculate aggregates
        total_queries = len(filtered_metrics)
        successful_queries = sum(1 for m in filtered_metrics if m["success"])

        avg_confidence = sum(m["confidence"] for m in filtered_metrics) / total_queries
        avg_latency = sum(
            m["performance_metrics"]["total_latency_ms"]
            for m in filtered_metrics
        ) / total_queries

        avg_grounding = sum(
            m["response_metrics"]["grounding_score"]
            for m in filtered_metrics
        ) / total_queries

        # Intent distribution
        intent_counts = {}
        for m in filtered_metrics:
            intent = m["intent"]
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": successful_queries / total_queries,
            "avg_confidence": avg_confidence,
            "avg_latency_ms": avg_latency,
            "avg_grounding_score": avg_grounding,
            "intent_distribution": intent_counts,
            "time_window_seconds": time_window
        }

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate evaluation report.

        Args:
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        aggregate = self.get_aggregate_metrics()

        report_lines = [
            "=" * 60,
            "VOICEASSIST PRO - EVALUATION REPORT",
            "=" * 60,
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Total Queries: {aggregate.get('total_queries', 0)}",
            "",
            "PERFORMANCE METRICS",
            "-" * 60,
            f"Success Rate: {aggregate.get('success_rate', 0):.2%}",
            f"Average Confidence: {aggregate.get('avg_confidence', 0):.2f}",
            f"Average Latency: {aggregate.get('avg_latency_ms', 0):.2f}ms",
            f"Average Grounding Score: {aggregate.get('avg_grounding_score', 0):.2f}",
            "",
            "INTENT DISTRIBUTION",
            "-" * 60
        ]

        for intent, count in aggregate.get("intent_distribution", {}).items():
            percentage = (count / aggregate.get('total_queries', 1)) * 100
            report_lines.append(f"{intent}: {count} ({percentage:.1f}%)")

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    def _save_metrics(self, query_id: str, metrics: Dict) -> None:
        """Save metrics to persistent storage."""
        if not self.storage_path:
            return

        metrics_file = self.storage_path / f"{query_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    collector = MetricsCollector(storage_path=Path("data/metrics"))

    # Record sample metrics
    collector.record_query_metrics(
        query_id="test-001",
        query="How do I reset my password?",
        intent="rag_query",
        response="To reset your password, click the Forgot Password link.",
        confidence=0.92,
        retrieved_docs=["Password reset procedure..."],
        latency_breakdown={
            "retrieval_ms": 150,
            "rerank_ms": 100,
            "llm_ms": 700
        },
        agent_name="RAGAgent",
        success=True
    )

    # Generate report
    report = collector.generate_report()
    print(report)
