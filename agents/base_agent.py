"""
Base Agent - Abstract base class for all agents
Defines common interface and shared functionality
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Provides common interface and shared functionality.
    """

    def __init__(self, name: str):
        """
        Initialize base agent.

        Args:
            name: Agent name/identifier
        """
        self.name = name
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_latency_ms = 0

    @abstractmethod
    def execute(
        self,
        query: str,
        context: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Execute agent logic.

        Args:
            query: User query
            context: Optional execution context
            **kwargs: Additional parameters

        Returns:
            Execution result dictionary
        """
        pass

    def _record_execution(
        self,
        success: bool,
        latency_ms: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Record execution metrics.

        Args:
            success: Whether execution succeeded
            latency_ms: Execution latency in milliseconds
            metadata: Optional execution metadata
        """
        self.execution_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.total_latency_ms += latency_ms

        logger.info(
            f"{self.name} execution: success={success}, "
            f"latency={latency_ms:.2f}ms"
        )

    def get_stats(self) -> Dict:
        """
        Get agent execution statistics.

        Returns:
            Statistics dictionary
        """
        avg_latency = (
            self.total_latency_ms / self.execution_count
            if self.execution_count > 0
            else 0
        )

        return {
            "agent_name": self.name,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": (
                self.success_count / self.execution_count
                if self.execution_count > 0
                else 0
            ),
            "average_latency_ms": avg_latency
        }

    def reset_stats(self) -> None:
        """Reset agent statistics."""
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_latency_ms = 0

    def _build_result(
        self,
        success: bool,
        response: str,
        confidence: float,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Build standardized result dictionary.

        Args:
            success: Whether execution succeeded
            response: Response text
            confidence: Response confidence score
            metadata: Optional metadata

        Returns:
            Result dictionary
        """
        return {
            "success": success,
            "response": response,
            "confidence": confidence,
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
