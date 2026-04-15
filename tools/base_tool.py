"""
Base Tool - Abstract base class for all tools
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, name: str, description: str):
        """
        Initialize base tool.

        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, query: str, context: Optional[Dict] = None) -> Any:
        """
        Execute tool logic.

        Args:
            query: User query or tool input
            context: Optional execution context

        Returns:
            Tool execution result
        """
        pass

    def get_info(self) -> Dict:
        """Get tool information."""
        return {
            "name": self.name,
            "description": self.description
        }
