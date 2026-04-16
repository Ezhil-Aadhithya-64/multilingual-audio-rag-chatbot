"""
Base Tool - Abstract base class for all tools
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, name: str, description: str, parameters_schema: Optional[Dict] = None):
        """
        Initialize base tool.

        Args:
            name: Tool name
            description: Tool description
            parameters_schema: JSON schema for tool parameters
        """
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema or {}

    @abstractmethod
    def execute(self, query: str, context: Optional[Dict] = None, **kwargs) -> Any:
        """
        Execute tool logic.

        Args:
            query: User query or tool input
            context: Optional execution context
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result
        """
        pass

    def get_schema(self) -> Dict:
        """
        Get tool schema for LLM function calling.
        
        Returns:
            Tool schema in OpenAI function calling format
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema
        }

    def get_info(self) -> Dict:
        """Get tool information."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema
        }
