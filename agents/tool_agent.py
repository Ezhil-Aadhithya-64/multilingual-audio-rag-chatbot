"""
Tool Agent - Executes structured operations using available tools
Handles queries requiring tool execution (search, summarization, etc.)
"""

import logging
from typing import Dict, Optional, List
import time

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ToolAgent(BaseAgent):
    """
    Tool execution agent.
    Routes queries to appropriate tools and orchestrates execution.
    """

    def __init__(self, tool_registry: Dict):
        """
        Initialize tool agent.

        Args:
            tool_registry: Dictionary of available tools
        """
        super().__init__("ToolAgent")
        self.tool_registry = tool_registry
        self.tool_execution_log = []

    def execute(
        self,
        query: str,
        context: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Execute tool-based query.

        Args:
            query: User query
            context: Optional session context
            **kwargs: Additional parameters (suggested_tools, etc.)

        Returns:
            Execution result with tool outputs
        """
        start_time = time.time()
        logger.info(f"ToolAgent executing query: {query[:100]}...")

        try:
            # Determine which tools to use
            tools_to_execute = self._select_tools(query, kwargs.get("suggested_tools"))

            if not tools_to_execute:
                return self._build_result(
                    success=False,
                    response="No appropriate tools found for this query.",
                    confidence=0.0,
                    metadata={"error": "no_tools_selected"}
                )

            # Execute tools
            tool_results = []
            for tool_name in tools_to_execute:
                tool_result = self._execute_tool(tool_name, query, context)
                tool_results.append(tool_result)

            # Synthesize results
            response, confidence = self._synthesize_results(query, tool_results)

            # Calculate latency
            total_latency = (time.time() - start_time) * 1000

            # Record execution
            self._record_execution(True, total_latency)

            # Log tool execution
            self._log_tool_execution(query, tools_to_execute, tool_results)

            return self._build_result(
                success=True,
                response=response,
                confidence=confidence,
                metadata={
                    "tools_used": tools_to_execute,
                    "tool_results": tool_results,
                    "latency_ms": total_latency
                }
            )

        except Exception as e:
            logger.error(f"ToolAgent execution failed: {e}")
            total_latency = (time.time() - start_time) * 1000
            self._record_execution(False, total_latency)

            return self._build_result(
                success=False,
                response=f"Tool execution failed: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def _select_tools(
        self,
        query: str,
        suggested_tools: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select appropriate tools for query.

        Args:
            query: User query
            suggested_tools: Optional pre-suggested tools from router

        Returns:
            List of tool names to execute
        """
        if suggested_tools:
            # Validate suggested tools exist
            valid_tools = [
                tool for tool in suggested_tools
                if tool in self.tool_registry
            ]
            if valid_tools:
                return valid_tools

        # Fallback to keyword-based selection
        query_lower = query.lower()
        selected_tools = []

        if any(word in query_lower for word in ["summarize", "summary"]):
            if "summarization" in self.tool_registry:
                selected_tools.append("summarization")

        if any(word in query_lower for word in ["search", "find", "filter"]):
            if "document_search" in self.tool_registry:
                selected_tools.append("document_search")

        if any(word in query_lower for word in ["count", "how many", "statistics"]):
            if "database_query" in self.tool_registry:
                selected_tools.append("database_query")

        return selected_tools

    def _execute_tool(
        self,
        tool_name: str,
        query: str,
        context: Optional[Dict]
    ) -> Dict:
        """
        Execute a single tool.

        Args:
            tool_name: Name of tool to execute
            query: User query
            context: Session context

        Returns:
            Tool execution result
        """
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return {
                "tool": tool_name,
                "success": False,
                "error": "Tool not found"
            }

        try:
            start_time = time.time()

            # Execute tool
            result = tool.execute(query, context)

            latency = (time.time() - start_time) * 1000

            return {
                "tool": tool_name,
                "success": True,
                "result": result,
                "latency_ms": latency
            }

        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            return {
                "tool": tool_name,
                "success": False,
                "error": str(e)
            }

    def _synthesize_results(
        self,
        query: str,
        tool_results: List[Dict]
    ) -> tuple[str, float]:
        """
        Synthesize tool results into coherent response.

        Args:
            query: Original query
            tool_results: List of tool execution results

        Returns:
            Tuple of (response, confidence)
        """
        successful_results = [r for r in tool_results if r["success"]]

        if not successful_results:
            return "All tool executions failed.", 0.0

        # Build response from tool outputs
        response_parts = []
        for result in successful_results:
            tool_name = result["tool"]
            tool_output = result["result"]

            if isinstance(tool_output, dict):
                response_parts.append(f"**{tool_name}**: {tool_output.get('summary', str(tool_output))}")
            else:
                response_parts.append(f"**{tool_name}**: {tool_output}")

        response = "\n\n".join(response_parts)

        # Calculate confidence based on success rate
        confidence = len(successful_results) / len(tool_results)

        return response, confidence

    def _log_tool_execution(
        self,
        query: str,
        tools_used: List[str],
        tool_results: List[Dict]
    ) -> None:
        """
        Log tool execution for audit trail.

        Args:
            query: User query
            tools_used: List of tools executed
            tool_results: Tool execution results
        """
        log_entry = {
            "timestamp": time.time(),
            "query": query,
            "tools_used": tools_used,
            "success_count": sum(1 for r in tool_results if r["success"]),
            "total_count": len(tool_results)
        }

        self.tool_execution_log.append(log_entry)

        # Keep only last 100 entries
        if len(self.tool_execution_log) > 100:
            self.tool_execution_log = self.tool_execution_log[-100:]

    def get_tool_execution_log(self, limit: int = 10) -> List[Dict]:
        """
        Get recent tool execution log.

        Args:
            limit: Number of entries to return

        Returns:
            List of log entries
        """
        return self.tool_execution_log[-limit:]


# Example usage
if __name__ == "__main__":
    print("ToolAgent module loaded")
