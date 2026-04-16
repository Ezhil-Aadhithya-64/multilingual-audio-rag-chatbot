"""
Tool Agent - Executes structured operations using available tools
Handles queries requiring tool execution (search, summarization, etc.)
Uses LLM-based tool selection for intelligent decision-making
"""

import logging
from typing import Dict, Optional, List
import time
import json
import re

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ToolAgent(BaseAgent):
    """
    Tool execution agent with LLM-based tool selection.
    Routes queries to appropriate tools and orchestrates execution.
    """

    def __init__(self, tool_registry: Dict, llm=None):
        """
        Initialize tool agent.

        Args:
            tool_registry: Dictionary of available tools
            llm: Language model for tool selection (optional)
        """
        super().__init__("ToolAgent")
        self.tool_registry = tool_registry
        self.llm = llm
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
            tool_selections = self._select_tools(query, kwargs.get("suggested_tools"))

            if not tool_selections:
                return self._build_result(
                    success=False,
                    response="No appropriate tools found for this query.",
                    confidence=0.0,
                    metadata={"error": "no_tools_selected"}
                )

            # Execute tools (with chaining support)
            tool_results = []
            chained_context = {}  # Context passed between tools
            
            for i, selection in enumerate(tool_selections):
                tool_name = selection["tool"]
                parameters = selection.get("parameters", {})
                
                # Pass results from previous tools as context
                if i > 0:
                    parameters["previous_results"] = tool_results
                    parameters["chained_context"] = chained_context
                
                tool_result = self._execute_tool(tool_name, query, context, **parameters)
                tool_results.append(tool_result)
                
                # Update chained context for next tool
                if tool_result["success"]:
                    result_data = tool_result.get("result", {})
                    if isinstance(result_data, dict):
                        chained_context[tool_name] = result_data
                    
                    # Check if this tool's output should feed into next tool
                    if self._should_chain_to_next(tool_name, tool_selections, i):
                        logger.info(f"Chaining {tool_name} output to next tool")

            # Synthesize results
            response, confidence = self._synthesize_results(query, tool_results, chained=len(tool_selections) > 1)

            # Calculate latency
            total_latency = (time.time() - start_time) * 1000

            # Record execution
            self._record_execution(True, total_latency)

            # Log tool execution
            tool_names = [s["tool"] for s in tool_selections]
            self._log_tool_execution(query, tool_names, tool_results)

            return self._build_result(
                success=True,
                response=response,
                confidence=confidence,
                metadata={
                    "tools_used": tool_names,
                    "tool_selections": tool_selections,
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
    ) -> List[Dict]:
        """
        Select appropriate tools for query using LLM.

        Args:
            query: User query
            suggested_tools: Optional pre-suggested tools from router

        Returns:
            List of tool selections with parameters
        """
        # Try LLM-based selection first
        if self.llm:
            try:
                tool_selections = self._llm_based_tool_selection(query)
                if tool_selections:
                    logger.info(f"LLM selected {len(tool_selections)} tools")
                    return tool_selections
            except Exception as e:
                logger.warning(f"LLM tool selection failed: {e}, falling back to keyword-based")
        
        # Fallback to keyword-based selection
        return self._keyword_based_tool_selection(query, suggested_tools)

    def _llm_based_tool_selection(self, query: str) -> List[Dict]:
        """
        Use LLM to select tools and extract parameters.
        
        Args:
            query: User query
            
        Returns:
            List of tool selections with parameters
        """
        # Build tool descriptions for LLM
        tool_descriptions = []
        for tool_name, tool in self.tool_registry.items():
            schema = tool.get_schema()
            tool_descriptions.append(
                f"- {schema['name']}: {schema['description']}\n"
                f"  Parameters: {json.dumps(schema['parameters'], indent=2)}"
            )
        
        tools_text = "\n".join(tool_descriptions)
        
        # Build prompt for tool selection
        prompt = f"""You are a tool selection assistant. Given a user query, decide which tools to use and what parameters to pass.

Available tools:
{tools_text}

User query: "{query}"

Instructions:
1. Analyze the query to determine which tool(s) are needed
2. Extract parameters from the query
3. Respond with a JSON array of tool calls

Response format:
[
  {{
    "tool": "tool_name",
    "parameters": {{
      "param1": "value1",
      "param2": "value2"
    }}
  }}
]

If no tools are needed, respond with an empty array: []

Tool selection:"""

        try:
            # Call LLM
            result = self.llm.generate_response(
                query=prompt,
                context_chunks=[]
            )
            
            response_text = result.get("response", "")
            
            # Extract JSON from response
            tool_selections = self._parse_tool_selection(response_text)
            
            # Validate tool selections
            valid_selections = []
            for selection in tool_selections:
                tool_name = selection.get("tool")
                if tool_name in self.tool_registry:
                    valid_selections.append(selection)
                else:
                    logger.warning(f"LLM selected invalid tool: {tool_name}")
            
            return valid_selections
            
        except Exception as e:
            logger.error(f"LLM tool selection failed: {e}")
            return []
    
    def _parse_tool_selection(self, response_text: str) -> List[Dict]:
        """
        Parse LLM response to extract tool selections.
        
        Args:
            response_text: LLM response
            
        Returns:
            List of tool selections
        """
        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            try:
                tool_selections = json.loads(json_match.group(0))
                if isinstance(tool_selections, list):
                    return tool_selections
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
        
        return []
    
    def _keyword_based_tool_selection(
        self,
        query: str,
        suggested_tools: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Fallback keyword-based tool selection.
        
        Args:
            query: User query
            suggested_tools: Optional pre-suggested tools
            
        Returns:
            List of tool selections
        """
        if suggested_tools:
            # Validate suggested tools exist
            valid_tools = [
                {"tool": tool, "parameters": {"query": query}}
                for tool in suggested_tools
                if tool in self.tool_registry
            ]
            if valid_tools:
                return valid_tools

        # Fallback to keyword-based selection
        query_lower = query.lower()
        selected_tools = []

        if any(word in query_lower for word in ["summarize", "summary", "overview"]):
            if "summarization" in self.tool_registry:
                selected_tools.append({
                    "tool": "summarization",
                    "parameters": {"query": query}
                })

        if any(word in query_lower for word in ["search", "find", "filter", "show"]):
            if "document_search" in self.tool_registry:
                selected_tools.append({
                    "tool": "document_search",
                    "parameters": {"query": query}
                })

        return selected_tools

    def _execute_tool(
        self,
        tool_name: str,
        query: str,
        context: Optional[Dict],
        **parameters
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

            # Execute tool with parameters
            result = tool.execute(query, context, **parameters)

            latency = (time.time() - start_time) * 1000

            return {
                "tool": tool_name,
                "success": True,
                "result": result,
                "parameters": parameters,
                "latency_ms": latency
            }

        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            return {
                "tool": tool_name,
                "success": False,
                "error": str(e)
            }

    def _should_chain_to_next(
        self,
        current_tool: str,
        tool_selections: List[Dict],
        current_index: int
    ) -> bool:
        """
        Determine if current tool output should chain to next tool.
        
        Args:
            current_tool: Name of current tool
            tool_selections: All selected tools
            current_index: Index of current tool
            
        Returns:
            True if should chain
        """
        # Check if there's a next tool
        if current_index >= len(tool_selections) - 1:
            return False
        
        next_tool = tool_selections[current_index + 1]["tool"]
        
        # Define chaining rules
        chaining_rules = {
            "document_search": ["summarization"],  # Search results can be summarized
            "database_query": ["summarization"],   # Query results can be summarized
        }
        
        return next_tool in chaining_rules.get(current_tool, [])
    
    def _synthesize_results(
        self,
        query: str,
        tool_results: List[Dict],
        chained: bool = False
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
        
        # If chained, focus on the last tool's output (it contains synthesized info)
        if chained and len(successful_results) > 1:
            last_result = successful_results[-1]
            tool_name = last_result["tool"]
            tool_output = last_result["result"]
            
            if isinstance(tool_output, dict):
                if "summary" in tool_output:
                    response_parts.append(tool_output["summary"])
                else:
                    response_parts.append(f"**{tool_name}**: {tool_output.get('summary', str(tool_output))}")
            else:
                response_parts.append(f"**{tool_name}**: {tool_output}")
            
            # Add metadata about chaining
            num_tools = len(successful_results)
            response_parts.append(f"\n\n(Result synthesized from {num_tools} tools)")
        else:
            # Non-chained: show all results
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
        
        # Boost confidence if chaining was successful
        if chained and len(successful_results) > 1:
            confidence = min(confidence * 1.1, 1.0)

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
