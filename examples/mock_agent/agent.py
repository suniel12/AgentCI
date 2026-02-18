
from typing import Any

def run_agent(input_text: str, tools: dict[str, Any] | None = None) -> str:
    """
    An agent that uses injected tools if available.
    """
    if not tools:
        return "No tools available."
    
    # Simple logic: look for a tool named "search"
    search_tool = tools.get("search")
    if search_tool:
        # Call the tool with the input text
        result = search_tool.call(query=input_text)
        return f"Tool result: {result}"
    
    return "Search tool not found."
