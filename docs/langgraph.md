# LangGraph Integration

Agent CI has first-class support for LangGraph.

## setup

1. Install the optional dependency:
   ```bash
   pip install "agentci[langgraph]"
   ```

2. Configure your `agentci.yaml`:
   ```yaml
   framework: "langgraph"
   agent: "my_graph:app"
   ```

The Agent CI runner will automatically instrument your LangGraph application to capture traces.
