---
name: agentci-testing
description: |
  Add trace-based regression tests for AI agents using AgentCI.
  Use when writing tests for agents built with LangGraph, Anthropic,
  or OpenAI Agents SDK. Covers tool call assertions, routing checks,
  cost guards, guardrail verification, and golden baseline diffing.
---

# AgentCI Testing Skill

When adding AgentCI tests to an agent project:

1. Install: `pip install agentci`
2. Create `tests/conftest.py` with framework-specific fixtures
3. Write test functions using `trace.tool_call_sequence`, `trace.get_handoffs()`, etc.
4. Record golden baselines with `agentci record`
5. Add CI with `agentci init`

## Test File Template

```python
import pytest
from agentci.capture import TraceContext
from agentci.assertions import assert_golden_match, assert_budget

def test_agent_routes_correctly():
    with TraceContext(agent_name="my_agent") as ctx:
        result = run_agent("I need help with billing")
        trace = ctx.trace

    handoffs = trace.get_handoffs()
    assert len(handoffs) == 1
    assert handoffs[-1].to_agent == "Billing Agent"

def test_agent_stays_under_budget():
    with TraceContext(agent_name="my_agent") as ctx:
        result = run_agent("simple query")
        trace = ctx.trace

    assert trace.total_cost_usd < 0.01

def test_agent_calls_expected_tools():
    with TraceContext(agent_name="my_agent") as ctx:
        result = run_agent("search the knowledge base")
        trace = ctx.trace

    assert "vector_search" in trace.tool_call_sequence

def test_golden_regression():
    with TraceContext(agent_name="my_agent") as ctx:
        result = run_agent("billing question")
        trace = ctx.trace

    assert_golden_match(trace, "golden_traces/test_billing.json")
```

## Mocking (Zero-Cost Testing)

```python
from agentci.mocks import AnthropicMocker, OpenAIMocker

# Anthropic
mocker = AnthropicMocker(mock_responses=[
    {"tool": "search", "input": {"query": "test"}},
    {"text": "Here are the results."},
])
agent.client = mocker.client

# OpenAI
mocker = OpenAIMocker(mock_responses=[
    {"tool": "search", "arguments": {"query": "test"}},
    {"text": "Here are the results."},
])
```

## CLI Commands

```bash
agentci init                    # Generate GitHub Actions + pre-push hook
agentci run                     # Run test suite
agentci run --json              # Machine-readable output
agentci record <test_name>      # Record golden baseline
agentci diff <test_name>        # Compare against golden
```

## Key Assertion Types

| Type | Usage |
|------|-------|
| `tool_called` | `assert "search" in trace.tool_call_sequence` |
| `cost_under` | `assert trace.total_cost_usd < 0.05` |
| `handoff_target` | `assert trace.get_handoffs()[-1].to_agent == "Agent"` |
| `output_contains` | `assert "text" in str(trace.spans[-1].output_data)` |
| `guardrails` | `assert "pii" not in trace.guardrails_triggered` |

See [AGENTS.md](../../../AGENTS.md) in the project root for full API reference.
