# AgentCI

Trace-based regression testing for AI agents. Catch semantic drift, tool call changes, and cost spikes before production.

[![Python](https://img.shields.io/pypi/pyversions/agentci)]()
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)]()
[![CI](https://github.com/agentci-org/agentci/actions/workflows/ci.yml/badge.svg)]()
[![AGENTS.md](https://img.shields.io/badge/AGENTS.md-supported-blue)](AGENTS.md)

You swap `gpt-4o-mini` for `gpt-4o`. Your support bot starts routing billing questions to the wrong agent. You change a retrieval prompt. Your RAG pipeline silently skips the vector search step. You don't find out until users complain.

AgentCI records what your agent *actually did* — every tool call, LLM invocation, routing decision, and cost — then diffs it against a known-good baseline. When something drifts, you see exactly what changed.

## See It Work

```bash
# Run your agent's test suite — all green
$ pytest tests/ -v
======================== test session starts =========================
tests/test_routing.py::test_billing_routes_correctly       PASSED
tests/test_routing.py::test_refund_flow_tool_sequence      PASSED
tests/test_routing.py::test_cost_under_budget              PASSED
tests/test_rag.py::test_retrieval_step_executed            PASSED
tests/test_rag.py::test_no_hallucination                   PASSED
==================== 5 passed in 0.8s (mock mode) ====================

# Now change one line — swap the model
-  model = "gpt-4o-mini"
+  model = "gpt-4o"

# Run again — AgentCI catches the drift
$ pytest tests/test_regression.py -v
====================== AgentCI Diff Report =======================
⚠️  ROUTING_CHANGED  "I can't log in and I'm being charged"
   baseline → Account Agent
   current  → Billing Agent

⚠️  COST_SPIKE  "Can I get a refund?"
   baseline: $0.0004  →  current: $0.0025  (6.2x increase)

✅  TOOLS_IDENTICAL  [lookup_invoice, process_refund]
✅  OUTPUT_SIMILAR   cosine_similarity=0.94

SUMMARY: 1 routing change, 1 cost spike across 5 test cases
=================================================================
```

## Quickstart

```bash
pip install agentci
git clone https://github.com/agentci-org/DemoAgents.git
cd DemoAgents
make test
```

You'll see 90+ tests pass across three demo agents — RAG, DevOps, and customer support routing — in under 2 seconds with zero API keys required.

## What It Catches

| What went wrong | How you'd find out today | How AgentCI catches it |
|----------------|--------------------------|----------------------|
| Model swap changes routing decisions | User complaints, days later | `ROUTING_CHANGED` diff with exact before/after agents |
| Prompt edit silently skips a tool call | Manual testing, maybe | `TOOLS_CHANGED` — "vector_search was called, now it's not" |
| New model costs 6x more per query | Surprise invoice at month-end | `COST_SPIKE` with threshold alerts and `@assert_budget` |
| RAG retriever returns irrelevant docs | Hallucinated answers in prod | LLM-as-judge assertions: `assert_llm_judge("answer is relevant")` |
| Guardrail stops firing after refactor | PII leaks through to specialist agent | `guardrails_triggered` trace assertions |
| Agent responds instead of routing | Support quality drops silently | `assert_handoff_count(expected=1)` catches the missing handoff |

## Works With Your Stack

AgentCI is framework-agnostic. Same assertions, same diff engine, same CLI — regardless of how you built your agent.

| Framework | Integration | Demo |
|-----------|------------|------|
| **OpenAI Agents SDK** | Native `AgentCITraceProcessor` — 2 lines to enable | [Support Router](https://github.com/agentci-org/DemoAgents/tree/main/examples/support-router) — multi-agent handoff with guardrails |
| **LangGraph / LangChain** | `ctx.attach_langgraph_state()` captures conditional edges and tool calls | [RAG Agent](https://github.com/agentci-org/DemoAgents/tree/main/examples/rag-agent) — retrieval + grading + generation pipeline |
| **Anthropic (raw)** | `AnthropicMocker` for zero-cost replay; native tool_use capture | [DevAgent](https://github.com/agentci-org/DemoAgents/tree/main/examples/dev-agent) — 8-tool sequential repo analysis |
| **Any Python agent** | Manual `Trace` / `Span` construction for custom frameworks | See [Core Concepts](#core-concepts) below |

All three demo agents run with **zero API keys** using AgentCI's mock system.

## Core Concepts

### Trace & Assert

Every agent run produces a `Trace` — a structured record of LLM calls, tool invocations, handoffs, and guardrail checks.

```python
from agentci.models import Trace

trace = run_your_agent("I was charged twice")

# What tools did the agent call?
assert "lookup_invoice" in trace.tool_call_sequence
assert "process_refund" in trace.tool_call_sequence

# Did it route correctly?
handoffs = trace.get_handoffs()
assert handoffs[0].to_agent == "Billing Agent"

# Did it stay under budget?
assert trace.total_cost_usd < 0.01

# Did guardrails fire when they should?
assert "pii_guardrail" not in trace.guardrails_triggered
```

### Golden Baselines

Record a known-good trace. Diff future runs against it.

```bash
# Record a baseline
$ agentci record --test-name "billing_flow" --output golden/

# Later, after a change, diff against it
$ agentci diff --test-name "billing_flow"

# AgentCI reports:
#   TOOLS_CHANGED — vector_search was called, now it's not
#   COST_SPIKE — $0.0004 → $0.0025 (6.2x)
#   ROUTING_CHANGED — Account Agent → Billing Agent
```

Or use the pytest-native approach with `assert_golden_match`:

```python
from agentci.assertions import assert_golden_match

def test_billing_regression():
    trace = run_your_agent("I was charged twice")
    assert_golden_match(trace, "golden/billing_flow.json")
```

### Zero-Cost Testing

Mock LLM responses for deterministic, instant, free test runs.

```python
from agentci.mocks import OpenAIMocker
from agents import set_default_openai_client

# Define a scripted sequence of tool calls + final response
mocker = OpenAIMocker([
    {"tool": "transfer_to_BillingAgent", "arguments": {}},
    {"tool": "lookup_invoice", "arguments": {"email": "user@test.com"}},
    {"text": "I found your invoice. The duplicate charge has been refunded."}
])

# Inject into the OpenAI Agents SDK — no API key needed
set_default_openai_client(mocker.client)

trace = run_your_agent("I was charged twice")
assert trace.tool_call_sequence == ["transfer_to_BillingAgent", "lookup_invoice"]
```

Works the same for Anthropic:

```python
from agentci.mocks import AnthropicMocker

mocker = AnthropicMocker([
    {"tool": "search_knowledge_base", "arguments": {"query": "billing"}},
    {"text": "Here's what I found about billing..."}
])
```

## Try Breaking It

Clone the [DemoAgents](https://github.com/agentci-org/DemoAgents) repo and try these:

**Break 1: Remove a specialist agent.** In `examples/support-router/support_router/agents/triage.py`, remove `account_agent` from the `handoffs` list. Run `pytest tests/test_routing.py`. Watch 6 tests fail with: `ROUTING_CHANGED: Expected 'Account Agent', got 'General Agent'`.

**Break 2: Disable retrieval guardrails.** In `examples/support-router/support_router/agents/triage.py`, remove the `input_guardrails` list. Run `pytest tests/test_guardrails.py`. Watch `test_off_topic_blocked` fail: `Expected guardrail 'relevance_guardrail' to trigger, but no guardrails fired.`

**Break 3: Inject bad context.** In `examples/rag-agent/agent.py`, modify the system prompt to remove the "only answer from context" instruction. Run `pytest tests/test_rag.py`. Watch the LLM-as-judge assertion catch the hallucination: `FAILED — Agent used pre-trained knowledge instead of declining.`

## CLI

```bash
agentci init         # Generate CI config (GitHub Actions + pre-push hooks)
agentci run          # Run all tests (pytest-compatible)
agentci record       # Record golden baseline traces
agentci diff         # Diff current run against baseline
agentci report       # Generate HTML report from results
```

## CI/CD

```bash
agentci init
```

Generates a GitHub Actions workflow and optional pre-push git hook. Every push runs your agent tests and diffs against golden baselines.

```yaml
# .github/workflows/agentci.yml (auto-generated)
name: AgentCI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e ".[all]"
      - run: pytest tests/ -v
      - run: agentci diff --baseline golden/
```

## For Coding Agents

If you're a coding agent (Claude Code, Cursor, Codex, Copilot) integrating AgentCI into a project, see [`AGENTS.md`](AGENTS.md) for structured setup instructions, import patterns, and the full assertion API reference.

Quick setup for any agent project:
1. `pip install agentci`
2. Copy the test template from AGENTS.md
3. Run `agentci init` to generate CI config
4. Run `agentci run` to execute tests

## Assertions

| Assertion | What it checks |
|-----------|---------------|
| `trace.tool_call_sequence` | Ordered list of tool names invoked during the run |
| `trace.total_cost_usd` | Total LLM cost in USD |
| `trace.get_handoffs()` | List of agent-to-agent routing decisions |
| `trace.guardrails_triggered` | Which safety guardrails fired |
| `trace.agents_involved` | Ordered list of agent names that executed |
| `assert_golden_match(trace, path)` | Diff against a saved golden baseline |
| `evaluate_assertion(type="llm_judge")` | LLM-as-judge evaluation of output quality |
| `evaluate_assertion(type="handoff_target")` | Verify routing destination |
| `evaluate_assertion(type="handoff_count")` | Verify number of handoffs |
| `@assert_budget(max_cost=0.10)` | Decorator: fail if run exceeds cost limit |
| `evaluate_assertion(type="tool_called")` | Verify a specific tool was invoked |
| `evaluate_assertion(type="output_contains")` | Check output includes expected content |
| `evaluate_assertion(type="cost_under")` | Check total cost below threshold |

## Diff Report Types

When you diff a current run against a golden baseline, AgentCI flags these change categories:

| Diff Type | Meaning |
|-----------|---------|
| `TOOLS_CHANGED` | Different tools were called vs. baseline |
| `ARGS_CHANGED` | Same tools, but arguments changed |
| `SEQUENCE_CHANGED` | Tools called in a different order |
| `COST_SPIKE` | Cost increased beyond threshold |
| `LATENCY_SPIKE` | Duration increased beyond threshold |
| `STOP_REASON_CHANGED` | LLM stopped for a different reason |
| `ROUTING_CHANGED` | Agent handoff went to a different target |
| `GUARDRAILS_CHANGED` | Different guardrails fired vs. baseline |
| `OUTPUT_CHANGED` | Final output semantically different |
| `AVAILABLE_HANDOFFS_CHANGED` | Set of reachable agents changed |

## Status

AgentCI is in **early release** (v0.1.x). The core trace model, diffing engine, assertion library, mock system, and CLI are stable. The API may evolve based on community feedback.

**What's here today:**
- Trace capture for LangGraph, Anthropic, and OpenAI Agents SDK
- Golden baseline recording and regression diffing
- LLM mocking for all three frameworks (zero API keys)
- LLM-as-judge assertions for subjective output grading
- Pytest plugin for native integration
- GitHub Actions generator via `agentci init`

**What's next:**
- Dashboard UI for trace visualization
- Trace export (OpenTelemetry format)
- More framework adapters (CrewAI, AutoGen)
- MCP server for native coding agent integration
- Agent Skills distribution for auto-discovery
- Hosted regression tracking

## Contributing

AgentCI is open source under the Apache 2.0 License. Issues, PRs, and feedback welcome.

If you build an agent and test it with AgentCI, I'd love to hear about it — open an issue or reach out.
