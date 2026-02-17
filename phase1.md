# Agent CI — Phase 1 Blueprint: The "Cost-Aware" Wedge

## Weeks 0–8 Detailed Execution Plan

**Objective:** Ship a working, pip-installable pytest plugin for AI agent regression testing. Reach 500+ GitHub stars. Prove that developers need CI-native testing for agent tool calls and cost behavior.

**Success Criteria (Week 8):**
- `pip install agentci` works cleanly on Python 3.10+
- A developer goes from install to first passing test in < 60 seconds
- The demo agent runs with zero API keys (using built-in mocks)
- GitHub Action template enables CI/CD in < 5 minutes
- Show HN post is live

---

## 1. Architecture Decisions (Decide Before Writing Code)

### 1.1 Core Principle: Trace-First, Not Test-First

Every piece of data flows through a single abstraction: the **Trace**. Even a single-agent, single-tool-call test produces a Trace with one Span. This is the critical decision that makes Phase 2 (multi-agent) an expansion rather than a rewrite.

```
Trace (one test execution)
  └── Span (one agent invocation)
        ├── LLM calls (input/output tokens, cost)
        ├── Tool calls (name, arguments, result)
        └── Child Spans (future: handoffs to other agents)
```

**Why this matters:** When you add `assert_handoff(agent_a, agent_b)` in Phase 2, you're just querying Spans within an existing Trace. No schema changes needed.

### 1.2 Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Language** | Python 3.10+ | Target audience is Python-first agent developers |
| **Data models** | Pydantic v2 | Industry standard for LLM tooling; fast validation; JSON serialization for golden traces |
| **CLI framework** | Click | More mature than Typer; better for plugins; used by Flask, pip, AWS CLI |
| **Terminal output** | Rich | Beautiful tables, progress bars, diff highlighting; standard in dev tools |
| **Test framework** | pytest plugin | Not a standalone runner — integrate into the tool developers already use |
| **Tracing** | OpenTelemetry SDK | Framework-agnostic; interoperable with Arize, Langfuse, Datadog in Phase 3 |
| **Serialization** | JSON (traces), YAML (test definitions) | JSON for machine data; YAML for human-authored config |
| **Packaging** | pyproject.toml + hatchling | Modern Python packaging; no setup.py |
| **CI template** | GitHub Actions | 90%+ of target audience; GitLab CI added if requested |

### 1.3 What NOT to Use

- **No database.** Golden traces are JSON files in the repo (version-controlled alongside code).
- **No web server.** No FastAPI, no Streamlit, no dashboard. CLI only.
- **No Docker.** Pure pip install. Zero infrastructure requirements.
- **No custom test runner.** pytest is the runner. You're a plugin, not a replacement.
- **No async in the CLI layer.** The test runner orchestrates async agent execution internally, but the CLI is synchronous. Keep it simple.

---

## 2. Data Model (The Foundation — Build This First)

### 2.1 Core Models (`src/agentci/models.py`)

```python
"""
Agent CI Core Data Models

These models define the universal trace format that all features
(diffing, cost tracking, assertions, reporting) consume.
Designed for Phase 1 (single-agent) but structured to support
Phase 2 (multi-agent) without schema changes.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import uuid


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


# ── Enums ──────────────────────────────────────────────

class SpanKind(str, Enum):
    """Type of span in the execution trace."""
    AGENT = "agent"          # An agent's full execution
    LLM_CALL = "llm_call"   # A single LLM API call
    TOOL_CALL = "tool_call"  # A single tool invocation
    HANDOFF = "handoff"      # Phase 2: agent-to-agent transfer


class DiffType(str, Enum):
    """Categories of detected changes between runs."""
    TOOLS_CHANGED = "tools_changed"       # Different tools called
    ARGS_CHANGED = "args_changed"         # Same tools, different arguments
    SEQUENCE_CHANGED = "sequence_changed" # Tools called in different order
    OUTPUT_CHANGED = "output_changed"     # Final output differs
    COST_SPIKE = "cost_spike"             # Cost exceeds threshold
    LATENCY_SPIKE = "latency_spike"       # Duration exceeds threshold
    STEPS_CHANGED = "steps_changed"       # Different number of LLM calls


class TestResult(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"    # Exception during execution, not a test failure


# ── Trace Components ───────────────────────────────────

class ToolCall(BaseModel):
    """A single tool/function call made by an agent."""
    tool_name: str
    arguments: dict[str, Any] = {}
    result: Any | None = None
    error: str | None = None          # If the tool call failed
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class LLMCall(BaseModel):
    """A single LLM API call within a span."""
    model: str = ""                    # e.g., "gpt-4o", "claude-sonnet-4-20250514"
    provider: str = ""                 # e.g., "openai", "anthropic"
    input_messages: list[dict[str, Any]] = []   # Stored for debugging, not diffing
    output_text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0             # Computed from token counts + pricing
    duration_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Span(BaseModel):
    """
    A unit of work in the execution trace.
    
    In Phase 1: One span per agent invocation.
    In Phase 2: Multiple spans form a DAG (agent A → handoff → agent B).
    """
    span_id: str = Field(default_factory=_new_id)
    parent_span_id: str | None = None  # Phase 2: enables tree structure
    kind: SpanKind = SpanKind.AGENT
    name: str = ""                     # Human-readable: "booking_agent", "search_tool"
    
    # Execution data
    input_data: Any = None             # What the agent/tool received
    output_data: Any = None            # What it returned
    
    # Collected events
    tool_calls: list[ToolCall] = []
    llm_calls: list[LLMCall] = []
    
    # Aggregated metrics (computed after execution)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    duration_ms: float = 0.0
    
    # Extensible metadata
    metadata: dict[str, Any] = {}

    def compute_metrics(self) -> None:
        """Roll up metrics from child LLM calls."""
        self.total_tokens_in = sum(c.tokens_in for c in self.llm_calls)
        self.total_tokens_out = sum(c.tokens_out for c in self.llm_calls)
        self.total_cost_usd = sum(c.cost_usd for c in self.llm_calls)


class Trace(BaseModel):
    """
    The complete execution record of a single test run.
    
    This is the universal data structure that every Agent CI feature
    consumes: diffing reads it, assertions query it, reports render it,
    and golden traces are serialized instances of it.
    """
    trace_id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # The execution data
    spans: list[Span] = []
    
    # Aggregated metrics
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    
    # Test metadata
    test_name: str = ""
    agent_name: str = ""
    framework: str = ""    # "langgraph", "crewai", "generic"
    metadata: dict[str, Any] = {}

    def compute_metrics(self) -> None:
        """Roll up metrics from all spans."""
        for span in self.spans:
            span.compute_metrics()
        self.total_cost_usd = sum(s.total_cost_usd for s in self.spans)
        self.total_tokens = sum(s.total_tokens_in + s.total_tokens_out for s in self.spans)
        self.total_llm_calls = sum(len(s.llm_calls) for s in self.spans)
        self.total_tool_calls = sum(len(s.tool_calls) for s in self.spans)

    @property
    def tool_call_sequence(self) -> list[str]:
        """Ordered list of tool names called across all spans."""
        calls = []
        for span in self.spans:
            calls.extend(tc.tool_name for tc in span.tool_calls)
        return calls
    
    @property
    def tool_call_details(self) -> list[ToolCall]:
        """All tool calls across all spans, in order."""
        calls = []
        for span in self.spans:
            calls.extend(span.tool_calls)
        return calls


# ── Test Definition Models ─────────────────────────────

class Assertion(BaseModel):
    """A single assertion to check against a trace."""
    type: str                          # "tool_called", "tool_not_called", 
                                       # "arg_equals", "arg_contains",
                                       # "cost_under", "steps_under",
                                       # "output_contains", "output_not_contains"
    tool: str | None = None            # Which tool (for tool-related assertions)
    field: str | None = None           # Which argument/field
    value: Any = None                  # Expected value
    threshold: float | None = None     # For numeric comparisons


class TestCase(BaseModel):
    """
    A single test scenario defined by the developer.
    
    Can be defined in Python (via decorators) or YAML (for non-code config).
    """
    name: str
    description: str = ""
    
    # Input to the agent
    input_data: Any = None             # String prompt, dict, or structured input
    
    # Expected behavior
    assertions: list[Assertion] = []
    
    # Cost/performance budgets
    max_cost_usd: float | None = None
    max_duration_ms: float | None = None
    max_steps: int | None = None       # Max LLM calls
    max_tool_calls: int | None = None
    
    # Golden trace reference
    golden_trace_path: str | None = None  # Path to saved golden trace JSON
    
    # Tags for filtering
    tags: list[str] = []


class TestSuite(BaseModel):
    """A collection of test cases, typically loaded from a YAML file."""
    name: str = "default"
    agent: str = ""                    # Import path: "myapp.agent:run_agent"
    framework: str = "generic"         # "langgraph", "crewai", "generic"
    tests: list[TestCase] = []
    
    # Suite-level defaults
    default_max_cost_usd: float | None = None
    default_max_steps: int | None = None


# ── Diff Models ────────────────────────────────────────

class DiffResult(BaseModel):
    """A single detected difference between current and golden trace."""
    diff_type: DiffType
    severity: str = "warning"          # "error", "warning", "info"
    message: str = ""
    details: dict[str, Any] = {}       # e.g., {"expected": "NYC", "actual": "New York"}


class RunResult(BaseModel):
    """The complete result of running a single test case."""
    test_name: str
    result: TestResult
    trace: Trace
    diffs: list[DiffResult] = []
    assertion_results: list[dict[str, Any]] = []
    error_message: str | None = None
    duration_ms: float = 0.0


class SuiteResult(BaseModel):
    """The complete result of running an entire test suite."""
    suite_name: str
    results: list[RunResult] = []
    total_passed: int = 0
    total_failed: int = 0
    total_errors: int = 0
    total_cost_usd: float = 0.0
    duration_ms: float = 0.0
```

### 2.2 Why This Model Is Phase-2-Ready

The `parent_span_id` field on Span enables tree structures:

```
Phase 1 (single agent):
Trace
  └── Span(kind=AGENT, name="booking_agent")
        ├── LLMCall(model="gpt-4o")
        ├── ToolCall(tool_name="search_flights")
        └── LLMCall(model="gpt-4o")

Phase 2 (multi-agent — same schema, no changes):
Trace
  └── Span(kind=AGENT, name="coordinator")
        ├── LLMCall(model="gpt-4o")
        ├── Span(kind=HANDOFF, name="handoff_to_researcher")  # parent_span_id = coordinator
        │     └── Span(kind=AGENT, name="researcher")
        │           ├── LLMCall(model="gpt-4o-mini")
        │           └── ToolCall(tool_name="web_search")
        └── Span(kind=HANDOFF, name="handoff_to_writer")
              └── Span(kind=AGENT, name="writer")
                    └── LLMCall(model="gpt-4o")
```

No schema migration. No rewrite. Just deeper trees.


---

## 3. Project Structure

```
agentci/
│
├── src/
│   └── agentci/
│       ├── __init__.py              # Version, public API exports
│       ├── cli.py                   # Click CLI: init, run, record, diff, report
│       ├── models.py                # All Pydantic models (Section 2 above)
│       ├── runner.py                # Test execution engine
│       ├── capture.py               # Trace capture via monkey-patching + OTEL
│       ├── diff_engine.py           # Golden trace comparison logic
│       ├── cost.py                  # Token counting + pricing tables
│       ├── assertions.py            # Built-in assertion evaluators
│       ├── mocks.py                 # MockTool class for zero-API-key testing
│       ├── report.py                # Rich CLI output + HTML report generation
│       ├── config.py                # Load agentci.yaml configuration
│       ├── pytest_plugin.py         # pytest integration (conftest fixtures)
│       ├── _version.py              # Single source of version truth
│       └── adapters/
│           ├── __init__.py          # Adapter registry
│           ├── base.py              # AbstractAdapter class
│           ├── langgraph.py         # LangGraph auto-instrumentation
│           └── generic.py           # Generic Python function adapter
│
├── tests/                           # Agent CI's own tests
│   ├── test_models.py
│   ├── test_diff_engine.py
│   ├── test_assertions.py
│   ├── test_cost.py
│   ├── test_runner.py
│   └── test_cli.py
│
├── examples/
│   ├── demo_agent/                  # Zero-dependency demo that works without API keys
│   │   ├── agent.py                 # Simple tool-calling agent using mocks
│   │   ├── agentci.yaml             # Test suite definition
│   │   └── golden/                  # Pre-saved golden traces
│   │       └── book_flight.golden.json
│   │
│   └── langgraph_example/           # LangGraph integration example
│       ├── agent.py
│       └── agentci.yaml
│
├── docs/
│   ├── quickstart.md                # 60-second getting started
│   ├── writing-tests.md             # How to define test cases
│   ├── golden-traces.md             # How diffing works
│   ├── cost-tracking.md             # Cost guardrails guide
│   ├── langgraph.md                 # LangGraph integration guide
│   └── ci-cd.md                     # GitHub Actions setup
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Agent CI's own CI
│       └── agentci-template.yml     # Copy-paste template for users
│
├── pyproject.toml                   # Package config, dependencies, entry points
├── README.md                        # The most important file in the project
├── VISION.md                        # "Agent CI" philosophy manifesto
├── LICENSE                          # Apache 2.0
├── CONTRIBUTING.md
└── CHANGELOG.md
```


---

## 4. Key Implementation Details

### 4.1 Trace Capture (`capture.py`)

The capture system works by monkey-patching LLM client libraries to intercept calls transparently:

```python
"""
Trace capture via monkey-patching.

Strategy: Wrap the OpenAI/Anthropic client's .create() methods to
automatically record every LLM call and tool invocation into a Trace
object. The developer doesn't change their agent code at all.

Phase 1: Patch openai.ChatCompletion and anthropic.Messages
Phase 2: Add OTEL span emission for interop with Arize/Langfuse
"""

import time
import contextvars
from agentci.models import Trace, Span, LLMCall, ToolCall, SpanKind
from agentci.cost import compute_cost

# Global context var — allows nested agent calls to share a trace
_active_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    '_active_trace', default=None
)
_active_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    '_active_span', default=None
)


class TraceContext:
    """
    Context manager that captures all LLM/tool activity into a Trace.
    
    Usage:
        with TraceContext(agent_name="booking_agent") as ctx:
            result = my_agent.run("Book a flight to NYC")
            trace = ctx.trace
    """
    
    def __init__(self, agent_name: str = "", test_name: str = ""):
        self.trace = Trace(agent_name=agent_name, test_name=test_name)
        self._patches = []
    
    def __enter__(self):
        # Create root span
        root_span = Span(kind=SpanKind.AGENT, name=self.trace.agent_name)
        self.trace.spans.append(root_span)
        
        # Set context vars
        _active_trace.set(self.trace)
        _active_span.set(root_span)
        
        # Apply monkey patches
        self._patch_openai()
        self._patch_anthropic()
        
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        # Compute duration
        self.trace.total_duration_ms = (time.perf_counter() - self._start_time) * 1000
        
        # Roll up metrics
        self.trace.compute_metrics()
        
        # Remove patches
        for restore_fn in self._patches:
            restore_fn()
        
        # Clear context
        _active_trace.set(None)
        _active_span.set(None)
    
    def _patch_openai(self):
        """Wrap openai.chat.completions.create to capture LLM calls."""
        try:
            import openai
            original_create = openai.resources.chat.completions.Completions.create
            
            def patched_create(self_client, *args, **kwargs):
                start = time.perf_counter()
                response = original_create(self_client, *args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                
                span = _active_span.get()
                if span is not None:
                    model = kwargs.get('model', getattr(response, 'model', ''))
                    usage = getattr(response, 'usage', None)
                    tokens_in = getattr(usage, 'prompt_tokens', 0) if usage else 0
                    tokens_out = getattr(usage, 'completion_tokens', 0) if usage else 0
                    
                    llm_call = LLMCall(
                        model=model,
                        provider="openai",
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=compute_cost("openai", model, tokens_in, tokens_out),
                        duration_ms=duration,
                    )
                    span.llm_calls.append(llm_call)
                    
                    # Capture tool calls from response
                    choices = getattr(response, 'choices', [])
                    if choices:
                        message = choices[0].message
                        tool_calls = getattr(message, 'tool_calls', None)
                        if tool_calls:
                            for tc in tool_calls:
                                import json
                                span.tool_calls.append(ToolCall(
                                    tool_name=tc.function.name,
                                    arguments=json.loads(tc.function.arguments),
                                ))
                
                return response
            
            openai.resources.chat.completions.Completions.create = patched_create
            self._patches.append(
                lambda: setattr(
                    openai.resources.chat.completions.Completions, 
                    'create', 
                    original_create
                )
            )
        except ImportError:
            pass  # OpenAI not installed — skip silently
    
    def _patch_anthropic(self):
        """Wrap anthropic.messages.create to capture LLM calls."""
        try:
            import anthropic
            original_create = anthropic.resources.messages.Messages.create
            
            def patched_create(self_client, *args, **kwargs):
                start = time.perf_counter()
                response = original_create(self_client, *args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                
                span = _active_span.get()
                if span is not None:
                    model = kwargs.get('model', getattr(response, 'model', ''))
                    usage = getattr(response, 'usage', None)
                    tokens_in = getattr(usage, 'input_tokens', 0) if usage else 0
                    tokens_out = getattr(usage, 'output_tokens', 0) if usage else 0
                    
                    llm_call = LLMCall(
                        model=model,
                        provider="anthropic",
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=compute_cost("anthropic", model, tokens_in, tokens_out),
                        duration_ms=duration,
                    )
                    span.llm_calls.append(llm_call)
                    
                    # Capture tool use blocks
                    for block in getattr(response, 'content', []):
                        if getattr(block, 'type', '') == 'tool_use':
                            span.tool_calls.append(ToolCall(
                                tool_name=block.name,
                                arguments=block.input if isinstance(block.input, dict) else {},
                            ))
                
                return response
            
            anthropic.resources.messages.Messages.create = patched_create
            self._patches.append(
                lambda: setattr(
                    anthropic.resources.messages.Messages, 
                    'create', 
                    original_create
                )
            )
        except ImportError:
            pass
```

### 4.2 Cost Tracking (`cost.py`)

```python
"""
Token cost computation.

Pricing is hardcoded (updated monthly). This is intentional —
an API call to fetch pricing would be a dependency and a failure point.
Users can override with custom pricing in agentci.yaml.
"""

# Prices per 1M tokens as of Feb 2026 (update monthly)
# Source: provider pricing pages
PRICING: dict[str, dict[str, tuple[float, float]]] = {
    # provider -> model -> (input_per_1M, output_per_1M)
    "openai": {
        "gpt-4o":       (2.50, 10.00),
        "gpt-4o-mini":  (0.15, 0.60),
        "gpt-4.1":      (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "o3-mini":      (1.10, 4.40),
    },
    "anthropic": {
        "claude-sonnet-4-20250514": (3.00, 15.00),
        "claude-haiku-4-5-20251001": (0.80, 4.00),
        "claude-opus-4-6":  (15.00, 75.00),
    },
}


def compute_cost(
    provider: str, 
    model: str, 
    tokens_in: int, 
    tokens_out: int
) -> float:
    """Compute USD cost for a single LLM call."""
    provider_pricing = PRICING.get(provider, {})
    
    # Try exact match first, then prefix match
    pricing = provider_pricing.get(model)
    if pricing is None:
        for model_key, price in provider_pricing.items():
            if model.startswith(model_key):
                pricing = price
                break
    
    if pricing is None:
        return 0.0  # Unknown model — don't crash, just skip cost
    
    input_cost = (tokens_in / 1_000_000) * pricing[0]
    output_cost = (tokens_out / 1_000_000) * pricing[1]
    return round(input_cost + output_cost, 6)
```

### 4.3 Golden Trace Diffing (`diff_engine.py`)

The diff engine is the core intellectual property. It compares two traces and categorizes every difference:

```python
"""
Golden Trace Diff Engine.

Compares a "current" trace against a "golden" (known-good) trace
and produces categorized diffs. The key insight: don't just say
"traces differ." Say exactly WHAT differs and WHY it matters.

Phase 1: Exact matching on tool names and arguments.
Phase 2: Add semantic matching via LLM-as-Judge for fuzzy comparison.
"""

from agentci.models import Trace, DiffResult, DiffType


def diff_traces(current: Trace, golden: Trace) -> list[DiffResult]:
    """
    Compare current trace against golden trace.
    Returns a list of specific, actionable differences.
    """
    diffs: list[DiffResult] = []
    
    # 1. TOOL SEQUENCE DIFF
    current_tools = current.tool_call_sequence
    golden_tools = golden.tool_call_sequence
    
    if current_tools != golden_tools:
        if set(current_tools) != set(golden_tools):
            # Different tools called entirely
            added = set(current_tools) - set(golden_tools)
            removed = set(golden_tools) - set(current_tools)
            diffs.append(DiffResult(
                diff_type=DiffType.TOOLS_CHANGED,
                severity="error",
                message=f"Tool set changed: +{added or 'none'} -{removed or 'none'}",
                details={
                    "golden_tools": golden_tools,
                    "current_tools": current_tools,
                    "added": list(added),
                    "removed": list(removed),
                }
            ))
        else:
            # Same tools, different order
            diffs.append(DiffResult(
                diff_type=DiffType.SEQUENCE_CHANGED,
                severity="warning",
                message=f"Tool call order changed",
                details={
                    "golden_sequence": golden_tools,
                    "current_sequence": current_tools,
                }
            ))
    
    # 2. ARGUMENT DIFF (for tools that appear in both)
    current_calls = current.tool_call_details
    golden_calls = golden.tool_call_details
    
    paired_calls = _pair_tool_calls(current_calls, golden_calls)
    for current_tc, golden_tc in paired_calls:
        arg_diffs = _diff_arguments(current_tc.arguments, golden_tc.arguments)
        if arg_diffs:
            diffs.append(DiffResult(
                diff_type=DiffType.ARGS_CHANGED,
                severity="warning",
                message=f"Arguments changed for '{current_tc.tool_name}'",
                details={
                    "tool": current_tc.tool_name,
                    "changes": arg_diffs,
                }
            ))
    
    # 3. COST DIFF
    if golden.total_cost_usd > 0:
        cost_ratio = current.total_cost_usd / golden.total_cost_usd
        if cost_ratio > 1.5:  # 50% cost increase threshold (configurable)
            diffs.append(DiffResult(
                diff_type=DiffType.COST_SPIKE,
                severity="error" if cost_ratio > 2.0 else "warning",
                message=f"Cost increased {cost_ratio:.1f}x: "
                        f"${golden.total_cost_usd:.4f} → ${current.total_cost_usd:.4f}",
                details={
                    "golden_cost": golden.total_cost_usd,
                    "current_cost": current.total_cost_usd,
                    "ratio": cost_ratio,
                }
            ))
    
    # 4. STEPS DIFF (number of LLM calls)
    if golden.total_llm_calls > 0:
        step_ratio = current.total_llm_calls / golden.total_llm_calls
        if step_ratio > 1.5:
            diffs.append(DiffResult(
                diff_type=DiffType.STEPS_CHANGED,
                severity="warning",
                message=f"LLM calls increased: {golden.total_llm_calls} → {current.total_llm_calls}",
                details={
                    "golden_steps": golden.total_llm_calls,
                    "current_steps": current.total_llm_calls,
                }
            ))
    
    return diffs


def _pair_tool_calls(current_calls, golden_calls):
    """
    Match tool calls between traces by name for comparison.
    Uses positional matching within each tool name group.
    """
    from collections import defaultdict
    
    current_by_name = defaultdict(list)
    golden_by_name = defaultdict(list)
    
    for tc in current_calls:
        current_by_name[tc.tool_name].append(tc)
    for tc in golden_calls:
        golden_by_name[tc.tool_name].append(tc)
    
    pairs = []
    for name in set(current_by_name) & set(golden_by_name):
        for c, g in zip(current_by_name[name], golden_by_name[name]):
            pairs.append((c, g))
    
    return pairs


def _diff_arguments(current_args: dict, golden_args: dict) -> list[dict]:
    """Produce a list of specific argument differences."""
    changes = []
    
    all_keys = set(current_args) | set(golden_args)
    for key in sorted(all_keys):
        current_val = current_args.get(key)
        golden_val = golden_args.get(key)
        
        if current_val != golden_val:
            changes.append({
                "field": key,
                "golden": golden_val,
                "current": current_val,
            })
    
    return changes
```

### 4.4 The Assertion System (`assertions.py`)

```python
"""
Built-in assertion evaluators.

Each assertion takes a Trace and returns (passed: bool, message: str).
Designed to be composable and extensible.
"""

from agentci.models import Trace, Assertion


def evaluate_assertion(assertion: Assertion, trace: Trace) -> tuple[bool, str]:
    """Dispatch an assertion to its evaluator."""
    evaluators = {
        "tool_called": _assert_tool_called,
        "tool_not_called": _assert_tool_not_called,
        "tool_call_count": _assert_tool_call_count,
        "arg_equals": _assert_arg_equals,
        "arg_contains": _assert_arg_contains,
        "cost_under": _assert_cost_under,
        "steps_under": _assert_steps_under,
        "output_contains": _assert_output_contains,
        "output_not_contains": _assert_output_not_contains,
    }
    
    evaluator = evaluators.get(assertion.type)
    if evaluator is None:
        return False, f"Unknown assertion type: {assertion.type}"
    
    return evaluator(assertion, trace)


def _assert_tool_called(a: Assertion, t: Trace) -> tuple[bool, str]:
    tools = t.tool_call_sequence
    if a.tool in tools:
        return True, f"✓ Tool '{a.tool}' was called"
    return False, f"✗ Tool '{a.tool}' was NOT called. Tools called: {tools}"


def _assert_tool_not_called(a: Assertion, t: Trace) -> tuple[bool, str]:
    tools = t.tool_call_sequence
    if a.tool not in tools:
        return True, f"✓ Tool '{a.tool}' was correctly not called"
    return False, f"✗ Tool '{a.tool}' was called but should not have been"


def _assert_tool_call_count(a: Assertion, t: Trace) -> tuple[bool, str]:
    count = t.tool_call_sequence.count(a.tool)
    expected = int(a.value)
    if count == expected:
        return True, f"✓ Tool '{a.tool}' called {count} time(s)"
    return False, f"✗ Tool '{a.tool}' called {count} time(s), expected {expected}"


def _assert_arg_equals(a: Assertion, t: Trace) -> tuple[bool, str]:
    for tc in t.tool_call_details:
        if tc.tool_name == a.tool:
            actual = tc.arguments.get(a.field)
            if actual == a.value:
                return True, f"✓ {a.tool}.{a.field} == {a.value}"
            return False, f"✗ {a.tool}.{a.field} == {actual}, expected {a.value}"
    return False, f"✗ Tool '{a.tool}' was not called"


def _assert_arg_contains(a: Assertion, t: Trace) -> tuple[bool, str]:
    for tc in t.tool_call_details:
        if tc.tool_name == a.tool:
            actual = str(tc.arguments.get(a.field, ""))
            if str(a.value) in actual:
                return True, f"✓ {a.tool}.{a.field} contains '{a.value}'"
            return False, f"✗ {a.tool}.{a.field} = '{actual}', missing '{a.value}'"
    return False, f"✗ Tool '{a.tool}' was not called"


def _assert_cost_under(a: Assertion, t: Trace) -> tuple[bool, str]:
    if t.total_cost_usd <= a.threshold:
        return True, f"✓ Cost ${t.total_cost_usd:.4f} ≤ ${a.threshold:.4f}"
    return False, f"✗ Cost ${t.total_cost_usd:.4f} > ${a.threshold:.4f} budget"


def _assert_steps_under(a: Assertion, t: Trace) -> tuple[bool, str]:
    if t.total_llm_calls <= int(a.threshold):
        return True, f"✓ LLM calls {t.total_llm_calls} ≤ {int(a.threshold)}"
    return False, f"✗ LLM calls {t.total_llm_calls} > {int(a.threshold)} limit"


def _assert_output_contains(a: Assertion, t: Trace) -> tuple[bool, str]:
    final_output = str(t.spans[-1].output_data) if t.spans else ""
    if str(a.value) in final_output:
        return True, f"✓ Output contains '{a.value}'"
    return False, f"✗ Output missing '{a.value}'"


def _assert_output_not_contains(a: Assertion, t: Trace) -> tuple[bool, str]:
    final_output = str(t.spans[-1].output_data) if t.spans else ""
    if str(a.value) not in final_output:
        return True, f"✓ Output correctly excludes '{a.value}'"
    return False, f"✗ Output unexpectedly contains '{a.value}'"
```

### 4.5 YAML Test Definition Format

This is what developers write. It should feel natural to anyone who's used pytest or Promptfoo:

```yaml
# agentci.yaml — Test suite definition
name: "booking_agent_tests"
agent: "myapp.agent:run_agent"     # Python import path to the agent function
framework: "langgraph"              # "langgraph", "crewai", "generic"

defaults:
  max_cost_usd: 0.10
  max_steps: 5

tests:
  - name: "book_flight_basic"
    description: "Agent should search for flights and book one"
    input: "Book me a flight from SFO to JFK on March 15"
    golden_trace: "golden/book_flight.golden.json"
    assertions:
      - type: tool_called
        tool: search_flights
      - type: arg_contains
        tool: search_flights
        field: origin
        value: "SFO"
      - type: tool_called
        tool: book_flight
      - type: cost_under
        threshold: 0.05
      - type: steps_under
        threshold: 4
  
  - name: "handle_no_results"
    description: "Agent should handle no flights found gracefully"
    input: "Book a flight from SFO to the Moon"
    assertions:
      - type: tool_called
        tool: search_flights
      - type: tool_not_called
        tool: book_flight
      - type: output_contains
        value: "sorry"
    tags: [edge_case]

  - name: "cost_guardrail"
    description: "Agent should not exceed cost budget on complex queries"
    input: "Find the cheapest multi-leg route from SFO to Tokyo with a layover in Hawaii"
    max_cost_usd: 0.08
    max_steps: 6
    assertions:
      - type: tool_called
        tool: search_flights
```

### 4.6 The Mock System (`mocks.py`)

```python
"""
Lightweight mock tools for zero-API-key testing.

Developers define mock responses in YAML or Python.
The demo agent ships with these pre-configured.
"""

from typing import Any, Callable
from agentci.models import ToolCall


class MockTool:
    """
    A fake tool that returns predefined responses.
    
    Usage:
        search = MockTool(
            name="search_flights",
            responses={
                "default": {"flights": [{"id": 1, "price": 350}]},
                "no_results": {"flights": []},
            }
        )
        
        # In agent code, replace real tool with mock:
        result = search.call(origin="SFO", destination="JFK")
    """
    
    def __init__(
        self, 
        name: str, 
        responses: dict[str, Any] | None = None,
        handler: Callable[..., Any] | None = None,
        stateful: bool = False,
    ):
        self.name = name
        self.responses = responses or {"default": {}}
        self.handler = handler
        self.stateful = stateful
        self._state: dict[str, Any] = {}
        self._call_history: list[dict[str, Any]] = []
        self._scenario: str = "default"
    
    def set_scenario(self, scenario: str) -> None:
        """Switch to a named response scenario."""
        self._scenario = scenario
    
    def call(self, **kwargs) -> Any:
        """Execute the mock tool, recording the call."""
        self._call_history.append({"arguments": kwargs})
        
        if self.handler:
            return self.handler(**kwargs, _state=self._state)
        
        return self.responses.get(self._scenario, self.responses["default"])
    
    @property
    def call_count(self) -> int:
        return len(self._call_history)
    
    def reset(self) -> None:
        self._call_history.clear()
        self._state.clear()
        self._scenario = "default"


class MockToolkit:
    """
    A collection of mock tools loaded from YAML.
    
    mocks.yaml:
        search_flights:
          default:
            flights:
              - id: 1
                price: 350
                airline: "United"
          no_results:
            flights: []
        
        book_flight:
          default:
            confirmation: "ABC123"
            status: "confirmed"
    """
    
    def __init__(self):
        self.tools: dict[str, MockTool] = {}
    
    @classmethod
    def from_yaml(cls, path: str) -> "MockToolkit":
        import yaml
        toolkit = cls()
        with open(path) as f:
            config = yaml.safe_load(f)
        
        for tool_name, responses in config.items():
            toolkit.tools[tool_name] = MockTool(
                name=tool_name,
                responses=responses,
            )
        
        return toolkit
    
    def get(self, name: str) -> MockTool:
        if name not in self.tools:
            raise KeyError(f"Mock tool '{name}' not found. Available: {list(self.tools)}")
        return self.tools[name]
    
    def set_all_scenarios(self, scenario: str) -> None:
        for tool in self.tools.values():
            tool.set_scenario(scenario)
    
    def reset_all(self) -> None:
        for tool in self.tools.values():
            tool.reset()
```

### 4.7 CLI Commands (`cli.py`)

```python
"""
Agent CI Command Line Interface.

Commands:
  agentci init          Scaffold a new test suite
  agentci run           Execute test suite
  agentci run --runs N  Statistical mode (run N times)
  agentci record        Run agent live, save golden trace
  agentci diff          Compare latest run against golden
  agentci report        Generate HTML report from last run
"""

import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
@click.version_option()
def cli():
    """Agent CI — Continuous Integration for AI Agents"""
    pass


@cli.command()
def init():
    """Scaffold a new Agent CI test suite."""
    # Creates: agentci.yaml, golden/, demo test case
    pass


@cli.command()
@click.option('--suite', '-s', default='agentci.yaml', help='Path to test suite YAML')
@click.option('--runs', '-n', default=1, help='Number of runs for statistical mode')
@click.option('--tag', '-t', multiple=True, help='Only run tests with these tags')
@click.option('--diff/--no-diff', default=True, help='Compare against golden traces')
@click.option('--html', type=click.Path(), help='Generate HTML report at this path')
@click.option('--fail-on-cost', type=float, help='Fail if total cost exceeds threshold')
@click.option('--ci', is_flag=True, help='CI mode: exit code 1 on any failure')
def run(suite, runs, tag, diff, html, fail_on_cost, ci):
    """Execute the test suite."""
    # Load suite → run tests → diff against golden → display results
    pass


@cli.command()
@click.argument('test_name')
@click.option('--suite', '-s', default='agentci.yaml')
@click.option('--output', '-o', help='Output path for golden trace')
def record(test_name, suite, output):
    """Run agent live and save the trace as a golden baseline."""
    # Execute the test → save trace as golden JSON
    pass


@cli.command()
@click.argument('test_name')
@click.option('--suite', '-s', default='agentci.yaml')
def diff(test_name, suite):
    """Show diff between latest run and golden trace."""
    pass


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True)
@click.option('--output', '-o', type=click.Path(), required=True)
def report(input, output):
    """Generate an HTML report from a JSON results file."""
    pass


if __name__ == '__main__':
    cli()
```

### 4.8 GitHub Action Template

```yaml
# .github/workflows/agentci.yml
# Copy this to your repo's .github/workflows/ directory

name: Agent CI Tests

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  agent-tests:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install agentci
          pip install -r requirements.txt
      
      - name: Run Agent CI tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}  # Optional if using mocks
        run: |
          agentci run --ci --fail-on-cost 0.50
      
      - name: Upload test report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: agentci-report
          path: agentci-report.html
```


---

## 5. Week-by-Week Execution Plan

### WEEK 1: Foundation + README-Driven Development

**Day 0 (before writing any code):**

- [ ] Choose the final name. Register on PyPI (even if empty package).
- [ ] Create the GitHub repo. Add Apache 2.0 license.
- [ ] Write VISION.md (the "Agent CI" manifesto — 500 words on why CI for agents is broken and how you're fixing it).

**Days 1–2: Write the README first.**

This is not optional. The README IS the product spec. Write it as if the tool already exists:
- Installation (`pip install agentci`)
- 60-second quickstart (show the full flow from install to first passing test)
- Feature overview with code examples
- YAML test definition format
- Golden trace diffing explanation
- Cost guardrails section
- GitHub Actions setup

Writing the README first forces you to make DX decisions before implementation decisions. If something is hard to explain in the README, it will be hard for developers to use.

**Days 3–5: Implement the data model.**

- [ ] Create the project structure (Section 3 above)
- [ ] Implement `models.py` — all Pydantic models from Section 2
- [ ] Write `tests/test_models.py` — validate serialization, compute_metrics, tool_call_sequence
- [ ] Implement `config.py` — YAML loader that hydrates TestSuite from agentci.yaml
- [ ] Implement `cost.py` — pricing tables and compute_cost function
- [ ] Set up pyproject.toml with all dependencies and entry points

**Deliverable by end of Week 1:** The data model compiles, serializes to JSON, and loads from YAML. You can create a Trace in Python and dump it to a .golden.json file. No execution yet — just the schema.


### WEEK 2: Core Runner + Trace Capture

**Days 1–2: Trace capture system.**

- [ ] Implement `capture.py` — the TraceContext context manager
- [ ] Implement OpenAI monkey-patch (capture LLM calls + tool calls)
- [ ] Implement Anthropic monkey-patch
- [ ] Test: Run a simple OpenAI function-calling agent inside TraceContext, verify the trace contains the correct tool calls and cost

**Days 3–4: Test runner.**

- [ ] Implement `runner.py` — loads a TestSuite, executes each TestCase by importing the agent function, wrapping it in TraceContext, and collecting results
- [ ] Implement `assertions.py` — all built-in assertion evaluators
- [ ] The runner should produce a SuiteResult with pass/fail for each test

**Day 5: Wire it together end-to-end.**

- [ ] Implement minimal `cli.py` with just the `run` command
- [ ] Test the full flow: define a test in YAML → `agentci run` → see pass/fail in terminal
- [ ] Use Rich for colorized output (green ✓ / red ✗)

**Deliverable by end of Week 2:** `agentci run` works against a simple Python agent. You can define tests in YAML, run them, and see colored pass/fail results. No diffing yet, no mocks, no `record` command.


### WEEK 3: Golden Trace Diffing

**Days 1–2: The diff engine.**

- [ ] Implement `diff_engine.py` — the full diff logic from Section 4.3
- [ ] Write extensive tests for diff_traces(): same traces (no diffs), added tools, removed tools, changed arguments, cost spikes, step count changes
- [ ] Ensure diff output is structured (DiffResult objects), not just strings

**Days 3–4: Integrate diffing into the runner.**

- [ ] Load golden traces from the path specified in TestCase.golden_trace_path
- [ ] After running a test, auto-diff against golden trace if path exists
- [ ] Display diffs in Rich-formatted terminal output with color coding:
  - 🔴 `TOOLS_CHANGED` — logic broke (error severity)
  - 🟡 `ARGS_CHANGED` — arguments drifted (warning)
  - 🟡 `SEQUENCE_CHANGED` — tool order changed (warning)
  - 🔴 `COST_SPIKE` — cost exceeded threshold (error)

**Day 5: Cost guardrails as assertions.**

- [ ] Implement the `--fail-on-cost` CLI flag
- [ ] Implement `max_cost_usd` and `max_steps` assertions from the YAML
- [ ] Test: an agent that exceeds the budget → test fails with clear cost breakdown

**Deliverable by end of Week 3:** The "smart diffing" works. `TOOLS_CHANGED` vs `COST_SPIKE` differentiation is live. A developer can save a golden trace and detect when their prompt change breaks tool behavior or inflates costs.


### WEEK 4: Statistical Mode + `record` Command

**Days 1–2: Statistical execution mode.**

- [ ] Implement `--runs N` flag in the runner
- [ ] Execute each test N times, collect all traces
- [ ] Compute and display: pass rate (e.g., "18/20 passed — 90%"), mean cost, cost standard deviation, tool call consistency rate
- [ ] Add `--pass-rate` flag (e.g., `--pass-rate 0.95` means fail if < 95% pass rate)

**Days 3–4: The `record` command.**

- [ ] Implement `agentci record <test_name>` — runs the agent live, captures the trace, saves it as `golden/<test_name>.golden.json`
- [ ] Pretty-print the recorded trace summary (tools called, cost, duration)
- [ ] Prompt the user: "Save this trace as golden baseline? [y/n]"
- [ ] This is crucial DX — developers shouldn't have to manually write golden traces

**Day 5: Refine CLI output polish.**

- [ ] Summary table after `agentci run` showing: test name, result, cost, diffs detected
- [ ] Total suite cost and duration at the bottom
- [ ] Exit code 1 when any test fails (for CI/CD)

**Deliverable by end of Week 4:** The tool is feature-complete for single-agent testing. Statistical mode works. The `record` command eliminates manual golden trace authoring. A developer can do the full workflow: record → modify prompt → run → see diff.


### WEEK 5: LangGraph Adapter + Mock System

**Days 1–3: LangGraph auto-instrumentation.**

- [ ] Implement `adapters/langgraph.py`
- [ ] The adapter should hook into LangGraph's execution to capture: which nodes executed, what state was passed between nodes, which tools each node called
- [ ] Map LangGraph's graph structure to the Trace → Span model: each node becomes a Span, edges become metadata
- [ ] Test against a real LangGraph agent (the ReAct pattern from LangGraph tutorials)

**Days 4–5: Mock tools.**

- [ ] Implement `mocks.py` — MockTool and MockToolkit classes from Section 4.6
- [ ] YAML-based mock definition (mocks.yaml)
- [ ] Integrate mocks into the test runner: if `framework: mock` or `use_mocks: true` in the YAML, auto-substitute mock tools
- [ ] Key requirement: the demo agent must work entirely offline using mocks — zero API keys, zero internet

**Deliverable by end of Week 5:** LangGraph agents can be tested with automatic trace capture. Mock tools enable offline testing. The demo agent runs without any API keys.


### WEEK 6: pytest Integration + CI Template

**Days 1–2: pytest plugin.**

- [ ] Implement `pytest_plugin.py` — register as a pytest plugin via entry points in pyproject.toml
- [ ] Provide fixtures: `agentci_trace` (capture context), `agentci_suite` (load from YAML)
- [ ] Provide the `@agentci.test` decorator for Python-defined tests (alternative to YAML)
- [ ] Ensure `pytest --agentci` flag activates the plugin
- [ ] Test discovery: auto-discover agentci.yaml files and register them as pytest items

**Example of Python-defined tests (decorator API):**

```python
# test_agent.py — discovered by pytest automatically
import agentci

@agentci.test(
    input="Book a flight from SFO to JFK",
    assert_tools=["search_flights", "book_flight"],
    max_cost=0.05,
    golden="golden/book_flight.golden.json",
)
def test_book_flight(agent):
    """Test that the booking agent completes a basic booking."""
    return agent.run("Book a flight from SFO to JFK")
```

**Days 3–4: GitHub Action template.**

- [ ] Create the `.github/workflows/agentci-template.yml` from Section 4.8
- [ ] Test it in a real GitHub repo (create a test repo, push, verify the action runs)
- [ ] Document the setup in `docs/ci-cd.md`
- [ ] Support both API-key-required and mock-only test modes in CI

**Day 5: The `init` command.**

- [ ] Implement `agentci init` — scaffolds the directory structure
- [ ] Creates: agentci.yaml (sample), golden/ directory, demo test case, mocks.yaml (sample)
- [ ] If it detects LangGraph in requirements.txt, auto-configure the LangGraph adapter
- [ ] This must feel magical — one command, complete setup

**Deliverable by end of Week 6:** Tests can be written in Python (decorators) or YAML. pytest discovers and runs Agent CI tests. GitHub Actions template works end-to-end. `agentci init` scaffolds a project in seconds.


### WEEK 7: Demo Agent + Documentation + Polish

**Days 1–2: The demo agent.**

- [ ] Build `examples/demo_agent/` — a complete, self-contained example
- [ ] The agent should: receive a travel booking request, call search_flights, evaluate results, call book_flight, return confirmation
- [ ] Implement using mock tools ONLY — no API keys required
- [ ] Include 3 test cases: happy path, no results found, cost budget exceeded
- [ ] Include pre-saved golden traces
- [ ] This is the "60-second quickstart" experience: `pip install agentci && cd examples/demo_agent && agentci run`

**Day 3: Documentation.**

- [ ] Finalize all docs (Section 3 file list)
- [ ] Quickstart guide (with screenshots/terminal output using Rich)
- [ ] Writing-tests guide (YAML format reference)
- [ ] Golden traces guide (what they are, how diffing works, when to update them)
- [ ] Cost tracking guide
- [ ] LangGraph integration guide (with real code examples)
- [ ] CI/CD guide

**Days 4–5: Polish and edge cases.**

- [ ] Error messages: every failure should explain what went wrong AND how to fix it
- [ ] Handle missing API keys gracefully (suggest mock mode)
- [ ] Handle invalid YAML with actionable parse errors
- [ ] Handle import errors for agent functions with clear messages
- [ ] Verify `pip install agentci` in a clean virtual environment
- [ ] Run the full test suite in CI (Agent CI testing itself)
- [ ] HTML report generation (simple single-page report with test results, diffs, costs)

**Deliverable by end of Week 7:** The entire product is polished, documented, and installable. The demo agent works in 60 seconds. Every error message is helpful. Documentation covers every feature.


### WEEK 8: Launch

**Day 1: Pre-launch checklist.**

- [ ] Final `pip install` test in clean environments (macOS, Ubuntu, Windows WSL)
- [ ] PyPI package published (version 0.1.0)
- [ ] GitHub repo is public with: README, LICENSE, CONTRIBUTING.md, CHANGELOG.md
- [ ] Create 3 GitHub Issues labeled "good first issue" for potential contributors
- [ ] Discord server created with channels: #general, #bugs, #feature-requests, #show-your-tests

**Day 2: Show HN launch.**

Post title: **"Show HN: Agent CI – Open-source CI for AI agents. Catch cost spikes and logic regressions."**

Post body structure:
- Problem: "I changed my agent's prompt to be more polite, and it silently stopped calling the right tools. My OpenAI bill tripled. Nothing caught it until a customer complained."
- Solution: "Agent CI is a pytest plugin that diffs your agent's tool-calling behavior against a golden baseline and enforces cost budgets in CI/CD."
- Demo: Link to the 60-second quickstart
- Differentiator: "Built trace-first for multi-agent systems (coming soon). Works with LangGraph today, framework-agnostic via OpenTelemetry."
- Ask: "I'd love feedback from anyone testing AI agents in production."

**Days 3–5: Post-launch engagement.**

- [ ] Respond to every HN comment within 2 hours
- [ ] Respond to every GitHub issue within 24 hours
- [ ] Post on Twitter/X: "Launched Agent CI on Hacker News today" with a terminal screenshot
- [ ] Post on r/MachineLearning and r/LangChain (if rules allow)
- [ ] Cross-post blog: "Why I built CI for AI agents" on dev.to or your personal blog
- [ ] Measure: stars, pip installs (PyPI stats), Discord joins, GitHub issues filed

**Deliverable by end of Week 8:** The product is live. Show HN is posted. Community channels are active. You're in feedback-gathering mode, preparing for Phase 2 based on real user pain points.


---

## 6. Dependencies (`pyproject.toml`)

```toml
[project]
name = "agentci"
version = "0.1.0"
description = "Continuous Integration for AI Agents. Catch cost spikes and logic regressions before production."
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.10"
authors = [{name = "Sunil Closure"}]
keywords = ["ai", "agents", "testing", "llm", "ci-cd", "regression"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Testing",
    "Framework :: Pytest",
]

dependencies = [
    "pydantic>=2.0",
    "click>=8.0",
    "rich>=13.0",
    "pyyaml>=6.0",
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
]

[project.optional-dependencies]
openai = ["openai>=1.0"]
anthropic = ["anthropic>=0.20"]
langgraph = ["langgraph>=0.1"]
all = ["agentci[openai,anthropic,langgraph]"]
dev = ["pytest>=8.0", "ruff", "mypy"]

[project.scripts]
agentci = "agentci.cli:cli"

[project.entry-points.pytest11]
agentci = "agentci.pytest_plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```


---

## 7. Success Metrics by Week

| Week | Stars | pip installs/day | Key Milestone |
|------|-------|------------------|---------------|
| 1 | 0 | 0 | Data model + README done |
| 2 | 0 | 0 | `agentci run` works end-to-end |
| 3 | 0 | 0 | Golden diffing + cost guardrails live |
| 4 | 0 | 0 | Statistical mode + record command |
| 5 | 0 | 0 | LangGraph adapter + mocks |
| 6 | 0 | 0 | pytest plugin + GitHub Action |
| 7 | 0 | 5–10 (beta testers) | Demo agent + docs complete |
| 8 | 50–200 | 20–50 | Show HN launch |
| 8+2w | 200–500 | 50–100 | Post-launch momentum |


---

## 8. Risks and Mitigations

| Risk | Probability | Mitigation |
|------|-------------|------------|
| OpenAI/Anthropic change their client API | Medium | Capture layer uses try/except; test against pinned versions; community reports breaks fast |
| LangGraph adapter breaks on new LangGraph release | High | Pin minimum version; join LangGraph Discord; treat adapter maintenance as weekly task |
| HN launch gets no traction | Medium | Have backup launch plan: Twitter threads, dev.to post, r/LangChain, LangChain Discord showcase |
| Scope creep (tempted to add dashboard) | High | This document is the scope contract. If it's not in the week-by-week plan, it doesn't ship in Phase 1. |
| Solo founder burnout at week 5–6 | Medium | Weeks 5-6 are the hardest (LangGraph adapter is complex). Build in 1 rest day per week. Ship imperfect adapters — iterate based on user feedback. |


---

## 9. What Explicitly Does NOT Ship in Phase 1

This is as important as what does ship:

- ❌ Web dashboard or any frontend
- ❌ CrewAI adapter (Phase 2, Weeks 9–14)
- ❌ Multi-agent assertions (assert_handoff, loop detection) — Phase 2
- ❌ Semantic/fuzzy matching on arguments — Phase 2
- ❌ VCR recorder (recording HTTP responses) — Phase 2
- ❌ Production trace ingestion (OTEL import) — Phase 3
- ❌ LLM-as-Judge evaluation — Phase 2
- ❌ Any payment/billing system
- ❌ User accounts or authentication
- ❌ Node.js/TypeScript SDK
- ❌ Support for non-Python agents
- ❌ Compliance or safety features
- ❌ Async CLI (internal runner can be async; CLI interface is sync)
