import pytest
from agentci.models import Trace, Span, ToolCall, SpanKind, DiffType
from agentci.diff_engine import diff_traces

def create_trace(
    tool_calls: list[tuple[str, dict]] = None,
    cost: float = 0.0,
    steps: int = 0
) -> Trace:
    """Helper to create a trace with specific tool calls and metrics."""
    t = Trace()
    t.total_cost_usd = cost
    t.total_llm_calls = steps
    
    if tool_calls:
        # Create a span to hold tool calls
        span = Span(kind=SpanKind.AGENT, name="test_agent")
        for name, args in tool_calls:
            span.tool_calls.append(ToolCall(tool_name=name, arguments=args))
        t.spans.append(span)
        t.total_tool_calls = len(tool_calls)
        
    return t

def test_no_diffs():
    """Identical traces should return empty diff list."""
    t1 = create_trace([("search", {"q": "foo"})], cost=0.01, steps=1)
    t2 = create_trace([("search", {"q": "foo"})], cost=0.01, steps=1)
    
    diffs = diff_traces(t2, t1)
    assert len(diffs) == 0

def test_tools_changed():
    """Detect added/removed tools."""
    golden = create_trace([("search", {}), ("book", {})])
    current = create_trace([("search", {}), ("cancel", {})])
    
    diffs = diff_traces(current, golden)
    assert len(diffs) == 1
    d = diffs[0]
    assert d.diff_type == DiffType.TOOLS_CHANGED
    assert d.severity == "error"
    assert "cancel" in d.details["added"]
    assert "book" in d.details["removed"]

def test_sequence_changed():
    """Detect same tools but different order."""
    golden = create_trace([("search", {}), ("book", {})])
    current = create_trace([("book", {}), ("search", {})])
    
    diffs = diff_traces(current, golden)
    assert len(diffs) == 1
    assert diffs[0].diff_type == DiffType.SEQUENCE_CHANGED
    assert diffs[0].severity == "warning"

def test_args_changed():
    """Detect changed arguments."""
    golden = create_trace([("search", {"q": "SFO"})])
    current = create_trace([("search", {"q": "JFK"})])
    
    diffs = diff_traces(current, golden)
    assert len(diffs) == 1
    d = diffs[0]
    assert d.diff_type == DiffType.ARGS_CHANGED
    assert d.details["tool"] == "search"
    assert d.details["changes"][0]["field"] == "q"
    assert d.details["changes"][0]["golden"] == "SFO"
    assert d.details["changes"][0]["current"] == "JFK"

def test_cost_spike():
    """Detect significant cost increase."""
    golden = create_trace(cost=0.10)
    current = create_trace(cost=0.50)  # 5x increase
    
    diffs = diff_traces(current, golden)
    assert len(diffs) == 1
    assert diffs[0].diff_type == DiffType.COST_SPIKE
    assert diffs[0].severity == "error"

def test_steps_increase():
    """Detect step count increase."""
    golden = create_trace(steps=5)
    current = create_trace(steps=10)  # 2x increase
    
    diffs = diff_traces(current, golden)
    assert len(diffs) == 1
    assert diffs[0].diff_type == DiffType.STEPS_CHANGED
