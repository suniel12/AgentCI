"""
Tests for data models.
"""
import json

import pytest

from agentci.models import Span, SpanKind, Trace


def test_trace_initialization():
    trace = Trace(test_name="example")
    assert trace.test_name == "example"
    assert len(trace.spans) == 0


# ── Span.attributes (Milestone 3.1) ─────────────────────────────────────────


class TestSpanAttributes:
    def test_attributes_default_empty_dict(self):
        """Span.attributes defaults to {} (backward-compatible)."""
        span = Span(name="tool_span", kind=SpanKind.TOOL_CALL)
        assert span.attributes == {}

    def test_attributes_can_be_set(self):
        """Span.attributes can hold arbitrary key/value pairs."""
        span = Span(
            name="retrieve_docs",
            kind=SpanKind.TOOL_CALL,
            attributes={
                "tool.args": {"query": "How do I install AgentCI?"},
                "tool.result": "pip install agentci",
            },
        )
        assert span.attributes["tool.args"]["query"] == "How do I install AgentCI?"
        assert span.attributes["tool.result"] == "pip install agentci"

    def test_attributes_round_trip_json(self):
        """Span.attributes serializes and deserializes through JSON correctly."""
        span = Span(
            name="search_tool",
            kind=SpanKind.TOOL_CALL,
            attributes={
                "tool.args": {"query": "pricing"},
                "tool.result": {"results": ["plan A", "plan B"]},
            },
        )
        serialized = span.model_dump_json()
        restored = Span.model_validate_json(serialized)
        assert restored.attributes["tool.args"]["query"] == "pricing"
        assert restored.attributes["tool.result"]["results"] == ["plan A", "plan B"]

    def test_attributes_in_trace_round_trip(self):
        """Span.attributes survives Trace-level serialization."""
        span = Span(
            name="my_tool",
            kind=SpanKind.TOOL_CALL,
            attributes={"tool.args.search_query": "weather"},
        )
        trace = Trace(spans=[span])
        json_str = trace.model_dump_json()
        restored_trace = Trace.model_validate_json(json_str)
        assert restored_trace.spans[0].attributes["tool.args.search_query"] == "weather"

    def test_attributes_accepts_nested_structures(self):
        """Span.attributes can store nested dicts and lists."""
        span = Span(
            name="complex_tool",
            kind=SpanKind.TOOL_CALL,
            attributes={
                "tool.args": {
                    "filters": ["active", "premium"],
                    "pagination": {"page": 1, "size": 10},
                }
            },
        )
        assert span.attributes["tool.args"]["pagination"]["page"] == 1
        assert "active" in span.attributes["tool.args"]["filters"]
