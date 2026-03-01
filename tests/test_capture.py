"""
Tests for capture.py: TraceContext, attach(), and langgraph_trace().
"""
import pytest
from unittest.mock import patch

from agentci.capture import TraceContext, langgraph_trace
from agentci.models import Trace


class TestTraceContextAttach:
    def test_attach_is_alias_for_attach_langgraph_state(self):
        """TraceContext.attach() delegates to attach_langgraph_state()."""
        with TraceContext(agent_name="test_agent") as ctx:
            called_with = []
            original = ctx.attach_langgraph_state

            def recording_attach(state):
                called_with.append(state)
                return original(state)

            ctx.attach_langgraph_state = recording_attach
            state = {"messages": []}
            ctx.attach(state)
            assert called_with == [state]

    def test_attach_populates_graph_state(self):
        """attach() via attach_langgraph_state sets trace.graph_state."""
        with TraceContext(agent_name="test_agent") as ctx:
            state = {"messages": [], "custom_key": "value"}
            ctx.attach(state)
        assert ctx.trace.graph_state == state


class TestLangGraphTrace:
    def test_returns_context_manager(self):
        """langgraph_trace() is a context manager that yields a TraceContext."""
        with langgraph_trace("my-agent") as ctx:
            assert isinstance(ctx, TraceContext)

    def test_trace_is_accessible_after_context(self):
        """ctx.trace is a Trace object after the context exits."""
        with langgraph_trace("my-agent") as ctx:
            pass
        assert isinstance(ctx.trace, Trace)

    def test_agent_name_is_set(self):
        """langgraph_trace passes agent_name to TraceContext."""
        with langgraph_trace("rag-agent") as ctx:
            pass
        assert ctx.trace.agent_name == "rag-agent"

    def test_empty_agent_name_is_allowed(self):
        """langgraph_trace works with no agent_name argument."""
        with langgraph_trace() as ctx:
            pass
        assert ctx.trace.agent_name == ""

    def test_metrics_computed_after_exit(self):
        """trace.total_cost_usd and total_llm_calls are set after context exits."""
        with langgraph_trace("my-agent") as ctx:
            pass
        # With no real LLM calls, these should be 0 (not None / unset)
        assert ctx.trace.total_cost_usd == 0.0
        assert ctx.trace.total_llm_calls == 0

    def test_attach_inside_context(self):
        """ctx.attach() can be called inside langgraph_trace context."""
        state = {"messages": [], "result": "hello"}
        with langgraph_trace("rag-agent") as ctx:
            ctx.attach(state)
        assert ctx.trace.graph_state == state
