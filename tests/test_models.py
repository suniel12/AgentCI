"""
Tests for data models.
"""
import pytest
from agentci.models import Trace, Span

def test_trace_initialization():
    trace = Trace(test_name="example")
    assert trace.test_name == "example"
    assert len(trace.spans) == 0
