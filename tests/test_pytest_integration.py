import pytest
import agentci
from agentci.models import Trace

def mock_agent_function(input_text):
    return "Processed: " + input_text

@agentci.test(
    max_cost_usd=0.01,
    assertions=[{"type": "cost_under", "threshold": 0.01}]
)
def test_decorated_agent(agentci_trace):
    """Verify that the decorator works and injects the trace."""
    assert isinstance(agentci_trace, Trace)
    result = mock_agent_function("test input")
    assert result == "Processed: test input"
    # Manual span creation to verify trace is active
    from agentci.models import Span
    agentci_trace.spans.append(Span(name="manual_span"))

@agentci.test()
def test_simple_decorator(agentci_trace):
    """Verify decorator works without arguments."""
    assert agentci_trace is not None
