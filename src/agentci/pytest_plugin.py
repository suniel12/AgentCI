"""
pytest integration plugin.

Provides fixtures and hooks for running AgentCI tests via pytest.
"""

import pytest
import functools
from typing import Any, Callable
from .models import TestCase, Assertion
from .capture import TraceContext
from .cost import compute_cost

# Global registry of tests defined via decorators
# We'll use this if we need to collect them, but pytest collection is usually file-based.
# For now, the decorator just wraps the function to run it inside AgentCI context.

def test(
    input: Any = None,
    assertions: list[Assertion | dict] | None = None,
    max_cost_usd: float | None = None,
    max_steps: int | None = None,
    golden_trace: str | None = None,
):
    """
    Decorator to mark a function as an AgentCI test.
    
    Usage:
        @agentci.test(max_cost_usd=0.10)
        def test_my_agent(agentci_trace):
            run_agent("Hello")
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Inject trace fixture if requested but not provided?
            # Actually, pytest handles fixture injection.
            # We just need to ensure the wrapper accepts it.
            
            # TODO: How do we get the agentci_trace here?
            # If the user asks for `agentci_trace` fixture, pytest provides it.
            # But we want to configure the trace with test metadata *before* the test runs?
            # Or we let the fixture handle it?
            
            return func(*args, **kwargs)

        # Mark as agentci test for collection
        wrapper.pytestmark = [pytest.mark.agentci]
        
        # Attach metadata to function for the fixture to find
        wrapper._agentci_config = {
            "input": input,
            "assertions": assertions,
            "max_cost_usd": max_cost_usd,
            "max_steps": max_steps,
            "golden_trace": golden_trace,
        }
        return wrapper
    return decorator


@pytest.fixture
def agentci_trace(request):
    """
    Fixture that activates trace capture for the duration of the test.
    """
    # Check if test function has @agentci.test config
    config = getattr(request.function, "_agentci_config", {})
    test_name = request.node.name
    
    with TraceContext(test_name=test_name) as ctx:
        yield ctx.trace
        
    # Post-test: validation and assertions
    # This runs AFTER the test function returns
    trace = ctx.trace
    
    # 1. Cost Budget
    if config.get("max_cost_usd") is not None:
        limit = config["max_cost_usd"]
        if trace.total_cost_usd > limit:
            pytest.fail(f"Cost ${trace.total_cost_usd:.4f} exceeded budget ${limit:.4f}")

    # 2. Assertions (if any defined in decorator)
    # Note: Evaluators are in agentci.assertions
    from .assertions import evaluate_assertion
    if config.get("assertions"):
        for a in config["assertions"]:
            if isinstance(a, dict):
                a = Assertion(**a) # Convert dict to model
            passed, msg = evaluate_assertion(a, trace)
            if not passed:
                pytest.fail(msg)

    # 3. Golden Trace Diff (if path provided)
    if config.get("golden_trace"):
        # TODO: Implement diff logic here using diff_engine
        pass


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "agentci: mark test as an AgentCI agent test"
    )
