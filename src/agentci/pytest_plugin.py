"""
pytest integration plugin.

Provides fixtures and hooks for running AgentCI tests via pytest.
"""

import pytest

@pytest.fixture
def agentci_trace():
    # TODO: Implement trace fixture
    yield None

def pytest_configure(config):
    # TODO: Register markers
    config.addinivalue_line(
        "markers", "agentci: mark test as an AgentCI agent test"
    )
