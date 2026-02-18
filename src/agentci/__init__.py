"""
Agent CI — Continuous Integration for AI Agents.

Catch cost spikes and logic regressions before production.
"""

try:
    from importlib.metadata import version
    __version__ = version("agentci")
except Exception:
    __version__ = "0.0.0"

# Expose the test decorator
from .pytest_plugin import test
from .capture import TraceContext
