"""
Agent CI — Continuous Integration for AI Agents.

Catch cost spikes and logic regressions before production.
"""

from importlib.metadata import version

try:
    __version__ = version("agentci")
except Exception:
    __version__ = "0.0.0"
