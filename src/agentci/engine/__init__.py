"""
AgentCI v2 Evaluation Engine.

Three-layer evaluation: Correctness (hard fail) → Path (soft warn) → Cost (soft warn).
"""

from .results import LayerResult, LayerStatus, QueryResult

__all__ = [
    "LayerResult",
    "LayerStatus",
    "QueryResult",
    "evaluate_query",
    "evaluate_spec",
]


def __getattr__(name: str):
    if name in ("evaluate_query", "evaluate_spec"):
        from .runner import evaluate_query, evaluate_spec  # noqa: F401
        return locals()[name]
    raise AttributeError(f"module 'agentci.engine' has no attribute {name!r}")
