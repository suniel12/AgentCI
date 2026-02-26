"""
AgentCI v2 Reporter.

Generates output in multiple formats and returns the appropriate exit code.

Exit code contract:
    0  — All correctness layers pass (warnings are annotations, not failures)
    1  — Any correctness failure or forbidden tool violation
    2  — Runtime/infrastructure error (set by caller)

Format options:
    console     — Human-readable rich output (default)
    github      — GitHub Actions annotations (::error:: / ::warning::)
    json        — Machine-readable JSON for dashboards
    prometheus  — Prometheus exposition format for Grafana
"""

from __future__ import annotations

import json
import os
from typing import Any

from agentci.engine.results import LayerStatus, QueryResult


# ── Public API ─────────────────────────────────────────────────────────────────


def report_results(
    results: list[QueryResult],
    format: str = "console",
    spec_file: str = "agentci_spec.yaml",
) -> int:
    """Generate output and return the appropriate exit code.

    Args:
        results:   List of QueryResult from the evaluation engine.
        format:    Output format: 'console', 'github', 'json', 'prometheus'.
        spec_file: Path to the spec file (used in GitHub annotation file references).

    Returns:
        Exit code: 0 = pass, 1 = correctness fail.
    """
    has_hard_failures = any(r.hard_fail for r in results)

    # Always emit annotations when running in GitHub Actions
    if format == "github" or _is_github_actions():
        _emit_github_annotations(results, spec_file)

    if format == "json":
        _emit_json(results)
    elif format == "prometheus":
        _emit_prometheus(results)
    else:
        _emit_console(results)

    return 1 if has_hard_failures else 0


# ── GitHub Actions Annotations ─────────────────────────────────────────────────


def _is_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _emit_github_annotations(results: list[QueryResult], spec_file: str) -> None:
    """Emit GitHub Actions workflow commands for inline PR feedback.

    Format: ::level file=<path>::<message>
    - ::error  → red, blocks merge
    - ::warning → yellow, non-blocking
    """
    for r in results:
        query_short = r.query[:60]

        if r.correctness.status == LayerStatus.FAIL:
            for msg in r.correctness.messages:
                print(f"::error file={spec_file}::[CORRECTNESS] {query_short}: {msg}")

        if r.path.status in (LayerStatus.FAIL, LayerStatus.WARN):
            level = "error" if r.path.status == LayerStatus.FAIL else "warning"
            for msg in r.path.messages:
                print(f"::{level} file={spec_file}::[PATH] {query_short}: {msg}")

        if r.cost.status == LayerStatus.WARN:
            for msg in r.cost.messages:
                print(f"::warning file={spec_file}::[COST] {query_short}: {msg}")


# ── Console Output ─────────────────────────────────────────────────────────────


def _emit_console(results: list[QueryResult]) -> None:
    """Rich console output with three-tier report per query."""
    for r in results:
        print(f"\n{'=' * 60}")
        print(f"Query: {r.query}")
        _print_layer("CORRECTNESS", r.correctness, fail_icon="❌", pass_icon="✅")
        _print_layer("PATH", r.path, fail_icon="⚠️", pass_icon="📈", warn_icon="⚠️")
        _print_layer("COST", r.cost, fail_icon="⚠️", pass_icon="💰", warn_icon="⚠️")

    total = len(results)
    passed = sum(1 for r in results if not r.hard_fail)
    warned = sum(1 for r in results if r.has_warnings and not r.hard_fail)
    failed = sum(1 for r in results if r.hard_fail)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed  |  {warned} warnings  |  {failed} failures")


def _print_layer(
    name: str,
    layer_result,
    fail_icon: str = "❌",
    pass_icon: str = "✅",
    warn_icon: str = "⚠️",
) -> None:
    status = layer_result.status
    if status == LayerStatus.PASS:
        icon = pass_icon
    elif status == LayerStatus.FAIL:
        icon = fail_icon
    elif status == LayerStatus.WARN:
        icon = warn_icon
    else:
        icon = "—"

    print(f"  {icon}  {name}: {status.value.upper()}")
    if status in (LayerStatus.FAIL, LayerStatus.WARN):
        for msg in layer_result.messages:
            print(f"       • {msg}")


# ── JSON Output ────────────────────────────────────────────────────────────────


def _emit_json(results: list[QueryResult]) -> None:
    """Structured JSON for dashboards and external tooling."""
    output: dict[str, Any] = {
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if not r.hard_fail),
            "failed": sum(1 for r in results if r.hard_fail),
            "warnings": sum(1 for r in results if r.has_warnings),
        },
        "results": [_serialize_result(r) for r in results],
    }
    print(json.dumps(output, indent=2))


def _serialize_result(r: QueryResult) -> dict[str, Any]:
    return {
        "query": r.query,
        "hard_fail": r.hard_fail,
        "has_warnings": r.has_warnings,
        "correctness": {
            "status": r.correctness.status.value,
            "messages": r.correctness.messages,
            "details": r.correctness.details,
        },
        "path": {
            "status": r.path.status.value,
            "messages": r.path.messages,
            "details": r.path.details,
        },
        "cost": {
            "status": r.cost.status.value,
            "messages": r.cost.messages,
            "details": r.cost.details,
        },
    }


# ── Prometheus Output ──────────────────────────────────────────────────────────


def _emit_prometheus(results: list[QueryResult]) -> None:
    """Prometheus exposition format for Grafana dashboards."""
    print("# AgentCI evaluation metrics")
    for r in results:
        label = r.query[:40].replace('"', '\\"').replace("\n", " ")
        ql = f'query="{label}"'

        # Correctness as boolean gauge
        val = 1 if r.correctness.status == LayerStatus.PASS else 0
        print(f'agentci_correctness_pass{{{ql}}} {val}')

        if "tool_recall" in r.path.details:
            print(f'agentci_tool_recall{{{ql}}} {r.path.details["tool_recall"]}')
        if "tool_precision" in r.path.details:
            print(f'agentci_tool_precision{{{ql}}} {r.path.details["tool_precision"]}')
        if "sequence_similarity" in r.path.details:
            print(f'agentci_sequence_similarity{{{ql}}} {r.path.details["sequence_similarity"]}')

        if "actual" in r.cost.details:
            actual = r.cost.details["actual"]
            print(f'agentci_cost_usd{{{ql}}} {actual["cost_usd"]}')
            print(f'agentci_latency_ms{{{ql}}} {actual["latency_ms"]}')
            print(f'agentci_total_tokens{{{ql}}} {actual["total_tokens"]}')
            print(f'agentci_llm_calls{{{ql}}} {actual["llm_calls"]}')
