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

# GitHub limits visible inline annotations per job; exceeding this silently
# drops annotations. Warnings are budget-capped; errors are always emitted.
MAX_INLINE_ANNOTATIONS: int = 10


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
    - ::error  → red, blocks merge (no cap — correctness failures always visible)
    - ::warning → yellow, non-blocking (capped at MAX_INLINE_ANNOTATIONS)

    Overflow warnings beyond the cap are written to GITHUB_STEP_SUMMARY so
    they remain accessible without silently disappearing.
    """
    warning_count: int = 0
    overflow_warnings: list[str] = []

    for r in results:
        query_short = r.query[:60]

        # Hard fails → always emit as ::error (no cap)
        if r.correctness.status == LayerStatus.FAIL:
            for msg in r.correctness.messages:
                print(f"::error file={spec_file}::[CORRECTNESS] {query_short}: {msg}")

        if r.path.status == LayerStatus.FAIL:
            for msg in r.path.messages:
                print(f"::error file={spec_file}::[PATH] {query_short}: {msg}")

        # Soft warnings → budget-capped
        if r.path.status == LayerStatus.WARN:
            for msg in r.path.messages:
                annotation = f"[PATH] {query_short}: {msg}"
                if warning_count < MAX_INLINE_ANNOTATIONS:
                    print(f"::warning file={spec_file}::{annotation}")
                    warning_count += 1
                else:
                    overflow_warnings.append(annotation)

        if r.cost.status == LayerStatus.WARN:
            for msg in r.cost.messages:
                annotation = f"[COST] {query_short}: {msg}"
                if warning_count < MAX_INLINE_ANNOTATIONS:
                    print(f"::warning file={spec_file}::{annotation}")
                    warning_count += 1
                else:
                    overflow_warnings.append(annotation)

    if overflow_warnings:
        _write_step_summary(overflow_warnings)


def _write_step_summary(messages: list[str]) -> None:
    """Write overflow warning messages to the GitHub Actions step summary.

    Appends a markdown table to $GITHUB_STEP_SUMMARY so that warnings
    beyond MAX_INLINE_ANNOTATIONS remain accessible in the Actions UI.
    """
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    try:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write("\n## AgentCI — Additional Warnings\n\n")
            fh.write("| # | Warning |\n")
            fh.write("|---|--------|\n")
            for i, msg in enumerate(messages, start=1):
                fh.write(f"| {i} | {msg} |\n")
    except OSError:
        pass  # Step summary write failure is non-fatal


# ── Console Output ─────────────────────────────────────────────────────────────


def _emit_console(results: list[QueryResult]) -> None:
    """Rich console output with three-tier report per query."""
    for r in results:
        print(f"\n{'=' * 60}")
        print(f"Query: {r.query}")
        _print_layer("CORRECTNESS", r.correctness, fail_icon="❌", pass_icon="✅")
        _print_layer("PATH", r.path, fail_icon="⚠️", pass_icon="📈", warn_icon="⚠️")
        _print_layer("COST", r.cost, fail_icon="⚠️", pass_icon="💰", warn_icon="⚠️")

        if r.hard_fail and getattr(r, "trace", None):
            _print_answer_preview(r.trace)
            _print_trace_summary(r.trace)

    total = len(results)
    passed = sum(1 for r in results if not r.hard_fail)
    warned = sum(1 for r in results if r.has_warnings and not r.hard_fail)
    failed = sum(1 for r in results if r.hard_fail)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed  |  {warned} warnings  |  {failed} failures")

def _print_answer_preview(trace: Any) -> None:
    """Show a truncated preview of the extracted answer for failed queries."""
    answer = ""
    meta_output = getattr(trace, "metadata", {}).get("final_output")
    if meta_output is not None:
        answer = str(meta_output)
    elif getattr(trace, "spans", None):
        last_span = trace.spans[-1]
        output = getattr(last_span, "output_data", None)
        if output is not None:
            answer = str(output)

    if not answer:
        print("  [ANSWER] (empty — no answer extracted from trace)")
        return

    # Collapse whitespace for compact display
    preview = " ".join(answer.split())
    max_len = 300
    if len(preview) > max_len:
        preview = preview[:max_len] + "..."
    print(f"  [ANSWER] {preview}")


def _print_trace_summary(trace: Any) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree
    from rich.text import Text

    console = Console()
    tree = Tree("[bold magenta]Trace Execution Summary[/]")
    for span in trace.spans:
        label = Text(f"{span.name} ", style="cyan bold")
        if span.kind == "tool_call":
            args = str(span.input_data)
            if len(args) > 80:
                args = args[:77] + "..."
            label.append(f"({args})", style="dim")
        else:
            label.append(f"[{span.kind}]", style="magenta dim")

        node = tree.add(label)
        if span.stop_reason == "error":
            node.add(Text(f"ERROR: {span.stop_reason}", style="bold red"))
        elif span.output_data:
            out = str(span.output_data)
            # Remove newlines for compact display
            out = " ".join(out.splitlines())
            if len(out) > 120:
                out = out[:117] + "..."
            node.add(Text(out, style="green"))

    console.print(Panel(tree, border_style="magenta", expand=False))

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

    # SKIP: inline reason on the status line, no bullet points
    if status == LayerStatus.SKIP:
        reason = layer_result.messages[0] if layer_result.messages else "not configured"
        print(f"  {icon}  {name}: SKIP ({reason})")
        return

    print(f"  {icon}  {name}: {status.value.upper()}")
    for msg in layer_result.messages:
        if status == LayerStatus.PASS:
            print(f"       ✓ {msg}")
        else:
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
