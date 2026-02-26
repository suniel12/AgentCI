"""
Unit tests for the AgentCI v2 Reporter.

Uses capsys to capture stdout output. Mocks os.environ for GitHub detection.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agentci.engine.reporter import report_results
from agentci.engine.results import LayerResult, LayerStatus, QueryResult


# ── Helpers ────────────────────────────────────────────────────────────────────


def skip_layer() -> LayerResult:
    return LayerResult(status=LayerStatus.SKIP, details={}, messages=["No spec"])


def pass_layer() -> LayerResult:
    return LayerResult(
        status=LayerStatus.PASS,
        details={},
        messages=["All checks passed"],
    )


def fail_layer(msg: str = "Something failed") -> LayerResult:
    return LayerResult(status=LayerStatus.FAIL, details={}, messages=[msg])


def warn_layer(msg: str = "Budget exceeded") -> LayerResult:
    return LayerResult(status=LayerStatus.WARN, details={}, messages=[msg])


def make_result(
    query: str = "Test query",
    correctness: LayerResult | None = None,
    path: LayerResult | None = None,
    cost: LayerResult | None = None,
) -> QueryResult:
    return QueryResult(
        query=query,
        correctness=correctness or pass_layer(),
        path=path or skip_layer(),
        cost=cost or skip_layer(),
    )


# ── Exit Codes ────────────────────────────────────────────────────────────────


class TestExitCodes:
    def test_all_pass_returns_zero(self, capsys):
        results = [make_result()]
        code = report_results(results)
        assert code == 0

    def test_correctness_fail_returns_one(self, capsys):
        results = [make_result(correctness=fail_layer())]
        code = report_results(results)
        assert code == 1

    def test_path_warn_only_returns_zero(self, capsys):
        results = [make_result(path=warn_layer())]
        code = report_results(results)
        assert code == 0

    def test_cost_warn_only_returns_zero(self, capsys):
        results = [make_result(cost=warn_layer())]
        code = report_results(results)
        assert code == 0

    def test_any_hard_fail_returns_one(self, capsys):
        results = [
            make_result(query="Q1"),
            make_result(query="Q2", correctness=fail_layer()),
            make_result(query="Q3"),
        ]
        code = report_results(results)
        assert code == 1

    def test_path_fail_forbidden_tool_returns_one(self, capsys):
        # Path layer FAIL (forbidden tool) counts as hard fail
        results = [make_result(correctness=fail_layer("forbidden tool used"))]
        code = report_results(results)
        assert code == 1

    def test_empty_results_returns_zero(self, capsys):
        code = report_results([])
        assert code == 0


# ── GitHub Annotations ────────────────────────────────────────────────────────


class TestGitHubAnnotations:
    def _env_github(self):
        return patch.dict("os.environ", {"GITHUB_ACTIONS": "true"})

    def test_correctness_fail_emits_error_annotation(self, capsys):
        with self._env_github():
            results = [make_result(correctness=fail_layer("Expected 'pip' not found"))]
            report_results(results, format="console", spec_file="spec.yaml")
        out = capsys.readouterr().out
        assert "::error file=spec.yaml::" in out
        assert "CORRECTNESS" in out

    def test_path_warn_emits_warning_annotation(self, capsys):
        with self._env_github():
            results = [make_result(path=warn_layer("Tool calls: 5 > max 3"))]
            report_results(results, format="console", spec_file="myspec.yaml")
        out = capsys.readouterr().out
        assert "::warning file=myspec.yaml::" in out
        assert "PATH" in out

    def test_cost_warn_emits_warning_annotation(self, capsys):
        with self._env_github():
            results = [make_result(cost=warn_layer("Tokens: 600 > max 500"))]
            report_results(results, format="console")
        out = capsys.readouterr().out
        assert "::warning" in out
        assert "COST" in out

    def test_github_format_always_emits_annotations(self, capsys):
        """format='github' emits annotations even without GITHUB_ACTIONS env."""
        results = [make_result(correctness=fail_layer("fail"))]
        report_results(results, format="github", spec_file="spec.yaml")
        out = capsys.readouterr().out
        assert "::error" in out

    def test_no_annotations_when_all_pass_and_not_github(self, capsys):
        with patch.dict("os.environ", {}, clear=True):
            results = [make_result()]
            report_results(results, format="console")
        out = capsys.readouterr().out
        assert "::" not in out


# ── JSON Output ────────────────────────────────────────────────────────────────


class TestJSONOutput:
    def test_json_contains_summary(self, capsys):
        results = [make_result(query="Q1"), make_result(query="Q2", correctness=fail_layer())]
        with patch.dict("os.environ", {}, clear=True):
            report_results(results, format="json")
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["summary"]["total"] == 2
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1

    def test_json_contains_per_result_details(self, capsys):
        results = [make_result(query="How do I install?")]
        with patch.dict("os.environ", {}, clear=True):
            report_results(results, format="json")
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data["results"]) == 1
        r = data["results"][0]
        assert r["query"] == "How do I install?"
        assert "correctness" in r
        assert "path" in r
        assert "cost" in r

    def test_json_summary_counts_warnings(self, capsys):
        results = [make_result(cost=warn_layer())]
        with patch.dict("os.environ", {}, clear=True):
            report_results(results, format="json")
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["summary"]["warnings"] == 1


# ── Console Output ─────────────────────────────────────────────────────────────


class TestConsoleOutput:
    def test_console_contains_query_text(self, capsys):
        with patch.dict("os.environ", {}, clear=True):
            results = [make_result(query="How do I install AgentCI?")]
            report_results(results)
        out = capsys.readouterr().out
        assert "How do I install AgentCI?" in out

    def test_console_summary_line_present(self, capsys):
        with patch.dict("os.environ", {}, clear=True):
            results = [make_result(), make_result(correctness=fail_layer())]
            report_results(results)
        out = capsys.readouterr().out
        assert "Results:" in out


# ── Prometheus Output ──────────────────────────────────────────────────────────


class TestPrometheusOutput:
    def test_prometheus_emits_correctness_gauge(self, capsys):
        with patch.dict("os.environ", {}, clear=True):
            results = [make_result(query="Install query")]
            report_results(results, format="prometheus")
        out = capsys.readouterr().out
        assert "agentci_correctness_pass" in out

    def test_prometheus_gauge_is_zero_on_fail(self, capsys):
        with patch.dict("os.environ", {}, clear=True):
            results = [make_result(query="Q", correctness=fail_layer())]
            report_results(results, format="prometheus")
        out = capsys.readouterr().out
        assert "agentci_correctness_pass" in out
        assert "} 0" in out

    def test_prometheus_emits_cost_metrics_when_present(self, capsys):
        cost_result = LayerResult(
            status=LayerStatus.PASS,
            details={"actual": {"cost_usd": 0.001, "latency_ms": 500.0, "total_tokens": 100, "llm_calls": 1}},
            messages=["OK"],
        )
        with patch.dict("os.environ", {}, clear=True):
            results = [make_result(cost=cost_result)]
            report_results(results, format="prometheus")
        out = capsys.readouterr().out
        assert "agentci_cost_usd" in out
        assert "agentci_latency_ms" in out
        assert "agentci_total_tokens" in out
