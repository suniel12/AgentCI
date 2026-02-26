"""
Unit tests for the Correctness Engine (Layer 1 — Hard Fail).

All LLM judge calls are mocked. No real API calls in this file.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentci.engine.correctness import evaluate_correctness
from agentci.engine.results import LayerStatus
from agentci.schema.spec_models import CorrectnessSpec, JudgeRubric


# ── Helpers ────────────────────────────────────────────────────────────────────


def judge_pass() -> dict:
    return {"passed": True, "score": 4, "label": "pass", "rationale": "Good"}


def judge_fail() -> dict:
    return {"passed": False, "score": 2, "label": "fail", "rationale": "Poor"}


def spec(**kwargs) -> CorrectnessSpec:
    return CorrectnessSpec(**kwargs)


# ── expected_in_answer ────────────────────────────────────────────────────────


class TestExpectedInAnswer:
    def test_term_found_passes(self):
        result = evaluate_correctness("pip install agentci", spec(expected_in_answer=["pip"]))
        assert result.status == LayerStatus.PASS

    def test_term_not_found_fails(self):
        result = evaluate_correctness("brew install agentci", spec(expected_in_answer=["pip"]))
        assert result.status == LayerStatus.FAIL
        assert any("pip" in m for m in result.messages)

    def test_case_insensitive(self):
        result = evaluate_correctness("PIP INSTALL AGENTCI", spec(expected_in_answer=["pip install"]))
        assert result.status == LayerStatus.PASS

    def test_multiple_terms_all_found(self):
        result = evaluate_correctness("pip install agentci", spec(expected_in_answer=["pip", "agentci"]))
        assert result.status == LayerStatus.PASS

    def test_multiple_terms_one_missing_fails(self):
        result = evaluate_correctness("pip install agentci", spec(expected_in_answer=["pip", "poetry"]))
        assert result.status == LayerStatus.FAIL


# ── not_in_answer ──────────────────────────────────────────────────────────────


class TestNotInAnswer:
    def test_absent_term_passes(self):
        result = evaluate_correctness("I cannot answer that", spec(not_in_answer=["degrees", "forecast"]))
        assert result.status == LayerStatus.PASS

    def test_present_term_fails(self):
        result = evaluate_correctness("Tokyo is 25 degrees today", spec(not_in_answer=["degrees"]))
        assert result.status == LayerStatus.FAIL
        assert any("degrees" in m for m in result.messages)

    def test_case_insensitive(self):
        result = evaluate_correctness("DEGREES Celsius", spec(not_in_answer=["degrees"]))
        assert result.status == LayerStatus.FAIL


# ── exact_match ────────────────────────────────────────────────────────────────


class TestExactMatch:
    def test_exact_match_passes(self):
        result = evaluate_correctness("yes", spec(exact_match="yes"))
        assert result.status == LayerStatus.PASS

    def test_exact_match_whitespace_stripped(self):
        result = evaluate_correctness("  yes  ", spec(exact_match="yes"))
        assert result.status == LayerStatus.PASS

    def test_exact_match_fails_on_difference(self):
        result = evaluate_correctness("no", spec(exact_match="yes"))
        assert result.status == LayerStatus.FAIL


# ── regex_match ────────────────────────────────────────────────────────────────


class TestRegexMatch:
    def test_regex_matches_passes(self):
        result = evaluate_correctness("Call us at 555-1234", spec(regex_match=r"\d{3}-\d{4}"))
        assert result.status == LayerStatus.PASS

    def test_regex_no_match_fails(self):
        result = evaluate_correctness("No phone number here", spec(regex_match=r"\d{3}-\d{4}"))
        assert result.status == LayerStatus.FAIL

    def test_regex_pattern_in_message(self):
        result = evaluate_correctness("no", spec(regex_match=r"\d+"))
        assert result.status == LayerStatus.FAIL
        assert r"\d+" in result.messages[0]


# ── json_schema ────────────────────────────────────────────────────────────────


class TestJsonSchema:
    def test_valid_json_matching_schema(self):
        result = evaluate_correctness(
            '{"name": "Alice"}',
            spec(json_schema={"type": "object", "properties": {"name": {"type": "string"}}}),
        )
        assert result.status == LayerStatus.PASS

    def test_invalid_json_fails(self):
        result = evaluate_correctness(
            "not json at all",
            spec(json_schema={"type": "object"}),
        )
        assert result.status == LayerStatus.FAIL
        assert any("JSON" in m for m in result.messages)


# ── LLM judge ─────────────────────────────────────────────────────────────────


class TestLLMJudge:
    def test_judge_pass_returns_pass(self):
        rubric = JudgeRubric(rule="Response is helpful", threshold=0.6)
        s = spec(llm_judge=[rubric])
        with patch("agentci.engine.correctness._run_judge_safe", return_value=judge_pass()):
            result = evaluate_correctness("Helpful answer", s)
        assert result.status == LayerStatus.PASS

    def test_judge_fail_returns_fail(self):
        rubric = JudgeRubric(rule="Response is helpful", threshold=0.6)
        s = spec(llm_judge=[rubric])
        with patch("agentci.engine.correctness._run_judge_safe", return_value=judge_fail()):
            result = evaluate_correctness("Bad answer", s)
        assert result.status == LayerStatus.FAIL

    def test_judge_skipped_when_deterministic_already_failed(self):
        rubric = JudgeRubric(rule="Response is helpful", threshold=0.6)
        s = spec(expected_in_answer=["pip"], llm_judge=[rubric])
        with patch("agentci.engine.correctness._run_judge_safe") as mock_judge:
            result = evaluate_correctness("No keyword here", s)
        mock_judge.assert_not_called()
        assert result.status == LayerStatus.FAIL

    def test_multiple_judges_all_must_pass(self):
        rubrics = [
            JudgeRubric(rule="Rule 1", threshold=0.6),
            JudgeRubric(rule="Rule 2", threshold=0.6),
        ]
        s = spec(llm_judge=rubrics)
        with patch(
            "agentci.engine.correctness._run_judge_safe",
            side_effect=[judge_pass(), judge_fail()],
        ):
            result = evaluate_correctness("Answer", s)
        assert result.status == LayerStatus.FAIL


# ── safety and hallucination sub-checks ───────────────────────────────────────


class TestSafetyAndHallucinationChecks:
    def test_safety_check_pass(self):
        rubric = JudgeRubric(rule="Response is safe", threshold=0.8)
        s = spec(safety_check=rubric)
        with patch("agentci.engine.correctness._run_judge_safe", return_value=judge_pass()):
            result = evaluate_correctness("Safe answer", s)
        assert result.status == LayerStatus.PASS

    def test_safety_check_fail(self):
        rubric = JudgeRubric(rule="Response is safe", threshold=0.8)
        s = spec(safety_check=rubric)
        with patch("agentci.engine.correctness._run_judge_safe", return_value=judge_fail()):
            result = evaluate_correctness("Unsafe answer", s)
        assert result.status == LayerStatus.FAIL

    def test_hallucination_check_fail(self):
        rubric = JudgeRubric(rule="No hallucinations", threshold=0.8)
        s = spec(hallucination_check=rubric)
        with patch("agentci.engine.correctness._run_judge_safe", return_value=judge_fail()):
            result = evaluate_correctness("Hallucinated answer", s)
        assert result.status == LayerStatus.FAIL


# ── no spec fields ─────────────────────────────────────────────────────────────


class TestEmptySpec:
    def test_empty_spec_passes(self):
        result = evaluate_correctness("Anything goes", spec())
        assert result.status == LayerStatus.PASS
        assert result.messages == ["All correctness checks passed"]


# ── details populated ─────────────────────────────────────────────────────────


class TestDetailsPopulated:
    def test_expected_in_answer_details(self):
        result = evaluate_correctness("pip install", spec(expected_in_answer=["pip", "brew"]))
        assert "expected_in_answer" in result.details
        assert "brew" in result.details["expected_in_answer"]["missing"]

    def test_not_in_answer_details(self):
        result = evaluate_correctness("hello world", spec(not_in_answer=["hello"]))
        assert "not_in_answer" in result.details
        assert "hello" in result.details["not_in_answer"]["found"]
