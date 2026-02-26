"""
Unit tests for agentci.engine.judge.

All LLM API calls are mocked — no real network requests made here.
See tests/integration/test_judge_live.py for real API tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentci.engine.judge import (
    JudgeVerdict,
    _build_judge_system_prompt,
    _build_judge_user_prompt,
    _parse_verdict,
    _run_ensemble,
    _score_threshold,
    run_judge,
)
from agentci.schema.spec_models import JudgeRubric


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_rubric(**kwargs) -> JudgeRubric:
    base = {"rule": "Response is helpful and accurate", "threshold": 0.6}
    base.update(kwargs)
    return JudgeRubric(**base)


def make_verdict(score: int = 4, label: str = "pass", rationale: str = "Good") -> JudgeVerdict:
    return JudgeVerdict(score=score, label=label, rationale=rationale)


# ── _score_threshold ──────────────────────────────────────────────────────────


class TestScoreThreshold:
    def test_zero_maps_to_one(self):
        assert _score_threshold(0.0) == 1

    def test_half_maps_to_three(self):
        assert _score_threshold(0.5) == 3

    def test_point_eight_maps_to_four(self):
        assert _score_threshold(0.8) == 4

    def test_one_maps_to_five(self):
        assert _score_threshold(1.0) == 5

    def test_point_two_maps_to_one(self):
        assert _score_threshold(0.2) == 1

    def test_point_six_maps_to_three(self):
        assert _score_threshold(0.6) == 3

    def test_point_seven_maps_to_four(self):
        assert _score_threshold(0.7) == 4


# ── _build_judge_system_prompt ────────────────────────────────────────────────


class TestBuildSystemPrompt:
    def test_contains_rubric_rule(self):
        rubric = make_rubric(rule="Check for accuracy")
        prompt = _build_judge_system_prompt(rubric)
        assert "Check for accuracy" in prompt

    def test_requires_json_output(self):
        rubric = make_rubric()
        prompt = _build_judge_system_prompt(rubric)
        assert "score" in prompt and "label" in prompt and "rationale" in prompt

    def test_includes_scale_anchors(self):
        rubric = make_rubric(scale=["1: Bad", "5: Perfect"])
        prompt = _build_judge_system_prompt(rubric)
        assert "1: Bad" in prompt
        assert "5: Perfect" in prompt

    def test_includes_few_shot_examples(self):
        rubric = make_rubric(
            few_shot_examples=[{"input": "q", "output": "a", "score": 4}]
        )
        prompt = _build_judge_system_prompt(rubric)
        assert "EXAMPLES" in prompt
        assert "score: 4" in prompt or "4" in prompt

    def test_no_scale_no_examples_clean_prompt(self):
        rubric = make_rubric()
        prompt = _build_judge_system_prompt(rubric)
        assert "SCORING ANCHORS" not in prompt
        assert "EXAMPLES" not in prompt


# ── _build_judge_user_prompt ──────────────────────────────────────────────────


class TestBuildUserPrompt:
    def test_contains_answer(self):
        rubric = make_rubric()
        prompt = _build_judge_user_prompt("My answer here", rubric, context=None)
        assert "My answer here" in prompt

    def test_contains_context_when_provided(self):
        rubric = make_rubric()
        prompt = _build_judge_user_prompt("answer", rubric, context="Retrieved doc")
        assert "Retrieved doc" in prompt
        assert "RETRIEVED CONTEXT" in prompt

    def test_no_context_section_when_none(self):
        rubric = make_rubric()
        prompt = _build_judge_user_prompt("answer", rubric, context=None)
        assert "RETRIEVED CONTEXT" not in prompt


# ── _parse_verdict ────────────────────────────────────────────────────────────


class TestParseVerdict:
    def test_parses_valid_json(self):
        raw = '{"score": 4, "label": "pass", "rationale": "Good response"}'
        verdict = _parse_verdict(raw)
        assert verdict.score == 4
        assert verdict.label == "pass"
        assert verdict.rationale == "Good response"

    def test_parses_json_in_markdown_block(self):
        raw = '```json\n{"score": 3, "label": "borderline", "rationale": "OK"}\n```'
        verdict = _parse_verdict(raw)
        assert verdict.score == 3

    def test_fallback_on_invalid_json(self):
        verdict = _parse_verdict("This is not JSON at all")
        assert verdict.label == "fail"
        assert verdict.score == 1
        assert "Failed to parse" in verdict.rationale


# ── run_judge ─────────────────────────────────────────────────────────────────


class TestRunJudge:
    def _mock_call_judge(self, verdict: JudgeVerdict):
        """Return a context manager that patches _call_judge."""
        return patch(
            "agentci.engine.judge._call_judge",
            return_value=verdict,
        )

    def test_passes_when_score_meets_threshold(self):
        rubric = make_rubric(threshold=0.6)  # threshold score = 3
        verdict = make_verdict(score=4, label="pass")
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["passed"] is True
        assert result["score"] == 4

    def test_fails_when_score_below_threshold(self):
        rubric = make_rubric(threshold=0.8)  # threshold score = 4
        verdict = make_verdict(score=2, label="fail")
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["passed"] is False

    def test_fails_when_score_equals_threshold_minus_one(self):
        rubric = make_rubric(threshold=0.6)  # threshold = 3
        verdict = make_verdict(score=2, label="fail")
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["passed"] is False

    def test_passes_when_score_exactly_at_threshold(self):
        rubric = make_rubric(threshold=0.6)  # threshold = 3
        verdict = make_verdict(score=3, label="pass")
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["passed"] is True

    def test_result_contains_expected_keys(self):
        rubric = make_rubric()
        verdict = make_verdict(score=4)
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert "passed" in result
        assert "score" in result
        assert "label" in result
        assert "rationale" in result
        assert "model" in result

    def test_uses_default_model(self):
        rubric = make_rubric()
        verdict = make_verdict(score=4)
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["model"] == "claude-sonnet-4-6"

    def test_uses_custom_model_from_config(self):
        rubric = make_rubric()
        verdict = make_verdict(score=4)
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric, config={"model": "gpt-4o-mini"})
        assert result["model"] == "gpt-4o-mini"

    def test_ensemble_not_triggered_by_default(self):
        rubric = make_rubric()
        verdict = make_verdict(score=4)
        with self._mock_call_judge(verdict) as mock:
            run_judge("answer", rubric)
        mock.assert_called_once()  # Only one call, no ensemble


# ── _run_ensemble ─────────────────────────────────────────────────────────────


class TestRunEnsemble:
    def _ensemble_config(self, models=None):
        return {
            "enabled": True,
            "models": models or ["model-a", "model-b", "model-c"],
            "strategy": "majority_vote",
        }

    def test_majority_pass(self):
        rubric = make_rubric(threshold=0.6)  # threshold score = 3
        verdicts = [
            make_verdict(score=4, label="pass"),
            make_verdict(score=4, label="pass"),
            make_verdict(score=2, label="fail"),
        ]
        with patch("agentci.engine.judge._call_judge", side_effect=verdicts):
            result = _run_ensemble("sys", "user", self._ensemble_config(), rubric)
        assert result["passed"] is True
        assert result["label"] == "pass"

    def test_majority_fail(self):
        rubric = make_rubric(threshold=0.6)
        verdicts = [
            make_verdict(score=2, label="fail"),
            make_verdict(score=2, label="fail"),
            make_verdict(score=4, label="pass"),
        ]
        with patch("agentci.engine.judge._call_judge", side_effect=verdicts):
            result = _run_ensemble("sys", "user", self._ensemble_config(), rubric)
        assert result["passed"] is False
        assert result["label"] == "fail"

    def test_ensemble_returns_avg_score(self):
        rubric = make_rubric(threshold=0.4)
        verdicts = [
            make_verdict(score=4, label="pass"),
            make_verdict(score=2, label="fail"),
            make_verdict(score=3, label="borderline"),
        ]
        with patch("agentci.engine.judge._call_judge", side_effect=verdicts):
            result = _run_ensemble("sys", "user", self._ensemble_config(), rubric)
        assert result["score"] == pytest.approx(3.0)

    def test_ensemble_result_contains_individual_verdicts(self):
        rubric = make_rubric()
        verdicts = [make_verdict(score=4), make_verdict(score=3), make_verdict(score=4)]
        with patch("agentci.engine.judge._call_judge", side_effect=verdicts):
            result = _run_ensemble("sys", "user", self._ensemble_config(), rubric)
        assert "individual_verdicts" in result
        assert len(result["individual_verdicts"]) == 3

    def test_ensemble_calls_all_models(self):
        rubric = make_rubric()
        models = ["model-x", "model-y", "model-z"]
        verdicts = [make_verdict(score=4)] * 3
        with patch("agentci.engine.judge._call_judge", side_effect=verdicts) as mock:
            _run_ensemble("sys", "user", self._ensemble_config(models), rubric)
        assert mock.call_count == 3
