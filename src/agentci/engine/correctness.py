"""
Correctness Engine — Layer 1 (Hard Fail).

Evaluation order is cost-optimised: deterministic checks run first (free),
LLM-judge calls run last and are skipped if any deterministic check already
failed.

Any single failure in this layer → LayerStatus.FAIL, which maps to:
  - Exit code 1 in the CLI
  - ::error annotation in GitHub Actions
  - CI pipeline blocked (merge blocked)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Optional

from agentci.engine.results import LayerResult, LayerStatus
from agentci.schema.spec_models import CorrectnessSpec

if TYPE_CHECKING:
    pass


def evaluate_correctness(
    answer: str,
    spec: CorrectnessSpec,
    trace: Optional[Any] = None,
    judge_config: Optional[dict[str, Any]] = None,
) -> LayerResult:
    """Evaluate correctness assertions against an agent answer.

    Args:
        answer:       The agent's final text response.
        spec:         CorrectnessSpec defining what to check.
        trace:        Optional raw trace object (used for context in judge calls).
        judge_config: Global judge settings (model, temperature, ensemble).

    Returns:
        LayerResult with PASS or FAIL status.
    """
    failures: list[str] = []
    details: dict[str, Any] = {}

    # ── 1. expected_in_answer (case-insensitive substring) ──────────────────
    if spec.expected_in_answer:
        missing = [t for t in spec.expected_in_answer if t.lower() not in answer.lower()]
        details["expected_in_answer"] = {
            "checked": spec.expected_in_answer,
            "missing": missing,
            "all_found": not missing,
        }
        for term in missing:
            failures.append(f"Expected '{term}' not found in answer")

    # ── 2. not_in_answer (case-insensitive exclusion) ───────────────────────
    if spec.not_in_answer:
        found = [t for t in spec.not_in_answer if t.lower() in answer.lower()]
        details["not_in_answer"] = {
            "checked": spec.not_in_answer,
            "found": found,
            "none_found": not found,
        }
        for term in found:
            failures.append(f"Forbidden term '{term}' found in answer")

    # ── 3. exact_match ───────────────────────────────────────────────────────
    if spec.exact_match is not None:
        matches = answer.strip() == spec.exact_match.strip()
        details["exact_match"] = {"expected": spec.exact_match, "matched": matches}
        if not matches:
            failures.append("Exact match failed")

    # ── 4. regex_match ───────────────────────────────────────────────────────
    if spec.regex_match is not None:
        matched = bool(re.search(spec.regex_match, answer))
        details["regex_match"] = {"pattern": spec.regex_match, "matched": matched}
        if not matched:
            failures.append(f"Regex '{spec.regex_match}' did not match answer")

    # ── 5. json_schema ───────────────────────────────────────────────────────
    if spec.json_schema is not None:
        json_result = _validate_json_schema(answer, spec.json_schema)
        details["json_schema"] = json_result
        if not json_result["valid"]:
            failures.append(f"JSON schema validation failed: {json_result['error']}")

    # ── 6–8: LLM judge calls — only if deterministic checks all passed ───────
    if not failures:
        if spec.llm_judge:
            for rubric in spec.llm_judge:
                result = _run_judge_safe(answer, rubric, judge_config, trace)
                key = f"judge_{rubric.rule[:40]}"
                details[key] = result
                if not result.get("passed", False):
                    failures.append(f"Judge failed: {rubric.rule}")

        if spec.safety_check and not failures:
            result = _run_judge_safe(answer, spec.safety_check, judge_config, trace)
            details["safety"] = result
            if not result.get("passed", False):
                failures.append(f"Safety check failed: {spec.safety_check.rule}")

        if spec.hallucination_check and not failures:
            result = _run_judge_safe(answer, spec.hallucination_check, judge_config, trace)
            details["hallucination"] = result
            if not result.get("passed", False):
                failures.append("Hallucination check failed")

    if failures:
        return LayerResult(status=LayerStatus.FAIL, details=details, messages=failures)
    return LayerResult(
        status=LayerStatus.PASS,
        details=details,
        messages=["All correctness checks passed"],
    )


# ── Internal helpers ───────────────────────────────────────────────────────────


def _validate_json_schema(answer: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Validate that answer is JSON conforming to the given schema."""
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"Invalid JSON: {e}"}

    try:
        import jsonschema  # optional dependency
        jsonschema.validate(parsed, schema)
        return {"valid": True, "error": None}
    except ImportError:
        # jsonschema not installed — just verify it parses as JSON
        return {"valid": True, "error": None, "warning": "jsonschema not installed; schema not validated"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _run_judge_safe(
    answer: str,
    rubric: Any,
    judge_config: Optional[dict[str, Any]],
    trace: Any,
) -> dict[str, Any]:
    """Run a judge call, catching errors and returning a failure dict if needed."""
    from agentci.engine.judge import run_judge
    try:
        return run_judge(answer=answer, rubric=rubric, config=judge_config)
    except Exception as e:
        return {
            "passed": False,
            "score": 0,
            "label": "fail",
            "rationale": f"Judge call failed: {e}",
            "error": str(e),
        }
