"""
Built-in assertion evaluators.

Each assertion takes a Trace and returns (passed: bool, message: str).
Designed to be composable and extensible.
"""

from .models import Trace, Assertion


def evaluate_assertion(assertion: Assertion, trace: Trace) -> tuple[bool, str]:
    """Dispatch an assertion to its evaluator."""
    evaluators = {
        "tool_called": _assert_tool_called,
        "tool_not_called": _assert_tool_not_called,
        "tool_call_count": _assert_tool_call_count,
        "arg_equals": _assert_arg_equals,
        "arg_contains": _assert_arg_contains,
        "cost_under": _assert_cost_under,
        "steps_under": _assert_steps_under,
        "output_contains": _assert_output_contains,
        "output_not_contains": _assert_output_not_contains,
    }
    
    evaluator = evaluators.get(assertion.type)
    if evaluator is None:
        return False, f"Unknown assertion type: {assertion.type}"
    
    return evaluator(assertion, trace)


def _assert_tool_called(a: Assertion, t: Trace) -> tuple[bool, str]:
    tools = t.tool_call_sequence
    if a.tool in tools:
        return True, f"✓ Tool '{a.tool}' was called"
    return False, f"✗ Tool '{a.tool}' was NOT called. Tools called: {tools}"


def _assert_tool_not_called(a: Assertion, t: Trace) -> tuple[bool, str]:
    tools = t.tool_call_sequence
    if a.tool not in tools:
        return True, f"✓ Tool '{a.tool}' was correctly not called"
    return False, f"✗ Tool '{a.tool}' was called but should not have been"


def _assert_tool_call_count(a: Assertion, t: Trace) -> tuple[bool, str]:
    count = t.tool_call_sequence.count(a.tool)
    expected = int(a.value)
    if count == expected:
        return True, f"✓ Tool '{a.tool}' called {count} time(s)"
    return False, f"✗ Tool '{a.tool}' called {count} time(s), expected {expected}"


def _assert_arg_equals(a: Assertion, t: Trace) -> tuple[bool, str]:
    for tc in t.tool_call_details:
        if tc.tool_name == a.tool:
            actual = tc.arguments.get(a.field)
            if actual == a.value:
                return True, f"✓ {a.tool}.{a.field} == {a.value}"
            return False, f"✗ {a.tool}.{a.field} == {actual}, expected {a.value}"
    return False, f"✗ Tool '{a.tool}' was not called"


def _assert_arg_contains(a: Assertion, t: Trace) -> tuple[bool, str]:
    for tc in t.tool_call_details:
        if tc.tool_name == a.tool:
            actual = str(tc.arguments.get(a.field, ""))
            if str(a.value) in actual:
                return True, f"✓ {a.tool}.{a.field} contains '{a.value}'"
            return False, f"✗ {a.tool}.{a.field} = '{actual}', missing '{a.value}'"
    return False, f"✗ Tool '{a.tool}' was not called"


def _assert_cost_under(a: Assertion, t: Trace) -> tuple[bool, str]:
    if t.total_cost_usd <= a.threshold:
        return True, f"✓ Cost ${t.total_cost_usd:.4f} ≤ ${a.threshold:.4f}"
    return False, f"✗ Cost ${t.total_cost_usd:.4f} > ${a.threshold:.4f} budget"


def _assert_steps_under(a: Assertion, t: Trace) -> tuple[bool, str]:
    if t.total_llm_calls <= int(a.threshold):
        return True, f"✓ LLM calls {t.total_llm_calls} ≤ {int(a.threshold)}"
    return False, f"✗ LLM calls {t.total_llm_calls} > {int(a.threshold)} limit"


def _assert_output_contains(a: Assertion, t: Trace) -> tuple[bool, str]:
    final_output = str(t.spans[-1].output_data) if t.spans else ""
    if str(a.value) in final_output:
        return True, f"✓ Output contains '{a.value}'"
    return False, f"✗ Output missing '{a.value}'"


def _assert_output_not_contains(a: Assertion, t: Trace) -> tuple[bool, str]:
    final_output = str(t.spans[-1].output_data) if t.spans else ""
    if str(a.value) not in final_output:
        return True, f"✓ Output correctly excludes '{a.value}'"
    return False, f"✗ Output unexpectedly contains '{a.value}'"
