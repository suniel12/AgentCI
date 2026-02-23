"""
Built-in assertion evaluators.

Each assertion takes a Trace and returns (passed: bool, message: str).
Designed to be composable and extensible.
"""

from agentci.models import Trace, Assertion
from typing import Callable, Any, TypeVar, cast
import os
import json
from functools import wraps
from typing import Callable, Any


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


# --- Phase 1 Learnings: Streamlined Golden Trace Assertions ---

def assert_golden_match(current_trace: Trace, golden_file_path: str, update_golden: bool = False, config_dir: str = ".") -> None:
    """
    Asserts that the current trace matches the saved golden trace.
    If update_golden is True (e.g. from --update-golden CLI flag), overwrites the file instead.
    
    Raises AssertionError if the diff engine finds errors (like tools removed, cost spiked).
    """
    from agentci.diff_engine import diff_traces
    from agentci.models import DiffType
    
    # Resolve the path relative to where tests run
    import pathlib
    resolved_path = pathlib.Path(config_dir) / golden_file_path
    if not resolved_path.suffix:
        resolved_path = resolved_path.with_suffix('.json')
        
    full_path = str(resolved_path)
        
    if update_golden:
        # Create parent directories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(current_trace.model_dump_json(indent=2))
        return

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Golden trace not found: {full_path}. Run with --update-golden to create it.")

    with open(full_path, "r") as f:
        golden_data = json.load(f)
        golden_trace = Trace.model_validate(golden_data)

    diffs = diff_traces(current_trace, golden_trace)
    errors = [d.message for d in diffs if d.severity == "error"]

    if errors:
        error_msg = "\n".join(f"- {e}" for e in errors)
        raise AssertionError(f"Agent Trace diverged from Golden Trace ({golden_file_path}):\n{error_msg}")


# --- Phase 0 Learnings: Declarative Guardrails ---

T = TypeVar('T')

def assert_budget(max_cost: float = float('inf'), max_tokens: int = 1000000) -> Callable[[Any], Any]:
    """
    Decorator to wrap an agent execution function and assert it stays within budget.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = func(*args, **kwargs)
            if not isinstance(trace, Trace):
                return trace
            
            if trace.total_cost_usd > max_cost:
                raise AssertionError(f"Budget Exceeded: Agent cost ${trace.total_cost_usd:.4f} > max allowed ${max_cost:.4f}")
            
            if trace.total_tokens > max_tokens:
                raise AssertionError(f"Budget Exceeded: Agent used {trace.total_tokens} tokens > max allowed {max_tokens}")
            
            return trace
        return wrapper
    return decorator


def truncate_tokens(max_tokens: int = 8000) -> Callable[[Any], Any]:
    """
    Decorator to automatically truncate a tool's string output to prevent LLM context limit blowouts.
    Roughly estimates tokens as characters // 4.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            
            # Simple heuristic truncation
            max_chars = max_tokens * 4
            
            if isinstance(result, str):
                if len(result) > max_chars:
                    return result[:max_chars] + f"\n\n...[TRUNCATED: Exceeded {max_tokens} token limit]"
                return result
            
            if isinstance(result, dict):
                import json
                try:
                    str_result = json.dumps(result)
                    if len(str_result) > max_chars:
                        return {"error": f"Tool output truncated. Exceeded {max_tokens} token limit. Original size: {len(str_result)} chars."}
                except (TypeError, ValueError):
                    pass
            
            return result
        return wrapper
    return decorator
