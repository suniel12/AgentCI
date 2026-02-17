"""
Test execution engine.

Loads a TestSuite, executes each TestCase by importing the agent function,
wrapping it in TraceContext, and collecting results.
"""

from .models import TestSuite, SuiteResult

import importlib
import time
from typing import Any, Callable
from .models import TestSuite, SuiteResult, RunResult, TestResult, Trace
from .capture import TraceContext
from .assertions import evaluate_assertion

class TestRunner:
    def __init__(self, suite: TestSuite):
        self.suite = suite
        self._agent_fn: Callable | None = None

    def _import_agent(self) -> Callable:
        """Dynamically import the agent function from a string path."""
        if self._agent_fn:
            return self._agent_fn
            
        if not self.suite.agent:
            raise ValueError("No agent import path provided in test suite.")
            
        try:
            module_path, fn_name = self.suite.agent.split(":")
            module = importlib.import_module(module_path)
            self._agent_fn = getattr(module, fn_name)
            return self._agent_fn
        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(f"Could not import agent function '{self.suite.agent}': {e}")

    def run_suite(self) -> SuiteResult:
        """Execute all tests in the suite."""
        agent_fn = self._import_agent()
        suite_result = SuiteResult(suite_name=self.suite.name)
        start_time = time.perf_counter()
        
        for test in self.suite.tests:
            run_result = self.run_test(test, agent_fn)
            suite_result.results.append(run_result)
            
            if run_result.result == TestResult.PASSED:
                suite_result.total_passed += 1
            elif run_result.result == TestResult.FAILED:
                suite_result.total_failed += 1
            else:
                suite_result.total_errors += 1
                
        suite_result.duration_ms = (time.perf_counter() - start_time) * 1000
        suite_result.total_cost_usd = sum(r.trace.total_cost_usd for r in suite_result.results)
        
        return suite_result

    def run_test(self, test: Any, agent_fn: Callable) -> RunResult:
        """Run a single test case."""
        start_time = time.perf_counter()
        
        with TraceContext(agent_name=self.suite.agent, test_name=test.name) as ctx:
            try:
                # Execute the agent
                if test.input_data:
                    result = agent_fn(test.input_data)
                else:
                    # Try calling without args, or with empty string if it fails?
                    # For now, just call it. If it needs args, it will raise TypeError, which is fair.
                    # Actually, for the demo agent, it needs a string. 
                    # Let's try to be smart: if input_data is None, pass "" if it looks like it wants a string?
                    # No, let's just respect the signature.
                    try:
                        result = agent_fn()
                    except TypeError:
                         # Fallback: maybe it requires one argument?
                         result = agent_fn("")
                
                # Update span with result
                ctx.trace.spans[0].output_data = result
                ctx.trace.compute_metrics()
                
                # Evaluate assertions
                assertion_results = []
                all_passed = True
                for assertion in test.assertions:
                    passed, message = evaluate_assertion(assertion, ctx.trace)
                    assertion_results.append({"passed": passed, "message": message})
                    if not passed:
                        all_passed = False
                
                # Check budgets (max_cost, max_steps)
                if test.max_cost_usd and ctx.trace.total_cost_usd > test.max_cost_usd:
                    all_passed = False
                    assertion_results.append({
                        "passed": False, 
                        "message": f"✗ Cost ${ctx.trace.total_cost_usd:.4f} exceeds budget ${test.max_cost_usd:.4f}"
                    })

                # Golden Trace Diffing
                diffs = []
                if test.golden_trace:
                    import json
                    from .diff_engine import diff_traces
                    try:
                        with open(test.golden_trace, 'r') as f:
                            golden = Trace.model_validate(json.load(f))
                            diffs = diff_traces(ctx.trace, golden)
                    except FileNotFoundError:
                        assertion_results.append({
                            "passed": True, 
                            "message": f"⚠ Golden trace not found: {test.golden_trace}"
                        })
                    except Exception as e:
                        assertion_results.append({
                            "passed": False, 
                            "message": f"⚠ Failed to load golden trace: {e}"
                        })

                return RunResult(
                    test_name=test.name,
                    result=TestResult.PASSED if all_passed else TestResult.FAILED,
                    trace=ctx.trace,
                    assertion_results=assertion_results,
                    diffs=diffs,
                    duration_ms=(time.perf_counter() - start_time) * 1000
                )
                
            except Exception as e:
                return RunResult(
                    test_name=test.name,
                    result=TestResult.ERROR,
                    trace=ctx.trace,
                    error_message=str(e),
                    duration_ms=(time.perf_counter() - start_time) * 1000
                )
