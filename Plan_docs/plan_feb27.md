Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: AgentCI v1 — Auto-Generate Tests on Init

 Context

 Developers who pip install ciagent and run agentci init today get a blank or hardcoded example
 agentci_spec.yaml. They still have to write all test queries by hand. The goal is to make agentci init
 --generate scan the agent project (code, knowledge base, existing tests) and use an LLM to write a real,
 useful agentci_spec.yaml automatically — so the first agentci test just works.

 Two supporting additions ship alongside this: cleaner assertion methods on Trace (trace.called(),
 trace.cost_under(), etc.) and a langgraph_trace() context manager that eliminates the boilerplate developers
 currently write in every test file.

 ---
 Part 1: Trace Assertion Helpers

 File: AgentCI/src/agentci/models.py

 Add five convenience methods to the Trace class. All delegate to existing fields — no new data, just readable
  APIs.

 def called(self, tool_name: str) -> bool:
     """True if tool_name appears anywhere in the execution trace."""
     return tool_name in self.tool_call_sequence

 def never_called(self, tool_name: str) -> bool:
     """True if tool_name was never called."""
     return tool_name not in self.tool_call_sequence

 def loop_count(self, tool_name: str) -> int:
     """Number of times tool_name was called (for loop detection)."""
     return self.tool_call_sequence.count(tool_name)

 def cost_under(self, threshold_usd: float) -> bool:
     """True if total cost is below threshold."""
     return self.total_cost_usd < threshold_usd

 def llm_calls_under(self, count: int) -> bool:
     """True if total LLM calls is below count."""
     return self.total_llm_calls < count

 Insert after the existing tool_call_details property (line ~197).

 ---
 Part 2: LangGraph Trace Shortcut

 File: AgentCI/src/agentci/capture.py

 2a. Add attach() alias on TraceContext:

 def attach(self, state: dict) -> None:
     """Alias for attach_langgraph_state — shorter to type."""
     self.attach_langgraph_state(state)

 2b. Add langgraph_trace() context manager:

 from contextlib import contextmanager

 @contextmanager
 def langgraph_trace(agent_name: str = ""):
     """
     Shortcut context manager for LangGraph agents.

     Usage:
         with langgraph_trace("rag-agent") as ctx:
             output, state = generate_answer_api(query)
             ctx.attach(state)
         trace = ctx.trace
     """
     with TraceContext(agent_name=agent_name) as ctx:
         yield ctx

 File: AgentCI/src/agentci/__init__.py

 Add to re-exports:
 from .capture import langgraph_trace
 And add "langgraph_trace" to __all__.

 ---
 Part 3: agentci init --generate (Main Feature)

 File: AgentCI/src/agentci/cli.py

 3a. Add --generate flag to the init command

 @click.option('--generate', is_flag=True,
               help='Scan project code + knowledge base and auto-generate agentci_spec.yaml using AI')

 3b. Project Scanner

 When --generate is passed, before writing the spec, call a new internal function _scan_project(project_dir)
 that returns a dict of context:

 def _scan_project(project_dir: Path) -> dict:
     """Scan project directory and return context for LLM test generation."""
     context = {
         "agent_files": [],       # Python source files with tool/node definitions
         "knowledge_base": [],    # KB file names + first 200 chars of each
         "existing_tests": [],    # Existing test file contents (truncated)
     }

 Scanning logic:

 1. Agent code: Walk *.py files, skip __pycache__, .venv, tests/. For each file, read and include if it
 contains keywords: @tool, def retrieve, def run, SystemMessage, bind_tools, add_node, add_edge, ChatOpenAI,
 ChatAnthropic. Include file name + first 150 lines.
 2. Knowledge base: Look for directories named any of: knowledge_base, kb, docs, data, knowledge. Read all
 *.md and *.txt files. Include: file path + first 200 characters of content.
 3. Existing tests: Find tests/ or test/ directory. Read *.py files. Include truncated content (first 100
 lines per file) so the LLM can see what's already covered.

 Respect a total context limit of ~12,000 tokens across all scanned content (truncate the longest files first
 if over limit).

 3c. LLM Query Generator

 New internal function _generate_queries(context: dict, runner_path: str, model: str) -> list[dict]:

 Prompt structure:

 You are an expert AI agent test engineer. Given the following agent project context,
 generate a diverse set of test queries for AgentCI's agentci_spec.yaml.

 AGENT CODE:
 {agent_files}

 KNOWLEDGE BASE TOPICS:
 {knowledge_base}

 EXISTING TEST COVERAGE:
 {existing_tests}

 Generate 10-15 test queries covering:
 1. Happy path (in-scope questions the agent should retrieve and answer)
 2. Out-of-scope (questions the agent must decline, max_tool_calls: 0)
 3. Edge cases (mixed intent, compound questions, unanswerable)
 4. At least 2 boundary cases (greeting, completely off-topic)

 For each query, produce a YAML block with:
 - query: the question string
 - description: one sentence explaining what this tests
 - tags: list of tags (smoke, in-scope, out-of-scope, edge-case, etc.)
 - path: expected_tools list OR max_tool_calls: 0 for decline cases
 - correctness: expected_in_answer keywords if deterministic, or llm_judge rule
 - cost: max_llm_calls budget

 Respond ONLY with valid YAML (a list of query objects, no surrounding text).

 LLM call: Use the anthropic library (already in optional deps). Default to claude-sonnet-4-6. Parse the YAML
 response with yaml.safe_load(). If parsing fails, fall back to the --example template and print a warning.

 3d. Updated init command flow with --generate

 agentci init --generate

 1. Print: "Scanning project..."
 2. Call _scan_project(Path(".")) → context dict
 3. Print summary: "Found 3 agent files, 10 knowledge base documents, 2 existing test files"
 4. Ask for runner path (same prompt as today)
 5. Print: "Generating test queries with AI..."
 6. Call _generate_queries(context, runner_path, model="claude-sonnet-4-6")
 7. Print: "Generated 12 test queries"
 8. Write agentci_spec.yaml with generated queries + standard header (version, agent name, judge_config,
 baseline_dir, runner)
 9. Continue with existing GitHub Actions + pre-push hook scaffolding

 If ANTHROPIC_API_KEY is not set, print a clear error:
 Error: --generate requires ANTHROPIC_API_KEY to be set.
 Set it and retry, or use `agentci init --example` for a hardcoded starter.

 ---
 Part 4: Update DemoAgents to Use New API

 File: DemoAgents/examples/rag-agent/demo_runner.py

 from agentci.capture import langgraph_trace
 from agent import generate_answer_api

 def run_for_agentci(query: str):
     with langgraph_trace("rag-agent") as ctx:
         output, state = generate_answer_api(query)
         ctx.attach(state)
         ctx.trace.metadata["final_output"] = str(output)
     return ctx.trace

 File: DemoAgents/examples/rag-agent/tests/test_rag.py

 Remove the 55-line RAGTrace class and trace_decorator. Replace with:

 from agentci.capture import langgraph_trace
 from agent import generate_answer_api

 def run_agent(question: str):
     with langgraph_trace("rag-agent") as ctx:
         output, state = generate_answer_api(question)
         ctx.attach(state)
         ctx.trace.metadata["final_output"] = str(output)
     return ctx.trace

 # Tests become:
 def test_retrieval_triggered_for_knowledge_question():
     trace = run_agent("How do I install AgentCI?")
     assert trace.called("retrieve_docs")

 def test_no_retrieval_for_greeting():
     trace = run_agent("Hello, how are you?")
     assert trace.never_called("retrieve_docs")

 def test_cost_within_budget():
     trace = run_agent("How do I install AgentCI?")
     assert trace.cost_under(0.01)

 def test_max_retries():
     trace = run_agent("What is the name of the top contributor...")
     assert trace.loop_count("rewrite_question") <= 3
     assert trace.cost_under(0.05)

 All other tests (test_grading_step_exists, test_out_of_scope_skips_retrieval, etc.) follow the same pattern —
  replace trace.tools_called with trace.called() / trace.never_called().

 ---
 Part 5: Tests for New Utilities

 File: AgentCI/tests/test_models.py (extend existing)

 Add a test class TestTraceAssertionHelpers:
 - Test called() returns True when tool in sequence, False otherwise
 - Test never_called() is inverse of called()
 - Test loop_count() counts correctly including duplicates
 - Test cost_under() with values above/below threshold
 - Test llm_calls_under() with values above/below count

 File: AgentCI/tests/test_capture.py (new or extend)

 - Test langgraph_trace() returns a context manager that yields a TraceContext
 - Test TraceContext.attach() is equivalent to calling attach_langgraph_state()

 ---
 Part 6: Update Testing Playbook

 File: AgentCI/Plan_docs/testing_playbook.md

 Update Part A (pytest section):
 - Step A2: Replace the old assertion examples with trace.called(), trace.never_called(), trace.cost_under(),
 trace.loop_count()
 - Show the clean run_agent() using langgraph_trace instead of the old TraceContext boilerplate

 Update intro: Add a Step 0 showing agentci init --generate as the starting point before tests are written.

 ---
 Part 7: Version Bump and Publish

 File: AgentCI/pyproject.toml — change version = "0.2.0" to version = "0.3.0"

 File: AgentCI/src/agentci/cli.py line 40 — change __version__ = "0.2.0" to __version__ = "0.3.0"

 Rebuild and upload to PyPI as ciagent 0.3.0:
 rm -rf dist/
 conda run -n agentci python -m build
 twine upload dist/* --username __token__ --password $PYPI_TOKEN

 ---
 Critical Files

 ┌─────────────────────────────────────────────────┬───────────────────────────────────────────────────────┐
 │                      File                       │                        Change                         │
 ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
 │                                                 │ Add called(), never_called(), loop_count(),           │
 │ AgentCI/src/agentci/models.py                   │ cost_under(), llm_calls_under() to Trace after line   │
 │                                                 │ 197                                                   │
 ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
 │ AgentCI/src/agentci/capture.py                  │ Add langgraph_trace() context manager +               │
 │                                                 │ TraceContext.attach() alias                           │
 ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
 │ AgentCI/src/agentci/__init__.py                 │ Re-export langgraph_trace, add to __all__             │
 ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
 │                                                 │ Add --generate flag to init command; add              │
 │ AgentCI/src/agentci/cli.py                      │ _scan_project() and _generate_queries() helpers; bump │
 │                                                 │  __version__ to 0.3.0                                 │
 ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
 │ AgentCI/tests/test_models.py                    │ Add TestTraceAssertionHelpers class                   │
 ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
 │ AgentCI/tests/test_capture.py                   │ Add tests for langgraph_trace and attach()            │
 ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
 │ DemoAgents/examples/rag-agent/demo_runner.py    │ Rewrite using langgraph_trace + ctx.attach()          │
 ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
 │ DemoAgents/examples/rag-agent/tests/test_rag.py │ Remove RAGTrace boilerplate, use trace.called() etc.  │
 ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
 │ AgentCI/Plan_docs/testing_playbook.md           │ Update Part A to show clean API; add Step 0 for       │
 │                                                 │ agentci init --generate                               │
 ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
 │ AgentCI/pyproject.toml                          │ Bump to 0.3.0                                         │
 └─────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘

 ---
 Reuse

 - tool_call_sequence property on Trace (line 171) — all new assertion methods delegate to this
 - total_cost_usd, total_llm_calls fields on Trace — cost_under() and llm_calls_under() delegate to these
 - TraceContext and attach_langgraph_state() in capture.py — langgraph_trace() wraps these
 - resolve_runner() in engine/parallel.py — already used by bootstrap, no change needed
 - judge.py LLM call infrastructure — can be referenced for how to call Anthropic API in _generate_queries()

 ---
 Verification

 1. Unit tests: conda run -n agentci python -m pytest tests/ -v — all existing 401+ tests pass, new Trace
 assertion tests pass
 2. RAG agent tests with clean API: cd DemoAgents/examples/rag-agent && pytest tests/test_rag.py -v — 16 tests
  pass using trace.called() instead of RAGTrace
 3. agentci test still works: agentci test in rag-agent directory passes 15/15
 4. Generate command (with API key): agentci init --generate in a project directory produces a populated
 agentci_spec.yaml with 10+ meaningful queries
 5. Import check: python -c "from agentci import langgraph_trace; from agentci.models import Trace; t =
 Trace(); assert not t.called('x'); print('OK')" passes
 6. Publish: pip install ciagent==0.3.0 then from agentci import langgraph_trace works