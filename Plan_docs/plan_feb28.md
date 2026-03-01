# Plan: AgentCI v0.4.0 — Guided Init, Mock Testing, Cost Estimates & Doctor

## Context

After shipping v0.3.5 (Trace helpers, `langgraph_trace()`, `init --generate`, bug fixes), end-to-end testing revealed that `agentci init --generate` produces poor-quality test queries. The LLM sees code structure but not domain knowledge, so it halluccinates expected answers (e.g., `'$199'`, `'99.5%'`) that don't match the actual KB. Additionally, users with no API key hit a dead end — there's no mock/dry-run path.

**Goal:** Make the first `agentci test` meaningful for any agent project, with or without an API key.

**What's already shipped (from feb27 plan):**
- Trace assertion helpers: `called()`, `never_called()`, `loop_count()`, `cost_under()`, `llm_calls_under()`
- `langgraph_trace()` context manager + `TraceContext.attach()` alias
- `agentci init --generate` with `_scan_project()` + `_generate_queries()`
- Empty query filter in `_generate_queries()`
- DemoAgents updated to new API
- Tests for all new utilities (444 passing)
- Published through v0.3.5

**What's open (this plan):**
- Guided init interview (replace blind LLM generation)
- Deeper KB sampling (ground expected answers in real content)
- Mock test mode (validate spec without API key)
- Cost estimate before execution
- `agentci doctor` health-check command
- Progressive spec building (generate 3, run, iterate)

---

## Priority Order

| Priority | Feature | Why |
|---|---|---|
| P0 | Guided init interview | Fixes the root cause — blind query generation |
| P0 | Deeper KB sampling | Grounds expected answers in real content |
| P1 | Mock test mode | Unblocks users without API keys |
| P1 | Cost estimate before execution | Prevents surprise bills |
| P2 | `agentci doctor` | Saves debugging time on setup issues |
| P2 | Progressive spec building | Better UX for iterating on spec |

---

## Part 1: Guided Init Interview (P0)

**File:** `AgentCI/src/agentci/cli.py` — modify `init` command

Replace the current silent `_scan_project()` → `_generate_queries()` flow with a 3-question interactive interview when `--generate` is used.

### Q1: "What does your agent do?"

```
? What does your agent do?
  (e.g., "Answers customer support questions about NovaCorp products using a knowledge base")
  >
```

Free text input via `click.prompt()`. Injected into the LLM generation prompt as `AGENT_DESCRIPTION`.

### Q2: "How do you want to run tests?"

```
? How do you want to run tests?
  [1] Live mode — run my agent with a real API key
  [2] Mock mode — validate my spec without API calls
  >
```

- **Live mode:** requires runner path + API key (current flow)
- **Mock mode:** generates spec + uses `MockTool`/`AnthropicMocker`/`OpenAIMocker` from `mocks.py` to produce synthetic traces matching the spec's expected paths

### Q3: Conditional follow-up based on agent description

The init command parses Q1 for agent-type keywords and asks a relevant follow-up:

**RAG / knowledge-base agent** (keywords: "knowledge", "document", "retriev", "RAG", "FAQ", "support"):
```
? Where is your knowledge base directory?
  Auto-detected: ./knowledge_base/ (10 files)
  [Enter to confirm, or type a different path]
  >

? Do you have existing golden Q&A pairs? (JSON or CSV with "question" and "answer" columns)
  Path (or press Enter to skip):
  >
```

**Tool-calling agent** (keywords: "tool", "function", "API", "action", "booking", "search"):
```
? What tools does your agent expose?
  Auto-detected from code: [search_flights, book_ticket, cancel_booking]
  [Enter to confirm, or type a comma-separated list]
  >
```

**Conversational / general** (fallback):
```
? What topics should your agent handle vs. decline?
  Handle (comma-separated):
  Decline (comma-separated):
  >
```

### Updated `_generate_queries()` prompt

The LLM prompt gets enriched with all interview answers:

```
AGENT DESCRIPTION:
{agent_description}   ← from Q1

AGENT TYPE: {agent_type}  ← inferred from Q1 keywords

KNOWLEDGE BASE CONTENT:
{kb_full_content}     ← deeper sampling (Part 2)

TOOLS AVAILABLE:
{tools_list}          ← from Q3 or auto-detected

GOLDEN Q&A PAIRS:
{golden_pairs}        ← from Q3 if provided

IMPORTANT: All expected_in_answer values MUST come directly from the knowledge
base content above. Do NOT invent or hallucinate facts. If a fact isn't in the
KB, use llm_judge instead of expected_in_answer.
```

### Non-interactive mode

For CI/scripting, support flags:
```
agentci init --generate \
  --agent-description "NovaCorp support bot" \
  --kb-path ./knowledge_base/ \
  --mode live
```

If all required flags are provided, skip the interactive prompts.

---

## Part 2: Deeper KB Sampling (P0)

**File:** `AgentCI/src/agentci/cli.py` — modify `_scan_project()`

### Current behavior
- Reads first 200 chars of each KB file
- Result: LLM sees file titles but not actual content

### New behavior
- Read full content of KB files up to 2000 chars each (10x increase)
- For files > 2000 chars, include first 1000 + last 500 chars (captures intro + conclusion)
- Prioritize smaller files (they're more focused, more useful per char)
- Sort KB files by size ascending before sampling
- Keep the existing `_MAX_CONTEXT_CHARS = 48_000` overall limit

### Change in `_scan_project()`

```python
# Knowledge base — deeper sampling
for kb_dir_name in ("knowledge_base", "kb", "docs", "data", "knowledge"):
    kb_dir = project_dir / kb_dir_name
    if kb_dir.is_dir():
        kb_files = sorted(
            [f for f in kb_dir.rglob("*") if f.suffix in (".md", ".txt")],
            key=lambda f: f.stat().st_size  # smallest first
        )
        for f in kb_files:
            content = f.read_text(errors="ignore")
            if len(content) > 2000:
                snippet = content[:1000] + "\n...\n" + content[-500:]
            else:
                snippet = content
            context["knowledge_base"].append({
                "path": str(f.relative_to(project_dir)),
                "snippet": snippet,
            })
        break
```

---

## Part 3: Mock Test Mode (P1)

**File:** `AgentCI/src/agentci/cli.py` — modify `test_cmd`
**File:** `AgentCI/src/agentci/engine/mock_runner.py` — new file

### Concept

When user selects mock mode during init, the spec gets a special runner:
```yaml
runner: "agentci.engine.mock_runner:mock_run"
mode: mock  # new field in spec
```

### Mock Runner

`mock_runner.py` generates synthetic traces that match the spec's expected behavior:

```python
def mock_run(query: str, query_spec: dict) -> Trace:
    """Generate a synthetic trace matching the spec's expectations."""
    trace = Trace(query=query, agent_name="mock")

    # Build tool calls from expected_tools in path spec
    expected_tools = query_spec.get("path", {}).get("expected_tools", [])
    for tool_name in expected_tools:
        trace.spans[0].tool_calls.append(
            ToolCall(tool_name=tool_name, arguments={}, result="[mock result]")
        )

    # Build expected answer from correctness spec
    expected_keywords = query_spec.get("correctness", {}).get("expected_in_answer", [])
    if expected_keywords:
        trace.final_output = f"Based on our documentation: {', '.join(expected_keywords)}."
    else:
        trace.final_output = "[Mock response — no expected keywords defined]"

    # Set cost within budget
    max_llm_calls = query_spec.get("cost", {}).get("max_llm_calls", 3)
    trace.spans[0].llm_calls = [
        LLMCall(model="mock", tokens_in=100, tokens_out=50, cost_usd=0.0)
        for _ in range(min(max_llm_calls, 2))
    ]

    return trace
```

### How mock mode works in `test_cmd`

```python
if spec.mode == "mock":
    from .engine.mock_runner import mock_run
    traces = {
        q.query: mock_run(q.query, q.model_dump())
        for q in spec.queries
    }
    console.print("[cyan]Running in mock mode — synthetic traces, no API calls[/]")
```

### Value to user

- Validates spec structure: "Are my queries well-formed? Do paths make sense?"
- Shows what a passing run looks like
- Identifies spec issues (missing expected_tools, conflicting assertions)
- Zero cost, no API key needed

---

## Part 4: Cost Estimate Before Execution (P1)

**File:** `AgentCI/src/agentci/cli.py` — modify `test_cmd`
**File:** `AgentCI/src/agentci/engine/cost_estimator.py` — new file

### Pricing Table

Hardcoded model pricing (updated periodically):

```python
MODEL_PRICING = {
    # Anthropic (per 1M tokens)
    "claude-sonnet-4-6":      {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5":       {"input": 0.80, "output": 4.00},
    "claude-opus-4-6":        {"input": 15.00, "output": 75.00},
    # OpenAI
    "gpt-4o":                 {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":            {"input": 0.15, "output": 0.60},
    "gpt-4.1":                {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini":           {"input": 0.40, "output": 1.60},
}

# Estimated tokens per query (based on typical RAG agent patterns)
DEFAULT_TOKENS_PER_QUERY = {
    "input": 2000,   # system prompt + KB retrieval + user query
    "output": 500,   # agent response
    "judge_input": 1500,  # judge prompt + agent response
    "judge_output": 200,  # judge verdict
}
```

### Estimate Function

```python
def estimate_cost(
    num_queries: int,
    agent_model: str = "gpt-4o",
    judge_model: str = "claude-sonnet-4-6",
    has_llm_judge: bool = True,
) -> dict:
    """Estimate cost for running agentci test."""
    agent_price = MODEL_PRICING.get(agent_model, MODEL_PRICING["gpt-4o"])
    judge_price = MODEL_PRICING.get(judge_model, MODEL_PRICING["claude-sonnet-4-6"])

    agent_cost = num_queries * (
        DEFAULT_TOKENS_PER_QUERY["input"] * agent_price["input"] / 1_000_000 +
        DEFAULT_TOKENS_PER_QUERY["output"] * agent_price["output"] / 1_000_000
    )

    judge_cost = 0
    if has_llm_judge:
        judge_cost = num_queries * (
            DEFAULT_TOKENS_PER_QUERY["judge_input"] * judge_price["input"] / 1_000_000 +
            DEFAULT_TOKENS_PER_QUERY["judge_output"] * judge_price["output"] / 1_000_000
        )

    return {
        "agent_cost": agent_cost,
        "judge_cost": judge_cost,
        "total_low": (agent_cost + judge_cost) * 0.5,   # optimistic
        "total_high": (agent_cost + judge_cost) * 2.0,   # pessimistic
        "total_estimate": agent_cost + judge_cost,
    }
```

### Integration in `test_cmd`

Before running queries, show estimate and ask for confirmation:

```
Estimated cost: $0.03–$0.12 for 15 queries
  Agent (gpt-4o): ~$0.04
  Judge (claude-sonnet-4-6): ~$0.02
Proceed? [Y/n]
```

Add `--yes` / `-y` flag to skip confirmation (for CI).

---

## Part 5: `agentci doctor` (P2)

**File:** `AgentCI/src/agentci/cli.py` — new command

Health-check command that validates the full setup before running tests.

### Checks

```
$ agentci doctor

AgentCI Doctor
  [pass] agentci_spec.yaml found and valid (15 queries)
  [pass] Runner 'demo_runner:run_traced' imports successfully
  [pass] ANTHROPIC_API_KEY is set
  [warn] OPENAI_API_KEY is not set (needed if agent uses OpenAI models)
  [pass] Knowledge base directory exists: ./knowledge_base/ (10 files)
  [pass] Python 3.10+ detected (3.13.1)
  [fail] numpy is not installed (required by agent dependencies)
  [pass] .github/workflows/agentci.yml exists

Result: 6 passed, 1 warning, 1 failure
  Fix: pip install numpy
```

### Implementation

```python
@cli.command(name="doctor")
@click.option('--config', '-c', default='agentci_spec.yaml', show_default=True)
def doctor_cmd(config):
    """Check your AgentCI setup for common issues."""
    checks = []

    # 1. Spec file exists and validates
    # 2. Runner imports successfully
    # 3. API keys present
    # 4. KB directory exists (if referenced)
    # 5. Python version >= 3.10
    # 6. Key dependencies installed (numpy, etc.)
    # 7. GitHub Actions workflow exists
    # 8. requirements.txt includes ciagent

    for check in checks:
        run_check(check)

    print_summary(checks)
```

Each check is a small function returning `(status, message, fix_hint)`. Extensible — new checks can be added without modifying existing ones.

---

## Part 6: Progressive Spec Building (P2)

**File:** `AgentCI/src/agentci/cli.py` — modify `init` with `--generate`

### Concept

Instead of generating 15 queries at once (and hoping they're all good), generate in batches:

### Flow

```
$ agentci init --generate

[Interview: Q1, Q2, Q3 as in Part 1]

Generating initial test queries...
Generated 3 smoke-test queries:

  1. "What is NovaCorp's refund policy?" [in-scope, smoke]
  2. "Hello, how are you?" [out-of-scope, boundary]
  3. "What's the pricing for Business plan?" [in-scope, smoke]

Run these 3 queries now to validate? [Y/n]

Running 3 queries...
  [pass] Query 1: refund policy — PASS
  [pass] Query 2: greeting — PASS
  [fail] Query 3: pricing — expected '$199' not found

Generate 10 more queries? [Y/n]
```

### Implementation

- Split `_generate_queries()` into two calls:
  - `_generate_smoke_queries(context, n=3)` — initial batch, conservative
  - `_generate_full_queries(context, smoke_results, n=12)` — informed by smoke results
- The second call's prompt includes smoke test results so the LLM avoids the same mistakes
- If user declines the second batch, save the spec with just the 3 smoke queries

---

## Implementation Order

### Phase A: v0.4.0 (P0 features)
1. Guided init interview (Part 1)
2. Deeper KB sampling (Part 2)
3. Tests for both
4. Publish v0.4.0

### Phase B: v0.5.0 (P1 features)
5. Mock test mode (Part 3)
6. Cost estimate before execution (Part 4)
7. Tests for both
8. Publish v0.5.0

### Phase C: v0.6.0 (P2 features)
9. `agentci doctor` (Part 5)
10. Progressive spec building (Part 6)
11. Tests for both
12. Publish v0.6.0

---

## Critical Files

| File | Change | Phase |
|---|---|---|
| `src/agentci/cli.py` | Guided interview in `init`, `--agent-description` / `--kb-path` / `--mode` flags, `doctor` command, progressive flow | A, C |
| `src/agentci/cli.py` (`_scan_project`) | Deeper KB sampling (2000 chars, size-sorted) | A |
| `src/agentci/cli.py` (`_generate_queries`) | Enriched prompt with agent description, KB content, grounding instruction | A |
| `src/agentci/engine/mock_runner.py` | New file — synthetic trace generator from spec expectations | B |
| `src/agentci/engine/cost_estimator.py` | New file — pricing table + estimate function | B |
| `src/agentci/schema/spec_models.py` | Add `mode: Optional[Literal["live", "mock"]]` to `AgentCISpec` | B |
| `src/agentci/mocks.py` | Already exists — reuse `MockTool`, `AnthropicMocker`, `OpenAIMocker` | B |
| `tests/test_cli_init.py` | Tests for guided interview, KB sampling, progressive flow | A, C |
| `tests/test_mock_runner.py` | Tests for mock trace generation | B |
| `tests/test_cost_estimator.py` | Tests for cost estimates | B |
| `tests/test_doctor.py` | Tests for doctor checks | C |
| `pyproject.toml` | Version bumps per phase | A, B, C |

---

## Verification

### Phase A (v0.4.0)
1. `agentci init --generate` prompts Q1, Q2, Q3 interactively
2. Generated queries have `expected_in_answer` values grounded in actual KB content
3. `agentci init --generate --agent-description "..." --kb-path ./kb/ --mode live` works non-interactively
4. All existing tests pass + new tests

### Phase B (v0.5.0)
1. `agentci init --generate` with mock mode → spec has `mode: mock`
2. `agentci test` with mock spec → 15/15 pass, zero API calls
3. `agentci test` with live spec → shows cost estimate, asks confirmation
4. `agentci test -y` skips confirmation

### Phase C (v0.6.0)
1. `agentci doctor` reports pass/warn/fail for all checks
2. `agentci doctor` catches missing numpy, bad runner import, missing API key
3. `agentci init --generate` generates 3 smoke queries first, runs them, then asks for more
4. Declining second batch saves spec with 3 queries only
