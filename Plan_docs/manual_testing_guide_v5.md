# AgentCI v0.5.0 — Manual Testing Guide

> **Goal:** Validate all v0.5.0 features end-to-end: 3-layer evaluation engine improvements, OR-logic keywords, diagnostic failure messages, verdict parsing robustness, label-based pass override, scan-first init flow, zero-API-key mock mode, `--golden-file`, context-aware next steps, mock test mode, cost estimates, `agentci doctor`, progressive spec building, and deeper KB sampling.

---

## Prerequisites

```bash
mkdir /tmp/agentci-test-v50 && cd /tmp/agentci-test-v50
python3 -m venv .venv && source .venv/bin/activate
pip install ciagent==0.5.0
```

Verify version:
```bash
agentci --version
# Expected: 0.5.0
```

Clone the DemoAgents repo (public):
```bash
git clone https://github.com/suniel12/Demoagents.git
```

Set at least one API key (only needed for live mode):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

---

## Part 1: `agentci doctor` — Health Check

The `doctor` command validates your environment before you run anything.

### Scenario D1 — No Spec File

```bash
cd /tmp/agentci-test-v50
agentci doctor
```

**Expected:**
- Exit code `1`
- Output contains: `not found`
- Should tell you there's no `agentci_spec.yaml`

---

### Scenario D2 — Valid Spec File

```bash
cat > agentci_spec.yaml << 'EOF'
version: 1.0
agent: test-agent
queries:
  - query: What is the pricing?
  - query: How do I get started?
EOF
agentci doctor
```

**Expected:**
- Output contains: `AgentCI Doctor`
- Shows Python version check
- Shows dependency checks (`pydantic`, `click`)
- Shows spec validation: `valid`
- Summary line with `passed`, `warnings`, `failures`

---

### Scenario D3 — Invalid Spec File

```bash
echo "not: valid: yaml: [" > agentci_spec.yaml
agentci doctor
```

**Expected:**
- Exit code `1`
- Spec validation fails with a YAML parse error

---

## Part 2: `agentci test --mock` — Zero-Cost Testing

Mock mode generates synthetic traces matching your spec expectations — no API keys, no LLM calls, zero cost.

### Scenario M1 — Basic Mock Run

```bash
cat > agentci_spec.yaml << 'EOF'
version: 1.0
agent: test-agent
queries:
  - query: What is X?
  - query: How does Y work?
  - query: Tell me about Z
EOF
agentci test --mock
```

**Expected:**
- Output contains `mock` (case-insensitive)
- Runs all 3 queries
- Exit code `0` or `1` (should run without crashing)
- No API calls made

---

### Scenario M2 — Mock with Expected Tools

```bash
cat > agentci_spec.yaml << 'EOF'
version: 1.0
agent: test-agent
queries:
  - query: What is the pricing?
    path:
      expected_tools: [search_docs, grade_answer]
    correctness:
      expected_in_answer: ["$199", "Business plan"]
EOF
agentci test --mock
```

**Expected:**
- Synthetic trace includes `search_docs` and `grade_answer` tool calls
- Output contains `$199` and `Business plan`
- Correctness and Path layers both PASS

---

### Scenario M3 — Mock with Cost Budget

```bash
cat > agentci_spec.yaml << 'EOF'
version: 1.0
agent: test-agent
queries:
  - query: test query
    cost:
      max_llm_calls: 8
EOF
agentci test --mock
```

**Expected:**
- LLM calls in trace stay within budget (mock uses `min(max, 2)`)
- Zero actual cost
- Default `max_llm_calls` is now 8 (was 3 in v0.4.x)

---

### Scenario M4 — Mock with Out-of-Scope (No Tools)

```bash
cat > agentci_spec.yaml << 'EOF'
version: 1.0
agent: test-agent
queries:
  - query: What is the weather?
    path:
      max_tool_calls: 0
EOF
agentci test --mock
```

**Expected:**
- Synthetic trace has 0 tool calls
- Should pass the path assertion

---

### Scenario M5 — Mock with OR-Logic Keywords (v0.5.0 New)

```bash
cat > agentci_spec.yaml << 'EOF'
version: 1.0
agent: test-agent
queries:
  - query: What frameworks does AgentCI support?
    correctness:
      any_expected_in_answer: ["LangGraph", "OpenAI Agents SDK", "Anthropic"]
EOF
agentci test --mock
```

**Expected:**
- Synthetic trace answer mentions at least ONE of the listed terms
- Correctness layer PASSES (OR logic — any single match is sufficient)
- Compare: `expected_in_answer` would require ALL three

---

## Part 3: `agentci init` — Scan-First Guided Interview

v0.4.1+ changes the init flow:
- **No more Q1** ("What does your agent do?") — agent type is auto-detected from code
- **Scan-first:** auto-scans project before asking any questions, shows summary
- **Q1 is now run mode** (mock/live) — was Q2 in v0.4.0
- **Q2 is conditional** (KB path / tools / topics) — was Q3 in v0.4.0

### Setup

```bash
cd /tmp/agentci-test-v50/Demoagents/examples/rag-agent
pip install -r requirements.txt
```

Create `.env` with your API key (only needed for live mode):
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

---

### Scenario I1 — Interactive Init (Scan-First Flow)

```bash
agentci init --generate --force
```

**Expected interview flow:**
1. **Auto-scan** — scans project, detects agent type, tools, KB directory
2. **Summary** — shows detected agent type (e.g., "rag"), tools found, KB files found
3. **Q1:** "How would you like to test?" — choose `mock` or `live`
4. **Q2 (conditional):** Based on auto-detected agent type:
   - RAG: confirms KB path (auto-detected `knowledge_base/`)
   - Tool: confirms detected tools
   - Conversational: asks for topics to handle/decline

**Expected output:**
- Generates `agentci_spec.yaml` with queries grounded in KB content
- Queries should use `any_expected_in_answer` (OR logic) for list-type answers (v0.5.0)
- `max_llm_calls` defaults to 8 for in-scope queries (v0.5.0)
- `expected_in_answer` values come from real KB files, not hallucinated

---

### Scenario I2 — Non-Interactive Init (CI Mode)

```bash
agentci init --generate --force \
  --kb-path ./knowledge_base \
  --mode mock
```

**Expected:**
- Skips all interactive prompts
- Agent type auto-detected from code (no `--agent-description` needed)
- Generates spec using the provided flags
- Good for CI pipelines where stdin is not available

> **Note:** `--agent-description` is deprecated since v0.4.1. It still works but shows a deprecation warning.

---

### Scenario I2b — Deprecated `--agent-description` Warning

```bash
agentci init --generate --force \
  --agent-description "RAG agent for AgentCI documentation" \
  --mode mock
```

**Expected:**
- Shows deprecation warning for `--agent-description`
- Still works (backwards compatible)
- Agent type auto-detected from code regardless of description

---

### Scenario I3 — Deep KB Sampling + OR-Logic Spec Generation (v0.5.0)

Verify the LLM generates specs with proper keyword logic:

```bash
# Check what the generated spec looks like
cat agentci_spec.yaml
```

**Expected (v0.5.0 changes):**
- Queries that expect lists/enumerations use `any_expected_in_answer` (OR logic)
- Queries where ALL terms are essential use `expected_in_answer` (AND logic)
- `cost.max_llm_calls` set to 8 for in-scope queries, 2-3 for out-of-scope
- Files sampled up to 2000 chars each (vs 200 in v0.3)
- The spec should have well-formed queries with path, correctness, and/or cost assertions

---

### Scenario I4 — Tool Detection

The RAG agent uses `@tool` decorated functions. Verify auto-detection:

```bash
# The init should have detected tools from the agent code
# Check the generated spec for expected_tools
grep -A2 "expected_tools" agentci_spec.yaml
```

**Expected:**
- Auto-detects tools like `retriever_tool`, `grade_documents` (or whatever tools the RAG agent defines)
- Generated spec includes these in `expected_tools` for relevant queries

---

## Part 4: Zero-API-Key Mock Init

Mock mode is truly zero-API-key since v0.4.1. Three paths for query input without LLM calls:

### Scenario Z1 — Mock Init with Golden File

```bash
cd /tmp/agentci-test-v50
unset ANTHROPIC_API_KEY OPENAI_API_KEY

cat > golden.json << 'EOF'
[
  {"question": "What is the pricing?", "answer": "The pricing starts at $199/mo"},
  {"question": "How do I install?", "answer": "Run pip install ciagent"}
]
EOF

mkdir -p myagent && cat > myagent/agent.py << 'EOF'
def chat(msg): pass
EOF

cd myagent
agentci init --generate --force --mode mock --golden-file ../golden.json
```

**Expected:**
- No API keys required
- Loads queries from `golden.json`
- Generated spec contains "What is the pricing?" and "How do I install?"
- `expected_in_answer` populated from the golden file answers

---

### Scenario Z2 — Mock Init with Interactive Queries

```bash
cd /tmp/agentci-test-v50/myagent
unset ANTHROPIC_API_KEY OPENAI_API_KEY

agentci init --generate --force --mode mock
```

When prompted "Enter queries interactively?", type `y`, then enter queries one per line, then `done`.

**Expected:**
- No API keys required
- Prompts for queries interactively
- Generated spec contains the queries you typed

---

### Scenario Z3 — Mock Init with Skeleton Template

```bash
cd /tmp/agentci-test-v50/myagent
unset ANTHROPIC_API_KEY OPENAI_API_KEY

agentci init --generate --force --mode mock
```

When prompted "Enter queries interactively?", type `n`.

**Expected:**
- No API keys required
- Generates skeleton spec with `# TODO:` placeholder queries
- Skeleton uses `any_expected_in_answer` and `cost.max_llm_calls: 8` (v0.5.0)
- Skeleton matches detected agent type (RAG/tool/conversational)

---

## Part 5: Context-Aware Next Steps

Shows context-aware "Next Steps" based on run mode and whether a CI workflow was created.

### Scenario N1 — Mock Mode Next Steps

```bash
agentci init --generate --force --mode mock
```

**Expected:**
- Next Steps suggests `agentci test --mock`
- Does NOT suggest `git push` or CI-related steps

---

### Scenario N2 — Live Mode Next Steps

```bash
agentci init --generate --force --mode live
```

**Expected:**
- Next Steps suggests `agentci test`
- If a `.github/workflows/` file was created, also suggests `git push`
- Mentions setting API key as CI secret

---

## Part 6: Progressive Spec Building

When using `--generate` with live mode, the init flow builds specs progressively:

1. **Smoke queries (3):** Quick validation batch
2. **Validation:** Checks smoke queries pass basic structure
3. **Full queries (10-12):** Informed by smoke results

### Scenario P1 — Progressive Build (Live Mode)

> **Note:** This requires a valid API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`).

```bash
cd /tmp/agentci-test-v50/Demoagents/examples/rag-agent
agentci init --generate --force --mode live
```

**Expected:**
- First generates 3 smoke queries
- Validates them
- Then generates 10-12 additional queries
- Total: ~13-15 queries in final spec

---

## Part 7: Cost Estimate Before Live Runs

When running live tests (not mock), AgentCI shows a cost estimate before execution.

### Scenario C1 — Cost Estimate Display

```bash
cd /tmp/agentci-test-v50/Demoagents/examples/rag-agent
agentci test
```

**Expected:**
- Shows estimated cost with `$` amounts before running
- Shows query count
- Shows range (low-high estimate)
- Prompts for confirmation: "Proceed? [y/N]"
- If you type `n`, exits without running

---

### Scenario C2 — Skip Confirmation with `--yes`

```bash
agentci test --yes
# or
agentci test -y
```

**Expected:**
- Skips the cost confirmation prompt
- Runs tests immediately (useful for CI)

---

### Scenario C3 — No Cost Prompt in Mock Mode

```bash
agentci test --mock
```

**Expected:**
- No cost estimate shown (mock mode is free)
- No confirmation prompt
- Runs immediately

---

## Part 8: 3-Layer Evaluation Engine (v0.5.0 Improvements)

### Scenario L1 — Diagnostic Failure Messages (v0.5.0 New)

When a keyword check fails, the failure message now includes the agent's actual answer for quick debugging.

```bash
cat > agentci_spec.yaml << 'EOF'
version: 1.0
agent: test-agent
queries:
  - query: How do I install?
    correctness:
      expected_in_answer: ["pip install ciagent"]
EOF
agentci test --mock
```

**Expected failure message format:**
```
Expected 'pip install ciagent' not found in answer
         Agent said: "<first 200 chars of the agent's response>"
```

This helps distinguish retrieval failures (wrong context fetched) from logic failures (right context, wrong answer).

---

### Scenario L2 — OR-Logic Keywords (`any_expected_in_answer`)

```bash
cat > agentci_spec.yaml << 'EOF'
version: 1.0
agent: test-agent
queries:
  - query: What frameworks are supported?
    correctness:
      any_expected_in_answer: ["LangGraph", "OpenAI", "Anthropic", "CrewAI"]
EOF
agentci test --mock
```

**Expected:**
- PASSES if the answer contains ANY ONE of the listed terms
- Failure message shows: `None of ["LangGraph", "OpenAI", "Anthropic", "CrewAI"] found in answer`
- Failure message includes: `Agent said: "..."`

---

### Scenario L3 — AND + OR Combined

```bash
cat > agentci_spec.yaml << 'EOF'
version: 1.0
agent: test-agent
queries:
  - query: How do I install with Python?
    correctness:
      expected_in_answer: ["Python"]
      any_expected_in_answer: ["3.10", "3.11", "3.12"]
EOF
agentci test --mock
```

**Expected:**
- Must contain "Python" (AND logic)
- Must contain at least one of "3.10", "3.11", "3.12" (OR logic)
- Both conditions must pass for the query to pass

---

### Scenario L4 — LLM Judge with Query Context

> **Note:** Requires API key for judge LLM calls.

```bash
cat > agentci_spec.yaml << 'EOF'
version: 1.0
agent: test-agent
judge_config:
  model: claude-sonnet-4-6
  temperature: 0
queries:
  - query: What is the pricing for the enterprise plan?
    correctness:
      llm_judge:
        - rule: "Response provides specific pricing information"
          threshold: 0.7
        - rule: "Response mentions enterprise-specific features"
          threshold: 0.6
EOF
agentci test --yes
```

**Expected:**
- Judge receives both the query AND the answer for evaluation
- Composite rubrics fail independently (actionable diagnostics)
- Each rubric shows score vs threshold in output

---

## Part 9: Additional CLI Commands (v0.5.0)

### Scenario A1 — `agentci validate`

```bash
cat > agentci_spec.yaml << 'EOF'
version: 1.0
agent: test-agent
queries:
  - query: Test question
EOF
agentci validate agentci_spec.yaml
```

**Expected:**
- Validates spec structure against the schema
- Reports valid or lists schema errors
- Does NOT execute any queries

---

### Scenario A2 — `agentci eval` (Standalone Evaluation)

```bash
agentci eval --config agentci_spec.yaml
```

**Expected:**
- Runs correctness evaluation without golden baselines
- Skips relative assertions (max_cost_multiplier, min_sequence_similarity)
- Useful for one-off evaluations

---

### Scenario A3 — `agentci save` and `agentci baselines`

```bash
# Save a versioned baseline (requires --agent, --version, --trace-file)
agentci save --agent test-agent --version v1-initial --trace-file trace.json --force-save

# List baselines for an agent (requires --agent)
agentci baselines --agent test-agent
```

**Expected:**
- `save`: Creates a versioned baseline with timestamp and query association
- `save` without `--force-save`: Runs correctness precheck before saving
- `save` requires `--agent`, `--version`, and `--trace-file` flags
- `baselines`: Shows table with version, timestamp, query, precheck status
- `baselines` requires `--agent` flag

---

## Part 10: End-to-End with DemoAgents RAG Agent

Full real-world validation using the cloned DemoAgents repo.

### Scenario E1 — Doctor Check

```bash
cd /tmp/agentci-test-v50/Demoagents/examples/rag-agent
agentci doctor
```

**Expected:** All checks pass — spec valid, deps installed, API key present, knowledge base detected.

---

### Scenario E2 — Mock Test (No API Cost)

```bash
agentci test --mock
```

**Expected:**
- Runs all queries from `agentci_spec.yaml` with synthetic traces
- Zero cost
- Validates spec structure is correct

---

### Scenario E3 — Live Test (With Cost Estimate)

```bash
agentci test
```

**Expected:**
1. Shows cost estimate for N queries
2. Prompts for confirmation
3. Runs all queries against the live agent
4. Shows 3-layer evaluation results (Correctness, Path, Cost)

---

### Scenario E4 — Re-generate Spec with Scan-First Interview (v0.5.0)

```bash
agentci init --generate --force
```

**Expected (v0.5.0 changes):**
- Auto-scan + summary (no Q1 about agent description)
- Q1: run mode (mock/live)
- Q2: conditional based on auto-detected agent type
- Deep KB sampling picks up actual pricing/feature content
- Generated queries use `any_expected_in_answer` for list-type answers
- `max_llm_calls` defaults to 8 (was 3 in v0.4.x)
- Overwrites existing spec (due to `--force`)

---

### Scenario E5 — Mock Then Live Comparison

```bash
# First run mock to validate spec structure
agentci test --mock

# Then run live to see real agent behavior
agentci test --yes
```

**Expected:**
- Mock run validates the spec is well-formed (all assertions make sense)
- Live run shows actual pass/fail results against the real agent
- Compare: mock should always pass (traces built to match), live may have failures

---

## Checklist

| # | Scenario | Feature | Status |
|---|----------|---------|--------|
| D1 | Doctor — no spec | `agentci doctor` | |
| D2 | Doctor — valid spec | `agentci doctor` | |
| D3 | Doctor — invalid spec | `agentci doctor` | |
| M1 | Basic mock run | `--mock` | |
| M2 | Mock with tools + keywords | `--mock` | |
| M3 | Mock with cost budget | `--mock` (max_llm_calls=8) | |
| M4 | Mock out-of-scope | `--mock` | |
| M5 | Mock with OR-logic keywords | `any_expected_in_answer` | |
| I1 | Interactive init (scan-first) | auto agent-type detection | |
| I2 | Non-interactive init | `--mode`, `--kb-path` | |
| I2b | Deprecated `--agent-description` | deprecation warning | |
| I3 | Deep KB sampling + OR-logic | `any_expected_in_answer` in spec | |
| I4 | Tool detection | `_detect_tools_from_code` | |
| Z1 | Mock init with golden file | `--golden-file` | |
| Z2 | Mock init interactive queries | zero-API-key mock | |
| Z3 | Mock init skeleton template | skeleton with OR-logic defaults | |
| N1 | Mock mode next steps | context-aware output | |
| N2 | Live mode next steps | context-aware output | |
| P1 | Progressive spec build | smoke -> full | |
| C1 | Cost estimate display | pre-execution estimate | |
| C2 | Skip with `--yes` | `-y` flag | |
| C3 | No cost in mock mode | mock bypass | |
| L1 | Diagnostic failure messages | "Agent said: ..." | |
| L2 | OR-logic keywords | `any_expected_in_answer` | |
| L3 | AND + OR combined | both keyword types | |
| L4 | Judge with query context | composite rubrics | |
| A1 | `agentci validate` | schema validation | |
| A2 | `agentci eval` | standalone evaluation | |
| A3 | `agentci save` + `baselines` | versioned baselines | |
| E1 | RAG agent doctor | end-to-end | |
| E2 | RAG agent mock test | end-to-end | |
| E3 | RAG agent live test | end-to-end | |
| E4 | RAG agent re-generate (v0.5.0) | OR-logic + max_llm_calls=8 | |
| E5 | Mock vs live comparison | end-to-end | |

---

## Running the Automated Test Suite

After manual testing, verify all unit tests still pass:

```bash
conda run -n agentci python -m pytest tests/ -v
# Expected: 539 tests passing, 4 skipped
```

Key test files for v0.5.0 features:
- `tests/test_mock_runner.py` — mock_run, run_mock_spec (8 tests)
- `tests/test_cost_estimator.py` — estimate_cost, format_estimate (11 tests)
- `tests/test_doctor.py` — doctor command (6 tests)
- `tests/test_cli.py` — guided init helpers, deep KB, mock command, improvements (40+ tests)
- `tests/test_judge.py` — verdict parsing strategies, label override, ensemble (20+ tests)
- `tests/test_correctness_engine.py` — keyword checks, OR-logic, diagnostic messages (30+ tests)

v0.5.0-specific test coverage:
- `TestParseVerdict` — 4-strategy verdict parsing (clean JSON, markdown, preamble, regex fallback)
- `TestRunJudge.test_label_pass_overrides_low_score` — label-based pass override
- `TestRunJudge.test_borderline_label_relies_on_score` — borderline handling
- `TestAnyExpectedInAnswer` — OR-logic keyword checks
- `test_failure_includes_agent_answer_preview` — diagnostic "Agent said:" messages
- `test_none_found_includes_agent_answer_preview` — OR-logic diagnostic messages

---

## What Changed from v0.4.1 to v0.5.0

| Change | v0.4.1 | v0.5.0 |
|--------|--------|--------|
| Default `max_llm_calls` | 3 (too low for RAG) | 8 (realistic for multi-step agents) |
| Keyword logic in generated specs | `expected_in_answer` (AND only) | `any_expected_in_answer` (OR) preferred for lists |
| Failure messages | "Expected 'X' not found" | "Expected 'X' not found\n Agent said: '...'" |
| Judge verdict parsing | Single-strategy | 4-strategy fallback (clean/markdown/preamble/regex) |
| Judge token limit | 256 (caused truncation) | 1024 (accommodates verbose rationale) |
| Judge query context | Not always passed | Always passed to judge for evaluation |
| Label-based pass override | Not implemented | Judge "pass" label overrides low score threshold |
| Skeleton spec template | `expected_in_answer` only | `any_expected_in_answer` + `cost.max_llm_calls: 8` |
| Calibration minimum | `max(3, observed*1.5)` | `max(8, observed*1.5)` |
| Test count | 509 | 539 |

---

## Exit Codes Reference

| Code | Meaning |
|------|---------|
| `0` | All correctness checks pass |
| `1` | One or more correctness failures |
| `2` | Infrastructure / runtime error |
