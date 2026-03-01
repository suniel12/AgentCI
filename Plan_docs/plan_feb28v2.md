# AgentCI v0.4.1 — Init Flow Improvements

## Context

During v0.4.0 manual testing, several UX issues were identified in `agentci init --generate`:
1. **Q1 is redundant** — "What does your agent do?" can be inferred from code scan
2. **Mock mode still needs API keys** — `_generate_smoke_queries()` calls LLM even in mock mode
3. **"Next Steps" is context-blind** — always shows git push instructions regardless of mode
4. **No `--golden-file` flag** — golden pairs only loadable interactively during RAG Q3
5. **Q2 rejects 'y'** — `choices=["live", "mock"]` doesn't accept 'y' as default confirmation

The goal: make mock mode truly zero-API-key, reduce questions, and make output smarter.

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/agentci/cli.py` | 4 new functions, rewrite `init()` flow (~500 lines) |
| `tests/test_cli.py` | 6+ new test classes, update 2 existing tests (~200 lines) |

---

## Implementation Plan

### Step 1: Add `_detect_agent_type_from_code()` (~30 lines)

Insert after `_detect_agent_type()` at line 174 in `src/agentci/cli.py`.

```python
def _detect_agent_type_from_code(context: dict, detected_tools: list, detected_kb: str | None) -> str:
```

**Logic:**
- `detected_kb is not None` → `"rag"`
- Agent files contain retrieval keywords (`"retriev"`, `"vector"`, `"embedding"`, `"search_docs"`) → `"rag"`
- `len(detected_tools) > 0` → `"tool"`
- Otherwise → `"conversational"`

Keep existing `_detect_agent_type(description)` for backwards compatibility.

### Step 2: Add `_prompt_for_queries_interactive()` (~30 lines)

New function for mock mode interactive query entry.

```python
def _prompt_for_queries_interactive() -> list[str]:
```

- Loop with `Prompt.ask("Enter a test query (or 'done' to finish)")`
- Break on empty input or `"done"`
- Return list of query strings

### Step 3: Add `_generate_skeleton_spec()` (~50 lines)

Generates a template `agentci_spec.yaml` with TODO placeholders based on scan results.

```python
def _generate_skeleton_spec(context, agent_type, detected_tools, detected_kb, runner_path) -> str:
```

- RAG: 3 template queries (in-scope KB question, out-of-scope, boundary)
- Tool: template query per detected tool + out-of-scope
- Conversational: 2 template queries (handle topic, decline topic)
- All with `# TODO:` comments guiding the user

### Step 4: Add `_build_next_steps()` (~30 lines)

Context-aware "Next Steps" output replacing the static block at lines 837-841.

```python
def _build_next_steps(run_mode: str, created_workflow: bool, has_queries: bool) -> str:
```

| Mode | has_queries | Output |
|------|------------|--------|
| mock | True | "Run `agentci test --mock` to validate" |
| mock | False | "Fill in TODO queries, then run `agentci test --mock`" |
| live | True | "Run `agentci test`" + git/CI steps if workflow created |
| live | False | "Fill in queries, then run `agentci test`" + git/CI steps |

### Step 5: Add `--golden-file` CLI flag

Add to `init()` command options:
```python
@click.option('--golden-file', default=None, type=click.Path(exists=True),
              help='Path to golden Q&A pairs (JSON/CSV) for mock mode')
```

Deprecate `--agent-description` with warning (keep for backwards compat).

### Step 6: Rewrite `init()` flow (lines 541-841)

**New flow when `--generate` is set:**

1. **Auto-scan first** (no questions yet)
   - `_scan_project()`, `_detect_tools_from_code()`, `_detect_kb_dir()`
   - `_detect_agent_type_from_code()` (new)
   - Print summary: agent type, tools found, KB files found

2. **Q1: Run mode** (was Q2)
   - `choices=["live", "mock"]`, default `"live"`
   - Accept `"y"` as alias for default

3. **API key guard** — only for live mode (move check after Q1 so mock never hits it)

4. **Q2: Conditional follow-up** (was Q3, renumbered)
   - RAG: confirm KB path (auto-detected), optional golden file
   - Tool: confirm detected tools
   - Conversational: optional topics

5. **Runner detection** — auto-detect, prompt for confirmation

6. **Query generation** — branched by mode:

   **Mock path (zero API keys):**
   - If `--golden-file` provided → load queries from file
   - Else → prompt "Enter queries interactively?"
     - Yes → `_prompt_for_queries_interactive()`
     - No → `_generate_skeleton_spec()` (TODO template)

   **Live path (API key required):**
   - Progressive build: smoke (3) → optional full (10-12)
   - Fallback to skeleton on LLM failure

7. **Write spec + workflow**

8. **Context-aware Next Steps** via `_build_next_steps()`

### Step 7: Update tests

**New tests:**
- `TestDetectAgentTypeFromCode` — rag/tool/conversational detection from code
- `test_init_generate_mock_no_api_key` — mock works without any API keys
- `test_init_generate_with_golden_file` — `--golden-file` loads queries
- `test_prompt_for_queries_interactive` — interactive loop works
- `test_build_next_steps_*` — context-aware output for each mode
- `test_init_agent_description_deprecated` — deprecation warning shown

**Updated tests:**
- `test_init_command_interactive` — update expected prompts (no more Q1 "What does your agent do?")

---

## Verification

1. **Unit tests:** `conda run -n agentci python -m pytest tests/test_cli.py -v`
2. **Full suite:** `conda run -n agentci python -m pytest tests/ -v` (expect 491+ passing)
3. **Manual E2E — mock mode (no API key):**
   ```bash
   unset ANTHROPIC_API_KEY OPENAI_API_KEY
   cd /tmp/agentci-test && agentci init --generate --mode mock
   # Should work without errors, no API calls
   ```
4. **Manual E2E — golden file:**
   ```bash
   agentci init --generate --mode mock --golden-file golden.json
   ```
5. **Manual E2E — live mode:** verify existing flow still works with API key
6. **Backwards compat:** `--agent-description` shows deprecation warning but doesn't break
