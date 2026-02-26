# Writing Tests with AgentCI

AgentCI tests are defined in `agentci_spec.yaml`. Each spec file contains
a list of `queries` — golden test cases the agent must pass.

## Quick Start

```yaml
version: 1
agent: my-agent
queries:
  - query: "How do I install the product?"
    correctness:
      expected_in_answer: ["pip install", "mypackage"]
```

Run with:
```bash
agentci validate agentci_spec.yaml
```

---

## Correctness: Brittle vs. Robust Assertions

### Brittle: Keyword Checks (use sparingly)

`expected_in_answer` and `not_in_answer` check for exact substrings. They fail
on paraphrases, translations, and formatting changes — brittle for anything
beyond smoke tests.

```yaml
# ⚠️  ADVANCED / BRITTLE — prefer llm_judge for production tests
correctness:
  expected_in_answer: ["pip install agentci"]
  not_in_answer: ["error", "failed"]
```

Use `expected_in_answer` only when:
- The output is machine-generated (JSON keys, specific codes)
- The exact string is guaranteed invariant (version numbers, IDs)
- As a fast pre-check before more expensive judge calls

### Robust: Composite LLM Judge Rubrics (recommended)

`llm_judge` is a list of `JudgeRubric` objects evaluated sequentially.
Each rubric asks an LLM to score the response on a 1–5 scale and checks
the score against a threshold. **Multiple rubrics target different quality
dimensions**, making failures actionable.

```yaml
# ✅  PREFERRED — rubrics survive paraphrasing and model updates
correctness:
  llm_judge:
    - rule: "Response provides clear, step-by-step installation instructions"
      threshold: 0.7
    - rule: "Instructions are specific to pip / Python packaging; no generic advice"
      threshold: 0.6
    - rule: "Response does not contain fabricated package names or incorrect commands"
      threshold: 0.8
```

**Why composite rubrics are better:**
- Survives paraphrasing ("install" vs. "set up")
- Each rubric fails independently → actionable diagnostics
- Thresholds let you tune sensitivity per criterion
- Deterministic checks still run first (cheaper), judges run last

### Before / After Migration Example

**Before (brittle):**
```yaml
correctness:
  expected_in_answer: ["pip install agentci", "Python 3.10"]
```

**After (robust):**
```yaml
correctness:
  llm_judge:
    - rule: "Response provides the pip install command for agentci"
      threshold: 0.7
    - rule: "Response specifies Python version requirements"
      threshold: 0.6
```

---

## Router Firewall Pattern

Use `min_tool_recall` to assert the retriever was actually called. This catches
"agent bypassed retrieval entirely" failures that keyword checks miss entirely.

See the annotated example: [`examples/patterns/router_firewall.yaml`](../DemoAgents/examples/patterns/router_firewall.yaml)

```yaml
path:
  expected_tools: [retrieve_docs]
  min_tool_recall: 1.0   # Hard-catches "retriever was skipped" failures
```

Use this whenever:
- Your agent has a retriever that should always be called for in-scope queries
- You want to verify routing logic (e.g., Triage → Billing agent)
- A safety prompt-firewall must not be bypassed

---

## The Three Evaluation Layers

| Layer | Field | Severity | Purpose |
|-------|-------|----------|---------|
| Correctness | `correctness:` | Hard FAIL | Answer quality — blocks CI |
| Path | `path:` | Soft WARN | Tool trajectory — annotation only |
| Cost | `cost:` | Soft WARN | Efficiency budget — annotation only |

Layer 1 (Correctness) failures → exit code 1, CI blocked.
Layer 2/3 failures → GitHub annotations only, CI passes.

---

## LLM Judge Configuration

```yaml
judge_config:
  model: claude-sonnet-4-6   # Judge model (separate from agent model)
  temperature: 0              # Always 0 for reproducibility
```

### Scale Anchors (optional but recommended)

Score anchors reduce judge variance by giving the LLM concrete examples:

```yaml
llm_judge:
  - rule: "Response politely declines out-of-scope questions"
    scale:
      - "1: Answers the question despite being out-of-scope"
      - "3: Declines but without explanation"
      - "5: Declines politely and offers to help with in-scope topics"
    threshold: 0.6
```

### Few-Shot Examples (optional)

```yaml
llm_judge:
  - rule: "Response is factually grounded"
    threshold: 0.8
    few_shot_examples:
      - input: "The sky is green"
        output: "The sky is blue"
        score: 1
      - input: "Water is H2O"
        output: "Water is H2O"
        score: 5
```

---

## Pattern Library

Copy-paste templates in `DemoAgents/examples/patterns/`:

| File | When to use |
|------|------------|
| `router_firewall.yaml` | Assert retriever was called; catch routing bypasses |
| `mixed_intent.yaml` | Query spans both in-scope and out-of-scope topics |
| `refute_premise.yaml` | User asks about a nonexistent feature (polite correction) |
| `doc_grounded.yaml` | Evaluate answer against a reference document |
