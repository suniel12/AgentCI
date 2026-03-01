# 🚀 AgentCI Beginner's Testing Playbook

Welcome! If you are new to **AgentCI** (Agent Continuous Integration), this guide is for you.

**What does AgentCI do?**
When developers build AI Agents (like chatbots that search through documents and answer questions), it is very hard to test them. AI is unpredictable. AgentCI records exactly what an AI agent does (called a "trace") and creates tests to make sure it doesn't misbehave — like searching for documents when the question has nothing to do with the knowledge base, or suddenly costing ten times more money to run.

**What AI Agent will we test?**
We will use a real, working AI chatbot called the **AgentCI RAG Agent**. It answers questions about AgentCI itself, using a folder of 10 documentation files as its knowledge base. When you ask it a question, it:
1. Decides whether the question is about AgentCI or not
2. If yes, searches the knowledge base for relevant documents
3. Checks whether what it found is actually useful
4. If not, rewrites the question and tries again (up to 3 times)
5. Writes a final answer using only what it found

**Two ways to test it**
AgentCI gives you two complementary tools — and this playbook shows both:

| | `pytest tests/test_rag.py` | `agentci test` |
|---|---|---|
| **Style** | Python code | YAML config |
| **What it checks** | Trace assertions + baseline drift | Three-layer evaluation (Correctness / Path / Cost) |
| **Best for** | Catching regression vs a saved snapshot | Defining expected behavior upfront |

---

## Step 0: Auto-Generate Your Test Spec (Optional)

If you are starting a new agent project and want AgentCI to write the initial test spec for you, run this **before** Step 1:

```bash
# In your agent project directory (must have ANTHROPIC_API_KEY set):
agentci init --generate
```

AgentCI will:
1. Scan your project for agent code (files containing `@tool`, `add_node`, `ChatOpenAI`, etc.)
2. Read your knowledge base documents (any `docs/`, `kb/`, or `knowledge_base/` folder)
3. Inspect your existing test files for coverage gaps
4. Call Claude to generate 10–15 diverse test queries covering happy path, out-of-scope, and edge cases
5. Write a ready-to-use `agentci_spec.yaml`

```
Scanning project...
Found 2 agent files, 10 knowledge base documents, 0 existing test files
Generating test queries with AI...
Generated 12 test queries
✓ Created agentci_spec.yaml
```

Then jump straight to `agentci test` — no manual query writing required.

If you don't have an API key yet, use the example template instead:
```bash
agentci init --example
```

---

## Step 1: Set Up Your Workspace

**Open your Terminal application on your Mac (Cmd + Space → type "Terminal").**

```bash
# 1. Create a new folder on your Desktop
mkdir -p ~/Desktop/agentci-rag-test

# 2. Go inside it
cd ~/Desktop/agentci-rag-test
```

---

## Step 2: Download the RAG Agent

```bash
git clone https://github.com/suniel12/Demoagents.git
cd Demoagents/examples/rag-agent
```

**What to expect:**
- You will see `Cloning into 'Demoagents'...` followed by download progress.
- Wait for it to finish before moving on.

---

## Step 3: Create a Clean Python Sandbox

```bash
# Create the sandbox
python3 -m venv .venv

# Step inside it
source .venv/bin/activate
```

**What to expect:**
- After the second command, `(.venv)` appears at the start of your terminal prompt. That means the sandbox is active.

---

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**What to expect:**
- A lot of downloading text. Wait for the blinking cursor to return.
- Key packages: **langgraph** (agent workflow engine), **langchain-openai** (AI model connector), **ciagent** (the AgentCI package), **pytest** (test runner).

---

## Step 5: Set Your OpenAI API Key

The RAG agent calls OpenAI to embed the knowledge base documents and answer questions. Set your key for this terminal session:

```bash
export OPENAI_API_KEY=sk-proj-...
```

*(This only lasts for the current terminal session — it is not saved anywhere on your computer.)*

---

## Part A: Testing with pytest

This approach uses Python code to capture a "trace" of everything the agent did, then asserts on it directly.

---

### Step A1: Run the Pytest Test Suite

```bash
pytest tests/test_rag.py -v
```

**What to expect (16 tests passing):**
```
tests/test_rag.py::test_retrieval_triggered_for_knowledge_question PASSED
tests/test_rag.py::test_no_retrieval_for_greeting PASSED
tests/test_rag.py::test_cost_within_budget PASSED
tests/test_rag.py::test_grading_step_exists PASSED
tests/test_rag.py::test_relevant_docs_pass_grading PASSED
tests/test_rag.py::test_out_of_scope_skips_retrieval PASSED
tests/test_rag.py::test_rewrite_triggered_for_vague_query PASSED
tests/test_rag.py::test_no_rewrite_for_clear_query PASSED
tests/test_rag.py::test_max_retries PASSED
tests/test_rag.py::test_execution_path_with_rewrite PASSED
tests/test_rag.py::test_compound_query_decomposed_within_budget PASSED
tests/test_rag.py::test_mixed_intent_not_decomposed PASSED
... (16 total) ...
```

---

### Step A2: What Are Those Tests Actually Checking?

Each test captures the agent's trace — the full record of what it did — and asserts on it in plain Python.

**Setting up a reusable `run_agent()` helper (v0.3 clean API):**
```python
from agentci.capture import langgraph_trace
from agent import generate_answer_api

def run_agent(question: str):
    with langgraph_trace("rag-agent") as ctx:
        output, state = generate_answer_api(question)
        ctx.attach(state)
        ctx.trace.metadata["final_output"] = str(output)
    return ctx.trace
```

**Did the agent search the knowledge base for a real question?**
```python
trace = run_agent("How do I install AgentCI?")
assert trace.called("retrieve_docs")
```

**Did it skip searching for a greeting?**
```python
trace = run_agent("Hello, how are you?")
assert trace.never_called("retrieve_docs")
```

**Did it stay under budget?**
```python
assert trace.cost_under(0.01)
```

**Did it refuse off-topic questions without searching?**
```python
trace = run_agent("What's the weather in Austin?")
assert trace.never_called("retrieve_docs")
```

**Did the rewrite loop stay bounded?**
```python
assert trace.loop_count("rewrite_question") <= 3
assert trace.cost_under(0.05)
```

**Trace assertion methods available on every `Trace` object:**

| Method | Description |
|---|---|
| `trace.called("tool_name")` | True if tool appears anywhere in the trace |
| `trace.never_called("tool_name")` | True if tool was never called |
| `trace.loop_count("tool_name")` | How many times the tool was called |
| `trace.cost_under(threshold_usd)` | True if total cost < threshold |
| `trace.llm_calls_under(count)` | True if total LLM calls < count |

---

### Step A3: Save a Golden Baseline

A "golden baseline" is a snapshot of every trace for every test query. Future runs compare against this snapshot and flag anything that *changed* — even if the change looks fine on the surface.

```bash
python save_baseline.py
```

This saves traces to `golden/rag-v1-gpt4o-mini.json`. Now run the tests again — the regression test at the bottom will also pass:

```bash
pytest tests/test_rag.py -v
```

---

### Step A4: Watch It Catch a Regression!

Let's break the agent and see the baseline catch it.

**1. Open `agent.py` in a text editor:**
```bash
code agent.py
```

**2. Use Find (Cmd + F) and search for this exact text:**
```
You are an AgentCI documentation assistant. You help users
```

You will find it inside the `generate_query_or_respond` function — this is the function that decides whether to search the knowledge base at all.

**3. Change that entire system message to:**
```
You are a helpful AI assistant. Answer all questions from your own knowledge. Do not use any tools.
```

**4. Save the file, then run the tests:**
```bash
pytest tests/test_rag.py -v
```

**What to expect — failures in red:**
```
FAILED tests/test_rag.py::test_retrieval_triggered_for_knowledge_question
    AssertionError: 'retrieve_docs' not in trace.tools_called

FAILED tests/test_rag.py::test_grading_step_exists
    AssertionError: 'grade_artifacts' not in trace.tools_called

FAILED tests/test_rag.py::test_regression_against_baseline
    Regression detected: Tool set changed: -{'retrieve_docs'}, LLM calls decreased: 3 → 1
```

That last failure is the **baseline diff engine** — it noticed the agent's behavior *changed*, without you having to spell out what to check. One sentence in a prompt silently killed the entire retrieval pipeline.

**5. Restore the original system message before continuing.**

---

## Part B: Testing with `agentci test`

This approach uses a YAML file (`agentci_spec.yaml`) instead of Python. AgentCI reads it and runs its three-layer evaluation engine automatically — no code required.

---

### Step B1: Look at the Spec File

Open `agentci_spec.yaml`. It defines 15 test cases in plain YAML. Here is one entry:

```yaml
- query: How do I install AgentCI?
  correctness:
    expected_in_answer:
    - pip install ciagent          # Layer 1: answer must contain this
  path:
    expected_tools:
    - retrieve_docs                # Layer 2: retriever must be called
    max_tool_calls: 5              # Layer 2: but not excessively
  cost:
    max_llm_calls: 5               # Layer 3: budget guard
```

No Python. AgentCI evaluates all three layers for every query automatically:
- **Layer 1 — Correctness:** did the answer contain the right keywords? Was it grounded (LLM judge)?
- **Layer 2 — Path:** which tools were called? Any forbidden tools? Max calls exceeded?
- **Layer 3 — Cost:** how many LLM calls? How many tokens? Did cost spike vs baseline?

---

### Step B2: Run the Spec-Based Evaluation

```bash
agentci test
```

*(By default this looks for `agentci_spec.yaml` in the current folder.)*

**What to expect:**
```
AgentCI Evaluation — rag-agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ How do I install AgentCI?              PASS
✅ What are the three evaluation layers?  PASS
✅ What's the CEO's favorite restaurant?  PASS  (0 tool calls — correct decline)
✅ Hello!                                 PASS
✅ What's the weather in Austin?          PASS
✅ How do I fail the CI pipeline...       PASS
...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
15/15 passed
```

---

### Step B3: Run Only Smoke Tests

The spec tags queries by category. To run only the core happy-path queries:

```bash
agentci test --tags smoke
```

Useful for a fast check before committing code.

---

### Step B4: Watch `agentci test` Catch the Same Broken Agent

Make the same change as Step A4 — change `generate_query_or_respond`'s system message to answer from its own knowledge. Then run:

```bash
agentci test
```

**What to expect:**
```
❌ How do I install AgentCI?
   PATH FAIL — expected_tools not called: retrieve_docs
   CORRECTNESS FAIL — 'pip install' not found in answer

❌ What are the three evaluation layers in AgentCI?
   PATH FAIL — expected_tools not called: retrieve_docs
   CORRECTNESS FAIL — 'correctness', 'path', 'cost' not found in answer

✅ What's the CEO's favorite restaurant?   PASS  (still correctly declines)
✅ Hello!                                  PASS
...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2/15 passed  |  13 failed
```

Notice: out-of-scope tests still pass (the agent still correctly refuses off-topic questions), but every in-scope question fails because retrieval was never triggered.

**Restore the original system message.**

---

## Part C: When to Use Each

| You want to… | Use |
|---|---|
| Catch drift from a known-good baseline | `pytest tests/test_rag.py` |
| Define expected behavior without writing Python | `agentci test` |
| Check that specific tools were (or weren't) called | Both |
| Use an LLM judge to evaluate answer quality | `agentci test` (built-in `llm_judge`) |
| Run in CI on every pull request | `agentci test` (returns exit code 0 or 1) |
| Debug one specific failing test | `pytest tests/test_rag.py -k test_name` |
| Run only smoke tests quickly | `agentci test --tags smoke` |

---

🎉 **You're Done!**

You downloaded a real AI agent, ran 16 trace-level tests with pytest, saved a golden baseline that detected behavioral drift, then ran AgentCI's native three-layer YAML-driven evaluation — and watched both approaches catch the same intentional regression from different angles.
